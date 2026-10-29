# -*- coding: utf-8 -*-
"""
train_grpo_agent.py
===================
用 GRPO + 可验证奖励(RLVR) 训练「多轮搜索智能体」的主入口。

为什么是这套自包含实现，而不是直接调 TRL/verl？
  - 透明、可复现：整条链路(采样→奖励→优势→带掩码的策略梯度)都在眼前，便于学习与排错，
    也是简历/面试里最值钱的部分——你能讲清楚 GRPO 每一步在干什么。
  - 版本稳健：只依赖 transformers + peft + torch 这些稳定 API，不被 TRL/vLLM 的版本矩阵牵着走。
  - 单卡友好：用 LoRA 把可训练参数压到几百万级；用「策略模型自己生成」做 rollout，
    天然 on-policy、无需把权重同步给外部采样器。
（追求更快/更大规模时，文档里给出了换成 TRL GRPOTrainer + vLLM 或 verl 的方案。）

训练循环（每个 step）：
  1. 取 B 个问题，每个问题用当前策略采样 G 条多轮轨迹(think/search/information/answer)；
  2. 对每条轨迹用 RLVR 打分(F1/EM + 格式 + 搜索)；
  3. 组内归一化得到优势 A_i = (r_i - mean)/(std+eps)；
  4. 一次前向算每个 token 的 logπ，按掩码(只在模型生成的 token 上)做 GRPO 策略梯度并反传；
  5. 更新 LoRA 参数。
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import random
import time

from prompts import build_chat_messages, format_information_block, STOP_STRINGS, apply_chat
from agent import parse_action
from rewards import compute_reward
from grpo import group_normalized_advantages, masked_grpo_loss
from retriever import load_retriever


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #
def load_policy(args, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"trust_remote_code": True}
    if args.bf16 and device == "cuda":
        model_kwargs["torch_dtype"] = torch.bfloat16
    if args.four_bit and device == "cuda":
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    if args.four_bit and device == "cuda":
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    if args.resume_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.resume_adapter, is_trainable=True)
        print(f"[train] warm-start from adapter: {args.resume_adapter}")
    else:
        model = get_peft_model(model, lora)
    if device == "cuda" and not args.four_bit:
        model = model.to(device)
    elif device == "cpu":
        model = model.to(device)
    model.print_trainable_parameters()
    return model, tokenizer


# --------------------------------------------------------------------------- #
# Rollout：用策略模型自己生成一条多轮轨迹（token 级，便于精确掩码）
# --------------------------------------------------------------------------- #
def sample_trajectory(model, tokenizer, question, gold, retriever, args, device):
    import torch

    messages = build_chat_messages(question, language=args.language)
    input_ids = apply_chat(tokenizer, messages).to(device)

    ids_list = input_ids[0].tolist()
    mask_list = [0] * len(ids_list)            # prompt 不参与损失
    traj_text = ""
    num_searches = 0
    finished = False
    eos_id = tokenizer.eos_token_id

    for _turn in range(args.max_turns):
        cur = torch.tensor([ids_list], device=device)
        attn = torch.ones_like(cur)
        with torch.no_grad():
            out = model.generate(
                input_ids=cur,
                attention_mask=attn,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                stop_strings=STOP_STRINGS,
                tokenizer=tokenizer,
                pad_token_id=eos_id,
                use_cache=True,
            )
        new_ids = out[0, len(ids_list):].tolist()
        # 去掉末尾可能的 pad/eos，避免污染掩码
        while new_ids and new_ids[-1] == eos_id:
            new_ids.pop()
        if not new_ids:
            break

        new_text = tokenizer.decode(new_ids, skip_special_tokens=False)
        ids_list += new_ids
        mask_list += [1] * len(new_ids)        # 模型生成 → 参与损失
        traj_text += new_text

        action, payload = parse_action(new_text)
        if action == "answer":
            finished = True
            break
        if action == "none":
            break
        # action == "search"
        num_searches += 1
        passages = retriever.search(payload, args.topk) if payload else []
        info_block = format_information_block(passages)
        info_ids = tokenizer.encode(info_block, add_special_tokens=False)
        ids_list += info_ids
        mask_list += [0] * len(info_ids)       # 工具注入 → 掩码屏蔽
        traj_text += info_block

        if len(ids_list) >= args.max_total_len:
            break

    reward, info = compute_reward(
        traj_text, gold, num_searches,
        answer_metric=args.answer_metric,
        w_answer=args.w_answer, w_format=args.w_format, w_search=args.w_search,
            length_penalty_after=args.length_penalty_after,
            length_penalty_per_word=args.length_penalty_per_word,
            no_search_penalty=args.no_search_penalty,
    )
    return {
        "ids": ids_list,
        "mask": mask_list,
        "reward": reward,
        "num_searches": num_searches,
        "finished": finished,
        "answer_score": info["answer_score"],
    }


# --------------------------------------------------------------------------- #
# 一个 micro-batch 的 GRPO 损失 + 反传
# --------------------------------------------------------------------------- #
def loss_on_microbatch(model, trajs, advs, tokenizer, args, device, scale):
    import torch

    maxlen = max(len(t["ids"]) for t in trajs)
    pad_id = tokenizer.pad_token_id
    B = len(trajs)
    input_ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((B, maxlen), dtype=torch.long)
    loss_mask = torch.zeros((B, maxlen), dtype=torch.long)
    for i, t in enumerate(trajs):
        L = len(t["ids"])
        input_ids[i, :L] = torch.tensor(t["ids"], dtype=torch.long)
        attn[i, :L] = 1
        loss_mask[i, :L] = torch.tensor(t["mask"], dtype=torch.long)
    input_ids = input_ids.to(device)
    attn = attn.to(device)
    loss_mask = loss_mask.to(device)
    advantages = torch.tensor(advs, dtype=torch.float32, device=device)

    out = model(input_ids=input_ids, attention_mask=attn)
    ref_logits = None
    if args.beta and args.beta > 0.0:
        # 「免费参考模型」技巧：关闭 LoRA 适配器即得到基座模型(参考)的 logits，无需第二份模型占显存
        with torch.no_grad():
            with model.disable_adapter():
                ref_logits = model(input_ids=input_ids, attention_mask=attn).logits

    loss, metrics = masked_grpo_loss(
        out.logits, input_ids, loss_mask, advantages,
        ref_logits=ref_logits, beta=args.beta,
    )
    (loss * scale).backward()
    return metrics


# --------------------------------------------------------------------------- #
# 主训练循环
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    # 模型/数据/检索
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--train_file", default="data/train/hotpotqa_train.jsonl")
    ap.add_argument("--corpus", default="data/corpus/hotpotqa_corpus.jsonl")
    ap.add_argument("--retriever", default="bm25s", help="simple | bm25s | dense")
    ap.add_argument("--output_dir", default="outputs/deepsearch-r1")
    ap.add_argument("--language", default="en")
    # GRPO 超参
    ap.add_argument("--group_size", type=int, default=5, help="每个问题采样多少条轨迹(G)")
    ap.add_argument("--prompts_per_step", type=int, default=4, help="每个 step 用多少个问题(B)")
    ap.add_argument("--micro_traj", type=int, default=2, help="损失前向的微批轨迹数(梯度累积)")
    ap.add_argument("--total_steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.0, help="KL 系数；0=不加载参考模型(省显存)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    # 轨迹/检索长度
    ap.add_argument("--max_turns", type=int, default=4)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--max_new_tokens", type=int, default=256, help="每回合最多生成多少 token")
    ap.add_argument("--max_total_len", type=int, default=3072, help="整条轨迹的 token 上限")
    # 奖励
    ap.add_argument("--answer_metric", default="f1", help="f1 | em | cover")
    ap.add_argument("--w_answer", type=float, default=1.0)
    ap.add_argument("--w_format", type=float, default=0.2)
    ap.add_argument("--w_search", type=float, default=0.1)
    # LoRA / 精度
    ap.add_argument("--resume_adapter", type=str, default=None, help="warm start from LoRA adapter path")
    ap.add_argument("--length_penalty_per_word", type=float, default=0.0, help="per-word penalty after threshold")
    ap.add_argument("--no_search_penalty", type=float, default=0.0, help="absolute penalty when num_searches==0")
    ap.add_argument("--length_penalty_after", type=int, default=5, help="word threshold for length penalty")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--four_bit", action="store_true", help="QLoRA 4bit，进一步省显存")
    # 杂项
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_steps", type=int, default=1)
    ap.add_argument("--save_steps", type=int, default=50)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import torch
    from transformers import set_seed
    set_seed(args.seed)
    random.seed(args.seed)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[train] device =", device)

    model, tokenizer = load_policy(args, device)
    retriever = load_retriever(args.corpus, backend=args.retriever)
    data = load_jsonl(args.train_file)
    print("[train] 训练样本 %d 条，语料来自 %s" % (len(data), args.corpus))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "train_log.jsonl")
    log_f = open(log_path, "a", encoding="utf-8")

    order = list(range(len(data)))
    random.shuffle(order)
    ptr = 0

    for step in range(1, args.total_steps + 1):
        t0 = time.time()
        model.eval()  # rollout 阶段不需要 dropout

        # 取 B 个问题
        batch_qs = []
        for _ in range(args.prompts_per_step):
            if ptr >= len(order):
                random.shuffle(order)
                ptr = 0
            batch_qs.append(data[order[ptr]])
            ptr += 1

        # 采样 B*G 条轨迹
        all_trajs = []
        all_rewards = []
        for ex in batch_qs:
            q = ex["question"]
            gold = ex.get("golden_answers") or ex.get("answer")
            for _g in range(args.group_size):
                tr = sample_trajectory(model, tokenizer, q, gold, retriever, args, device)
                all_trajs.append(tr)
                all_rewards.append(tr["reward"])

        # 组内优势
        advantages = group_normalized_advantages(all_rewards, args.group_size)

        # 更新（梯度累积）
        model.train()
        optimizer.zero_grad()
        idxs = list(range(len(all_trajs)))
        # 只在有非零优势的轨迹上更新（全零组没有梯度，跳过省算力）
        nonzero = [i for i in idxs if abs(advantages[i]) > 1e-8]
        agg = {"loss": 0.0, "kl": 0.0, "n_tokens": 0}
        if nonzero:
            chunks = [nonzero[i:i + args.micro_traj] for i in range(0, len(nonzero), args.micro_traj)]
            for ch in chunks:
                trajs = [all_trajs[i] for i in ch]
                advs = [advantages[i] for i in ch]
                m = loss_on_microbatch(model, trajs, advs, tokenizer, args, device,
                                       scale=1.0 / len(chunks))
                agg["loss"] += m["loss"] / len(chunks)
                agg["kl"] += m["kl"] / len(chunks)
                agg["n_tokens"] += m["n_tokens"]
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()

        # 统计
        mean_reward = sum(all_rewards) / len(all_rewards)
        mean_ans = sum(t["answer_score"] for t in all_trajs) / len(all_trajs)
        mean_search = sum(t["num_searches"] for t in all_trajs) / len(all_trajs)
        frac_finished = sum(1 for t in all_trajs if t["finished"]) / len(all_trajs)
        rec = {
            "step": step,
            "reward": round(mean_reward, 4),
            "answer_score": round(mean_ans, 4),
            "num_searches": round(mean_search, 3),
            "frac_finished": round(frac_finished, 3),
            "loss": round(agg["loss"], 4),
            "kl": round(agg["kl"], 5),
            "sec": round(time.time() - t0, 1),
        }
        if step % args.log_steps == 0:
            print("[step %d] reward=%.4f ans=%.4f search=%.2f finish=%.2f loss=%.4f (%.1fs)" % (
                step, mean_reward, mean_ans, mean_search, frac_finished, agg["loss"], rec["sec"]))
        log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log_f.flush()

        if step % args.save_steps == 0 or step == args.total_steps:
            save_dir = os.path.join(args.output_dir, "checkpoint-%d" % step)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print("[train] 已保存 LoRA 适配器 -> %s" % save_dir)

    log_f.close()
    print("[train] 训练完成。最终适配器在 %s" % args.output_dir)


if __name__ == "__main__":
    main()
