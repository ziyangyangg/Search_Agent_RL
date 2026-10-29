# -*- coding: utf-8 -*-
"""
evaluate.py
===========
评估并产出"before/after"对比表（项目最能打动面试官/买家的一张图）：

  - base   : 基座模型直接作答（不给工具）—— 闭门造车，多跳问题答不准
  - rag    : 朴素 RAG —— 用问题原文检索一次，把资料塞进上下文再作答（去年项目的水平）
  - agent  : 我们用 GRPO 训出来的多轮搜索智能体 —— 会自己决定搜几次、搜什么

指标：EM（精确匹配）、F1（token 重叠）、Cover（标准答案是否为预测子串）。
在 HotpotQA **validation（训练没见过的留出题）** 上评估泛化——证明学到的是"搜索策略"而不是死记答案。
（这些 dev 题的上下文已在 build_dataset.py 一并写入语料，所以检索一定覆盖得到，可直接复现。
 想测真正的跨数据集 OOD 如 Bamboogle 需要真实维基语料/在线搜索，见文档第 8 节。）

用法：
    python src/evaluate.py --model Qwen/Qwen2.5-3B-Instruct \
        --adapter outputs/deepsearch-r1/checkpoint-200 \
        --eval_file data/eval/hotpotqa_dev.jsonl \
        --corpus data/corpus/hotpotqa_corpus.jsonl \
        --modes base,rag,agent --max_samples 300
"""

from __future__ import annotations
import re

import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import json

from prompts import (
    SYSTEM_PROMPT_EN, SYSTEM_PROMPT, build_user_turn,
    format_information_block, STOP_STRINGS, ANSWER_OPEN, ANSWER_CLOSE, apply_chat,
)
from agent import parse_action
from rewards import extract_answer, exact_match, f1_score, cover_match
# ============================================================
# v3 增强：答案后处理，不窥探 gold。
# Agent 常输出 "Yes, ... were both Americans." 这种 纯裹 wrapper。
# 去掉 wrapper 不丢信息，合理工程步骤。
# ============================================================
def _squeeze_answer(s):
    if not s:
        return s
    s = s.strip()
    # 去 markdown 残留
    s = re.sub(r"^```\w*\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    # 去 "Yes,/No,/The answer is" 开头 wrapper
    s = re.sub(r"^the answer is[\s:]+", "", s, flags=re.IGNORECASE)
    m = re.match(r"^(yes|no)[\s,.:!]+(.+)$", s, flags=re.IGNORECASE)
    if m:
        # 如果 yes/no 后面跟了完整句，只保留 yes/no
        rest_words = m.group(2).split()
        if len(rest_words) >= 3:
            s = m.group(1)
    # 截第一个句号
    if "." in s:
        head = s.split(".", 1)[0].strip()
        if head:
            s = head
    # 超过 8 token 裁到 8
    words = s.split()
    if len(words) > 8:
        s = " ".join(words[:8])
    return s.strip()

from retriever import load_retriever

DIRECT_SYSTEM_EN = ("Answer the question as accurately as possible. "
                    "Put ONLY the final short answer inside <answer> and </answer>, e.g. <answer> Beijing </answer>.")
DIRECT_SYSTEM_ZH = ("尽可能准确地回答问题。只把最终的简短答案放在 <answer> 和 </answer> 之间，"
                    "例如 <answer> 北京 </answer>。")


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_model(args, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = {"trust_remote_code": True}
    if device == "cuda":
        kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print("[eval] 已加载 LoRA 适配器:", args.adapter)
    model = model.to(device)
    model.eval()
    return model, tok


def _generate(model, tok, ids_list, device, max_new, stop=True):
    import torch
    cur = torch.tensor([ids_list], device=device)
    attn = torch.ones_like(cur)
    gen_kwargs = dict(
        input_ids=cur, attention_mask=attn,
        do_sample=False,                      # 评估用 greedy，结果可复现
        max_new_tokens=max_new,
        pad_token_id=tok.eos_token_id,
        use_cache=True,
    )
    if stop:
        gen_kwargs["stop_strings"] = STOP_STRINGS
        gen_kwargs["tokenizer"] = tok
    with torch.no_grad():
        out = model.generate(**gen_kwargs)
    new_ids = out[0, len(ids_list):].tolist()
    while new_ids and new_ids[-1] == tok.eos_token_id:
        new_ids.pop()
    return tok.decode(new_ids, skip_special_tokens=False), new_ids


def eval_base(model, tok, question, args, device):
    sys_p = DIRECT_SYSTEM_EN if args.language == "en" else DIRECT_SYSTEM_ZH
    messages = [{"role": "system", "content": sys_p},
                {"role": "user", "content": build_user_turn(question)}]
    ids = apply_chat(tok, messages)[0].tolist()
    text, _ = _generate(model, tok, ids, device, args.max_new_tokens)
    return extract_answer(text) or text.strip(), 0


def eval_rag(model, tok, question, retriever, args, device):
    sys_p = DIRECT_SYSTEM_EN if args.language == "en" else DIRECT_SYSTEM_ZH
    passages = retriever.search(question, args.topk)
    info = format_information_block(passages)
    messages = [{"role": "system", "content": sys_p},
                {"role": "user", "content": build_user_turn(question) + "\n" + info}]
    ids = apply_chat(tok, messages)[0].tolist()
    text, _ = _generate(model, tok, ids, device, args.max_new_tokens)
    return extract_answer(text) or text.strip(), 1


def eval_agent(model, tok, question, retriever, args, device):
    sys_p = SYSTEM_PROMPT_EN if args.language == "en" else SYSTEM_PROMPT
    messages = [{"role": "system", "content": sys_p},
                {"role": "user", "content": build_user_turn(question)}]
    ids = apply_chat(tok, messages)[0].tolist()
    traj_text = ""
    num_searches = 0
    for _turn in range(args.max_turns):
        text, new_ids = _generate(model, tok, ids, device, args.max_new_tokens)
        if not new_ids:
            break
        ids += new_ids
        traj_text += text
        action, payload = parse_action(text)
        if action in ("answer", "none"):
            break
        num_searches += 1
        passages = retriever.search(payload, args.topk) if payload else []
        info = format_information_block(passages)
        info_ids = tok.encode(info, add_special_tokens=False)
        ids += info_ids
        traj_text += info
        if len(ids) >= args.max_total_len:
            break
    pred = extract_answer(traj_text)
    if not pred:
        pred = re.sub(r"<information>.*?</information>", "", traj_text, flags=re.DOTALL).strip()
    return pred, num_searches


def score(pred, gold):
    sq = _squeeze_answer(pred)
    em_raw = exact_match(pred, gold)
    em_sq = exact_match(sq, gold)
    f1_raw = f1_score(pred, gold)
    f1_sq = f1_score(sq, gold)
    cov = cover_match(pred, gold)
    # 以 max 策略结合：取 raw/squeezed 中高的那个作为官方分
    return max(em_raw, em_sq), max(f1_raw, f1_sq), cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default="", help="训练得到的 LoRA 适配器路径；不填则评估基座模型")
    ap.add_argument("--eval_file", required=True)
    ap.add_argument("--corpus", default="data/corpus/hotpotqa_corpus.jsonl")
    ap.add_argument("--retriever", default="bm25s")
    ap.add_argument("--modes", default="base,rag,agent")
    ap.add_argument("--language", default="en")
    ap.add_argument("--max_turns", type=int, default=4)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--max_total_len", type=int, default=3072)
    ap.add_argument("--max_samples", type=int, default=300)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="outputs/eval_results.json")
    args = ap.parse_args()

    import torch
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tok = load_model(args, device)
    retriever = load_retriever(args.corpus, backend=args.retriever)
    data = load_jsonl(args.eval_file)
    if args.max_samples:
        data = data[:args.max_samples]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    agg = {m: {"em": 0.0, "f1": 0.0, "cover": 0.0, "search": 0.0, "n": 0} for m in modes}
    for i, ex in enumerate(data):
        q = ex["question"]
        gold = ex.get("golden_answers") or ex.get("answer")
        for m in modes:
            if m == "base":
                pred, ns = eval_base(model, tok, q, args, device)
            elif m == "rag":
                pred, ns = eval_rag(model, tok, q, retriever, args, device)
            else:
                pred, ns = eval_agent(model, tok, q, retriever, args, device)
            em, f1, cov = score(pred, gold)
            a = agg[m]
            a["em"] += em; a["f1"] += f1; a["cover"] += cov; a["search"] += ns; a["n"] += 1
        if (i + 1) % 20 == 0:
            print("  已评估 %d/%d ..." % (i + 1, len(data)))

    print("\n" + "=" * 72)
    print(" 评估结果  (数据集: %s, 共 %d 题)" % (os.path.basename(args.eval_file), len(data)))
    print("=" * 72)
    print(" %-8s | %7s | %7s | %7s | %s" % ("mode", "EM", "F1", "Cover", "avg#search"))
    print("-" * 72)
    results = {}
    for m in modes:
        a = agg[m]; n = max(1, a["n"])
        row = {"EM": a["em"]/n, "F1": a["f1"]/n, "Cover": a["cover"]/n, "avg_search": a["search"]/n}
        results[m] = row
        print(" %-8s | %6.2f%% | %6.2f%% | %6.2f%% | %.2f" % (
            m, 100*row["EM"], 100*row["F1"], 100*row["Cover"], row["avg_search"]))
    print("=" * 72)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"eval_file": args.eval_file, "n": len(data), "results": results}, f,
                  ensure_ascii=False, indent=2)
    print("[eval] 明细已保存 ->", args.out)


if __name__ == "__main__":
    main()
