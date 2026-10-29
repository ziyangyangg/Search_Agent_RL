# -*- coding: utf-8 -*-
"""
build_dataset.py
================
一步到位地准备「问答数据」和「检索语料」，并**保证两者对齐**（这是本项目能训得动的关键）。

为什么要一个脚本同时产出 QA 和语料？
  本项目用 HotpotQA 的 distractor 设定：每个问题自带 10 段上下文(含 2 段金标 + 8 段干扰)。
  我们必须保证——**你用来训练/评估的那些问题，它们的金标段落确实在语料里**，否则智能体怎么搜都搜不到，
  奖励永远是 0、根本学不动。把"取问题"和"取该问题的上下文入库"放在同一次遍历里，就从根本上杜绝了不对齐。

数据来源：`hotpotqa/hotpot_qa` 的 `distractor` 配置（已是 Parquet，**不要传 trust_remote_code**，
否则在 datasets>=4.0 会报错）。每个样本：question / answer / context{title:[...], sentences:[[...]]}。

输出：
  QA jsonl   每行 {"id","question","golden_answers":[...]}
  语料 jsonl  每行 {"id","contents"}（retriever.py 直接可读）

用法（脚本 scripts/01_prepare_data.sh 已串好）：
  # 训练集：取 train 前 4000 题，QA 写 train 文件，上下文写语料(覆盖)
  python src/build_dataset.py --split train --max_samples 4000 \
      --qa_out data/train/hotpotqa_train.jsonl \
      --corpus_out data/corpus/hotpotqa_corpus.jsonl --corpus_mode overwrite
  # 验证集：取 validation 前 500 题，QA 写 dev 文件，上下文**追加**进同一个语料
  python src/build_dataset.py --split validation --max_samples 500 \
      --qa_out data/eval/hotpotqa_dev.jsonl \
      --corpus_out data/corpus/hotpotqa_corpus.jsonl --corpus_mode append
  # 结果：语料 = train前4000题 ∪ dev前500题 的所有上下文段落 → 训练和评估的问题全部被覆盖。
"""

from __future__ import annotations

import os

# 国内镜像；必须在 import datasets 之前
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import hashlib
import json


def _hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", help="train | validation")
    ap.add_argument("--max_samples", type=int, default=4000, help="取该 split 的前 N 题，0=全部")
    ap.add_argument("--qa_out", required=True)
    ap.add_argument("--corpus_out", required=True)
    ap.add_argument("--corpus_mode", default="overwrite", choices=["overwrite", "append"])
    args = ap.parse_args()

    from datasets import load_dataset
    # 关键：不要传 trust_remote_code（HotpotQA 已是 Parquet；该参数在 datasets>=4.0 会报错）
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=args.split)

    os.makedirs(os.path.dirname(args.qa_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.corpus_out), exist_ok=True)

    # 语料去重：append 模式下先把已有语料的 id 读进来，避免重复写
    seen = set()
    if args.corpus_mode == "append" and os.path.exists(args.corpus_out):
        with open(args.corpus_out, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        seen.add(json.loads(line)["id"])
                    except Exception:  # noqa
                        pass

    qa_f = open(args.qa_out, "w", encoding="utf-8")
    corpus_f = open(args.corpus_out, "a" if args.corpus_mode == "append" else "w", encoding="utf-8")

    n_qa = 0
    n_corpus_new = 0
    for i, ex in enumerate(ds):
        if args.max_samples and n_qa >= args.max_samples:
            break
        question = (ex.get("question") or "").strip()
        answer = (ex.get("answer") or "").strip()
        if not question or not answer:
            continue
        qid = str(ex.get("id") or ex.get("_id") or ("hotpot-%s-%d" % (args.split, i)))

        # 1) 写 QA
        qa_f.write(json.dumps({"id": qid, "question": question, "golden_answers": [answer]},
                              ensure_ascii=False) + "\n")
        n_qa += 1

        # 2) 把该题的所有上下文段落写进语料（保证该题的金标段落一定在库里）
        ctx = ex.get("context") or {}
        titles = ctx.get("title") or []
        sentences = ctx.get("sentences") or []
        for title, sents in zip(titles, sentences):
            passage = (str(title).strip() + ": " + " ".join(s.strip() for s in sents)).strip()
            if len(passage) < 5:
                continue
            h = _hash(passage)
            if h in seen:
                continue
            seen.add(h)
            corpus_f.write(json.dumps({"id": h, "contents": passage}, ensure_ascii=False) + "\n")
            n_corpus_new += 1

    qa_f.close()
    corpus_f.close()
    print("[build_dataset] split=%s  写入 QA %d 条 -> %s" % (args.split, n_qa, args.qa_out))
    print("[build_dataset] 新增语料段落 %d 条 (%s) -> %s" % (n_corpus_new, args.corpus_mode, args.corpus_out))
    print("[build_dataset] 语料当前去重总段落数 ≈ %d" % len(seen))


if __name__ == "__main__":
    main()
