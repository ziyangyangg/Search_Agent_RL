# -*- coding: utf-8 -*-
"""
build_dataset_local.py
======================
build_dataset.py 的「断网适配版」（实验记录 坑4/坑6）：跳过 `datasets.load_dataset`
（hf-mirror/代理环境下会被 /api 的 308 重定向卡死），改用 pyarrow 直接读你已经 curl
下载好的 HotpotQA distractor parquet 分片，产出与 build_dataset.py **完全相同格式**的
QA jsonl + 语料 jsonl（对齐逻辑、去重逻辑也完全相同）。

先下载 3 个 parquet（共 ~344MB，约 1 分钟；文件名实测，404 过的不要再试 of-00001 的 train）：
  curl -L -o train-00000-of-00002.parquet 'https://hf-mirror.com/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/train-00000-of-00002.parquet'
  curl -L -o train-00001-of-00002.parquet 'https://hf-mirror.com/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/train-00001-of-00002.parquet'
  curl -L -o validation-00000-of-00001.parquet 'https://hf-mirror.com/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet'

用法（参数与 build_dataset.py 对齐，只多一个 --parquet_dir 指向 parquet 所在目录）：
  # 训练集：取 train 前 4000 题，QA 写 train 文件，上下文写语料(覆盖)
  python src/build_dataset_local.py --split train --max_samples 4000 --parquet_dir . \
      --qa_out data/train/hotpotqa_train.jsonl \
      --corpus_out data/corpus/hotpotqa_corpus.jsonl --corpus_mode overwrite
  # 验证集：取 validation 前 500 题，QA 写 dev 文件，上下文**追加**进同一个语料
  python src/build_dataset_local.py --split validation --max_samples 500 --parquet_dir . \
      --qa_out data/eval/hotpotqa_dev.jsonl \
      --corpus_out data/corpus/hotpotqa_corpus.jsonl --corpus_mode append
  # 期望产物与主路径相同：4000 / 500 / 41897 行（wc -l 验证）
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os


def _hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


def iter_rows(parquet_dir: str, split: str):
    """按分片顺序逐行 yield dict，行结构与 datasets.load_dataset 的样本一致
    （id / question / answer / context{title:[...], sentences:[[...]]}）。"""
    import pyarrow.parquet as pq

    pattern = os.path.join(parquet_dir, "%s-*.parquet" % split)
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(
            "[build_dataset_local] 找不到 %s ——先按本文件头部注释 curl 下载 parquet" % pattern
        )
    print("[build_dataset_local] 读取 %d 个分片: %s" % (len(files), ", ".join(files)))
    for fp in files:
        table = pq.read_table(fp)
        for batch in table.to_batches(max_chunksize=2048):
            for row in batch.to_pylist():
                yield row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", help="train | validation")
    ap.add_argument("--max_samples", type=int, default=4000, help="取该 split 的前 N 题，0=全部")
    ap.add_argument("--parquet_dir", default=".", help="curl 下载的 parquet 分片所在目录")
    ap.add_argument("--qa_out", required=True)
    ap.add_argument("--corpus_out", required=True)
    ap.add_argument("--corpus_mode", default="overwrite", choices=["overwrite", "append"])
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.qa_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.corpus_out), exist_ok=True)

    # 语料去重：append 模式下先把已有语料的 id 读进来，避免重复写（与 build_dataset.py 相同）
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
    for i, ex in enumerate(iter_rows(args.parquet_dir, args.split)):
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
    print("[build_dataset_local] split=%s  写入 QA %d 条 -> %s" % (args.split, n_qa, args.qa_out))
    print("[build_dataset_local] 新增语料段落 %d 条 (%s) -> %s" % (n_corpus_new, args.corpus_mode, args.corpus_out))
    print("[build_dataset_local] 语料当前去重总段落数 ≈ %d" % len(seen))


if __name__ == "__main__":
    main()
