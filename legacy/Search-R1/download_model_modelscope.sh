#!/bin/bash
# 魔搭(ModelScope国内源) + aria2c 多线程 下载 Qwen2.5-3B-Instruct 全部文件
# autoDL复现专用: HF海外CDN慢节点会卡死; 魔搭单连接限速~1MB/s; aria2c -x16 多线程拉到~200MB/s
set -e
MODEL_DIR=/root/autodl-tmp/hf_models/Qwen2.5-3B-Instruct
MS_BASE="https://modelscope.cn/models/Qwen/Qwen2.5-3B-Instruct/resolve/master"
mkdir -p "$MODEL_DIR"; cd "$MODEL_DIR"
for f in config.json generation_config.json merges.txt \
         model-00001-of-00002.safetensors model-00002-of-00002.safetensors \
         model.safetensors.index.json tokenizer.json tokenizer_config.json vocab.json \
         LICENSE README.md; do
  aria2c -x16 -s16 -c --console-log-level=warn --summary-interval=0 --allow-overwrite=true \
    -d "$MODEL_DIR" -o "$f" "$MS_BASE/$f" >/dev/null 2>&1 || { echo "FAILED: $f"; exit 1; }
  echo "  ok: $f = $(stat -c%s "$f") bytes"
done
echo "ALL_MODEL_FILES_DONE"
