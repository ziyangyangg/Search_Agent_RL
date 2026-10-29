#!/bin/bash
# ============================================================================
# reuse_setup.sh — autoDL 镜像复用一键复原脚本（系统盘，随镜像保留）
# 场景：用保存的镜像新建机器后，数据盘是空的。本脚本把"数据层"复原到训练就绪。
# 用法：bash /root/Search-R1/reuse_setup.sh
# 复原内容：corpus(下载+解tar+normalize) + 模型(下载) + ~/.cache软链 + 起BM25
# 不需复原：conda环境/代码/脚本/BM25索引/parquet —— 这些在系统盘，随镜像已保留
# ============================================================================
set -e

echo "==================== [0/6] 前置检查（确认在镜像里、系统盘资产都在）===================="
# 这些是系统盘资产，镜像里必须已存在；缺了说明镜像没存好，照 doc04 重建
for must in \
  /root/miniconda3/envs/verl_env/bin/python \
  /root/miniconda3/envs/dsr1/bin/python \
  /root/Search-R1/data/nq_hotpotqa/train.parquet \
  /root/Search-R1/data/nq_hotpotqa/test.parquet \
  /root/Search-R1/data/bm25_index/bm25 \
  /root/Search-R1/train_grpo.sh \
  /root/Search-R1/train_smoke_1card.sh \
  /root/Search-R1/retrieval_launch_bm25.sh ; do
  if [ -e "$must" ]; then echo "  [OK ] $must"; else echo "  [缺!] $must  <- 镜像没存好，照 doc04 重建该项"; fi
done

echo "==================== [1/6] 重建数据盘目录 + ~/.cache 软链 ===================="
mkdir -p /root/autodl-tmp/data/corpus /root/autodl-tmp/hf_models /root/autodl-tmp/verl_checkpoints /root/autodl-tmp/.cache_root
# ~/.cache 软链到数据盘（HF datasets 会在这里生成 arrow cache，6G+，不能落 30G 系统盘）
if [ ! -L /root/.cache ]; then rm -rf /root/.cache; ln -s /root/autodl-tmp/.cache_root /root/.cache; fi
echo "  /root/.cache -> $(readlink /root/.cache)"
# 注意：/root/Search-R1/data/corpus 和 /root/Search-R1/hf_models 本身就是指向数据盘的软链（随镜像保留），
#       数据下到 /root/autodl-tmp/... 后，通过 Search-R1 路径自动可见，【不要】再在里面建软链（会嵌套）

echo "==================== [2/6] 下载 + 解tar + normalize wiki-18 corpus（约15min）===================="
cd /root/autodl-tmp/data/corpus
TARGET=/root/autodl-tmp/data/corpus/wiki-18.jsonl
if [ -f "$TARGET" ] && [ "$(wc -l < "$TARGET")" = "21015324" ]; then
  echo "  corpus 已存在且行数正确(21015324)，跳过"
else
  source /etc/network_turbo 2>/dev/null || true
  # (1) 下载 .gz（文件名是 .jsonl.gz 但内容实为 tar 归档，HF 上传时打的包）。
  # aria2c -x16 多线程：HF Xet CDN(cas-bridge.xethub.hf.co) 正常时 ~50-65MiB/s，4.8G 约 1-2min（实测）。
  # aria2c 自带 --max-tries=5 会对瞬时掉线自动重试。
  # 🔧 唯一加固（不猜任何时间/速度阈值、刻意不碰快路径）：套一层循环，
  #    【仅当】aria2c 非零退出且 .gz 仍未下完时，才 -c 断点续传重跑一次——防止某次瞬时网络错误把整脚本 set -e 杀掉。
  #    CDN 正常时第 1 次 ~2min 就下完，循环只跑一轮，行为与原始脚本完全一致。
  #    【不要】加 --lowest-speed-limit（实测它会在 aria2c 升速的头十几秒把"暂时<阈值"的连接全 abort，反让快下载升不上速拖崩）；
  #    【不要】加 timeout N（会误杀"正常但偏慢"的下载，阈值多少全是猜）。
  CORPUS_URL="https://hf-mirror.com/datasets/PeterJinGo/wiki-18-corpus/resolve/main/wiki-18.jsonl.gz"
  GZ=/root/autodl-tmp/data/corpus/wiki-18.jsonl.gz
  set +e
  dl_ok=0
  for attempt in $(seq 1 10); do
    aria2c -x16 -s16 -c --console-log-level=warn --summary-interval=15 --allow-overwrite=true \
      --max-tries=5 --retry-wait=3 --connect-timeout=15 \
      -d /root/autodl-tmp/data/corpus -o wiki-18.jsonl.gz "$CORPUS_URL"
    if [ -f "$GZ" ] && [ ! -f "$GZ.aria2" ]; then dl_ok=1; echo "  ✅ corpus .gz 下载完整（第 $attempt 次尝试）"; break; fi
    echo "  ⟳ aria2c 非零退出但 .gz 未下完，3s 后 -c 断点续传重跑（第 $attempt 次）..."
    sleep 3
  done
  set -e
  [ "$dl_ok" = 1 ] || { echo "  [致命] corpus 多次重试仍未下完，手动 aria2c -c 续传或换时段重跑 reuse_setup.sh"; exit 1; }
  # (2) gunzip 出 tar
  gunzip -f wiki-18.jsonl.gz && mv wiki-18.jsonl wiki-18.tar
  # (3) tar 解出真 jsonl
  tar -xOf wiki-18.tar > wiki-18.raw.jsonl && rm -f wiki-18.tar
  # (4) ⚠️ normalize：hf-mirror 版 escape 风格 pyarrow.read_json 不接受，必须 stdlib json 规范化（见 doc08 E21）
  /root/miniconda3/envs/verl_env/bin/python - <<'PYEOF'
import json
with open("wiki-18.raw.jsonl") as fin, open("wiki-18.jsonl","w") as fout:
    for line in fin:
        fout.write(json.dumps(json.loads(line), ensure_ascii=False) + "\n")
print("normalize done")
PYEOF
  rm -f wiki-18.raw.jsonl
fi
echo "  corpus: $(wc -l < $TARGET) 行 (应 21015324)"
echo "  通过 Search-R1 路径可见性: $(ls -lL /root/Search-R1/data/corpus/wiki-18.jsonl | awk '{print $5}') bytes"

echo "==================== [3/6] 下载 Qwen2.5-3B-Instruct（魔搭ModelScope+aria2c，约1min）===================="
# ⚠️ 不再用 huggingface-cli：HF 海外 CDN 慢节点会卡死（实测 model-00001 下到 50M 僵死、进程不退）。
# 改用魔搭(国内源) + aria2c -x16 多线程：实测 ~200MB/s，整模型 6G 约 1min。详见 download_model_modelscope.sh。
MODEL_DIR=/root/autodl-tmp/hf_models/Qwen2.5-3B-Instruct
S1=$(stat -c%s "$MODEL_DIR/model-00001-of-00002.safetensors" 2>/dev/null || echo 0)
S2=$(stat -c%s "$MODEL_DIR/model-00002-of-00002.safetensors" 2>/dev/null || echo 0)
# 校验两个分片字节数都达标 + index 存在才跳过（旧版只看 00002 存在 → 00001 残缺也误判跳过）
if [ "$S1" = "3968658944" ] && [ "$S2" = "2203268048" ] && [ -f "$MODEL_DIR/model.safetensors.index.json" ]; then
  echo "  模型两分片+index已完整(字节校验通过)，跳过"
else
  echo "  模型缺失/不完整(00001=$S1 00002=$S2)，用魔搭源+aria2c下载..."
  bash /root/Search-R1/download_model_modelscope.sh
fi
echo "  模型通过 Search-R1 路径可见: $(ls /root/Search-R1/hf_models/Qwen2.5-3B-Instruct/*.safetensors 2>&1 | wc -l) 个 safetensors"

echo "==================== [4/6] 起 BM25 retrieval server（约70s）===================="
pkill -f retrieval_server.py 2>/dev/null || true
sleep 2
> /tmp/retrieval_server.log
nohup bash /root/Search-R1/retrieval_launch_bm25.sh > /tmp/retrieval_server.log 2>&1 < /dev/null &
echo "  BM25 启动中，poll 就绪信号..."
for t in $(seq 1 30); do
  sleep 10
  if grep -q "Uvicorn running" /tmp/retrieval_server.log 2>/dev/null; then echo "  BM25 READY"; break; fi
  if grep -qE "Traceback|RuntimeError|Error" /tmp/retrieval_server.log 2>/dev/null; then
    echo "  BM25 启动失败，看日志："; tail -20 /tmp/retrieval_server.log; exit 1; fi
  echo "    ...等待中 ($((t*10))s)"
done

echo "==================== [5/6] BM25 冒烟自检（curl Marie Curie）===================="
curl -s -X POST http://127.0.0.1:8000/retrieve -H 'Content-Type: application/json' \
  -d '{"queries":["Marie Curie"],"topk":3,"return_scores":true}' | head -c 300
echo

echo "==================== [6/6] 完成。下一步训练 ===================="
echo "  单卡验证(推荐先跑，3步看 advantages/pg_loss 是否健康):"
echo "    cd /root/Search-R1 && nohup bash train_smoke_1card.sh > /tmp/train_smoke.log 2>&1 &"
echo "    然后: grep -E 'step:1|advantages|pg_loss' /tmp/train_smoke.log"
echo "  多卡真实复现 44.80%(需 >=7 卡):"
echo "    cd /root/Search-R1 && nohup bash train_grpo.sh > /tmp/verl.log 2>&1 &"
echo "  评估(完整 500 题 base/rag/agent，eval 已在镜像 /root/Search-R1/eval/，无需 scp；单卡 base 模型 ~2-2.5h):"
echo "    bash /root/Search-R1/eval/eval_full.sh           # 默认 base模型+500题+base,rag,agent；评ckpt: eval_full.sh <ckpt路径>"
echo "    看进度: grep '已评估' /tmp/eval_full_500.log | tail -1     看结果: cat /tmp/eval_full_500.json"
