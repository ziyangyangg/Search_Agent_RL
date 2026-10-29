#!/bin/bash
# 一眼看全状态: 训练/检索server/GPU/最新step指标
# 用法: bash /root/Search-R1/status.sh
echo "========== 状态总览 $(date '+%H:%M:%S') =========="
echo "--- 训练进程 ---"
pgrep -fa "verl.trainer.main_ppo" | head -1 || echo "  (无训练在跑)"
echo "--- BM25 retrieval server (port 8000) ---"
if pgrep -f retrieval_server.py >/dev/null; then
  echo "  up (PID $(pgrep -f retrieval_server.py | head -1))"
else
  echo "  DOWN —— 训练前必须先起: nohup bash /root/Search-R1/retrieval_launch_bm25.sh > /tmp/retrieval_server.log 2>&1 &"
fi
echo "--- GPU ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "  (无 GPU / 无卡模式)"
echo "--- CPU 内存 (容器限 120G) ---"
free -g | awk '/Mem:/{print "  used "$3"G / total "$2"G"}'
echo "--- 最新训练 step 指标 ---"
if [ -f /tmp/train_smoke.log ]; then
  grep -E "step:[0-9]+ - " /tmp/train_smoke.log | tail -1 | grep -oE "step:[0-9]+|critic/score/mean:[0-9.]+|critic/rewards/mean:[0-9.-]+|critic/advantages/max:[0-9.]+|actor/grad_norm:[0-9.]+|env/finish_ratio:[0-9.]+|timing_s/step:[0-9.]+" | tr '\n' ' '
  echo
else
  echo "  (还没有 /tmp/train_smoke.log)"
fi
