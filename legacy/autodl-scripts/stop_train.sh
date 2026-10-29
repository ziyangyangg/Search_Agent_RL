#!/bin/bash
# 终止训练（只杀训练，保留 BM25 retrieval server）
# 用法: bash /root/Search-R1/stop_train.sh
echo "[stop_train] 杀 verl 训练进程 + ray worker ..."
pkill -9 -f "verl.trainer.main_ppo" 2>/dev/null
pkill -9 -f "ray::WorkerDict" 2>/dev/null
pkill -9 -f "ray::main_task" 2>/dev/null
sleep 2
/root/miniconda3/envs/verl_env/bin/ray stop --force >/dev/null 2>&1
sleep 1
if pgrep -f "verl.trainer.main_ppo" >/dev/null; then
  echo "[stop_train] 还有残留，强杀"; pkill -9 -f "main_ppo" 2>/dev/null; sleep 1
fi
echo "[stop_train] 训练进程: $(pgrep -f verl.trainer.main_ppo | head -1 || echo 已清空)"
echo "[stop_train] BM25 server (保留): $(pgrep -f retrieval_server.py | head -1 || echo 未运行)"
echo "[stop_train] GPU 显存: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null)"
