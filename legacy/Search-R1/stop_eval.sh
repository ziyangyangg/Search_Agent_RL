#!/bin/bash
# ============================================================================
# stop_eval.sh — 停止正在跑的完整评估（evaluate.py，对应 eval_full.sh）
# 用法: bash /root/Search-R1/stop_eval.sh
# 说明: eval 是 nohup 后台进程；本脚本按 pid 文件 + 进程名双重兜底杀干净，并确认 GPU 释放。
#       ⚠️ evaluate.py 只在跑完最后才写 json，半途停 = 该次结果作废（要重跑）。
# ============================================================================
echo "[stop_eval] 停止 eval(evaluate.py) ..."
PID=$(cat /tmp/eval_full.pid 2>/dev/null)
if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
  kill -9 "$PID" 2>/dev/null && echo "[stop_eval] 已按 pid 杀 PID=$PID"
else
  echo "[stop_eval] /tmp/eval_full.pid 无效或进程已停，转用进程名兜底"
fi
# 兜底：按命令名杀所有 evaluate.py（覆盖手动启动、改了 pid 文件等情况）
pkill -9 -f "src/evaluate.py" 2>/dev/null
sleep 2
if pgrep -f "src/evaluate.py" >/dev/null; then
  echo "[stop_eval] ⚠️ 仍有残留，强杀一次"; pkill -9 -f "evaluate.py" 2>/dev/null; sleep 1
fi
if pgrep -f "src/evaluate.py" >/dev/null; then
  echo "[stop_eval] ❌ 仍有 evaluate.py 进程: $(pgrep -af 'src/evaluate.py' | head -1)"
else
  echo "[stop_eval] ✅ eval 已完全停止"
fi
echo "[stop_eval] GPU 显存: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
