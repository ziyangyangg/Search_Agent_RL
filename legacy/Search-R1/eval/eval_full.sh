#!/bin/bash
# ============================================================================
# eval_full.sh — 完整评估脚本（HotpotQA dev 500题，base/rag/agent 三档对照）
# 场景：在 hotpotqa 小语料(41897 doc)+bm25s 上跑完整 500 题，产出 before/after 对照表。
#       与本项目 eval/src 保持一致（同 evaluate.py + 同数据 + 同 prompt/metric）。
# ----------------------------------------------------------------------------
# 用法：
#   bash /root/Search-R1/eval/eval_full.sh                         # 默认: base模型, 500题, base,rag,agent
#   bash /root/Search-R1/eval/eval_full.sh <模型/ckpt路径>          # 评训练好的 ckpt
#   bash /root/Search-R1/eval/eval_full.sh <模型> <题数> <modes>    # 自定义
#   例: bash eval_full.sh /root/Search-R1/verl_checkpoints/xxx/actor 500 agent
# 评 LoRA 适配器: 在下方命令加 --adapter <LoRA路径>（base 权重用 --model）
# ----------------------------------------------------------------------------
# ⚠️ 耗时(autoDL 单卡 A800)：base 模型 agent 没学会早停 → 慢。实测 ~4 题/分钟：
#      base,rag,agent 500题 ≈ 2-2.5 小时；只 agent 也差不多(agent 是大头)；
#      只 base,rag ≈ 30-40 分钟；只 base ≈ 10 分钟。
#   训练好的 ckpt(会早停) agent 快很多(~17min/500题)。
# ⚠️ 必须 nohup 后台跑：evaluate.py 只在最后写 json，半途断了全白跑。本脚本已 nohup。
# ============================================================================
MODEL="${1:-/root/Search-R1/hf_models/Qwen2.5-3B-Instruct}"
MAXN="${2:-500}"
MODES="${3:-base,rag,agent}"
OUT="${4:-/tmp/eval_full_500.json}"
LOG="${OUT%.json}.log"

cd /root/Search-R1/eval || { echo "[eval_full] 找不到 /root/Search-R1/eval"; exit 1; }

# 评估用 bm25s 小语料，不依赖 8000 端口 BM25 server；若它在跑会白占内存，可停(可选)：
# pkill -f retrieval_server.py 2>/dev/null

echo "[eval_full] 模型 = $MODEL"
echo "[eval_full] 题数 = $MAXN   modes = $MODES"
echo "[eval_full] 输出 = $OUT    日志 = $LOG"
rm -f "$OUT" "$LOG"

nohup env CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/dsr1/bin/python src/evaluate.py \
  --model "$MODEL" \
  --eval_file data/eval/hotpotqa_dev.jsonl \
  --corpus data/corpus/hotpotqa_corpus.jsonl \
  --modes "$MODES" \
  --max_samples "$MAXN" \
  --max_turns 4 --topk 3 \
  --out "$OUT" > "$LOG" 2>&1 &
PID=$!
echo "$PID" > /tmp/eval_full.pid
echo "[eval_full] 已后台启动 PID=$PID（nohup 安全，可断 SSH）"
echo ""
echo "  看进度:  grep '已评估' $LOG | tail -1        # 每20题打印一次, 实测~4题/分钟"
echo "  看是否完成: grep -q '明细已保存' $LOG && echo DONE"
echo "  看结果:  cat $OUT                            # 完成后产出 base/rag/agent 三档 EM/F1/Cover/avg_search"
echo "  断SSH后回来查: ps -p \$(cat /tmp/eval_full.pid) && tail -5 $LOG"
