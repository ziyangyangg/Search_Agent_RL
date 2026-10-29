#!/bin/bash
# allsteps.sh — 展示训练日志里【所有】step 的关键指标表格
#   (status.sh 只显示最新一条；本脚本列出全部 step + timing 统计，方便找慢步)
# 用法: bash /root/Search-R1/allsteps.sh [日志路径]
#   不传参: 自动找 /tmp/train_smoke.log(单卡冒烟) → /tmp/verl.log(多卡正本)
#   传参:   bash /root/Search-R1/allsteps.sh /tmp/verl.log
LOG="$1"
if [ -z "$LOG" ]; then
  for c in /tmp/train_smoke.log /tmp/verl.log; do [ -f "$c" ] && LOG="$c" && break; done
fi
[ -z "$LOG" ] && { echo "找不到训练日志(试过 /tmp/train_smoke.log /tmp/verl.log)；用法: bash allsteps.sh [日志路径]"; exit 1; }
[ -f "$LOG" ] || { echo "日志不存在: $LOG"; exit 1; }

steps=$(grep -aE "step:[0-9]+ - " "$LOG")
[ -z "$steps" ] && { echo "[$LOG] 还没有 step 指标(训练可能在 init/rollout 阶段，稍等再看)"; exit 0; }

echo "===== 所有 step 指标 | $LOG | $(date "+%H:%M:%S") ====="
printf "%-5s %-10s %-9s %-9s %-10s %-11s %-8s %-9s\n" step "timing(s)" reward adv_max grad_norm valid_srch finish resp_len
printf "%-5s %-10s %-9s %-9s %-10s %-11s %-8s %-9s\n" ----- ---------- -------- -------- ---------- ----------- ------- --------

val() { printf "%s" "$1" | grep -oE "$2:[-0-9.]+" | head -1 | sed "s/.*://"; }
printf "%s\n" "$steps" | while IFS= read -r ln; do
  st=$(printf "%s" "$ln" | grep -oE "step:[0-9]+ -" | head -1 | grep -oE "[0-9]+")
  printf "%-5s %-10s %-9s %-9s %-10s %-11s %-8s %-9s\n" \
    "${st:-?}" \
    "$(val "$ln" "timing_s/step")" \
    "$(val "$ln" "critic/rewards/mean")" \
    "$(val "$ln" "critic/advantages/max")" \
    "$(val "$ln" "actor/grad_norm")" \
    "$(val "$ln" "env/number_of_valid_search")" \
    "$(val "$ln" "env/finish_ratio")" \
    "$(val "$ln" "response_length/mean")"
done

echo "-------------------------------------------------------------------------"
printf "%s\n" "$steps" | grep -oE "timing_s/step:[0-9.]+" | sed "s/.*://" | awk "
  {n++; s+=\$1; if(min==\"\"||\$1<min)min=\$1; if(\$1>max)max=\$1}
  END{ if(n>0) printf \"共 %d 步 | timing: 平均 %.1fs / 最快 %.1fs / 最慢 %.1fs\n\", n, s/n, min, max }"
