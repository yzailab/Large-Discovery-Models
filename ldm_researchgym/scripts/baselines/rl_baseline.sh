#!/usr/bin/env bash
# 基线验证：均匀采样 buffer + SAC，cheetah-run 100K 步，2 个种子并行。
# 任务表里 SAC 在这个环境是 346.61。我们这套栈如果落在 250-450，就算复现成功，
# 后面搜出来的东西才可信；差太远说明训练栈有 bug，必须先修。
set -u
RG_ROOT="${RG_ROOT:-$HOME/ResearchGym}"
T="$RG_ROOT/tasks/test/improving-replay-buffers"
read -r -a GPUS <<< "${RG_RL_GPUS:-0 1}"
cd "$T" || exit 1
echo "=== SAC 基线 cheetah-run 100K ==="; date "+%F %T"
i=0
for seed in 0 1; do
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  CUDA_VISIBLE_DEVICES=$gpu "$T/.venv/bin/python" rl_runner.py \
      --domain cheetah --task run --seed $seed --steps 100000 \
      --buffer ldm_buffer --out "$T/results/baseline" > "/tmp/rl_base_seed${seed}.log" 2>&1 &
  i=$((i + 1))
done
wait
date "+%F %T"
echo "=== 结果 ==="
for f in "$T"/results/baseline/*.json; do
  "$T/.venv/bin/python" -c "
import json,sys
d=json.load(open(sys.argv[1]))
print(seed %d 最终回报 %.2f 用时 %.1f 分钟 % (d[seed], d[metrics][average_return], d[wall_clock_minutes]))
print( 曲线:,  .join(%d:%.0f % (c[step], c[average_return]) for c in d[curve]))
" "$f"
done
echo "=== RL_BASELINE DONE ==="
