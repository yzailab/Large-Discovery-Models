#!/usr/bin/env bash
# 一个候选的完整评估：5 数据集 × 5 折 = 25 个归因作业，8 卡并行。
# 由 LDM 评估器通过 --ldm_run_command 在 workspace 里调用（cwd = workspace）。
set -u
RG_ROOT="${RG_ROOT:-$HOME/ResearchGym}"   # 由 rg.adapter 传进来；单跑时自己设
PY="${RG_TSE_PYTHON:-$RG_ROOT/tasks/test/time-series-explanation/.venv/bin/python}"   # venv 不随 workspace 复制
export PYTHONPATH="$PWD"
NGPU="${RG_TSE_NGPU:-8}"
mkdir -p logs

# PAM 最慢（单折约 497 秒），先发让它与快的重叠，关键路径才不会串起来
i=0
for d in PAM boiler freezer wafer epilepsy; do
  for cv in 0 1 2 3 4; do
    gpu=$((i % NGPU)); i=$((i+1))
    CUDA_VISIBLE_DEVICES=$gpu "$PY" real/main.py \
      --model_type state --explainers ldm --data "$d" --fold "$cv" \
      --testbs 50 --areas 0.1 --top 0 --device cuda:0 \
      --output-file "logs/state_${d}_${cv}_0_ldm.csv" \
      > "/tmp/evalcand_${d}_${cv}.log" 2>&1 &
    while [ "$(jobs -rp | wc -l)" -ge "$NGPU" ]; do sleep 2; done
  done
done
wait

# 静默失败可见化：评估器只看 CSV，缺文件会被当成"这一格没跑"而不是"跑失败了"
fail=0
for d in PAM boiler freezer wafer epilepsy; do
  for cv in 0 1 2 3 4; do
    [ -s "logs/state_${d}_${cv}_0_ldm.csv" ] || { echo "MISSING ${d}_${cv}"; fail=$((fail+1)); }
  done
done
echo "候选评估完成：25 个作业，缺失 $fail 个"
exit 0
