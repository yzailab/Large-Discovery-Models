#!/usr/bin/env bash
# 一个候选的完整评估：cheetah-run，100K 环境步，2 个种子并行，用任务自带的 grader 打分。
# 由 LDM 评估器通过 --run-command 在 workspace 里调用（cwd = workspace/input）。
# 恒为 0 退出：候选跑坏了交给覆盖率闸门判 invalid，非零退出留给"评估器本身坏了"。
set -u
RG_ROOT="${RG_ROOT:-$HOME/ResearchGym}"   # 由 rg.adapter 传进来；单跑时自己设
TASK="$RG_ROOT/tasks/test/improving-replay-buffers"
PY="$TASK/.venv/bin/python"                 # venv 不随 workspace 复制
export MUJOCO_GL=egl                        # 状态观测不渲染，但 dm_control 导入时要选后端
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=4                    # MuJoCo 物理在 CPU 上，别让两个种子抢满核

read -r -a GPUS <<< "${RG_RL_GPUS:-0 1}"
# 别的项目会中途占卡（2026-09-13 晚就被占过一次，整场作废），开跑前按空闲显存筛一遍
NEED_MIB="${RG_RL_MIN_FREE_MIB:-20000}"
FREE=()
for g in "${GPUS[@]}"; do
  free=$(nvidia-smi --id="$g" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || echo 0)
  if [ "${free:-0}" -ge "$NEED_MIB" ]; then FREE+=("$g"); else echo "跳过 GPU$g（空闲 ${free}MiB）"; fi
done
[ "${#FREE[@]}" -gt 0 ] && GPUS=("${FREE[@]}")
NG=${#GPUS[@]}

SEEDS=(${RG_RL_SEEDS:-0 1})
STEPS="${RG_RL_STEPS:-100000}"
SLOT="${RG_RL_SLOT:-ldm}"
MODULE="${RG_RL_MODULE:-ldm_buffer}"
OUT="results/${SLOT}"

rm -rf "$OUT"                                # 上一个候选的 JSON 会被 grader 一起吃掉
mkdir -p "$OUT"

i=0
for seed in "${SEEDS[@]}"; do
  gpu="${GPUS[$((i % NG))]}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" rl_runner.py \
      --domain cheetah --task run --seed "$seed" --steps "$STEPS" \
      --buffer "$MODULE" --out "$OUT" \
      > "/tmp/evalcand_rl_${SLOT}_seed${seed}.log" 2>&1 &
  i=$((i + 1))
done
wait

# 静默失败可见化：grader 少一个种子就只按剩下的算，先把缺哪个说清楚
fail=0
for seed in "${SEEDS[@]}"; do
  f="${OUT}/cheetah-run_seed${seed}.json"
  if [ ! -s "$f" ]; then
    echo "MISSING seed ${seed}"
    tail -4 "/tmp/evalcand_rl_${SLOT}_seed${seed}.log" 2>/dev/null | sed "s/^/  [seed ${seed}] /"
    fail=$((fail + 1))
  else
    echo "seed ${seed}: $("$PY" -c "import json,sys;print('%.2f' % json.load(open(sys.argv[1]))['metrics']['average_return'])" "$f")"
  fi
done
echo "候选评估完成：${#SEEDS[@]} 个种子，缺失 $fail 个"
exit 0
