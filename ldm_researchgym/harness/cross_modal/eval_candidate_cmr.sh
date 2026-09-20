#!/usr/bin/env bash
# 一个候选的完整评估：COCO-C 16 种腐蚀 × severity 5，i2t，1000 图子集，多卡并行。
# 由 LDM 评估器通过 --run-command 在 workspace 里调用（cwd = workspace/input）。
# 恒为 0 退出：候选跑坏了交给覆盖率闸门判 invalid，非零退出留给"评估器本身坏了"。
set -u
RG_ROOT="${RG_ROOT:-$HOME/ResearchGym}"   # 由 rg.adapter 传进来；单跑时自己设
TASK="$RG_ROOT/tasks/test/cross-modal-retrieval"
TORCHRUN="$TASK/.venv/bin/torchrun"          # venv 不随 workspace 复制
export PYTHONPATH="$PWD"
export HF_ENDPOINT="https://hf-mirror.com"    # 这台机器到不了 huggingface.co
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

# dataset/ 和 weights/ 没进 git，workspace 拷贝里没有；软链回任务目录
for d in dataset weights; do
  [ -e "$PWD/$d" ] || ln -s "$TASK/$d" "$PWD/$d"
done

CORR=(gaussian_noise shot_noise impulse_noise speckle_noise defocus_blur glass_blur \
      motion_blur zoom_blur snow frost fog brightness contrast elastic_transform \
      pixelate jpeg_compression)
read -r -a GPUS <<< "${RG_CMR_GPUS:-0 1 2 3}"
# 别的项目会中途占走卡：2026-09-13 晚上有个作业占满了 GPU4，而 16 个腐蚀里
# 下标 ≡0 (mod 4) 的四个全排给它，于是每个候选都缺同样 4 列、整行被 grader 丢掉，
# 一整场 campaign 15 次评估全废。开跑前按空闲显存筛一遍，占用超过阈值的卡直接不用。
NEED_MIB="${RG_CMR_MIN_FREE_MIB:-30000}"
FREE_GPUS=()
for g in "${GPUS[@]}"; do
  free=$(nvidia-smi --id="$g" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || echo 0)
  if [ "${free:-0}" -ge "$NEED_MIB" ]; then FREE_GPUS+=("$g"); else echo "跳过 GPU$g（空闲 ${free}MiB < ${NEED_MIB}MiB，别人在用）"; fi
done
if [ "${#FREE_GPUS[@]}" -gt 0 ]; then GPUS=("${FREE_GPUS[@]}"); else echo "警告：没有一张卡满足空闲阈值，按原列表硬跑"; fi
NG=${#GPUS[@]}
CFG_DIR="${RG_CMR_CFG_DIR:-$TASK/configs/QS/i2t/coco_image_c_sub}"
STEPS="${RG_CMR_TTA_STEPS:-3}"
METHOD="${RG_CMR_METHOD:-ldm}"            # 基线复核时换成 tent/sar/read/shot
# 基线和 campaign 并发跑时会抢 torchrun 的 master_port，换个基址就能错开
PORT_BASE="${RG_CMR_PORT_BASE:-29600}"
OUT="output/${METHOD}"

rm -rf "$OUT"                              # 上一个候选的 evaluate.txt 会被 grade.py 一起吃掉
mkdir -p "$OUT"

# 一个腐蚀跑一遍。GPU 和端口按腐蚀在 CORR 里的原始下标分配，补跑时也照旧，
# 这样并发的另一场评估（基线/别的 campaign）端口不会跟着重排撞上来。
run_one() {
  local idx="$1" c="$2" gpu port
  gpu="${GPUS[$((idx % NG))]}"
  port=$((PORT_BASE + idx))
  # main.py 自己会建 output_dir，但它建完到写 evaluate.txt 之间隔着整场评估，
  # 这中间目录要是没了就是"跑完了却写不进去"——2026-09-12 的四个基线全栽在这，
  # 每个都缺了一整波的列。先建好，再由下面的补跑兜底。
  mkdir -p "$OUT/$c"
  CUDA_VISIBLE_DEVICES="$gpu" "$TORCHRUN" --nproc_per_node=1 --master_port="$port" main.py \
    --config "$CFG_DIR/COCO_IP_${c}_5.yaml" \
    --output_dir "${OUT}/${c}" \
    --retrieval i2t --method "$METHOD" --tta_steps "$STEPS" \
    > "/tmp/evalcand_cmr_${METHOD}_${c}.log" 2>&1
}

index_of() {
  local c="$1" k
  for k in "${!CORR[@]}"; do [ "${CORR[$k]}" = "$c" ] && { echo "$k"; return; }; done
  echo 0
}

run_pass() {
  local n=0 c
  for c in "$@"; do
    run_one "$(index_of "$c")" "$c" &
    n=$((n + 1))
    if (( n % NG == 0 )); then wait; fi
  done
  wait
}

missing_list() {
  local c
  for c in "${CORR[@]}"; do
    [ -s "${OUT}/${c}/evaluate.txt" ] || echo "$c"
  done
}

# grade.py 缺一列就整行丢掉，而缺列多半是上面那个目录竞争、不是候选本身坏了。
# 补跑一遍，把"评估器抖了一下"和"候选真的跑不起来"分开。
pending=("${CORR[@]}")
for attempt in 1 2; do
  run_pass "${pending[@]}"
  mapfile -t pending < <(missing_list)
  [ "${#pending[@]}" -eq 0 ] && break
  if [ "$attempt" -lt 2 ]; then
    echo "第 ${attempt} 遍后仍缺 ${#pending[@]} 个，补跑：${pending[*]}"
  fi
done

# 静默失败可见化：grade.py 缺一列就整行丢掉，这里先把缺哪列说清楚
fail=0
for c in "${CORR[@]}"; do
  f="${OUT}/${c}/evaluate.txt"
  if [ ! -s "$f" ]; then
    echo "MISSING ${c}"; tail -3 "/tmp/evalcand_cmr_${METHOD}_${c}.log" 2>/dev/null | sed "s/^/  [${c}] /"
    fail=$((fail + 1))
  else
    echo "${c}: $(tail -1 "$f" | sed "s/,.*//")"
  fi
done
echo "候选评估完成：16 种腐蚀，缺失 $fail 个"
exit 0
