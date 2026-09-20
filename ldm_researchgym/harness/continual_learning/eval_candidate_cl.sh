#!/usr/bin/env bash
# 一个候选的完整评估：CIFAR100 十任务 × 3 个种子，3 张卡并行。
# 由 LDM 评估器通过 --ldm_run_command 在 workspace 里调用（cwd = workspace）。
# 恒为 0 退出：候选跑坏了要由覆盖率闸门判成 invalid，非零退出留给"评估器本身坏了"。
set -u
RG_ROOT="${RG_ROOT:-$HOME/ResearchGym}"   # 由 rg.adapter 传进来；单跑时自己设
PY="${RG_CL_PYTHON:-$RG_ROOT/tasks/test/continual-learning/.venv/bin/python}"   # venv 不随 workspace 复制
export PYTHONPATH="$PWD"
export HF_ENDPOINT="https://hf-mirror.com"        # 这台机器到不了 huggingface.co
export HF_HUB_DOWNLOAD_TIMEOUT=60
export TOKENIZERS_PARALLELISM=false
SEEDS=(1992 1993 1994)
read -r -a GPUS <<< "${RG_CL_GPUS:-0 1 2}"          # 三个种子各一张卡；按机器设 RG_CL_GPUS

"$PY" -c "import patch_cifar_md5 as p; p.patch()" 2>/dev/null

# trainer.py 用 exists()+makedirs() 建日志目录，三个种子同时起会撞 FileExistsError；先建好
mkdir -p exps "logs/ldm/cifar224/0/10"
for i in "${!SEEDS[@]}"; do
  seed="${SEEDS[$i]}"
  cat > "exps/_ldm_${seed}.json" <<JSON
{
  "prefix": "reproduce", "dataset": "cifar224",
  "memory_size": 0, "memory_per_class": 0, "fixed_memory": false, "shuffle": true,
  "init_cls": 10, "increment": 10,
  "model_name": "ldm", "backbone_type": "pretrained_vit_b16_224",
  "device": ["0"], "seed": [${seed}],
  "tuned_epoch": 5, "init_lr": 0.01, "batch_size": 48,
  "weight_decay": 0.0005, "min_lr": 0,
  "ffn_num": 64, "optimizer": "sgd", "vpt_type": "shallow", "prompt_token_num": 5
}
JSON
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "$PY" main.py --config "exps/_ldm_${seed}.json" \
    > "/tmp/evalcand_cl_${seed}.log" 2>&1 &
done
wait

# 静默失败可见化：评分器只看 .log，缺文件会被当成"这一格没跑"而不是"跑失败了"
LOGDIR="logs/ldm/cifar224/0/10"
fail=0
for seed in "${SEEDS[@]}"; do
  f="${LOGDIR}/reproduce_${seed}_pretrained_vit_b16_224.log"
  if [ ! -s "$f" ]; then
    echo "MISSING seed ${seed}"; fail=$((fail+1))
  elif ! grep -q "Average Accuracy (CNN)" "$f"; then
    echo "NO METRICS seed ${seed}"; fail=$((fail+1))
  else
    n=$(grep -c "Average Accuracy (CNN)" "$f")
    echo "seed ${seed}: ${n}/10 个任务有指标"
    [ "$n" -ge 10 ] || fail=$((fail+1))
  fi
  tail -3 "/tmp/evalcand_cl_${seed}.log" 2>/dev/null | sed "s/^/  [${seed}] /"
done
echo "候选评估完成：3 个种子，缺失/不完整 $fail 个"
exit 0
