#!/usr/bin/env bash
# 参照跑：SimpleCIL / CIFAR100 十任务，既拿基线数字也量单次评估的墙钟时间
set -u
RG_ROOT="${RG_ROOT:-$HOME/ResearchGym}"
cd "$RG_ROOT/tasks/test/continual-learning" || exit 1
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_DOWNLOAD_TIMEOUT=60
"$PWD/.venv/bin/python" -c "import patch_cifar_md5 as p; p.patch()" 2>/dev/null
date "+start %F %T"
CUDA_VISIBLE_DEVICES="${RG_CL_REF_GPU:-0}" "$PWD/.venv/bin/python" main.py --config exps/_ref_simplecil_c100.json 2>&1 \
  | grep -viE "it/s\]|s/it\]|%\|" | tail -60
echo "退出码 ${PIPESTATUS[0]}"
date "+end %F %T"
echo "=== CL_REF DONE ==="
