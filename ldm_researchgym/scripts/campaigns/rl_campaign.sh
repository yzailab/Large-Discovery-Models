#!/usr/bin/env bash
# LDM × ResearchGym / improving-replay-buffers 的正式 campaign。
# 用法：bash rl_campaign.sh [smoke|main]
# 子协议：cheetah-run，10 万环境步，2 个种子。均匀采样基线实测 257（302/211，种子噪声 ±45）。
# 论文表里 SAC 347 / REDQ 607 / SynthER 729——那是他们的实现，不能和这里的数并列。
# 训练栈是我们补的（上游只给了 54 行空壳），见 ../../harness/replay_buffer/rl_runner.py。
# 路径与端点见 ../common.sh。
set -u
MODE="${1:-main}"
. "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
ldm_require_endpoint

export RG_RL_GPUS="${RG_RL_GPUS:-0 1}"
export RG_RL_STEPS="${RG_RL_STEPS:-100000}"

if [ "$MODE" = "smoke" ]; then ITERS=1; RES=3; HOURS=1; else ITERS=15; RES=4; HOURS=12; fi

echo "=== RL campaign: $MODE (iterations=$ITERS reservoir=$RES gpus=$RG_RL_GPUS steps=$RG_RL_STEPS) ==="
date '+%F %T'
ldm_campaign tasks/test/improving-replay-buffers "$ITERS" "$RES" "$HOURS" 1 2 "bash eval_candidate_rl.sh"
echo "退出码 $?"; date '+%F %T'
echo "=== RL_CAMPAIGN ${MODE} DONE ==="
