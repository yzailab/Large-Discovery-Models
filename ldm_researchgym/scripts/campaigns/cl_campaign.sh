#!/usr/bin/env bash
# LDM × ResearchGym / continual-learning 的正式 campaign。
# 用法：bash cl_campaign.sh [smoke|main]
# 路径与端点见 ../common.sh。
set -u
MODE="${1:-main}"
. "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
ldm_require_endpoint

export RG_CL_GPUS="${RG_CL_GPUS:-0 1 2}"
# CL 的契约长，复读风险更高，这里的 frequency_penalty 比默认再高一档。
export LDM_LLM_EXTRA_BODY_JSON="${LDM_LLM_EXTRA_BODY_JSON_CL:-{\"frequency_penalty\":0.3}}"

if [ "$MODE" = "smoke" ]; then ITERS=1; RES=3; HOURS=1; else ITERS=15; RES=4; HOURS=8; fi

echo "=== campaign: $MODE (iterations=$ITERS reservoir=$RES) ==="; date '+%F %T'
# 3 个种子都要跑满：少一个种子就按无效候选处理，否则崩在难种子上反而抬高均分。
ldm_campaign tasks/test/continual-learning "$ITERS" "$RES" "$HOURS" 1 3 "bash eval_candidate_cl.sh"
echo "退出码 $?"; date '+%F %T'
echo "=== CL_CAMPAIGN ${MODE} DONE ==="
