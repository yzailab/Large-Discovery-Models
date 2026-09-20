#!/usr/bin/env bash
# LDM × ResearchGym / time-series-explanation 的正式 campaign。
# 用法：bash tse_campaign.sh [smoke|main]
# 子协议：5 个真实数据集 × 5 折 = 25 个归因作业，单个候选约 11 分钟（8 卡并行）。
# 实测：LDM 0.5417 vs 任务自带的 IG 0.4600 / GradSHAP 0.4536。
# 路径与端点见 ../common.sh。
set -u
MODE="${1:-main}"
. "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
ldm_require_endpoint

export RG_TSE_NGPU="${RG_TSE_NGPU:-8}"

if [ "$MODE" = "smoke" ]; then ITERS=1; RES=3; HOURS=1; else ITERS="${TSE_ITERS:-30}"; RES=8; HOURS=12; fi

echo "=== TSE campaign: $MODE (iterations=$ITERS reservoir=$RES ngpu=$RG_TSE_NGPU) ==="; date '+%F %T'
# 5 个数据集每个 5 折都要跑满：在最难的 PAM 上崩掉反而会抬高均分，缺格按无效处理。
ldm_campaign tasks/test/time-series-explanation "$ITERS" "$RES" "$HOURS" 5 5 "bash eval_candidate.sh"
echo "退出码 $?"; date '+%F %T'
echo "=== TSE_CAMPAIGN ${MODE} DONE ==="
