#!/usr/bin/env bash
# LDM × ResearchGym / cross-modal-retrieval 的正式 campaign。
# 用法：bash cmr_campaign.sh [smoke|main|abl|long]
#   smoke = 1 轮真评估器，验通 install→eval→grade→标量化整条链
#   main  = 15 轮正式跑
#   abl   = 消融：reservoir=1，提案什么就评什么，等于把 GP-UCB 选择层摘掉
#   long  = 30 轮，用来判断 15 轮时第 3 轮就不再进步是预算不够还是真的到顶
# 路径与端点见 ../common.sh（RG_ROOT / LDM_ROOT / LDM_LLM_URL / LDM_LLM_API_KEY）。
set -u
MODE="${1:-main}"
. "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
ldm_require_endpoint

# 评估脚本开跑前按空闲显存筛卡：被别人占走一张卡会让每个候选都缺同一批格子，整场作废。
export RG_CMR_GPUS="${RG_CMR_GPUS:-0 1 2 3}"
export RG_CMR_MIN_FREE_MIB="${RG_CMR_MIN_FREE_MIB:-30000}"

if   [ "$MODE" = "smoke" ]; then ITERS=1;  RES=3; HOURS=1
elif [ "$MODE" = "abl"   ]; then ITERS=15; RES=1; HOURS=10
elif [ "$MODE" = "long"  ]; then ITERS="${CMR_ITERS:-30}"; RES=4; HOURS=14
else                             ITERS=15; RES=4; HOURS=10; fi

echo "=== campaign: $MODE (iterations=$ITERS reservoir=$RES gpus=$RG_CMR_GPUS) ==="; date '+%F %T'
ldm_campaign tasks/test/cross-modal-retrieval "$ITERS" "$RES" "$HOURS" 16 1 "bash eval_candidate_cmr.sh"
echo "退出码 $?"; date '+%F %T'
echo "=== CMR_CAMPAIGN ${MODE} DONE ==="
