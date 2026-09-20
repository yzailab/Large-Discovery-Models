#!/usr/bin/env bash
# 一屏看完今晚所有 LDM 作业的状态。本地只 ssh 一次拿这一屏，不留常驻连接。
set -u
echo "时间 $(date '+%F %T')  |  GPU $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | tr '\n' ' ')"
echo "在跑: $(pgrep -af 'rg.entry|ldm_researchgym|cmr_baselines.sh|overnight.sh' | sed 's/.*\(rg.entry\|ldm_researchgym\|cmr_baselines.sh\|overnight.sh\).*/\1/' | sort | uniq -c | tr '\n' ' ')"
echo
python3 "$(dirname "${BASH_SOURCE[0]}")/report_data.py" 2>/dev/null | grep -E "cmr_|tse_|cl " | sed 's/^/  /'
echo
LOGS="${LDM_LOG_DIR:-$HOME}"
echo "-- 排班链 --"; tail -3 "$LOGS/overnight.log" 2>/dev/null | sed 's/^/  /'
echo "-- 基线 --"; grep -E "^=== 基线|avg=|BASELINES DONE" "$LOGS/cmr_baselines.log" 2>/dev/null | tail -4 | sed 's/^/  /'
for f in ${LDM_STATUS_LOGS:-cl_campaign cmr_campaign rl_campaign}; do
  [ -f "$LOGS/$f.log" ] && { echo "-- $f --"; tail -2 "$LOGS/$f.log" | sed 's/^/  /'; }
done
