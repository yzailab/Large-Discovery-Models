#!/usr/bin/env bash
# 同一套 1000 图协议下把其余基线补齐，凑出可比的对照表。
# 用法：bash cmr_baselines.sh [方法...]，默认 source_ref sar read shot
set -u
RG_ROOT="${RG_ROOT:-$HOME/ResearchGym}"
T="$RG_ROOT/tasks/test/cross-modal-retrieval"
cd "$T" || exit 1
METHODS=("$@"); [ ${#METHODS[@]} -eq 0 ] && METHODS=(source_ref sar read shot)
for m in "${METHODS[@]}"; do
  echo "=== 基线 $m ==="; date '+%F %T'
  RG_CMR_METHOD="$m" bash eval_candidate_cmr.sh
  ./.venv/bin/python grading/grade.py --root "output/$m" --table qs_image \
      --json-out "/tmp/cmr_${m}_summary.json" --no-markdown 2>&1 | tail -3
  ./.venv/bin/python - "$m" <<'PY'
import json, sys
m = sys.argv[1]
s = json.load(open(f"/tmp/cmr_{m}_summary.json"))
for r in s["rows"]:
    print(f"{m:11s} {r['display']} avg={r['average']:.2f}")
    print("   ", " ".join(f"{c['label']}={c['value']}" for c in r["columns"]))
PY
done
echo "=== CMR_BASELINES DONE ==="; date '+%F %T'
