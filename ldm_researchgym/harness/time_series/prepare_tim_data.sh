#!/usr/bin/env bash
# Stage the five real-world datasets for tasks/test/time-series-explanation.
#
#   bash prepare_tim_data.sh <task-root>
#
# Produces <task-root>/data/{PAM,epilepsy,boiler,Wafer,FreezerRegularTrain}
# with exactly the filenames datasets/*.py look for. Verified 2026-09-02.
set -euo pipefail

TASK_ROOT="${1:-.}"
DATA_DIR="${TASK_ROOT}/data"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

mkdir -p "${DATA_DIR}"

# --- Harvard Dataverse (doi:10.7910/DVN/B0DEQJ) -----------------------------
# The readme links the file.xhtml landing pages; the API path below is the
# actual download and needs no login.
declare -a DV=("7495583:PAM:PAM" "7495586:Epilepsy:epilepsy" "7495589:Boiler:boiler")
for entry in "${DV[@]}"; do
  file_id="${entry%%:*}"; rest="${entry#*:}"
  archive_name="${rest%%:*}"; target_name="${rest##*:}"
  if [ -d "${DATA_DIR}/${target_name}" ]; then
    echo "skip ${target_name} (already present)"
    continue
  fi
  echo "fetching ${archive_name}..."
  curl -fsSL --retry 8 --retry-delay 5 --retry-all-errors --connect-timeout 30 -o "${WORK}/${archive_name}.tar.gz" \
    "https://dataverse.harvard.edu/api/access/datafile/${file_id}"
  tar -xzf "${WORK}/${archive_name}.tar.gz" -C "${WORK}"
  # Dataverse ships capitalised directory names; two loaders want lowercase.
  mv "${WORK}/${archive_name}" "${DATA_DIR}/${target_name}"
done

# --- UCR archive (Wafer, FreezerRegularTrain) -------------------------------
# 316 MB, password-protected zip. Ships .tsv; the loaders open .txt.
if [ ! -f "${DATA_DIR}/Wafer/Wafer_TRAIN.txt" ] || \
   [ ! -f "${DATA_DIR}/FreezerRegularTrain/FreezerRegularTrain_TRAIN.txt" ]; then
  echo "fetching UCRArchive_2018.zip (316 MB)..."
  curl -fsSL --retry 8 --retry-delay 5 --retry-all-errors --connect-timeout 30 -o "${WORK}/ucr.zip" \
    "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip"
  for name in Wafer FreezerRegularTrain; do
    mkdir -p "${DATA_DIR}/${name}"
    unzip -o -q -P someone -j "${WORK}/ucr.zip" "UCRArchive_2018/${name}/*" \
      -d "${DATA_DIR}/${name}"
    for split in TRAIN TEST; do
      mv -f "${DATA_DIR}/${name}/${name}_${split}.tsv" \
            "${DATA_DIR}/${name}/${name}_${split}.txt"
    done
  done
fi

echo "done. staged under ${DATA_DIR}:"
du -sh "${DATA_DIR}"
