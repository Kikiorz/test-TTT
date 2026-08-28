#!/usr/bin/env bash
set -euo pipefail

RMBENCH_ROOT=/workspace/RMBench
RMBENCH_ASSETS_ROOT=${RMBENCH_ROOT}/assets
HF_BIN=/venv/main/bin/hf
MIN_FREE_KIB=$((8 * 1024 * 1024))

if [[ ! -d "${RMBENCH_ROOT}/.git" ]]; then
    echo "Missing RMBench checkout: ${RMBENCH_ROOT}" >&2
    exit 1
fi

available_kib=$(df --output=avail "${RMBENCH_ROOT}" | tail -n 1 | tr -d ' ')
if (( available_kib < MIN_FREE_KIB )); then
    echo "Need at least 8 GiB free before download; found ${available_kib} KiB" >&2
    exit 1
fi

"${HF_BIN}" download TianxingChen/RMBench \
    --repo-type dataset \
    --include 'embodiments/**' \
    --local-dir "${RMBENCH_ASSETS_ROOT}" \
    --max-workers 8

"${HF_BIN}" download TianxingChen/RMBench \
    --repo-type dataset \
    --include 'objects/**' \
    --local-dir "${RMBENCH_ASSETS_ROOT}" \
    --max-workers 8

"${HF_BIN}" download TianxingChen/RMBench \
    --repo-type dataset \
    --include 'data/cover_blocks/demo_clean/**' \
    --local-dir "${RMBENCH_ROOT}" \
    --max-workers 8

test -f "${RMBENCH_ROOT}/data/cover_blocks/demo_clean/data/episode0.hdf5"
test -d "${RMBENCH_ASSETS_ROOT}/embodiments/aloha-agilex"
test -d "${RMBENCH_ASSETS_ROOT}/objects/003_cover"

echo "RMBench cover_blocks download complete"
du -sh \
    "${RMBENCH_ASSETS_ROOT}/embodiments" \
    "${RMBENCH_ASSETS_ROOT}/objects" \
    "${RMBENCH_ROOT}/data/cover_blocks"
df -h "${RMBENCH_ROOT}"
