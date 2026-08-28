#!/usr/bin/env bash
set -euo pipefail

METHOD1_ROOT=/workspace/test-TTT/policy/Method1_lerobot-pi0-ttt
RAW_DATA_DIR=/workspace/RMBench/data/cover_blocks/demo_clean/data
OUTPUT_DIR=/workspace/data_rmbench_lerobot/cover_blocks_demo_clean
REPO_ID=rmbench/cover_blocks_demo_clean

test -f "${METHOD1_ROOT}/examples/rmbench/convert_cover_blocks_to_lerobot.py"
test "$(find "${RAW_DATA_DIR}" -maxdepth 1 -type f -name 'episode*.hdf5' | wc -l)" -eq 50
if [[ -e "${OUTPUT_DIR}" || -L "${OUTPUT_DIR}" ]]; then
    echo "Refusing to overwrite existing conversion output: ${OUTPUT_DIR}" >&2
    exit 1
fi

mkdir -p "$(dirname -- "${OUTPUT_DIR}")"
cd "${METHOD1_ROOT}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export UV_CACHE_DIR=/workspace/uv_cache

uv run --frozen --no-sync --with h5py \
    python examples/rmbench/convert_cover_blocks_to_lerobot.py \
    --raw-data-dir "${RAW_DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --repo-id "${REPO_ID}" \
    --episode-count 50 \
    --image-writer-threads 8

"${METHOD1_ROOT}/.venv/bin/python" - "${OUTPUT_DIR}" "${REPO_ID}" <<'PY'
import sys
from pathlib import Path

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

root = Path(sys.argv[1])
metadata = LeRobotDatasetMetadata(sys.argv[2], root=root)
if metadata.total_episodes != 50 or metadata.total_frames != 51077:
    raise SystemExit(
        f"Unexpected converted dataset size: episodes={metadata.total_episodes}, "
        f"frames={metadata.total_frames}"
    )
print(
    "RMBench LeRobot conversion complete:",
    f"episodes={metadata.total_episodes}",
    f"frames={metadata.total_frames}",
    f"fps={metadata.fps}",
)
PY

du -sh "${OUTPUT_DIR}"
df -h "${OUTPUT_DIR}"
