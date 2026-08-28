#!/usr/bin/env bash

# Extend the completed RememberShape5 Stage-2 checkpoint with a fresh
# action-head/TTT optimization stage for 100 complete sequence epochs. Save
# both +50 and +100 epoch checkpoints, evaluate them with the same canonical
# official 50-episode seed stream, and write a new comparison without
# modifying any earlier training or evaluation result.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
MODE="${1:-plan}"
EXECUTE="${EXECUTE:-0}"

die() {
  echo "RememberShape5 SmolVLA-TTT Stage3: $*" >&2
  exit 2
}

case "${MODE}" in
  plan|run) ;;
  *) die "mode must be plan or run" ;;
esac

DATASET_REPO_ID=remember_shape_5_vla_v0
ENV_ID=RememberShape5-VLA-v0
EXPECTED_EPISODES=250
EXTRA_EPOCHS=100
MIDPOINT_EPOCHS=50

GPU_A="${GPU_A:-2}"
GPU_B="${GPU_B:-3}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-256}"
SEQUENCE_STRIDE="${SEQUENCE_STRIDE:-256}"
TBPTT_SEGMENT_LENGTH="${TBPTT_SEGMENT_LENGTH:-4}"
TTT_HIDDEN_DIM="${TTT_HIDDEN_DIM:-4096}"
TTT_LAYERS="${TTT_LAYERS:-[12,13,14,15]}"
REGISTER_TOKENS="${REGISTER_TOKENS:-16}"
EFFECTIVE_GATE_INIT="${EFFECTIVE_GATE_INIT:-0.001}"
N_ACTION_STEPS="${N_ACTION_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29662}"
LOG_FREQ="${LOG_FREQ:-10}"
SEED="${SEED:-1000}"
WAIT_SECONDS="${WAIT_SECONDS:-30}"

DATASET_ROOT="${DATASET_ROOT:-/workspace/data_mikasa_robo/data_lerobot/${DATASET_REPO_ID}}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-/workspace/experiments/method1_smolvla_ttt_sequence_outer_v1_50x50/remember_shape5/stage2_action_head/checkpoints/last/pretrained_model}"
EXPECTED_SOURCE_SHA256="${EXPECTED_SOURCE_SHA256:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/experiments/method1_smolvla_ttt_sequence_outer_v1_stage2plus100/remember_shape5/stage3_action_head}"
MANIFEST="${MANIFEST:-${OUTPUT_DIR}/stage3_manifest.json}"

EVALUATION_ROOT="${EVALUATION_ROOT:-/workspace/evaluations/mikasa_official50_20260827/remember_shape5}"
MIDPOINT_EVAL="${EVALUATION_ROOT}/stage2plus50_action_head/eval.json"
FINAL_EVAL="${EVALUATION_ROOT}/stage2plus100_action_head/eval.json"
MIDPOINT_OFFICIAL="${EVALUATION_ROOT}/stage2plus50_action_head/official"
FINAL_OFFICIAL="${EVALUATION_ROOT}/stage2plus100_action_head/official"
COMPARISON_INPUT="${EVALUATION_ROOT}/comparison_with_baseline250.json"
COMPARISON_OUTPUT="${EVALUATION_ROOT}/comparison_with_baseline250_and_stage2_extension.json"

TRAIN_PYTHON="${TRAIN_PYTHON:-/venv/main/bin/python3}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/venv/main/bin/accelerate}"
EVAL_PYTHON="${EVAL_PYTHON:-/workspace/MIKASA-Robo/.venv/bin/python}"
EVALUATOR="${EVALUATOR:-${SCRIPT_DIR}/evaluate_mikasa.py}"

for integer_name in \
  GPU_A GPU_B BATCH_SIZE NUM_PROCESSES SEQUENCE_LENGTH SEQUENCE_STRIDE \
  TBPTT_SEGMENT_LENGTH TTT_HIDDEN_DIM REGISTER_TOKENS NUM_WORKERS \
  N_ACTION_STEPS PREFETCH_FACTOR MAIN_PROCESS_PORT LOG_FREQ SEED WAIT_SECONDS; do
  integer_value="${!integer_name}"
  [[ "${integer_value}" =~ ^[0-9]+$ ]] \
    || die "${integer_name} must be a non-negative integer; got '${integer_value}'"
done

[[ "${GPU_A}" != "${GPU_B}" ]] || die "GPU_A and GPU_B must differ"
(( BATCH_SIZE == 8 )) || die "the canonical continuation requires BATCH_SIZE=8"
(( NUM_PROCESSES == 2 )) || die "the canonical continuation requires NUM_PROCESSES=2"
(( SEQUENCE_LENGTH > 0 && SEQUENCE_STRIDE > 0 )) \
  || die "SEQUENCE_LENGTH and SEQUENCE_STRIDE must be positive"
(( SEQUENCE_STRIDE <= SEQUENCE_LENGTH )) \
  || die "SEQUENCE_STRIDE cannot exceed SEQUENCE_LENGTH"
(( TBPTT_SEGMENT_LENGTH > 0 && TBPTT_SEGMENT_LENGTH <= SEQUENCE_LENGTH )) \
  || die "TBPTT_SEGMENT_LENGTH must be in 1..SEQUENCE_LENGTH"
(( N_ACTION_STEPS == 1 )) \
  || die "the bundled MIKASA TTT evaluator is fixed to K1; Stage3 requires N_ACTION_STEPS=1"
(( MAIN_PROCESS_PORT > 0 && MAIN_PROCESS_PORT <= 65535 )) \
  || die "MAIN_PROCESS_PORT must be in 1..65535"
(( WAIT_SECONDS > 0 )) || die "WAIT_SECONDS must be positive"
[[ "${EFFECTIVE_GATE_INIT}" == "0.001" ]] \
  || die "the canonical continuation requires EFFECTIVE_GATE_INIT=0.001"

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/workspace/data_mikasa_robo}"
export PYTHONPATH="${REPO_ROOT}/src:/workspace/MIKASA-Robo${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl

RESOLVED_STATS=""
if [[ -x "${TRAIN_PYTHON}" && -f "${DATASET_ROOT}/meta/info.json" ]]; then
  RESOLVED_STATS="$("${TRAIN_PYTHON}" - \
      "${DATASET_REPO_ID}" "${DATASET_ROOT}" "${EXPECTED_EPISODES}" \
      "${SEQUENCE_LENGTH}" "${SEQUENCE_STRIDE}" "${BATCH_SIZE}" \
      "${NUM_PROCESSES}" <<'PY'
import json
import sys
from pathlib import Path

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

repo_id, root_raw, expected_raw, length_raw, stride_raw, batch_raw, world_raw = sys.argv[1:]
root = Path(root_raw)
expected = int(expected_raw)
sequence_length = int(length_raw)
sequence_stride = int(stride_raw)
batch_size = int(batch_raw)
world_size = int(world_raw)

meta = LeRobotDatasetMetadata(repo_id, root=root)
episodes = meta.episodes
if episodes is None:
    raise SystemExit("dataset metadata does not contain episode boundaries")
if isinstance(episodes, dict):
    rows = [
        (index, int(start), int(end))
        for index, (start, end) in enumerate(
            zip(episodes["dataset_from_index"], episodes["dataset_to_index"], strict=True)
        )
    ]
else:
    rows = [
        (
            int(row.get("episode_index", index)),
            int(row["dataset_from_index"]),
            int(row["dataset_to_index"]),
        )
        for index, row in enumerate(episodes)
    ]
rows.sort(key=lambda item: item[0])
indices = [item[0] for item in rows]
if indices != list(range(expected)):
    raise SystemExit(f"expected episode indices 0..{expected - 1}; got {indices[:1]}..{indices[-1:]}")
lengths = [end - start for _, start, end in rows]
if any(length <= 0 for length in lengths):
    raise SystemExit("all selected episodes must have positive length")
if sequence_stride > sequence_length:
    raise SystemExit("sequence_stride must not exceed sequence_length")
windows_per_episode = [
    (length + sequence_stride - 1) // sequence_stride for length in lengths
]
windows = sum(windows_per_episode)
global_batch_size = batch_size * world_size
window_count_histogram = {
    window_count: windows_per_episode.count(window_count)
    for window_count in sorted(set(windows_per_episode))
}
steps_per_epoch = (windows + global_batch_size - 1) // global_batch_size
print(
    json.dumps(
        {
            "episodes": len(indices),
            "episodes_json": json.dumps(indices, separators=(",", ":")),
            "frames": sum(lengths),
            "windows": windows,
            "steps_per_epoch": steps_per_epoch,
            "window_count_histogram": window_count_histogram,
            "unused_batch_slots": steps_per_epoch * global_batch_size - windows,
            "min_episode_length": min(lengths),
            "max_episode_length": max(lengths),
            "sequence_length": sequence_length,
        },
        separators=(",", ":"),
    )
)
PY
  )" || die "could not resolve the canonical sequence epoch"
fi

if [[ -n "${RESOLVED_STATS}" ]]; then
  IFS=$'\t' read -r \
    NUM_EPISODES NUM_FRAMES NUM_WINDOWS STEPS_PER_EPOCH UNUSED_BATCH_SLOTS \
    MIN_EPISODE_LENGTH MAX_EPISODE_LENGTH EPISODES_JSON \
    <<< "$("${TRAIN_PYTHON}" -c '
import json, sys
x = json.loads(sys.argv[1])
print(x["episodes"], x["frames"], x["windows"], x["steps_per_epoch"],
      x["unused_batch_slots"], x["min_episode_length"], x["max_episode_length"],
      x["episodes_json"], sep="\t")
' "${RESOLVED_STATS}")"
  EXTRA_STEPS=$((STEPS_PER_EPOCH * EXTRA_EPOCHS))
  SAVE_INTERVAL=$((STEPS_PER_EPOCH * MIDPOINT_EPOCHS))
  EXPECTED_SOURCE_STEPS=$((STEPS_PER_EPOCH * 50))
else
  [[ "${MODE}" == plan ]] \
    || die "run mode requires locally resolvable dataset metadata"
  NUM_EPISODES="<250>"
  NUM_FRAMES="<N_FRAMES>"
  NUM_WINDOWS="<N_WINDOWS>"
  STEPS_PER_EPOCH="<STEPS_PER_EPOCH>"
  UNUSED_BATCH_SLOTS="<UNUSED_BATCH_SLOTS>"
  MIN_EPISODE_LENGTH="<MIN_EPISODE_LENGTH>"
  MAX_EPISODE_LENGTH="<MAX_EPISODE_LENGTH>"
  EPISODES_JSON="<ALL_250_EPISODE_INDICES>"
  EXTRA_STEPS="<STEPS_PER_EPOCH*100>"
  SAVE_INTERVAL="<STEPS_PER_EPOCH*50>"
  EXPECTED_SOURCE_STEPS="<STEPS_PER_EPOCH*50>"
fi

echo "RememberShape5 SmolVLA-TTT Stage3 continuation"
echo "source=${SOURCE_CHECKPOINT} source_total_ttt_epochs=100"
echo "dataset=${DATASET_REPO_ID} episodes=${NUM_EPISODES} frames=${NUM_FRAMES} windows=${NUM_WINDOWS}"
echo "episode_length=${MIN_EPISODE_LENGTH}..${MAX_EPISODE_LENGTH} unused_batch_slots=${UNUSED_BATCH_SLOTS}"
echo "batch/device=${BATCH_SIZE} processes=${NUM_PROCESSES} steps/sequence_epoch=${STEPS_PER_EPOCH}"
echo "inference_action_steps=${N_ACTION_STEPS} fast_updates=once_per_action_chunk_inference"
echo "stage3=action_head gate_learned_init=${EFFECTIVE_GATE_INIT} optimizer=fresh scheduler=fresh extra_epochs=${EXTRA_EPOCHS} steps=${EXTRA_STEPS}"
echo "checkpoints=+50ep@${SAVE_INTERVAL},+100ep@${EXTRA_STEPS} output=${OUTPUT_DIR}"
echo "gpu_pair=${GPU_A},${GPU_B} evaluations=${MIDPOINT_EVAL},${FINAL_EVAL}"

if [[ "${MODE}" == plan ]]; then
  echo "Plan only: no process or output was created."
  exit 0
fi

[[ "${EXECUTE}" == 1 ]] || die "run mode requires EXECUTE=1"
[[ -x "${TRAIN_PYTHON}" ]] || die "training Python not found: ${TRAIN_PYTHON}"
[[ -x "${ACCELERATE_BIN}" ]] || die "accelerate not found: ${ACCELERATE_BIN}"
[[ -x "${EVAL_PYTHON}" ]] || die "MIKASA Python not found: ${EVAL_PYTHON}"
[[ -f "${EVALUATOR}" ]] || die "MIKASA evaluator not found: ${EVALUATOR}"
[[ -f "${REPO_ROOT}/src/lerobot/scripts/lerobot_train.py" ]] \
  || die "REPO_ROOT is not a Method1 LeRobot checkout: ${REPO_ROOT}"
[[ -f "${DATASET_ROOT}/meta/info.json" ]] || die "dataset metadata missing: ${DATASET_ROOT}"
command -v nvidia-smi >/dev/null || die "nvidia-smi is required"

validate_source() {
  "${TRAIN_PYTHON}" - \
    "${SOURCE_CHECKPOINT}" "${EXPECTED_EPISODES}" "${EXPECTED_SOURCE_SHA256}" \
    "${EXPECTED_SOURCE_STEPS}" "${SEQUENCE_LENGTH}" "${SEQUENCE_STRIDE}" \
    "${TBPTT_SEGMENT_LENGTH}" "${TTT_HIDDEN_DIM}" "${REGISTER_TOKENS}" \
    "${EFFECTIVE_GATE_INIT}" "${N_ACTION_STEPS}" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1]).resolve()
expected_episodes = int(sys.argv[2])
expected_sha256 = sys.argv[3]
expected_source_steps = int(sys.argv[4])
expected_sequence_length = int(sys.argv[5])
expected_sequence_stride = int(sys.argv[6])
expected_tbptt = int(sys.argv[7])
expected_hidden = int(sys.argv[8])
expected_registers = int(sys.argv[9])
expected_gate = float(sys.argv[10])
expected_action_steps = int(sys.argv[11])
for filename in (
    "config.json",
    "model.safetensors",
    "train_config.json",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
):
    path = checkpoint / filename
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing source checkpoint file: {path}")
config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
train = json.loads((checkpoint / "train_config.json").read_text(encoding="utf-8"))
state_path = checkpoint.parent / "training_state" / "training_step.json"
state = json.loads(state_path.read_text(encoding="utf-8"))
checks = {
    "type": config.get("type") == "smolvla_ttt",
    "stage": config.get("ttt_training_stage") == "action_head",
    "gate config": math.isclose(
        float(config.get("ttt_effective_gate_init", -1)), expected_gate, abs_tol=1e-12
    ),
    "action steps": int(config.get("n_action_steps", -1)) == expected_action_steps,
    "sequence length": int(config.get("sequence_length", -1)) == expected_sequence_length,
    "sequence stride": int(config.get("sequence_stride", -1)) == expected_sequence_stride,
    "TBPTT": int(config.get("tbptt_segment_length", -1)) == expected_tbptt,
    "TTT hidden": int(config.get("ttt_hidden_dim", -1)) == expected_hidden,
    "TTT layers": config.get("ttt_layer_indices") == [12, 13, 14, 15],
    "registers": int(config.get("ttt_num_register_tokens", -1)) == expected_registers,
    "second order": config.get("ttt_second_order") is True,
    "selected sequence state": config.get("ttt_sequence_state_semantics")
    == "sequence_outer_step_v1",
    "source configured steps": int(train.get("steps", -1)) == expected_source_steps,
    "source completed steps": int(state.get("step", -1)) == expected_source_steps,
    "source batch": int(train.get("batch_size", -1)) == 8,
    "all episodes": train.get("dataset", {}).get("episodes") == list(range(expected_episodes)),
}
failed = [name for name, ok in checks.items() if not ok]
if failed:
    raise SystemExit(f"source Stage2 checkpoint failed canonical validation: {failed}")
for processor_name in ("policy_preprocessor.json", "policy_postprocessor.json"):
    processor = json.loads((checkpoint / processor_name).read_text(encoding="utf-8"))
    for step in processor.get("steps", []):
        state_file = step.get("state_file")
        if state_file and not (checkpoint / state_file).is_file():
            raise SystemExit(f"processor references missing state file: {state_file}")
digest = hashlib.sha256()
with (checkpoint / "model.safetensors").open("rb") as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
digest = digest.hexdigest()
if expected_sha256 and digest != expected_sha256:
    raise SystemExit(f"source Stage2 SHA mismatch: {digest} != {expected_sha256}")
print(f"Validated source Stage2 checkpoint: {checkpoint} sha256={digest}")
PY
}

validate_output_checkpoint() {
  local checkpoint=$1 expected_step=$2
  "${TRAIN_PYTHON}" - \
    "${checkpoint}" "${expected_step}" "${EXTRA_STEPS}" \
    "${SOURCE_CHECKPOINT}" "${OUTPUT_DIR}" "${EXPECTED_EPISODES}" \
    "${SEQUENCE_LENGTH}" "${SEQUENCE_STRIDE}" "${TBPTT_SEGMENT_LENGTH}" \
    "${TTT_HIDDEN_DIM}" "${REGISTER_TOKENS}" "${EFFECTIVE_GATE_INIT}" \
    "${N_ACTION_STEPS}" <<'PY'
import json
import math
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1]).resolve()
expected_step = int(sys.argv[2])
configured_steps = int(sys.argv[3])
source = Path(sys.argv[4]).resolve()
output_dir = Path(sys.argv[5]).resolve()
expected_episodes = int(sys.argv[6])
expected_sequence_length = int(sys.argv[7])
expected_sequence_stride = int(sys.argv[8])
expected_tbptt = int(sys.argv[9])
expected_hidden = int(sys.argv[10])
expected_registers = int(sys.argv[11])
expected_gate = float(sys.argv[12])
expected_action_steps = int(sys.argv[13])
for filename in (
    "config.json",
    "model.safetensors",
    "train_config.json",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
):
    path = checkpoint / filename
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing Stage3 checkpoint file: {path}")
config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
train = json.loads((checkpoint / "train_config.json").read_text(encoding="utf-8"))
state = json.loads(
    (checkpoint.parent / "training_state" / "training_step.json").read_text(encoding="utf-8")
)
pretrained = Path(train.get("policy", {}).get("pretrained_path", "")).resolve()
checks = {
    "type": config.get("type") == "smolvla_ttt",
    "stage": config.get("ttt_training_stage") == "action_head",
    "gate config": math.isclose(
        float(config.get("ttt_effective_gate_init", -1)), expected_gate, abs_tol=1e-12
    ),
    "action steps": int(config.get("n_action_steps", -1)) == expected_action_steps,
    "sequence length": int(config.get("sequence_length", -1)) == expected_sequence_length,
    "sequence stride": int(config.get("sequence_stride", -1)) == expected_sequence_stride,
    "TBPTT": int(config.get("tbptt_segment_length", -1)) == expected_tbptt,
    "TTT hidden": int(config.get("ttt_hidden_dim", -1)) == expected_hidden,
    "TTT layers": config.get("ttt_layer_indices") == [12, 13, 14, 15],
    "registers": int(config.get("ttt_num_register_tokens", -1)) == expected_registers,
    "second order": config.get("ttt_second_order") is True,
    "selected sequence state": config.get("ttt_sequence_state_semantics")
    == "sequence_outer_step_v1",
    "configured steps": int(train.get("steps", -1)) == configured_steps,
    "checkpoint step": int(state.get("step", -1)) == expected_step,
    "batch": int(train.get("batch_size", -1)) == 8,
    "all episodes": train.get("dataset", {}).get("episodes") == list(range(expected_episodes)),
    "source lineage": pretrained == source or output_dir in pretrained.parents,
}
failed = [name for name, ok in checks.items() if not ok]
if failed:
    raise SystemExit(f"Stage3 checkpoint failed canonical validation: {failed}")
PY
}

checkpoint_step() {
  "${TRAIN_PYTHON}" - "${OUTPUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

last = (Path(sys.argv[1]) / "checkpoints" / "last").resolve()
state = json.loads((last / "training_state" / "training_step.json").read_text(encoding="utf-8"))
print(int(state["step"]))
PY
}

wait_for_gpu_pair_idle() {
  local uuid_a uuid_b active
  uuid_a="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "${GPU_A}")"
  uuid_b="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "${GPU_B}")"
  while true; do
    active="$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null || true)"
    if ! grep -Fq "${uuid_a}" <<<"${active}" && ! grep -Fq "${uuid_b}" <<<"${active}"; then
      return 0
    fi
    echo "Waiting for physical GPUs ${GPU_A},${GPU_B} to become compute-idle."
    sleep "${WAIT_SECONDS}"
  done
}

LAUNCH=(
  "${ACCELERATE_BIN}" launch
  --num_machines=1
  --num_processes="${NUM_PROCESSES}"
  --multi_gpu
  --main_process_port="${MAIN_PROCESS_PORT}"
  --mixed_precision="${MIXED_PRECISION}"
  --dynamo_backend=no
  -m lerobot.scripts.lerobot_train
)

TRAIN_COMMAND=(
  "${LAUNCH[@]}"
  --dataset.repo_id="${DATASET_REPO_ID}"
  --dataset.root="${DATASET_ROOT}"
  --dataset.episodes="${EPISODES_JSON}"
  --dataset.video_backend=pyav
  --dataset.return_uint8=true
  --policy.type=smolvla_ttt
  --policy.pretrained_path="${SOURCE_CHECKPOINT}"
  --policy.device=cuda
  --policy.push_to_hub=false
  --policy.n_action_steps="${N_ACTION_STEPS}"
  --policy.sequence_length="${SEQUENCE_LENGTH}"
  --policy.sequence_stride="${SEQUENCE_STRIDE}"
  --policy.tbptt_segment_length="${TBPTT_SEGMENT_LENGTH}"
  --policy.ttt_hidden_dim="${TTT_HIDDEN_DIM}"
  --policy.ttt_layer_indices="${TTT_LAYERS}"
  --policy.ttt_base_inner_lr=0.1
  --policy.ttt_effective_gate_init="${EFFECTIVE_GATE_INIT}"
  --policy.ttt_second_order=true
  --policy.ttt_num_register_tokens="${REGISTER_TOKENS}"
  --policy.ttt_training_stage=action_head
  --policy.compile_model=false
  --batch_size="${BATCH_SIZE}"
  --num_workers="${NUM_WORKERS}"
  --prefetch_factor="${PREFETCH_FACTOR}"
  --persistent_workers=true
  --steps="${EXTRA_STEPS}"
  --log_freq="${LOG_FREQ}"
  --save_checkpoint=true
  --save_freq="${SAVE_INTERVAL}"
  --eval_freq=0
  --resume=false
  --wandb.enable=false
  --seed="${SEED}"
  --output_dir="${OUTPUT_DIR}"
)

write_manifest() {
  local midpoint_checkpoint=$1 final_checkpoint=$2
  "${TRAIN_PYTHON}" - \
    "${MANIFEST}" "${SOURCE_CHECKPOINT}" "${midpoint_checkpoint}" "${final_checkpoint}" \
    "${STEPS_PER_EPOCH}" "${SAVE_INTERVAL}" "${EXTRA_STEPS}" "${N_ACTION_STEPS}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(
    target,
    source_raw,
    midpoint_raw,
    final_raw,
    steps_per_epoch,
    midpoint_step,
    final_step,
    action_steps,
) = sys.argv[1:]
source = Path(source_raw).resolve()
midpoint = Path(midpoint_raw).resolve()
final = Path(final_raw).resolve()

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

payload = {
    "schema": "method1_smolvla_ttt_stage2_extension_v3",
    "task_id": "remember_shape5",
    "env_id": "RememberShape5-VLA-v0",
    "policy_type": "smolvla_ttt",
    "training_stage": "action_head",
    "sequence_state_semantics": "sequence_outer_step_v1",
    "inference_action_steps": int(action_steps),
    "fast_update_cadence": "once per action-chunk inference",
    "source_checkpoint": str(source),
    "source_model_sha256": sha256(source / "model.safetensors"),
    "source_total_ttt_sequence_epochs": 100,
    "extra_sequence_epochs": 100,
    "total_ttt_sequence_epochs": 200,
    "steps_per_sequence_epoch": int(steps_per_epoch),
    "extra_optimizer_steps": int(final_step),
    "optimizer_transition": "fresh AdamW at the Stage2-to-Stage3 warm-start boundary",
    "scheduler_transition": "fresh warmup+cosine at the Stage2-to-Stage3 warm-start boundary",
    "gate": "unfrozen; learned Stage2 gate weights are inherited",
    "per_device_batch_size": 8,
    "world_size": 2,
    "global_batch_size": 16,
    "all_official_demonstrations": True,
    "checkpoints": {
        "stage2plus50_action_head": {
            "checkpoint": str(midpoint),
            "model_sha256": sha256(midpoint / "model.safetensors"),
            "extra_sequence_epochs": 50,
            "optimizer_step": int(midpoint_step),
        },
        "stage2plus100_action_head": {
            "checkpoint": str(final),
            "model_sha256": sha256(final / "model.safetensors"),
            "extra_sequence_epochs": 100,
            "optimizer_step": int(final_step),
        },
    },
}
path = Path(target)
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_name(f".{path.name}.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
temporary.replace(path)
PY
}

run_eval() {
  local label=$1 checkpoint=$2 gpu=$3 output=$4 official=$5
  mkdir -p "$(dirname -- "${output}")"
  echo "Starting canonical ${label} evaluation on physical GPU ${gpu}."
  CUDA_VISIBLE_DEVICES="${gpu}" "${EVAL_PYTHON}" "${EVALUATOR}" \
    --policy-kind ttt \
    --checkpoint "${checkpoint}" \
    --dataset-repo-id "${DATASET_REPO_ID}" \
    --dataset-root "${DATASET_ROOT}" \
    --task "${ENV_ID}" \
    --num-episodes 50 \
    --start-seed 4242424242 \
    --torch-seed 7000 \
    --sim-backend gpu \
    --device cuda \
    --output "${output}" \
    --official-output-dir "${official}"
}

validate_source
wait_for_gpu_pair_idle

if [[ -f "${OUTPUT_DIR}/checkpoints/last/pretrained_model/model.safetensors" ]]; then
  CURRENT_STEP="$(checkpoint_step)"
  if (( CURRENT_STEP == EXTRA_STEPS )); then
    echo "Validated final Stage3 checkpoint already exists; skipping training."
    validate_output_checkpoint "${OUTPUT_DIR}/checkpoints/last/pretrained_model" "${EXTRA_STEPS}"
  elif (( CURRENT_STEP > 0 && CURRENT_STEP < EXTRA_STEPS )); then
    validate_output_checkpoint "${OUTPUT_DIR}/checkpoints/last/pretrained_model" "${CURRENT_STEP}"
    RESUME_CONFIG="$(readlink -f "${OUTPUT_DIR}/checkpoints/last")/pretrained_model/train_config.json"
    echo "Resuming interrupted Stage3 from step ${CURRENT_STEP}/${EXTRA_STEPS}."
    CUDA_VISIBLE_DEVICES="${GPU_A},${GPU_B}" \
      "${LAUNCH[@]}" --resume=true --config_path="${RESUME_CONFIG}"
    validate_output_checkpoint "${OUTPUT_DIR}/checkpoints/last/pretrained_model" "${EXTRA_STEPS}"
  else
    die "unsafe Stage3 checkpoint step ${CURRENT_STEP}"
  fi
elif [[ -e "${OUTPUT_DIR}" || -L "${OUTPUT_DIR}" ]]; then
  die "partial Stage3 output exists without a resumable checkpoint: ${OUTPUT_DIR}"
else
  echo "Starting Stage3 extra 100 sequence epochs on physical GPUs ${GPU_A},${GPU_B}."
  cd "${REPO_ROOT}"
  CUDA_VISIBLE_DEVICES="${GPU_A},${GPU_B}" "${TRAIN_COMMAND[@]}"
  validate_output_checkpoint "${OUTPUT_DIR}/checkpoints/last/pretrained_model" "${EXTRA_STEPS}"
fi

printf -v MIDPOINT_STEP_DIR '%06d' "${SAVE_INTERVAL}"
printf -v FINAL_STEP_DIR '%06d' "${EXTRA_STEPS}"
MIDPOINT_CHECKPOINT="$(readlink -f "${OUTPUT_DIR}/checkpoints/${MIDPOINT_STEP_DIR}/pretrained_model")"
FINAL_CHECKPOINT="$(readlink -f "${OUTPUT_DIR}/checkpoints/${FINAL_STEP_DIR}/pretrained_model")"
validate_output_checkpoint "${MIDPOINT_CHECKPOINT}" "${SAVE_INTERVAL}"
validate_output_checkpoint "${FINAL_CHECKPOINT}" "${EXTRA_STEPS}"
write_manifest "${MIDPOINT_CHECKPOINT}" "${FINAL_CHECKPOINT}"
wait_for_gpu_pair_idle

run_eval stage2plus50_action_head \
  "${MIDPOINT_CHECKPOINT}" "${GPU_A}" "${MIDPOINT_EVAL}" "${MIDPOINT_OFFICIAL}" &
midpoint_eval_pid=$!
run_eval stage2plus100_action_head \
  "${FINAL_CHECKPOINT}" "${GPU_B}" "${FINAL_EVAL}" "${FINAL_OFFICIAL}" &
final_eval_pid=$!
status=0
wait "${midpoint_eval_pid}" || status=1
wait "${final_eval_pid}" || status=1
(( status == 0 )) || die "one or both Stage3 evaluations failed; partial files are restart-safe"

[[ -f "${COMPARISON_INPUT}" ]] \
  || die "baseline-250 comparison is missing: ${COMPARISON_INPUT}"
"${EVAL_PYTHON}" - \
  "${COMPARISON_INPUT}" "${MIDPOINT_EVAL}" "${FINAL_EVAL}" "${MANIFEST}" \
  "${COMPARISON_OUTPUT}" "${EVALUATION_ROOT}" "${N_ACTION_STEPS}" <<'PY'
import json
import sys
from pathlib import Path

comparison_path, midpoint_path, final_path, manifest_path, output_path, root = map(Path, sys.argv[1:7])
ttt_cadence = int(sys.argv[7])
comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
benchmark_commit = "509b875f3d207c287497c0a897661062de928bb0"
expected_env_seeds = list(range(4242424242, 4242424292))
expected_torch_seeds = list(range(7000, 7050))
existing_specs = {
    "baseline_native_k50": (root / "baseline_native_k50" / "eval.json", 50),
    "stage1_ttt_only": (root / "stage1_ttt_only" / "eval.json", ttt_cadence),
    "stage2_action_head": (root / "stage2_action_head" / "eval.json", ttt_cadence),
    "baseline_250ep_warmstart_k50": (root / "baseline_250ep_warmstart_k50" / "eval.json", 50),
}

def validate_eval(path: Path, cadence: int, expected_checkpoint: str | None = None):
    payload = json.loads(path.read_text(encoding="utf-8"))
    identity = payload["evaluation_identity"]
    result = payload["results"][0]
    checks = {
        "env": result.get("env_id") == "RememberShape5-VLA-v0",
        "episodes": result.get("n_episodes") == 50 and len(result.get("successes", [])) == 50,
        "start seed": result.get("start_seed") == 4242424242,
        "torch seed": result.get("torch_seed") == 7000,
        "env seed stream": result.get("episode_seeds") == expected_env_seeds,
        "torch seed stream": result.get("episode_torch_seeds") == expected_torch_seeds,
        "cadence": result.get("execution_action_steps") == cadence,
        "benchmark commit": result.get("benchmark_commit") == benchmark_commit,
        "sim backend": result.get("sim_backend") == "gpu",
        "observation mode": result.get("obs_mode") == "rgb",
        "reward mode": result.get("reward_mode") == "normalized_dense",
        "identity checkpoint": identity.get("checkpoint") == result.get("model", {}).get("checkpoint"),
        "identity episodes": identity.get("n_episodes") == 50,
        "identity start seed": identity.get("start_seed") == 4242424242,
        "identity torch seed": identity.get("torch_seed") == 7000,
        "identity cadence": identity.get("execution_action_steps") == cadence,
    }
    if expected_checkpoint is not None:
        checks["expected checkpoint"] = identity.get("checkpoint") == expected_checkpoint
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise SystemExit(f"non-canonical evaluation {path}: {failed}")
    return payload, result

for label, (path, cadence) in existing_specs.items():
    validate_eval(path, cadence)

midpoint_checkpoint = manifest["checkpoints"]["stage2plus50_action_head"]["checkpoint"]
final_checkpoint = manifest["checkpoints"]["stage2plus100_action_head"]["checkpoint"]
midpoint, midpoint_result = validate_eval(midpoint_path, ttt_cadence, midpoint_checkpoint)
final, final_result = validate_eval(final_path, ttt_cadence, final_checkpoint)
if midpoint["evaluation_identity"]["checkpoint_model_sha256"] != manifest["checkpoints"]["stage2plus50_action_head"]["model_sha256"]:
    raise SystemExit("midpoint evaluation SHA does not match Stage3 manifest")
if final["evaluation_identity"]["checkpoint_model_sha256"] != manifest["checkpoints"]["stage2plus100_action_head"]["model_sha256"]:
    raise SystemExit("final evaluation SHA does not match Stage3 manifest")

rows = [
    item for item in comparison["models"]
    if item.get("label") not in {"stage2plus50_action_head", "stage2plus100_action_head"}
]
for label, payload, result in (
    ("stage2plus50_action_head", midpoint, midpoint_result),
    ("stage2plus100_action_head", final, final_result),
):
    rows.append(
        {
            "label": label,
            "checkpoint": payload["evaluation_identity"]["checkpoint"],
            "checkpoint_model_sha256": payload["evaluation_identity"]["checkpoint_model_sha256"],
            "sr": result["sr"],
            "mean_return": result["mean_return"],
            "n_episodes": result["n_episodes"],
            "start_seed": result["start_seed"],
            "execution_action_steps": result["execution_action_steps"],
            "training_extension": manifest,
        }
    )

_, stage2 = validate_eval(existing_specs["stage2_action_head"][0], ttt_cadence)
stage2_successes = [bool(value) for value in stage2["successes"]]
paired = {}
for label, result in (
    ("stage2plus50_action_head", midpoint_result),
    ("stage2plus100_action_head", final_result),
):
    new_successes = [bool(value) for value in result["successes"]]
    paired[label] = {
        "reference": "stage2_action_head",
        "both_success": sum(a and b for a, b in zip(stage2_successes, new_successes, strict=True)),
        "new_only_success": sum((not a) and b for a, b in zip(stage2_successes, new_successes, strict=True)),
        "reference_only_success": sum(a and (not b) for a, b in zip(stage2_successes, new_successes, strict=True)),
        "both_failure": sum((not a) and (not b) for a, b in zip(stage2_successes, new_successes, strict=True)),
        "delta_sr": float(result["sr"]) - float(stage2["sr"]),
        "delta_mean_return": float(result["mean_return"]) - float(stage2["mean_return"]),
    }

payload = dict(comparison)
payload["models"] = rows
payload["stage2_extension_note"] = (
    "Both extension checkpoints inherit the completed Stage2 weights and use a fresh AdamW/cosine "
    "optimization stage with the gate unfrozen; they are not an in-place optimizer-state resume."
)
payload["paired_vs_stage2"] = paired
if output_path.exists():
    existing = json.loads(output_path.read_text(encoding="utf-8"))
    if existing == payload:
        print(f"Stage2-extension comparison already complete: {output_path}")
        raise SystemExit(0)
    raise SystemExit(f"refusing to overwrite a different existing comparison: {output_path}")
temporary = output_path.with_name(f".{output_path.name}.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
temporary.replace(output_path)
print(f"Stage2-extension comparison complete: {output_path}")
PY

echo "Completed RememberShape5 Stage3:"
echo "  midpoint_checkpoint=${MIDPOINT_CHECKPOINT}"
echo "  final_checkpoint=${FINAL_CHECKPOINT}"
echo "  midpoint_evaluation=${MIDPOINT_EVAL}"
echo "  final_evaluation=${FINAL_EVAL}"
echo "  comparison=${COMPARISON_OUTPUT}"
