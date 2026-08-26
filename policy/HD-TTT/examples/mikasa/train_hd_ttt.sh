#!/usr/bin/env bash

# Reproducible MIKASA HD-TTT training recipe.
#
# The compatibility defaults keep the original suffix writer and legacy HD
# objective when HD is disabled.  When ``HD_ENABLED=true`` and the protocol is
# not explicitly supplied, the script selects the v2 attribution protocol and
# enables the action-effect term; this prevents the common failure mode where
# the v2 label builder is paired with a legacy policy config.  The paper
# commands still spell out every structural choice (including
# ``TTT_WRITER_MODE=prefix_only``) so a checkpoint is self-auditing.
#
# A window-keyed HD artifact stores a complete replay context for each
# selected window, so the window cap is an exact contract rather than an
# approximate frame-label shortcut.  For strict full-history labels, use the
# frame-level artifact emitted by ``build_hd_labels.py`` and set
# ``MAX_WINDOWS_PER_EPISODE=1`` with ``HISTORY_WARMUP_LENGTH=full``.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/test-TTT/policy/HD-TTT}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/MIKASA-Robo/.venv/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/workspace/MIKASA-Robo/.venv/bin/accelerate}"
DATASET_REPO_ID="${DATASET_REPO_ID:?set DATASET_REPO_ID}"
DATASET_ROOT="${DATASET_ROOT:?set DATASET_ROOT}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR}"
PRETRAINED_PATH="${PRETRAINED_PATH:-lerobot/smolvla_base}"
LABEL_PATH="${LABEL_PATH:-}"
EPOCHS="${EPOCHS:-150}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
# Keep the benchmark recipe on bf16 by default, but allow a reproducible
# fp32/no-autocast diagnostic (and an explicit mixed-precision ablation)
# without copying or hand-editing this launcher.  This is an execution
# setting, not an HD-TTT algorithm parameter.
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
# Optional accelerate rank log fan-out for distributed bring-up.  Empty keeps
# the normal single combined log; when set, ``--tee``/``--log_dir`` preserve
# each rank's traceback without changing model math.
ACCELERATE_TEE="${ACCELERATE_TEE:-}"
ACCELERATE_LOG_DIR="${ACCELERATE_LOG_DIR:-}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-64}"
SEQUENCE_STRIDE="${SEQUENCE_STRIDE:-64}"
MAX_WINDOWS_PER_EPISODE="${MAX_WINDOWS_PER_EPISODE:-4}"
TBPTT_SEGMENT_LENGTH="${TBPTT_SEGMENT_LENGTH:-32}"
HISTORY_WARMUP_LENGTH="${HISTORY_WARMUP_LENGTH:-64}"
if [[ "${HISTORY_WARMUP_LENGTH}" == "full" ]]; then
  # draccus decodes Optional[int] from JSON null, not the Python spelling
  # ``None``.  Keep the shell sentinel human-readable, but pass a value the
  # training CLI can actually decode.
  HISTORY_WARMUP_ARG="null"
else
  HISTORY_WARMUP_ARG="${HISTORY_WARMUP_LENGTH}"
fi
# Window-keyed HD labels carry their numeric replay context in metadata and
# are checked against this value at startup.  If LABEL_PATH points to the
# frame-level full-episode artifact from ``build_hd_labels.py``, the ``full``
# sentinel above is the correct setting; for a window-keyed artifact use its
# exact numeric context length instead.
TTT_HIDDEN_DIM="${TTT_HIDDEN_DIM:-1024}"
TTT_LAYERS="${TTT_LAYERS:-[12,13,14,15]}"
REGISTER_TOKENS="${REGISTER_TOKENS:-16}"
# Leave the mode unresolved until ``HD_ENABLED`` is known below.  A plain
# ``HD_ENABLED=true`` invocation must enter the paper's observation-only
# writer; silently falling back to the legacy suffix writer would make the
# resulting run action-noise dependent while still advertising the v2 label
# protocol.  Explicit ``TTT_WRITER_MODE`` remains authoritative, and clean /
# legacy runs retain the suffix compatibility default.
TTT_WRITER_MODE="${TTT_WRITER_MODE:-}"
# The formal HD/v2 recipe uses the bounded inner update by default.  Set this
# to ``false`` only for an explicit legacy/clean ablation; v2 action-effect
# supervision below rejects that combination because its labels must match the
# student's recurrence.
TTT_STABLE_INNER_UPDATE="${TTT_STABLE_INNER_UPDATE:-true}"
RESIZE="${RESIZE:-[224,224]}"
TRAINING_STAGE="${TRAINING_STAGE:-ttt_only}"
HD_ENABLED="${HD_ENABLED:-false}"
# The core v2 method distills hindsight into writer content/effect; it does
# not rely on a separately tuned online gate.  The learned prefix gate stays
# available as an explicit ablation via HD_LEARNED_GATE=true.
HD_LEARNED_GATE="${HD_LEARNED_GATE:-false}"
HD_PHASE_MODE="${HD_PHASE_MODE:-deployment}"
HD_EVENT_BLOCK_SIZE="${HD_EVENT_BLOCK_SIZE:-4}"
HD_MAX_EVENTS="${HD_MAX_EVENTS:-0}"
HD_GROUNDING_MIN_FUTURE_FRAMES="${HD_GROUNDING_MIN_FUTURE_FRAMES:-64}"
HD_ATTRIBUTION_THRESHOLD="${HD_ATTRIBUTION_THRESHOLD:-0.0}"
HD_HCA_WEIGHT="${HD_HCA_WEIGHT:-1.0}"
HD_H2L_WEIGHT="${HD_H2L_WEIGHT:-1.0}"
HD_GROUNDING_WEIGHT="${HD_GROUNDING_WEIGHT:-}"
HD_INVARIANCE_WEIGHT="${HD_INVARIANCE_WEIGHT:-0.25}"
HD_WRITE_GATE_WEIGHT="${HD_WRITE_GATE_WEIGHT:-1.0}"
HD_COUNTERFACTUAL_MARGIN="${HD_COUNTERFACTUAL_MARGIN:-0.0}"
# Select the structural writer default only after the HD switch is available.
# This is an entry-point safety rule, not a tunable algorithm parameter:
# callers can still request the suffix path explicitly for a registered
# ablation/legacy checkpoint.
if [[ -z "${TTT_WRITER_MODE}" ]]; then
  if [[ "${HD_ENABLED,,}" == "true" ]]; then
    TTT_WRITER_MODE="prefix_only"
  else
    TTT_WRITER_MODE="suffix"
  fi
fi
# Keep the old protocol for clean/legacy runs, but make an HD run safe by
# defaulting to the paper v2 contract.  The ``+x`` test distinguishes an
# omitted variable from an explicit legacy ablation.
HD_ENABLED_NORMALIZED="${HD_ENABLED,,}"
if [[ -z "${HD_ATTRIBUTION_PROTOCOL+x}" ]]; then
  if [[ "${HD_ENABLED_NORMALIZED}" == "true" ]]; then
    HD_ATTRIBUTION_PROTOCOL="v2_relative_antithetic_robust"
  else
    HD_ATTRIBUTION_PROTOCOL="legacy_raw_hinge_max"
  fi
fi
if [[ -z "${HD_EFFECT_WEIGHT+x}" ]]; then
  HD_EFFECT_WEIGHT="0.0"
  if [[ "${HD_ENABLED_NORMALIZED}" == "true" ]]; then
    HD_EFFECT_WEIGHT="1.0"
  fi
fi
if [[ -z "${HD_GROUNDING_WEIGHT}" ]]; then
  # v2 effect replay already supplies the causal true/wrong intervention;
  # detached reader grounding is retained for legacy/no-effect runs and
  # explicit ablations, not stacked into the paper objective.
  if [[ "${HD_ENABLED_NORMALIZED}" == "true" && "${HD_EFFECT_WEIGHT}" != "0" && "${HD_EFFECT_WEIGHT}" != "0.0" ]]; then
    HD_GROUNDING_WEIGHT="0.0"
  else
    HD_GROUNDING_WEIGHT="1.0"
  fi
fi
# The action-effect term differentiates through the inner fast-weight update.
# Select the required second-order path automatically for the v2 paper run;
# clean/legacy runs keep the cheaper first-order default.  An explicit value
# remains available for ablations, but v2 with effect supervision is rejected
# by the policy if it is set to false rather than silently changing the method.
if [[ -z "${TTT_SECOND_ORDER+x}" ]]; then
  if [[ "${HD_ENABLED_NORMALIZED}" == "true" && "${HD_EFFECT_WEIGHT}" != "0" && "${HD_EFFECT_WEIGHT}" != "0.0" ]]; then
    TTT_SECOND_ORDER="true"
  else
    TTT_SECOND_ORDER="false"
  fi
fi
SAVE_FREQ="${SAVE_FREQ:-500}"
LOG_FREQ="${LOG_FREQ:-50}"
SEED="${SEED:-1000}"
RESUME="${RESUME:-false}"
CONFIG_PATH="${CONFIG_PATH:-}"
# Optional comma-free JSON/list syntax understood by draccus, e.g. ``[0]`` or
# ``[0,1,2]``.  This is useful for a bounded smoke run and for reproducible
# per-shard experiments; when unset all 250 demonstrations are used.
DATASET_EPISODES="${DATASET_EPISODES:-}"

case "${TTT_WRITER_MODE}" in
  suffix|prefix_only) ;;
  *)
    echo "TTT_WRITER_MODE must be 'suffix' or 'prefix_only', got '${TTT_WRITER_MODE}'" >&2
    exit 2
    ;;
esac
case "${TTT_STABLE_INNER_UPDATE,,}" in
  true|false)
    TTT_STABLE_INNER_UPDATE="${TTT_STABLE_INNER_UPDATE,,}"
    ;;
  *)
    echo "TTT_STABLE_INNER_UPDATE must be true or false, got '${TTT_STABLE_INNER_UPDATE}'" >&2
    exit 2
    ;;
esac
case "${HD_ATTRIBUTION_PROTOCOL}" in
  legacy|legacy_raw_hinge_max)
    HD_ATTRIBUTION_PROTOCOL="legacy_raw_hinge_max"
    ;;
  v2|v2_relative_antithetic_robust)
    HD_ATTRIBUTION_PROTOCOL="v2_relative_antithetic_robust"
    ;;
  *)
    echo "HD_ATTRIBUTION_PROTOCOL must be legacy or v2, got '${HD_ATTRIBUTION_PROTOCOL}'" >&2
    exit 2
    ;;
esac
if ! [[ "${HD_EFFECT_WEIGHT}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "HD_EFFECT_WEIGHT must be a non-negative number, got '${HD_EFFECT_WEIGHT}'" >&2
  exit 2
fi
if [[ "${HD_ATTRIBUTION_PROTOCOL}" == "v2_relative_antithetic_robust" \
      && ! "${HD_EFFECT_WEIGHT}" =~ ^0*([.]0*)?$ \
      && "${TTT_STABLE_INNER_UPDATE}" != "true" ]]; then
  echo "v2 action-effect training requires TTT_STABLE_INNER_UPDATE=true; got '${TTT_STABLE_INNER_UPDATE}'" >&2
  exit 2
fi
if [[ "${HD_ATTRIBUTION_PROTOCOL}" == "legacy_raw_hinge_max" && "${HD_EFFECT_WEIGHT}" != "0" && "${HD_EFFECT_WEIGHT}" != "0.0" ]]; then
  echo "warning: legacy attribution labels do not contain v2 action-effect targets; HD_EFFECT_WEIGHT=${HD_EFFECT_WEIGHT} will have no effect" >&2
fi

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/workspace/data_mikasa_robo}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

if [[ "${MAX_WINDOWS_PER_EPISODE}" == "none" ]]; then
  # Same Optional[int] contract as ttt_history_warmup_length above.
  MAX_WINDOWS_ARG="null"
else
  MAX_WINDOWS_ARG="${MAX_WINDOWS_PER_EPISODE}"
fi

# Count the exact sequence windows consumed by TailPreservingSequenceDataset
# so ``steps`` really denotes the requested number of dataset epochs.
WINDOW_STATS="$(${PYTHON_BIN} - "${DATASET_ROOT}" "${SEQUENCE_LENGTH}" "${SEQUENCE_STRIDE}" "${MAX_WINDOWS_ARG}" "${DATASET_EPISODES}" <<'PY'
import json
import sys
from pathlib import Path

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

root = Path(sys.argv[1])
length = int(sys.argv[2])
stride = int(sys.argv[3])
cap = None if sys.argv[4] in {"None", "null"} else int(sys.argv[4])
selected = None
if sys.argv[5]:
    selected = {int(value) for value in json.loads(sys.argv[5].replace("'", '"'))}
meta = LeRobotDatasetMetadata(root.name, root=root)
episodes = meta.episodes
total = 0
min_episode_length = None
max_episode_length = 0
for episode_index, row in enumerate(episodes):
    if selected is not None and episode_index not in selected:
        continue
    n = int(row["dataset_to_index"]) - int(row["dataset_from_index"])
    min_episode_length = n if min_episode_length is None else min(min_episode_length, n)
    max_episode_length = max(max_episode_length, n)
    # The one-window full-history recipe must not silently discard a suffix if
    # a future dataset contains an episode longer than the configured replay
    # capacity. Fail before launching distributed training.
    if cap == 1 and n > length:
        raise SystemExit(
            "max_windows_per_episode=1 requires sequence_length >= every selected "
            f"episode length; episode {episode_index} has {n} frames but sequence_length={length}"
        )
    offsets = list(range(0, n, stride))
    if cap is not None and len(offsets) > cap:
        last_full_offset = max(n - length, 0)
        full_offsets = list(range(0, last_full_offset + 1, stride))
        if not full_offsets or full_offsets[-1] != last_full_offset:
            full_offsets.append(last_full_offset)
        # Match TailPreservingSequenceDataset's deterministic linspace/round
        # selection exactly, including its terminal full window.
        positions = [round(i * (len(full_offsets) - 1) / (cap - 1)) for i in range(cap)] if cap > 1 else [0]
        offsets = [full_offsets[position] for position in sorted(set(positions))]
    total += len(offsets)
print(total, min_episode_length, max_episode_length)
PY
)"
WINDOWS="$("${PYTHON_BIN}" -c 'import sys; print(sys.argv[1].split()[0])' "${WINDOW_STATS}")"
MIN_EPISODE_LENGTH="$("${PYTHON_BIN}" -c 'import sys; print(sys.argv[1].split()[1])' "${WINDOW_STATS}")"
MAX_EPISODE_LENGTH="$("${PYTHON_BIN}" -c 'import sys; print(sys.argv[1].split()[2])' "${WINDOW_STATS}")"
STEPS_PER_EPOCH=$(( (WINDOWS + NUM_PROCESSES - 1) / NUM_PROCESSES ))
STEPS=$(( STEPS_PER_EPOCH * EPOCHS ))

echo "MIKASA HD-TTT: windows=${WINDOWS}, episode_length=${MIN_EPISODE_LENGTH}..${MAX_EPISODE_LENGTH}, steps/epoch=${STEPS_PER_EPOCH}, epochs=${EPOCHS}, steps=${STEPS}, resume=${RESUME}, writer=${TTT_WRITER_MODE}, stable_inner_update=${TTT_STABLE_INNER_UPDATE}, attribution=${HD_ATTRIBUTION_PROTOCOL}, second_order=${TTT_SECOND_ORDER}, grounding_min_future=${HD_GROUNDING_MIN_FUTURE_FRAMES}, margin=${HD_COUNTERFACTUAL_MARGIN}, effect_weight=${HD_EFFECT_WEIGHT}, grounding_weight=${HD_GROUNDING_WEIGHT}"
echo "mixed_precision=${MIXED_PRECISION}"

COMMON_ARGS=(
  --dataset.repo_id="${DATASET_REPO_ID}"
  --dataset.root="${DATASET_ROOT}"
  --dataset.video_backend=pyav
  --dataset.return_uint8=true
  --policy.type=smolvla_ttt
  --policy.pretrained_path="${PRETRAINED_PATH}"
  --policy.device=cuda
  --policy.push_to_hub=false
  --policy.sequence_length="${SEQUENCE_LENGTH}"
  --policy.sequence_stride="${SEQUENCE_STRIDE}"
  --policy.max_windows_per_episode="${MAX_WINDOWS_ARG}"
  --policy.tbptt_segment_length="${TBPTT_SEGMENT_LENGTH}"
  --policy.ttt_history_warmup_length="${HISTORY_WARMUP_ARG}"
  --policy.ttt_hidden_dim="${TTT_HIDDEN_DIM}"
  --policy.ttt_second_order="${TTT_SECOND_ORDER}"
  --policy.ttt_stable_inner_update="${TTT_STABLE_INNER_UPDATE}"
  --policy.ttt_layer_indices="${TTT_LAYERS}"
  --policy.ttt_num_register_tokens="${REGISTER_TOKENS}"
  --policy.ttt_writer_mode="${TTT_WRITER_MODE}"
  --policy.ttt_training_stage="${TRAINING_STAGE}"
  --policy.resize_imgs_with_padding="${RESIZE}"
  --policy.hd_ttt_enabled="${HD_ENABLED}"
  --policy.hd_learned_write_gate="${HD_LEARNED_GATE}"
  --policy.hd_phase_mode="${HD_PHASE_MODE}"
  --policy.hd_event_block_size="${HD_EVENT_BLOCK_SIZE}"
  --policy.hd_max_events="${HD_MAX_EVENTS}"
  --policy.hd_grounding_min_future_frames="${HD_GROUNDING_MIN_FUTURE_FRAMES}"
  --policy.hd_attribution_threshold="${HD_ATTRIBUTION_THRESHOLD}"
  --policy.hd_hca_weight="${HD_HCA_WEIGHT}"
  --policy.hd_h2l_weight="${HD_H2L_WEIGHT}"
  --policy.hd_effect_weight="${HD_EFFECT_WEIGHT}"
  --policy.hd_grounding_weight="${HD_GROUNDING_WEIGHT}"
  --policy.hd_invariance_weight="${HD_INVARIANCE_WEIGHT}"
  --policy.hd_write_gate_weight="${HD_WRITE_GATE_WEIGHT}"
  --policy.hd_counterfactual_margin="${HD_COUNTERFACTUAL_MARGIN}"
  --policy.hd_attribution_protocol="${HD_ATTRIBUTION_PROTOCOL}"
  --batch_size=1
  --num_workers="${NUM_WORKERS:-4}"
  --prefetch_factor="${PREFETCH_FACTOR:-2}"
  --persistent_workers=true
  --steps="${STEPS}"
  --log_freq="${LOG_FREQ}"
  --save_checkpoint=true
  --save_freq="${SAVE_FREQ}"
  --eval_freq=0
  --resume="${RESUME}"
  --wandb.enable=false
  --seed="${SEED}"
  --output_dir="${OUTPUT_DIR}"
)

if [[ -n "${LABEL_PATH}" ]]; then
  COMMON_ARGS+=(--dataset.hd_label_path="${LABEL_PATH}")
fi
if [[ -n "${DATASET_EPISODES}" ]]; then
  COMMON_ARGS+=(--dataset.episodes="${DATASET_EPISODES}")
fi
if [[ -n "${CONFIG_PATH}" ]]; then
  COMMON_ARGS+=(--config_path="${CONFIG_PATH}")
fi

LAUNCH=(
  "${ACCELERATE_BIN}" launch
  --num_machines=1
  --num_processes="${NUM_PROCESSES}"
  --mixed_precision="${MIXED_PRECISION}"
  --dynamo_backend=no
)
if [[ -n "${ACCELERATE_TEE}" ]]; then
  LAUNCH+=(--tee="${ACCELERATE_TEE}")
fi
if [[ -n "${ACCELERATE_LOG_DIR}" ]]; then
  LAUNCH+=(--log_dir="${ACCELERATE_LOG_DIR}")
fi
if (( NUM_PROCESSES > 1 )); then
  LAUNCH+=(--multi_gpu)
fi
LAUNCH+=(-m lerobot.scripts.lerobot_train)

cd "${REPO_ROOT}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'DRY RUN:'
  printf ' %q' "${LAUNCH[@]}" "${COMMON_ARGS[@]}"
  printf '\n'
  exit 0
fi
exec "${LAUNCH[@]}" "${COMMON_ARGS[@]}"
