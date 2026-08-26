"""Executable-contract tests for the CreditTTT shell launcher.

The launcher performs a dataset metadata preflight before constructing the
student command.  These tests replace only that metadata reader with a tiny
stub, allowing the step/epoch checks to run without downloading a MIKASA
dataset or starting distributed training.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest


LAUNCHER = Path(__file__).parents[1] / "examples" / "mikasa" / "train_credit_ttt.sh"


def _make_fake_python(tmp_path: Path, *, batches_per_rank: int = 4) -> Path:
    """Return a Python shim that answers only the launcher metadata probe."""

    shim = tmp_path / "fake-python"
    # The preflight invokes ``python -`` with exactly ten metadata arguments;
    # all other invocations (the ``-c`` JSON extractors and sidecar writer) are
    # delegated to the real interpreter so the test exercises the shell path
    # and metadata serialization as it would run in production.
    payload = {
        "schema": "credit_ttt_training_batch_v1",
        "windows": 4,
        "raw_windows": 4,
        "min_episode_length": 4,
        "max_episode_length": 4,
        "sequence_length": 4,
        "sequence_stride": 4,
        "fps": 30,
        "total_batches": batches_per_rank,
        "batches_per_rank": batches_per_rank,
        "batch_size": 1,
        "per_device_batch_size": 1,
        "world_size": 1,
        "global_batch_size": 1,
        "equal_length_batching": False,
        "sampler": "accelerate_default",
        "ddp_flow_weighting": "historical_rank_mean",
        "no_temporal_padding": True,
        "stats_available": True,
        "offset_domains": 1,
        "bucket_count": None,
        "buckets": [],
        "groups_before_ddp": batches_per_rank,
        "ddp_repeated_groups": 0,
        "total_groups": batches_per_rank,
        "bucket_fill_repeated_rows": 0,
        "ddp_repeated_rows": 0,
        "total_repeated_rows": 0,
        "effective_rows": batches_per_rank,
        "repeat_rate": 0.0,
        "steps_per_epoch_per_rank": batches_per_rank,
    }
    shim.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "import subprocess\n"
        "import sys\n"
        f"REAL = {sys.executable!r}\n"
        f"PAYLOAD = {json.dumps(payload)!r}\n"
        "if len(sys.argv) == 12 and sys.argv[1] == '-':\n"
        "    sys.stdin.read()\n"
        "    print(PAYLOAD)\n"
        "else:\n"
        "    raise SystemExit(subprocess.call([REAL, *sys.argv[1:]]))\n",
        encoding="utf-8",
    )
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR)
    return shim


def _make_noop_accelerate(tmp_path: Path) -> Path:
    """Return an executable that absorbs the printed student command."""

    shim = tmp_path / "fake-accelerate"
    shim.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR)
    return shim


def _run_launcher(
    tmp_path: Path,
    *,
    steps: int | None = None,
    allow_short_run: bool = False,
    min_sequence_epochs: int = 0,
    batches_per_rank: int = 4,
) -> subprocess.CompletedProcess[str]:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    base_checkpoint = tmp_path / "base"
    base_checkpoint.touch()
    label_path = tmp_path / "credit_pairs.pt"
    label_path.touch()
    output_root = tmp_path / "output"
    shim = _make_fake_python(tmp_path, batches_per_rank=batches_per_rank)
    fake_accelerate = _make_noop_accelerate(tmp_path)
    env = os.environ.copy()
    for name in (
        "STEPS",
        "EPOCHS",
        "MIN_SEQUENCE_EPOCHS",
        "ALLOW_SHORT_RUN",
        "TRAIN_EPISODES",
    ):
        env.pop(name, None)
    env.update(
        {
            "PYTHON_BIN": str(shim),
            "ACCELERATE_BIN": str(fake_accelerate),
            "DATASET_ROOT": str(dataset_root),
            "BASE_CHECKPOINT": str(base_checkpoint),
            "LABEL_PATH": str(label_path),
            "OUTPUT_ROOT": str(output_root),
            "TRAINING_METADATA_PATH": str(output_root / "training_metadata.json"),
            "TRAIN_EPISODES": "0,1,2,3",
            "TRAIN_EPISODE_START": "0",
            "TRAIN_EPISODE_END": "4",
            "FEATURE_EPISODE_END": "4",
            "NUM_PROCESSES": "1",
            "BATCH_SIZE": "1",
            "EPOCHS": "2",
            "MIN_SEQUENCE_EPOCHS": str(min_sequence_epochs),
            "ALLOW_SHORT_RUN": "1" if allow_short_run else "0",
            "SEQUENCE_LENGTH": "auto",
            "SEQUENCE_STRIDE": "auto",
            "MAX_WINDOWS_PER_EPISODE": "1",
            "HISTORY_WARMUP_LENGTH": "null",
            # The real launcher protects executable stages behind EXECUTE=1.
            # The fake accelerate binary above makes that execution side-effect
            # free while still exercising the complete preflight/metadata path.
            "EXECUTE": "1",
        }
    )
    if steps is not None:
        env["STEPS"] = str(steps)
    return subprocess.run(
        [str(LAUNCHER), "student"],
        cwd=LAUNCHER.parents[2],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_launcher_implicit_steps_are_complete_epochs_and_recorded(tmp_path: Path) -> None:
    result = _run_launcher(tmp_path, min_sequence_epochs=2)
    assert result.returncode == 0, result.stderr
    metadata = json.loads(
        (tmp_path / "output" / "training_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["steps"] == 8
    assert metadata["steps_per_epoch"] == 4
    assert metadata["steps_was_explicit"] is False
    assert metadata["complete_sequence_epochs"] == 2


def test_launcher_accepts_explicit_complete_epoch_steps(tmp_path: Path) -> None:
    result = _run_launcher(tmp_path, steps=12, min_sequence_epochs=2)
    assert result.returncode == 0, result.stderr
    metadata = json.loads(
        (tmp_path / "output" / "training_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["steps"] == 12
    assert metadata["steps_was_explicit"] is True
    assert metadata["complete_sequence_epochs"] == 3


def test_launcher_rejects_explicit_partial_epoch_in_canonical_mode(tmp_path: Path) -> None:
    result = _run_launcher(tmp_path, steps=6, min_sequence_epochs=0)
    assert result.returncode != 0
    assert "complete number of sequence epochs" in result.stderr


def test_launcher_allows_partial_explicit_steps_only_for_named_smoke(tmp_path: Path) -> None:
    result = _run_launcher(tmp_path, steps=1, allow_short_run=True)
    assert result.returncode == 0, result.stderr
    metadata = json.loads(
        (tmp_path / "output" / "training_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["steps_was_explicit"] is True
    assert metadata["complete_sequence_epochs"] is None


@pytest.mark.parametrize("bad_steps", [0, -1, "not-an-int"])
def test_launcher_rejects_non_positive_explicit_steps(tmp_path: Path, bad_steps: object) -> None:
    result = _run_launcher(tmp_path, steps=bad_steps)  # type: ignore[arg-type]
    assert result.returncode != 0
    assert "STEPS must be a positive integer" in result.stderr
