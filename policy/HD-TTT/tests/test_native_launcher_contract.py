"""Contract tests for the task-trained Native SmolVLA launcher."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path


LAUNCHER = Path(__file__).parents[1] / "examples" / "mikasa" / "train_native_smolvla.sh"


def _make_fake_python(tmp_path: Path, *, all_indices: list[int] | None = None) -> Path:
    """Shim metadata imports while delegating metadata serialization to Python."""

    indices = all_indices or [0, 1, 2, 3]
    stats = {
        "schema": "native_smolvla_training_batch_v1",
        "dataset_repo_id": "fake_dataset",
        "dataset_root": str(tmp_path / "dataset"),
        "available_episode_indices": indices,
        "train_episode_indices": indices,
        "num_episodes": len(indices),
        "num_frames": 40,
        "fps": 30,
        "per_device_batch_size": 2,
        "world_size": 2,
        "global_batch_size": 4,
        "raw_batches": 20,
        "steps_per_epoch": 10,
        "effective_frame_slots_per_epoch": 40,
        "sampler_repeated_slots_per_epoch": 0,
        "all_official_demos": True,
    }
    shim = tmp_path / "fake-python"
    shim.write_text(
        "#!/usr/bin/env python3\n"
        "import json, subprocess, sys\n"
        f"REAL = {sys.executable!r}\n"
        f"STATS = {json.dumps(stats)!r}\n"
        "# Resolver invocation has eight arguments after the '-' marker.\n"
        "if len(sys.argv) == 10 and sys.argv[1] == '-':\n"
        "    sys.stdin.read()\n"
        "    print(STATS)\n"
        "else:\n"
        "    raise SystemExit(subprocess.call([REAL, *sys.argv[1:]]))\n",
        encoding="utf-8",
    )
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR)
    return shim


def _make_noop_accelerate(tmp_path: Path) -> Path:
    shim = tmp_path / "fake-accelerate"
    shim.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR)
    return shim


def _run(tmp_path: Path, *, extra: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta" / "info.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "output"
    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(_make_fake_python(tmp_path)),
            "ACCELERATE_BIN": str(_make_noop_accelerate(tmp_path)),
            "DATASET_REPO_ID": "fake_dataset",
            "DATASET_ROOT": str(dataset),
            "OUTPUT_DIR": str(output),
            "TRAINING_METADATA_PATH": str(tmp_path / "native.json"),
            "NUM_PROCESSES": "2",
            "BATCH_SIZE": "2",
            "EPOCHS": "2",
            "MIN_EPOCHS": "0",
            "EXECUTE": "1",
        }
    )
    if extra:
        env.update(extra)
    return subprocess.run(
        [str(LAUNCHER), "run"],
        cwd=LAUNCHER.parents[2],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_native_launcher_emits_standard_policy_only(tmp_path: Path) -> None:
    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    metadata = json.loads((tmp_path / "native.json").read_text(encoding="utf-8"))
    assert metadata["policy_type"] == "smolvla"
    assert metadata["ttt_enabled"] is False
    assert metadata["hd_ttt_enabled"] is False
    assert metadata["complete_frame_epochs"] == 2
    assert "--policy.type=smolvla" in result.stdout
    assert "smolvla_ttt" not in result.stdout
    assert "hd_label_path" not in result.stdout


def test_native_launcher_requires_explicit_partial_opt_in(tmp_path: Path) -> None:
    result = _run(tmp_path, extra={"TRAIN_EPISODES": "0,1"})
    assert result.returncode != 0
    assert "ALLOW_PARTIAL=1" in result.stderr


def test_native_launcher_refuses_short_canonical_budget(tmp_path: Path) -> None:
    result = _run(tmp_path, extra={"EPOCHS": "1", "MIN_EPOCHS": "2"})
    assert result.returncode != 0
    assert "requires at least 2 epochs" in result.stderr
