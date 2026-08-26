from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_helper(script_name: str):
    if script_name == "evaluate_smolvla_baseline.py":
        # The baseline adapter intentionally reuses the observation bridge via
        # the script-style ``evaluate_smolvla_ttt`` import.
        ttt_path = ROOT / "examples" / "mikasa" / "evaluate_smolvla_ttt.py"
        ttt_spec = importlib.util.spec_from_file_location("evaluate_smolvla_ttt", ttt_path)
        assert ttt_spec is not None and ttt_spec.loader is not None
        ttt_module = importlib.util.module_from_spec(ttt_spec)
        sys.modules["evaluate_smolvla_ttt"] = ttt_module
        ttt_spec.loader.exec_module(ttt_module)
    path = ROOT / "examples" / "mikasa" / script_name
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._set_torch_seed


@pytest.mark.parametrize(
    "script_name",
    ["evaluate_smolvla_ttt.py", "evaluate_smolvla_baseline.py"],
)
def test_optional_mikasa_torch_seed_replays_cpu_noise(script_name: str) -> None:
    set_torch_seed = _load_helper(script_name)

    set_torch_seed(1234)
    first = torch.randn(8)
    set_torch_seed(1234)
    second = torch.randn(8)
    assert torch.equal(first, second)

    set_torch_seed(None)


@pytest.mark.parametrize(
    "script_name",
    ["evaluate_smolvla_ttt.py", "evaluate_smolvla_baseline.py"],
)
def test_mikasa_torch_seed_rejects_negative(script_name: str) -> None:
    set_torch_seed = _load_helper(script_name)
    with pytest.raises(ValueError, match="non-negative"):
        set_torch_seed(-1)
