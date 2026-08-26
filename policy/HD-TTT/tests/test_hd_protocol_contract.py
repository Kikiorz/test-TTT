"""Unit tests for the immutable Hindsight-TTT v2 label provenance contract."""

import pytest

from lerobot.scripts.lerobot_train import _validate_hd_v2_label_contract


_V2_LABEL_KEYS = {
    "hd_teacher_effect",
    "hd_effect_rho",
    "hd_effect_write_gate",
    "hd_effect_valid",
}


def _v2_metadata(**overrides):
    metadata = {
        "attribution_protocol": "v2_relative_antithetic_robust",
        "attribution_slot_mode": "slot0",
        "attribution_replays": 2,
        "effect_target": "plus_noise_full_minus_wrong",
        "effect_branches": 2,
    }
    metadata.update(overrides)
    return metadata


def test_v2_contract_accepts_canonical_metadata_and_required_columns() -> None:
    assert (
        _validate_hd_v2_label_contract(_v2_metadata(), label_keys=_V2_LABEL_KEYS)
        == "v2_relative_antithetic_robust"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("attribution_slot_mode", "all_slots"),
        ("attribution_replays", 1),
        ("attribution_replays", True),
        ("effect_target", "antithetic_mean_full_minus_wrong"),
        ("effect_branches", 0),
        ("effect_branches", -1),
        ("effect_branches", True),
    ],
)
def test_v2_contract_rejects_protocol_mismatch(field, value) -> None:
    with pytest.raises(ValueError, match="HD v2 label protocol contract mismatch"):
        _validate_hd_v2_label_contract(_v2_metadata(**{field: value}), label_keys=_V2_LABEL_KEYS)


def test_v2_contract_allows_positive_legacy_branch_budget() -> None:
    """Branch count is a compatibility budget; only positivity is required."""

    metadata = _v2_metadata(effect_branches=1)
    assert _validate_hd_v2_label_contract(metadata, label_keys=_V2_LABEL_KEYS).startswith("v2_")


def test_v2_contract_rejects_missing_effect_columns() -> None:
    with pytest.raises(ValueError, match="missing action-effect columns"):
        _validate_hd_v2_label_contract(
            _v2_metadata(),
            label_keys={"hd_teacher_effect", "hd_effect_valid"},
        )


def test_legacy_contract_remains_unchanged() -> None:
    assert (
        _validate_hd_v2_label_contract(
            {"attribution_protocol": "legacy_raw_hinge_max"},
            label_keys=set(),
        )
        == "legacy_raw_hinge_max"
    )
