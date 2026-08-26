"""Regression checks for the provenance emitted by the MIKASA evaluator."""

from __future__ import annotations

from types import SimpleNamespace

from examples.mikasa.benchmark_credit_ttt_v3 import CANONICAL_V3_PROTOCOL_IDENTITY
from examples.mikasa.evaluate_smolvla_ttt import _credit_ttt_protocol_identity


def test_evaluator_emits_complete_canonical_credit_identity() -> None:
    config = SimpleNamespace(
        credit_ttt_enabled=True,
        hd_attribution_protocol="credit_ttt_v3_query_effect",
        hd_v3_intervention="delete",
        ttt_writer_mode="prefix_only",
        ttt_second_order=True,
    )
    identity = _credit_ttt_protocol_identity(config)
    assert identity is not None
    for key, expected in CANONICAL_V3_PROTOCOL_IDENTITY.items():
        assert identity[key] == expected


def test_evaluator_omits_credit_identity_for_clean_policy() -> None:
    config = SimpleNamespace(credit_ttt_enabled=False)
    assert _credit_ttt_protocol_identity(config) is None
