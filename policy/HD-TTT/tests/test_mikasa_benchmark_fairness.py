"""Regression tests for the cadence-fair CreditTTT benchmark contract."""

from __future__ import annotations

import json
from pathlib import Path

from examples.mikasa import benchmark_credit_ttt_v3 as benchmark


def _manifest(tmp_path: Path, *, n_episodes: int = 2) -> dict:
    parser = benchmark._build_parser()
    args = parser.parse_args(
        [
            "manifest",
            "--output",
            str(tmp_path / "manifest.json"),
            "--repo-root",
            str(tmp_path),
            "--python-bin",
            "python",
            "--n-episodes",
            str(n_episodes),
            "--start-seed",
            "100",
            "--torch-seed",
            "200",
            "--train-seeds",
            "1000",
        ]
    )
    return benchmark.build_manifest(args)


def test_manifest_separates_native_cadences(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    methods = {item["id"]: item for item in manifest["methods"]}

    assert methods[benchmark.NATIVE_VARIANT_CHUNK]["comparison_scope"] == (
        "cadence_mismatched_reference"
    )
    assert methods[benchmark.NATIVE_VARIANT_K1]["comparison_scope"] == "matched_cadence"
    assert methods[benchmark.NATIVE_VARIANT_K1]["expected_action_chunk_size"] == 1
    assert methods[benchmark.NATIVE_VARIANT_K1]["expected_model_action_horizon"] == 50

    native_k1 = [
        command
        for command in manifest["commands"]
        if command["method_id"] == benchmark.NATIVE_VARIANT_K1
    ]
    native_k50 = [
        command
        for command in manifest["commands"]
        if command["method_id"] == benchmark.NATIVE_VARIANT_CHUNK
    ]
    assert len(native_k1) == len(native_k50) == 2
    assert all(
        "--execution-action-steps" in command["argv"] for command in native_k1
    )
    assert all(
        "--execution-action-steps" not in command["argv"] for command in native_k50
    )
    assert manifest["checkpoints"][benchmark.NATIVE_VARIANT_K1] == manifest["checkpoints"][
        benchmark.NATIVE_VARIANT_CHUNK
    ]


def _native_record(
    *,
    env_id: str,
    seeds: list[int],
    k1: bool,
) -> dict:
    steps = 1 if k1 else 50
    variant = benchmark.NATIVE_VARIANT_K1 if k1 else benchmark.NATIVE_VARIANT_CHUNK
    cadence = benchmark.NATIVE_CADENCE_RECEDING if k1 else benchmark.NATIVE_CADENCE_CHUNK
    return {
        "env_id": env_id,
        "benchmark_protocol": benchmark.OFFICIAL_PROTOCOL,
        "successes": [True, False],
        "episode_seeds": seeds,
        "action_chunk_size": steps,
        "model_action_horizon": 50,
        "execution_action_steps": steps,
        "execution_cadence": cadence,
        "model": {
            "method": "SmolVLA",
            "policy_type": "smolvla",
            "ttt_enabled": False,
            "benchmark_variant": variant,
            "model_action_horizon": 50,
            "execution_action_steps": steps,
            "execution_cadence": cadence,
        },
    }


def _ttt_record(*, env_id: str, seeds: list[int], credit: bool) -> dict:
    model = {
        "method": "CreditTTT" if credit else "clean-TTT",
        "policy_type": "smolvla_ttt",
        "hd_ttt_enabled": bool(credit),
        "model_action_horizon": 50,
        "execution_action_steps": 1,
        "execution_cadence": benchmark.NATIVE_CADENCE_RECEDING,
    }
    if credit:
        model.update(
            {
                "protocol_version": "creditttt_qh2l_v3",
                "protocol_id": "credit_ttt_v3",
                "credit_ttt_protocol": dict(benchmark.CANONICAL_V3_PROTOCOL_IDENTITY),
            }
        )
    return {
        "env_id": env_id,
        "benchmark_protocol": benchmark.OFFICIAL_PROTOCOL,
        "successes": [True, False],
        "episode_seeds": seeds,
        "action_chunk_size": 1,
        "model_action_horizon": 50,
        "execution_action_steps": 1,
        "execution_cadence": benchmark.NATIVE_CADENCE_RECEDING,
        "model": model,
    }


def _write_run(root: Path, method: str, train_seed: str, task: dict, record: dict) -> None:
    path = root / method / f"train_seed_{train_seed}" / task["id"] / "eval.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"results": [record]}) + "\n", encoding="utf-8")


def test_aggregate_does_not_make_k50_a_primary_comparison(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    results_root = tmp_path / "results"
    seeds = manifest["evaluation"]["episode_seeds"]
    for task in manifest["tasks"]:
        _write_run(
            results_root,
            benchmark.NATIVE_VARIANT_CHUNK,
            "fixed",
            task,
            _native_record(env_id=task["env_id"], seeds=seeds, k1=False),
        )
        _write_run(
            results_root,
            benchmark.NATIVE_VARIANT_K1,
            "fixed",
            task,
            _native_record(env_id=task["env_id"], seeds=seeds, k1=True),
        )
        _write_run(
            results_root,
            "clean_ttt",
            "1000",
            task,
            _ttt_record(env_id=task["env_id"], seeds=seeds, credit=False),
        )
        _write_run(
            results_root,
            "credit_ttt",
            "1000",
            task,
            _ttt_record(env_id=task["env_id"], seeds=seeds, credit=True),
        )

    aggregate = benchmark.aggregate_results(manifest, results_root, n_bootstrap=20, seed=7)
    assert "CreditTTT_vs_native_smolvla_k1" in aggregate["pairwise"]
    assert "CreditTTT_vs_native_smolvla" not in aggregate["pairwise"]
    assert "CreditTTT_vs_native_smolvla" in aggregate["pairwise_exploratory"]
    assert aggregate["pairwise_exploratory"]["CreditTTT_vs_native_smolvla"]["primary"] is False
    assert aggregate["fairness"]["native_k1_complete"] is True


def test_strict_check_requires_native_k1_control() -> None:
    non_strict = benchmark.run_go_no_go_checks(
        {},
        aggregate_payload={
            "fairness": {
                "native_k1_complete": False,
                "native_k1_missing_tasks": ["color"],
            }
        },
        strict=False,
    )
    strict = benchmark.run_go_no_go_checks(
        {},
        aggregate_payload={
            "fairness": {
                "native_k1_complete": False,
                "native_k1_missing_tasks": ["color"],
            }
        },
        strict=True,
    )
    non_strict_gate = next(
        item
        for item in non_strict["checks"]
        if item["id"] == "benchmark_native_k1_cadence_control"
    )
    strict_gate = next(
        item
        for item in strict["checks"]
        if item["id"] == "benchmark_native_k1_cadence_control"
    )
    assert non_strict_gate["status"] == "INCONCLUSIVE"
    assert strict_gate["status"] == "FAIL"
