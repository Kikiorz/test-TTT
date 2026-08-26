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
            "--task-set",
            "legacy_two",
        ]
    )
    return benchmark.build_manifest(args)


def test_published_four_task_profile_schema(tmp_path: Path) -> None:
    parser = benchmark._build_parser()
    args = parser.parse_args(
        [
            "manifest",
            "--output",
            str(tmp_path / "published_manifest.json"),
            "--repo-root",
            str(tmp_path),
            "--python-bin",
            "python",
            "--n-episodes",
            "1",
            "--train-seeds",
            "1000",
            "--task-set",
            "published_four",
        ]
    )
    manifest = benchmark.build_manifest(args)
    assert manifest["task_set"] == "published_four"
    assert manifest["protocol_id"] == benchmark.PUBLISHED_FOUR_TASK_PROTOCOL_ID
    assert [task["id"] for task in manifest["tasks"]] == [
        "shell_touch",
        "intercept_medium",
        "remember_color3",
        "remember_color9",
    ]
    assert [task["env_id"] for task in manifest["tasks"]] == [
        "ShellGameTouch-VLA-v0",
        "InterceptMedium-VLA-v0",
        "RememberColor3-VLA-v0",
        "RememberColor9-VLA-v0",
    ]
    assert set(manifest["mechanism_audits"]["delay_bins_by_task"]) == {
        "shell_touch",
        "intercept_medium",
        "remember_color3",
        "remember_color9",
    }
    assert manifest["mechanism_audits"]["delay_bins"] == ["1-16"]
    # Four tasks × one train seed × four method variants.  Native K=50 and K=1
    # still point to the same frozen checkpoint, differing only in cadence.
    assert len(manifest["commands"]) == 16


def test_published_profile_uses_all_official_demos() -> None:
    """The canonical four-task recipe must not reserve an implicit 20% split."""
    for task in benchmark.PUBLISHED_COMPARABLE_TASKS:
        assert task["demo_count"] == 250
        assert task["train_demo_indices"] == [0, 249]
        assert task["validation_demo_indices"] == []

    parser = benchmark._build_parser()
    args = parser.parse_args(
        ["manifest", "--output", "manifest.json", "--repo-root", ".", "--task-set", "published_four"]
    )
    manifest = benchmark.build_manifest(args)
    split = manifest["training"]["demo_split"]
    assert "all official" in split["train_and_label"]
    assert manifest["training"]["official_demo_count"] == 250
    assert manifest["training"]["all_official_demos_used"] is True
    assert manifest["training"]["validation_affects_training"] is False


def test_task_checkpoint_maps_are_frozen_per_task(tmp_path: Path) -> None:
    parser = benchmark._build_parser()
    task_ids = [task["id"] for task in benchmark.PUBLISHED_COMPARABLE_TASKS]
    maps = {
        "native": {task_id: f"/ckpt/native/{task_id}" for task_id in task_ids},
        "clean": {task_id: f"/ckpt/clean/{task_id}" for task_id in task_ids},
        "credit": {task_id: f"/ckpt/credit/{task_id}" for task_id in task_ids},
    }
    map_paths = {}
    for name, payload in maps.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        map_paths[name] = path
    args = parser.parse_args(
        [
            "manifest",
            "--output",
            str(tmp_path / "mapped_manifest.json"),
            "--repo-root",
            str(tmp_path),
            "--python-bin",
            "python",
            "--task-set",
            "published_four",
            "--n-episodes",
            "1",
            "--train-seeds",
            "1000",
            "--native-checkpoints-json",
            str(map_paths["native"]),
            "--clean-checkpoints-json",
            str(map_paths["clean"]),
            "--credit-checkpoints-json",
            str(map_paths["credit"]),
        ]
    )
    manifest = benchmark.build_manifest(args)
    assert manifest["checkpoint_scope"]["native_smolvla"] == "per_task"
    assert manifest["checkpoint_scope"]["clean_ttt"] == "per_task"
    assert manifest["checkpoint_scope"]["credit_ttt"] == "per_task"
    for command in manifest["commands"]:
        method = command["method_id"]
        task_id = command["task_id"]
        expected_prefix = {
            "native_smolvla": "/ckpt/native/",
            "native_smolvla_k1": "/ckpt/native/",
            "clean_ttt": "/ckpt/clean/",
            "credit_ttt": "/ckpt/credit/",
        }[method]
        checkpoint = command["argv"][command["argv"].index("--checkpoint") + 1]
        assert checkpoint == expected_prefix + task_id
        assert manifest["checkpoints_by_task"][method][task_id] == checkpoint


def test_incomplete_task_checkpoint_map_fails_closed(tmp_path: Path) -> None:
    parser = benchmark._build_parser()
    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps({"shell_touch": "/ckpt/shell"}), encoding="utf-8")
    args = parser.parse_args(
        [
            "manifest",
            "--output",
            str(tmp_path / "manifest.json"),
            "--repo-root",
            str(tmp_path),
            "--python-bin",
            "python",
            "--task-set",
            "published_four",
            "--clean-checkpoints-json",
            str(partial),
        ]
    )
    try:
        benchmark.build_manifest(args)
    except ValueError as exc:
        assert "incomplete" in str(exc)
    else:  # pragma: no cover - defensive assertion for the test itself
        raise AssertionError("partial task checkpoint map was accepted")


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
