#!/usr/bin/env python
"""Exercise one real Cover Blocks observation -> RPC -> qpos simulator step."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml


RMBENCH_ROOT = Path("/workspace/RMBench")
INSTRUCTION = (
    "On the table, red, green, and blue blocks are arranged randomly along with three lids. "
    "From the current viewpoint, cover the blocks from left to right using the lids, and then "
    "uncover them again in the sequence red, green, and blue."
)


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def task_arguments() -> dict:
    args = load_yaml(RMBENCH_ROOT / "task_config/demo_clean.yml")
    embodiments = load_yaml(RMBENCH_ROOT / "task_config/_embodiment_config.yml")
    cameras = load_yaml(RMBENCH_ROOT / "task_config/_camera_config.yml")

    embodiment = args["embodiment"][0]
    robot_file = (RMBENCH_ROOT / embodiments[embodiment]["file_path"]).resolve()
    robot_config = load_yaml(robot_file / "config.yml")
    camera_config = cameras[args["camera"]["head_camera_type"]]
    args.update(
        {
            "task_name": "cover_blocks",
            "task_config": "demo_clean",
            "ckpt_setting": "rpc_scene_step_smoke",
            "left_robot_file": str(robot_file),
            "right_robot_file": str(robot_file),
            "left_embodiment_config": robot_config,
            "right_embodiment_config": robot_config,
            "dual_arm_embodied": True,
            "head_camera_h": camera_config["h"],
            "head_camera_w": camera_config["w"],
            "eval_mode": True,
            "eval_video_log": False,
            "collect_data": False,
        }
    )
    args["data_type"]["pointcloud"] = False
    return args


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parsed = parser.parse_args()

    # Import simulator-heavy modules only in the guarded main process. RMBench's
    # robot planner uses multiprocessing spawn and re-imports this file.
    from envs.cover_blocks import cover_blocks
    from policy.smolvla_ttt.deploy_policy import eval as policy_eval
    from script.eval_policy_client import ModelClient

    client = ModelClient(host="127.0.0.1", port=parsed.port, timeout=30)
    task = cover_blocks()
    try:
        task.setup_demo(now_ep_num=0, seed=100000, is_test=True, **task_arguments())
        task.set_instruction(instruction=INSTRUCTION)
        client.call("reset_model")

        observation = task.get_obs()
        before = int(task.take_action_cnt)
        action = np.asarray(policy_eval(task, client, observation))
        after = int(task.take_action_cnt)
        if action.shape != (14,) or not np.isfinite(action).all():
            raise RuntimeError(f"Invalid RPC action: shape={action.shape}")
        if before != 0 or after != 1:
            raise RuntimeError(
                f"take_action counter did not advance exactly once: {before}->{after}"
            )

        next_observation = task.get_obs()
        qpos = np.asarray(next_observation["joint_action"]["vector"])
        if qpos.shape != (14,) or not np.isfinite(qpos).all():
            raise RuntimeError(f"Invalid post-action qpos: shape={qpos.shape}")
        print(
            "RMBENCH_FULL_RPC_SCENE_STEP_OK",
            f"counter={before}->{after}",
            f"action_range=({action.min():.6f},{action.max():.6f})",
            f"post_qpos_range=({qpos.min():.6f},{qpos.max():.6f})",
            flush=True,
        )
    finally:
        client.close()
        task.close_env(clear_cache=True)


if __name__ == "__main__":
    main()
