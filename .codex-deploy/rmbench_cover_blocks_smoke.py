#!/usr/bin/env python
"""Create one real Cover Blocks scene and read its policy observation."""

from pathlib import Path

import numpy as np
import yaml

from envs import CONFIGS_PATH
from envs.cover_blocks import cover_blocks


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def main() -> None:
    root = Path(__file__).resolve().parent
    args = load_yaml(root / "task_config" / "demo_clean.yml")
    embodiments = load_yaml(root / "task_config" / "_embodiment_config.yml")
    cameras = load_yaml(root / "task_config" / "_camera_config.yml")

    embodiment = args["embodiment"][0]
    robot_file = (root / embodiments[embodiment]["file_path"]).resolve()
    robot_config = load_yaml(robot_file / "config.yml")
    camera_config = cameras[args["camera"]["head_camera_type"]]

    args.update(
        {
            "task_name": "cover_blocks",
            "task_config": "demo_clean",
            "ckpt_setting": "smoke",
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
    # SmolVLA consumes only head RGB and qpos. Point-cloud sampling would add a
    # policy-irrelevant PyTorch3D dependency without changing task dynamics.
    args["data_type"]["pointcloud"] = False

    task = cover_blocks()
    try:
        task.setup_demo(now_ep_num=0, seed=100000, is_test=True, **args)
        observation = task.get_obs()
        rgb = np.asarray(observation["observation"]["head_camera"]["rgb"])
        qpos = np.asarray(observation["joint_action"]["vector"])
        if rgb.shape != (camera_config["h"], camera_config["w"], 3):
            raise RuntimeError(f"Unexpected head RGB shape: {rgb.shape}")
        if qpos.shape != (14,) or not np.isfinite(qpos).all():
            raise RuntimeError(f"Unexpected qpos: shape={qpos.shape}")
        print(
            "COVER_BLOCKS_SCENE_OK",
            f"rgb={rgb.shape}/{rgb.dtype}",
            f"qpos={qpos.shape}/{qpos.dtype}",
            f"step_limit={task.step_lim}",
            flush=True,
        )
    finally:
        task.close_env(clear_cache=True)


if __name__ == "__main__":
    main()
