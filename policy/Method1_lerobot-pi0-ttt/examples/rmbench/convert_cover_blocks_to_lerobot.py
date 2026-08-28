#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert RMBench ``cover_blocks/demo_clean`` HDF5 episodes to LeRobot v3.

RMBench records the current dual-arm qpos at every observation. Consequently,
each emitted transition uses ``qpos[t]`` as ``observation.state`` and
``qpos[t + 1]`` as ``action``. The final raw observation has no future action
and is intentionally not emitted.
"""

import argparse
import logging
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata

RMBENCH_FPS = 30
RMBENCH_EPISODES = 50
TASK_INSTRUCTION = (
    "On the table, red, green, and blue blocks are arranged randomly along with three lids. "
    "From the current viewpoint, cover the blocks from left to right using the lids, and then "
    "uncover them again in the sequence red, green, and blue."
)
IMAGE_KEY = "observation.images.head_camera"
QPOS_KEY = "joint_action/vector"
QPOS_NAMES = [
    "left_joint_0",
    "left_joint_1",
    "left_joint_2",
    "left_joint_3",
    "left_joint_4",
    "left_joint_5",
    "left_gripper",
    "right_joint_0",
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
    "right_joint_4",
    "right_joint_5",
    "right_gripper",
]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        required=True,
        help="Directory containing RMBench episode0.hdf5 ... episode49.hdf5.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New, non-existing directory in which to create the LeRobot dataset.",
    )
    parser.add_argument(
        "--repo-id",
        default="rmbench/cover_blocks_demo_clean",
        help="Repository id stored in dataset metadata (no Hub upload is performed).",
    )
    parser.add_argument(
        "--episode-count",
        type=positive_int,
        default=RMBENCH_EPISODES,
        help=f"Number of contiguous episodes to convert (default: all {RMBENCH_EPISODES}).",
    )
    parser.add_argument(
        "--max-frames-per-episode",
        type=positive_int,
        default=None,
        help="Development-only cap on emitted transitions per episode; omit for full conversion.",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=4,
        help="Threads used for temporary image writes before video encoding (default: 4).",
    )
    args = parser.parse_args()
    if args.image_writer_threads < 0:
        parser.error("--image-writer-threads must be non-negative")
    return args


def decode_rgb(encoded_image: bytes | np.bytes_ | np.ndarray, source: str) -> np.ndarray:
    """Decode one RMBench image while preserving its original RGB channel order.

    RMBench writes SAPIEN RGB arrays directly with ``cv2.imencode``.  Although
    OpenCV labels the decoded array BGR, its numeric channel order is therefore
    already the original SAPIEN RGB order.  Applying BGR-to-RGB here would swap
    red and blue relative to live evaluator observations.
    """
    if isinstance(encoded_image, (bytes, np.bytes_)):
        encoded_array = np.frombuffer(encoded_image, dtype=np.uint8)
    else:
        encoded_array = np.asarray(encoded_image, dtype=np.uint8).reshape(-1)

    image_rgb = cv2.imdecode(encoded_array, cv2.IMREAD_COLOR)
    if image_rgb is None:
        raise ValueError(f"Failed to decode RGB frame from {source}")
    return np.ascontiguousarray(image_rgb)


def expected_native_qpos(hdf5_file: h5py.File) -> np.ndarray:
    """Reconstruct RMBench's native [L6, L-grip, R6, R-grip] qpos."""
    left_arm = np.asarray(hdf5_file["joint_action/left_arm"])
    left_gripper = np.asarray(hdf5_file["joint_action/left_gripper"])[:, None]
    right_arm = np.asarray(hdf5_file["joint_action/right_arm"])
    right_gripper = np.asarray(hdf5_file["joint_action/right_gripper"])[:, None]
    return np.concatenate((left_arm, left_gripper, right_arm, right_gripper), axis=1)


def validate_episode(path: Path, expected_image_shape: tuple[int, int, int] | None) -> tuple[int, int, int]:
    """Validate schema, native qpos ordering, finiteness, and head-camera decoding."""
    required_keys = (
        QPOS_KEY,
        "joint_action/left_arm",
        "joint_action/left_gripper",
        "joint_action/right_arm",
        "joint_action/right_gripper",
        "observation/head_camera/rgb",
    )
    with h5py.File(path, "r") as hdf5_file:
        missing = [key for key in required_keys if key not in hdf5_file]
        if missing:
            raise KeyError(f"{path} is missing required datasets: {missing}")

        qpos = np.asarray(hdf5_file[QPOS_KEY])
        if qpos.ndim != 2 or qpos.shape[1] != len(QPOS_NAMES):
            raise ValueError(f"{path}:{QPOS_KEY} must have shape [T, 14], got {qpos.shape}")
        if qpos.shape[0] < 2:
            raise ValueError(f"{path} needs at least two observations, got {qpos.shape[0]}")
        if not np.isfinite(qpos).all():
            raise ValueError(f"{path}:{QPOS_KEY} contains non-finite values")

        native_qpos = expected_native_qpos(hdf5_file)
        if native_qpos.shape != qpos.shape or not np.allclose(native_qpos, qpos, rtol=1e-6, atol=1e-6):
            raise ValueError(
                f"{path}:{QPOS_KEY} does not match native [left_arm, left_gripper, "
                "right_arm, right_gripper] ordering"
            )

        images = hdf5_file["observation/head_camera/rgb"]
        if len(images) != qpos.shape[0]:
            raise ValueError(
                f"{path} has {qpos.shape[0]} qpos observations but {len(images)} head-camera frames"
            )
        image = decode_rgb(images[0], f"{path}:head_camera[0]")

    image_shape = tuple(image.shape)
    if image_shape[-1] != 3:
        raise ValueError(f"{path} head-camera frame must have three RGB channels, got {image_shape}")
    if expected_image_shape is not None and image_shape != expected_image_shape:
        raise ValueError(
            f"{path} head-camera shape {image_shape} differs from expected {expected_image_shape}"
        )
    return image_shape


def resolve_and_validate_episodes(
    raw_data_dir: Path, episode_count: int
) -> tuple[list[Path], tuple[int, int, int]]:
    if not raw_data_dir.is_dir():
        raise NotADirectoryError(f"Raw data directory does not exist: {raw_data_dir}")

    episode_paths = [raw_data_dir / f"episode{index}.hdf5" for index in range(episode_count)]
    missing = [path for path in episode_paths if not path.is_file()]
    if missing:
        preview = ", ".join(str(path) for path in missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise FileNotFoundError(f"Missing {len(missing)} requested RMBench episode(s): {preview}{suffix}")

    image_shape = None
    for path in tqdm(episode_paths, desc="Preflight", unit="episode"):
        image_shape = validate_episode(path, image_shape)
    assert image_shape is not None
    return episode_paths, image_shape


def make_features(image_shape: tuple[int, int, int]) -> dict[str, dict]:
    height, width, channels = image_shape
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(QPOS_NAMES),),
            "names": QPOS_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(QPOS_NAMES),),
            "names": QPOS_NAMES,
        },
        IMAGE_KEY: {
            "dtype": "video",
            "shape": (height, width, channels),
            "names": ["height", "width", "channels"],
        },
    }


def convert_episode(
    dataset: LeRobotDataset,
    episode_path: Path,
    image_shape: tuple[int, int, int],
    max_frames_per_episode: int | None,
) -> int:
    with h5py.File(episode_path, "r") as hdf5_file:
        qpos = np.asarray(hdf5_file[QPOS_KEY], dtype=np.float32)
        images = hdf5_file["observation/head_camera/rgb"]
        transition_count = qpos.shape[0] - 1
        if max_frames_per_episode is not None:
            transition_count = min(transition_count, max_frames_per_episode)

        for frame_index in tqdm(
            range(transition_count),
            desc=episode_path.stem,
            unit="frame",
            leave=False,
        ):
            image = decode_rgb(images[frame_index], f"{episode_path}:head_camera[{frame_index}]")
            if tuple(image.shape) != image_shape:
                raise ValueError(
                    f"{episode_path}:head_camera[{frame_index}] has shape {image.shape}; expected {image_shape}"
                )

            dataset.add_frame(
                {
                    "observation.state": qpos[frame_index].copy(),
                    "action": qpos[frame_index + 1].copy(),
                    IMAGE_KEY: image,
                    "task": TASK_INSTRUCTION,
                }
            )

    dataset.save_episode()
    return transition_count


def convert(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {args.output_dir}. "
            "Choose a new path so an interrupted or existing dataset is never overwritten."
        )

    episode_paths, image_shape = resolve_and_validate_episodes(args.raw_data_dir, args.episode_count)
    logging.info("Validated %d episodes with RGB shape %s", len(episode_paths), image_shape)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=args.output_dir,
        robot_type="aloha_agilex",
        fps=RMBENCH_FPS,
        features=make_features(image_shape),
        use_videos=True,
        image_writer_threads=args.image_writer_threads,
        batch_encoding_size=1,
    )

    total_frames = 0
    for episode_path in tqdm(episode_paths, desc="Converting", unit="episode"):
        total_frames += convert_episode(
            dataset,
            episode_path,
            image_shape,
            args.max_frames_per_episode,
        )

    # LeRobot 0.5.2 requires an explicit finalization to close parquet writers,
    # flush metadata, and make the newly created dataset readable.
    dataset.finalize()

    metadata = LeRobotDatasetMetadata(repo_id=args.repo_id, root=args.output_dir)
    if metadata.total_episodes != len(episode_paths) or metadata.total_frames != total_frames:
        raise RuntimeError(
            "Final dataset metadata does not match the conversion: "
            f"episodes={metadata.total_episodes}/{len(episode_paths)}, "
            f"frames={metadata.total_frames}/{total_frames}"
        )
    logging.info(
        "Created %s with %d episodes and %d transitions",
        args.output_dir,
        metadata.total_episodes,
        metadata.total_frames,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    convert(parse_args())


if __name__ == "__main__":
    main()
