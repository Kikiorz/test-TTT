from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset

from lerobot.policies.smolvla_ttt.hd_dataset import HindsightLabelDataset
from lerobot.policies.smolvla_ttt.sequence import (
    SEQUENCE_OFFSET_KEY,
    TailPreservingSequenceDataset,
    sequence_collate_fn,
)
from lerobot.processor.converters import batch_to_transition, transition_to_batch


class _EpisodeDataset(Dataset):
    def __init__(self, episode_lengths: list[int]) -> None:
        starts = []
        ends = []
        cursor = 0
        for length in episode_lengths:
            starts.append(cursor)
            cursor += length
            ends.append(cursor)
        self.episodes = None
        self.meta = SimpleNamespace(
            episodes={"dataset_from_index": starts, "dataset_to_index": ends}
        )
        self._length = cursor

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"observation.state": torch.tensor([float(index)])}


class _SelectedEpisodeDataset(_EpisodeDataset):
    def __init__(self) -> None:
        super().__init__([3, 4, 2])
        self.episodes = [1, 2]
        self._length = 6
        self.repo_id = "synthetic"


def test_columnar_labels_attach_and_preserve_dataset_metadata(tmp_path) -> None:
    dataset = _EpisodeDataset([3, 2])
    path = tmp_path / "labels.pt"
    torch.save(
        {
            "hd_rho": torch.arange(5, dtype=torch.float32),
            "hd_teacher_velocity": torch.ones(5, 4, 7),
        },
        path,
    )

    labeled = HindsightLabelDataset(dataset, path)

    assert labeled.meta is dataset.meta
    assert labeled.label_keys == ("hd_rho", "hd_teacher_velocity")
    assert labeled[4]["hd_rho"].item() == 4
    assert labeled[1]["hd_teacher_velocity"].shape == (4, 7)


def test_episode_records_and_nested_metadata_are_supported(tmp_path) -> None:
    dataset = _EpisodeDataset([3, 2])
    path = tmp_path / "labels.pt"
    torch.save(
        {
            "labels": {"rho": torch.arange(5, dtype=torch.float32)},
            "metadata": {
                "episode_index": torch.tensor([0, 0, 0, 1, 1]),
                "frame_index": torch.tensor([0, 1, 2, 0, 1]),
            },
        },
        path,
    )

    labeled = HindsightLabelDataset(dataset, path)
    assert [labeled[i]["hd_rho"].item() for i in range(5)] == list(range(5))


def test_explicit_null_attribution_protocol_is_legacy(tmp_path) -> None:
    """Early JSON/torch artifacts may encode the optional protocol as null."""

    dataset = _EpisodeDataset([2])
    path = tmp_path / "labels.pt"
    torch.save(
        {
            "hd_rho": torch.ones(2),
            "metadata": {"attribution_protocol": None},
        },
        path,
    )

    labeled = HindsightLabelDataset(dataset, path)

    assert labeled.hd_attribution_protocol == "legacy_raw_hinge_max"


def test_full_attribution_matrix_is_reduced_to_future_weight(tmp_path) -> None:
    dataset = _EpisodeDataset([4])
    matrix = torch.zeros(4, 4)
    matrix[:, 2] = torch.arange(1, 5, dtype=torch.float32)
    path = tmp_path / "labels.pt"
    torch.save({"C": matrix}, path)

    labeled = HindsightLabelDataset(dataset, path)
    assert labeled[2]["hd_attribution"].item() == 10
    assert labeled[0]["hd_attribution"].item() == 0


def test_full_source_labels_are_reindexed_for_selected_episodes(tmp_path) -> None:
    dataset = _SelectedEpisodeDataset()
    path = tmp_path / "labels.pt"
    torch.save({"hd_rho": torch.arange(9, dtype=torch.float32)}, path)

    labeled = HindsightLabelDataset(dataset, path)
    assert [labeled[i]["hd_rho"].item() for i in range(6)] == [3, 4, 5, 6, 7, 8]


def test_provenance_rich_superset_labels_are_reindexed_for_selected_episodes(tmp_path) -> None:
    """V3 train views may reuse one artifact containing held-out episodes."""

    dataset = _SelectedEpisodeDataset()
    path = tmp_path / "labels_with_metadata.pt"
    torch.save(
        {
            "hd_rho": torch.arange(9, dtype=torch.float32),
            "global_index": torch.arange(9, dtype=torch.int64),
            "metadata": {"dataset_repo_id": "synthetic"},
        },
        path,
    )

    labeled = HindsightLabelDataset(dataset, path)
    assert [labeled[i]["hd_rho"].item() for i in range(6)] == [3, 4, 5, 6, 7, 8]


def test_labels_survive_sequence_collation_and_processor_transition(tmp_path) -> None:
    dataset = _EpisodeDataset([3])
    path = tmp_path / "labels.pt"
    torch.save({"hd_rho": torch.arange(3, dtype=torch.float32)}, path)
    labeled = HindsightLabelDataset(dataset, path)

    sequences = TailPreservingSequenceDataset(labeled, sequence_length=8, sequence_stride=8)
    batch = sequence_collate_fn([sequences[0]])
    assert batch["hd_rho"].shape == (3,)

    transition = batch_to_transition(
        {
            "observation.state": batch["observation.state"],
            "hd_rho": batch["hd_rho"],
            SEQUENCE_OFFSET_KEY: batch[SEQUENCE_OFFSET_KEY],
        }
    )
    round_trip = transition_to_batch(transition)
    torch.testing.assert_close(round_trip["hd_rho"], batch["hd_rho"])
    torch.testing.assert_close(round_trip[SEQUENCE_OFFSET_KEY], batch[SEQUENCE_OFFSET_KEY])


def test_sequence_offset_is_episode_local_and_includes_warmup() -> None:
    dataset = _EpisodeDataset([20])
    sequences = TailPreservingSequenceDataset(
        dataset,
        sequence_length=4,
        sequence_stride=4,
        history_warmup_length=3,
    )

    # Window 1 targets frame 4 and replays frames 1..3, so pair indices must
    # be interpreted relative to episode frame 1 rather than selected-view 0.
    assert sequences.window_sequence_offsets[:3] == [0, 1, 5]
    samples = sequences[1]
    assert [sample[SEQUENCE_OFFSET_KEY].item() for sample in samples] == [1] * len(samples)
    collated = sequence_collate_fn([samples])
    assert collated[SEQUENCE_OFFSET_KEY].ndim == 0
    assert collated[SEQUENCE_OFFSET_KEY].item() == 1


def test_sequence_offset_resets_at_episode_boundary() -> None:
    dataset = _EpisodeDataset([3, 4])
    sequences = TailPreservingSequenceDataset(
        dataset,
        sequence_length=4,
        sequence_stride=4,
        history_warmup_length=3,
    )

    # The second episode starts at selected-view index 3, but its first window
    # is episode-local frame 0 and must not inherit the previous episode's 3.
    assert sequences.window_sequence_offsets == [0, 0]
    assert [sample["observation.state"].item() for sample in sequences[1]] == [3, 4, 5, 6]


def test_sequence_collate_rejects_mixed_offsets() -> None:
    dataset = _EpisodeDataset([4])
    sequences = TailPreservingSequenceDataset(dataset, sequence_length=4, sequence_stride=4)
    samples = sequences[0]
    samples[0][SEQUENCE_OFFSET_KEY] = torch.tensor(1, dtype=torch.int64)
    with pytest.raises(ValueError, match="share .*sequence_offset"):
        sequence_collate_fn([samples])


def test_window_keyed_labels_preserve_the_exact_replay_context(tmp_path) -> None:
    dataset = _EpisodeDataset([6])
    path = tmp_path / "window_labels.pt"
    source_indices = torch.tensor([1, 2, 3], dtype=torch.int64)
    torch.save(
        {
            "windows": [
                {
                    "target_global_index": 2,
                    "history_start_source": 1,
                    "source_indices": source_indices,
                    "length": 3,
                    "labels": {
                        "hd_write_gate": torch.tensor([0.2, 0.3, 0.4]),
                        "hd_counterfactual_write_gate": torch.tensor([0.0, 1.0, 1.0]),
                        "hd_writer_valid": torch.ones(3, dtype=torch.bool),
                    },
                }
            ],
            "metadata": {
                "window_local": True,
                "window_keyed": True,
                "sequence_length": 2,
                "sequence_stride": 2,
                "context_length": 1,
                "max_windows_per_episode": None,
                "phase_mode": "deployment",
            },
        },
        path,
    )
    labeled = HindsightLabelDataset(dataset, path)
    assert labeled.hd_window_keyed
    sequences = TailPreservingSequenceDataset(
        labeled,
        sequence_length=2,
        sequence_stride=2,
        history_warmup_length=1,
    )
    samples = sequences[1]
    assert [sample["observation.state"].item() for sample in samples] == [1, 2, 3]
    # The warm-up row keeps the window-specific counterfactual gate instead of
    # inheriting the gate from its own target window.
    torch.testing.assert_close(
        torch.stack([sample["hd_counterfactual_write_gate"] for sample in samples]),
        torch.tensor([0.0, 1.0, 1.0]),
    )
    torch.testing.assert_close(
        torch.stack([sample["hd_write_gate"] for sample in samples]),
        torch.tensor([0.2, 0.3, 0.4]),
    )


def test_strict_mode_reports_uncovered_frames(tmp_path) -> None:
    dataset = _EpisodeDataset([3])
    path = tmp_path / "labels.pt"
    torch.save({"hd_rho": torch.ones(2)}, path)

    with pytest.raises(ValueError, match="do not cover"):
        HindsightLabelDataset(dataset, path)
