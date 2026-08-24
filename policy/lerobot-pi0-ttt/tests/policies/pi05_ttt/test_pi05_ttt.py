from types import SimpleNamespace

import torch
from torch.utils.data import Dataset

from lerobot.policies.pi05_ttt.configuration_pi05_ttt import PI05TTTConfig
from lerobot.policies.pi05_ttt.sequence import TailPreservingSequenceDataset
from lerobot.policies.pi05_ttt.ttt import TTTMLPLayer


class _EpisodeDataset(Dataset):
    def __init__(self, episode_lengths: list[int]) -> None:
        starts = []
        ends = []
        cursor = 0
        for length in episode_lengths:
            starts.append(cursor)
            cursor += length
            ends.append(cursor)
        self._length = cursor
        self.episodes = None
        self.meta = SimpleNamespace(
            episodes={"dataset_from_index": starts, "dataset_to_index": ends}
        )

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, int]:
        return {"frame_index": index}


def test_tail_preserving_sequence_windows_cover_every_frame_once_at_stride_256() -> None:
    episode_lengths = [258, 251, 294, 265, 267, 290, 407, 259, 245, 186]
    dataset = _EpisodeDataset(episode_lengths)
    sequences = TailPreservingSequenceDataset(dataset, sequence_length=256, sequence_stride=256)

    covered = []
    for index in range(len(sequences)):
        covered.extend(sample["frame_index"] for sample in sequences[index])

    assert covered == list(range(sum(episode_lengths)))
    assert max(length for _, length in sequences.window_specs) == 256
    assert any(length < 256 for _, length in sequences.window_specs)


def test_stage_one_gate_is_fixed_at_effective_point_zero_five() -> None:
    layer = TTTMLPLayer(
        dim=8,
        hidden_dim=16,
        effective_gate_init=0.05,
        gate_trainable=False,
    )
    assert not layer.gate.requires_grad
    torch.testing.assert_close(layer.effective_gate, torch.full((8,), 0.05), rtol=0, atol=1e-7)


def test_stage_config_controls_gate_and_action_head_trainability() -> None:
    pretrain = PI05TTTConfig(ttt_training_stage="ttt_only")
    posttrain = PI05TTTConfig(ttt_training_stage="action_head")
    assert not pretrain.trains_gate
    assert not pretrain.trains_action_head
    assert posttrain.trains_gate
    assert posttrain.trains_action_head
