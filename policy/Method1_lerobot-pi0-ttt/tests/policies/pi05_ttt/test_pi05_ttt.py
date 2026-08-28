import copy
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset

from lerobot.policies.pi05_ttt.configuration_pi05_ttt import PI05TTTConfig
from lerobot.policies.pi05_ttt.modeling_pi05_ttt import PI05TTTPolicy, PI05TTTPytorch
from lerobot.policies.pi05_ttt.sequence import TailPreservingSequenceDataset
from lerobot.policies.pi05_ttt.ttt import TTTMLPLayer
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


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
        self.meta = SimpleNamespace(episodes={"dataset_from_index": starts, "dataset_to_index": ends})

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


@pytest.mark.parametrize("training_stage", ["ttt_only", "action_head"])
def test_gate_is_learnable_and_initialized_to_configured_near_zero_value_in_both_stages(
    training_stage: str,
) -> None:
    config = PI05TTTConfig(ttt_training_stage=training_stage)
    layer = TTTMLPLayer(
        dim=8,
        hidden_dim=16,
        effective_gate_init=config.ttt_effective_gate_init,
        gate_trainable=config.trains_gate,
        second_order=False,
    )

    assert config.ttt_effective_gate_init == 0.001
    assert config.trains_gate
    assert config.trains_action_head is (training_stage == "action_head")
    assert layer.gate.requires_grad
    torch.testing.assert_close(layer.effective_gate, torch.full((8,), 0.001), rtol=0, atol=1e-7)

    outputs, _ = layer(torch.randn(1, 2, 3, 8), create_graph=False)
    outputs.square().mean().backward()
    assert layer.gate.grad is not None


def test_config_rejects_invalid_action_execution_cadence() -> None:
    with pytest.raises(ValueError, match="n_action_steps must be positive"):
        PI05TTTConfig(n_action_steps=0)
    with pytest.raises(ValueError, match="cannot be greater than chunk_size"):
        PI05TTTConfig(chunk_size=4, n_action_steps=5)


def test_gate_zero_is_an_exact_residual_bypass_even_when_memory_updates() -> None:
    torch.manual_seed(0)
    layer = TTTMLPLayer(dim=8, hidden_dim=16, effective_gate_init=0.001, second_order=False)
    with torch.no_grad():
        layer.gate.zero_()

    inputs = torch.randn(1, 3, 2, 8)
    outputs, state = layer(inputs, update=True, create_graph=False)

    assert torch.equal(outputs, inputs)
    assert state.position.tolist() == [2]


def test_fast_state_carries_across_detached_tbptt_segments() -> None:
    torch.manual_seed(1)
    layer = TTTMLPLayer(dim=8, hidden_dim=16, effective_gate_init=0.001, second_order=False)
    inputs = torch.randn(1, 6, 2, 8)

    joint_outputs, joint_state = layer(inputs, update=True, create_graph=False)
    first_outputs, first_state = layer(inputs[:, :3], update=True, create_graph=False)
    second_outputs, split_state = layer(
        inputs[:, 3:],
        state=first_state.detach(),
        update=True,
        create_graph=False,
    )

    torch.testing.assert_close(
        torch.cat([first_outputs, second_outputs], dim=1),
        joint_outputs,
        rtol=0,
        atol=0,
    )
    for split_tensor, joint_tensor in zip(split_state.tensors(), joint_state.tensors(), strict=True):
        torch.testing.assert_close(split_tensor, joint_tensor, rtol=0, atol=0)
    assert split_state.position.tolist() == joint_state.position.tolist() == [5]


def test_action_padding_values_do_not_change_fast_state_and_full_mask_does_not_advance() -> None:
    torch.manual_seed(27)
    layer = TTTMLPLayer(dim=8, hidden_dim=16, second_order=False)
    reference_layer = copy.deepcopy(layer)
    inputs = torch.randn(1, 2, 4, 8)
    changed_padding = inputs.clone()
    changed_padding[:, 0, 2:] = torch.randn_like(changed_padding[:, 0, 2:]) * 1_000
    changed_padding[:, 1] = torch.randn_like(changed_padding[:, 1]) * 1_000
    token_mask = torch.tensor([[[True, True, False, False], [False, False, False, False]]])

    outputs, state = layer(inputs, token_mask=token_mask, create_graph=False)
    changed_outputs, changed_state = reference_layer(
        changed_padding,
        token_mask=token_mask,
        create_graph=False,
    )

    torch.testing.assert_close(changed_outputs[:, 0, :2], outputs[:, 0, :2], rtol=0, atol=0)
    for changed_tensor, reference_tensor in zip(changed_state.tensors(), state.tensors(), strict=True):
        torch.testing.assert_close(changed_tensor, reference_tensor, rtol=0, atol=0)
    # The first timestep updates W0; the fully padded second timestep is not a
    # selected trajectory observation and must neither update nor advance RoPE.
    assert changed_state.position.tolist() == state.position.tolist() == [0]


def test_padded_actions_are_attention_keys_for_no_query_but_padded_queries_stay_readable() -> None:
    class _CaptureBackbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            q_proj = torch.nn.Linear(1, 1, bias=False)
            self.paligemma = SimpleNamespace(
                model=SimpleNamespace(
                    language_model=SimpleNamespace(
                        layers=[SimpleNamespace(self_attn=SimpleNamespace(q_proj=q_proj))]
                    )
                )
            )
            self.attention_mask = None

        def forward(self, *, inputs_embeds, attention_mask, **kwargs):
            del kwargs
            self.attention_mask = attention_mask
            return inputs_embeds, None

    model = PI05TTTPytorch.__new__(PI05TTTPytorch)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(chunk_size=3)
    model.paligemma_with_expert = _CaptureBackbone()
    model.action_out_proj = torch.nn.Identity()

    def embed_prefix(self, images, img_masks, tokens, masks):
        del self, images, img_masks, tokens, masks
        return (
            torch.zeros(1, 2, 3),
            torch.ones(1, 2, dtype=torch.bool),
            torch.zeros(1, 2, dtype=torch.bool),
        )

    def embed_suffix(self, x_t, time):
        del self, x_t, time
        return (
            torch.zeros(1, 3, 3),
            torch.ones(1, 3, dtype=torch.bool),
            torch.ones(1, 3, dtype=torch.bool),
            torch.zeros(1, 3),
        )

    model.embed_prefix = MethodType(embed_prefix, model)
    model.embed_suffix = MethodType(embed_suffix, model)
    model.forward(
        [],
        [],
        torch.zeros(1, 1, dtype=torch.long),
        torch.ones(1, 1, dtype=torch.bool),
        torch.zeros(1, 3, 3),
        torch.zeros(1, 3, 3),
        torch.zeros(1),
        action_token_mask=torch.tensor([[True, False, False]]),
        expert_layer_callback=lambda layer_index, hidden_states: hidden_states,
    )

    allowed = model.paligemma_with_expert.attention_mask[:, 0] == 0
    # The last two action slots are padding and therefore cannot be keys for
    # prefix, valid-action, or padded-action queries.
    assert not allowed[:, :, -2:].any()
    # Their query rows retain readable prefix/valid-action keys, preventing an
    # all-masked softmax row and its NaNs.
    assert allowed[:, -2:, :].any(dim=-1).all()


def test_outer_loss_and_ttt_updates_ignore_padded_action_slots() -> None:
    class _FixedLossModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer(
                "losses",
                torch.tensor(
                    [
                        [[1.0, 3.0], [5.0, 7.0], [100.0, 100.0]],
                        [[200.0, 200.0], [200.0, 200.0], [200.0, 200.0]],
                    ]
                ),
            )
            self.update_mask = None
            self.token_mask = None

        @staticmethod
        def sample_noise(shape, device):
            return torch.zeros(shape, device=device)

        @staticmethod
        def sample_time(batch_size, device):
            return torch.zeros(batch_size, device=device)

        def forward_with_state(self, *args, update_mask, token_mask, **kwargs):
            del args, kwargs
            self.update_mask = update_mask
            self.token_mask = token_mask
            return self.losses.clone(), {0: "state"}

    policy = PI05TTTPolicy.__new__(PI05TTTPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        chunk_size=3,
        output_features={ACTION: SimpleNamespace(shape=(2,))},
    )
    policy.model = _FixedLossModel()

    def preprocess_images(self, batch):
        del self, batch
        return [], []

    def prepare_action(self, batch):
        del self
        return batch[ACTION]

    policy._preprocess_images = MethodType(preprocess_images, policy)
    policy.prepare_action = MethodType(prepare_action, policy)
    batch = {
        ACTION: torch.zeros(2, 3, 2),
        "action_is_pad": torch.tensor(
            [[False, False, True], [True, True, True]],
            dtype=torch.bool,
        ),
        OBS_LANGUAGE_TOKENS: torch.zeros(2, 1, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 1, dtype=torch.bool),
    }

    loss, loss_dict, fast_states = policy.forward_sequence_segment(
        batch,
        sequence_shape=(1, 2),
    )

    torch.testing.assert_close(loss, torch.tensor(4.0))
    assert loss_dict["loss_per_dim"] == pytest.approx([3.0, 5.0])
    assert fast_states == {0: "state"}
    assert policy.model.update_mask.tolist() == [[True, False]]
    assert policy.model.token_mask.tolist() == [[[True, True, False], [False, False, False]]]

    per_sample_loss, _, _ = policy.forward_sequence_segment(
        batch,
        sequence_shape=(1, 2),
        reduction="none",
    )
    torch.testing.assert_close(per_sample_loss, torch.tensor([4.0, 0.0]))


def test_flow_denoising_advances_fast_state_only_once_per_action_chunk() -> None:
    class _PrefixOnlyBackbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.paligemma = SimpleNamespace(
                model=SimpleNamespace(
                    language_model=SimpleNamespace(config=SimpleNamespace(_attn_implementation="eager"))
                )
            )

        def forward(self, *, inputs_embeds, **kwargs):
            del kwargs
            return inputs_embeds, {}

    model = PI05TTTPytorch.__new__(PI05TTTPytorch)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(chunk_size=2, max_action_dim=3, num_inference_steps=4)
    model.paligemma_with_expert = _PrefixOnlyBackbone()
    model.ttt_layers = torch.nn.ModuleDict({"0": TTTMLPLayer(dim=4, hidden_dim=8, second_order=False)})

    def embed_prefix(self, images, img_masks, tokens, masks):
        del self, images, img_masks, tokens, masks
        return (
            torch.zeros(1, 1, 4),
            torch.ones(1, 1, dtype=torch.bool),
            torch.zeros(1, 1, dtype=torch.bool),
        )

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        expert_layer_callback,
    ):
        del self, prefix_pad_masks, past_key_values, timestep
        expert_layer_callback(0, torch.ones(x_t.shape[0], 2, 4))
        return torch.zeros_like(x_t)

    model.embed_prefix = MethodType(embed_prefix, model)
    model._denoise_step_with_callback = MethodType(denoise_step, model)
    _, fast_states = model.sample_actions_with_state(
        [],
        [],
        torch.zeros(1, 1, dtype=torch.long),
        torch.ones(1, 1, dtype=torch.bool),
        noise=torch.zeros(1, 2, 3),
        num_steps=4,
    )

    assert fast_states[0].position.tolist() == [0]


class _CheckpointPolicy(PI05TTTPolicy):
    missing_keys: list[str] = []
    unexpected_keys: list[str] = []

    def __init__(self, config, **kwargs) -> None:
        del kwargs
        self.config = config

    def _fix_pytorch_state_dict_keys(self, state_dict, config):
        del config
        return state_dict

    def load_state_dict(self, state_dict, strict=False):
        del state_dict, strict
        return list(self.missing_keys), list(self.unexpected_keys)

    def eval(self):
        return self


def _write_checkpoint_config(path: Path, raw_config: dict) -> Path:
    path.mkdir()
    (path / "config.json").write_text(json.dumps(raw_config), encoding="utf-8")
    return path


def test_base_checkpoint_may_initialize_missing_ttt_parameters_but_ttt_checkpoint_may_not(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import safetensors.torch
    import transformers.utils

    monkeypatch.setattr(
        transformers.utils,
        "cached_file",
        lambda repo, filename, **kwargs: str(Path(repo) / filename),
    )
    monkeypatch.setattr(safetensors.torch, "load_file", lambda path: {})
    _CheckpointPolicy.missing_keys = ["model.ttt_layers.14.q_proj.weight"]
    _CheckpointPolicy.unexpected_keys = []

    base_path = _write_checkpoint_config(tmp_path / "base", {"type": "pi05"})
    loaded = _CheckpointPolicy.from_pretrained(base_path)
    assert isinstance(loaded.config, PI05TTTConfig)

    ttt_path = _write_checkpoint_config(tmp_path / "ttt", {"type": "pi05_ttt"})
    with pytest.raises(RuntimeError, match="model.ttt_layers.14.q_proj.weight"):
        _CheckpointPolicy.from_pretrained(ttt_path)


def test_ttt_checkpoint_restores_structural_fields_but_keeps_requested_training_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import safetensors.torch
    import transformers.utils

    monkeypatch.setattr(
        transformers.utils,
        "cached_file",
        lambda repo, filename, **kwargs: str(Path(repo) / filename),
    )
    monkeypatch.setattr(safetensors.torch, "load_file", lambda path: {})
    _CheckpointPolicy.missing_keys = []
    _CheckpointPolicy.unexpected_keys = []
    source_fields = {
        "ttt_hidden_dim": 64,
        "ttt_base_inner_lr": 0.03,
        "ttt_effective_gate_init": 0.02,
        "ttt_rope_theta": 5_000.0,
        "ttt_second_order": False,
        "ttt_start_layer": 4,
        "ttt_layer_indices": [4, 5],
    }
    checkpoint_path = _write_checkpoint_config(
        tmp_path / "ttt_fields",
        {"type": "pi05_ttt", **source_fields},
    )
    requested = PI05TTTConfig(
        ttt_hidden_dim=32,
        ttt_layer_indices=[1],
        ttt_training_stage="action_head",
    )

    loaded = _CheckpointPolicy.from_pretrained(checkpoint_path, config=requested)

    for field_name, expected in source_fields.items():
        assert getattr(loaded.config, field_name) == expected
    assert loaded.config.ttt_training_stage == "action_head"
    assert loaded.config.pretrained_path == checkpoint_path
