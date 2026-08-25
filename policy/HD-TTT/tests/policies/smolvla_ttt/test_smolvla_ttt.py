from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset

from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.policies.smolvla_ttt.configuration_smolvla_ttt import SmolVLATTTConfig
from lerobot.policies.smolvla_ttt.hd_ttt import (
    build_episode_event_block_mask,
    compute_hindsight_attribution,
    counterfactual_grounding_loss,
    local_kvb_loss,
)
from lerobot.policies.smolvla_ttt.modeling_smolvla_ttt import (
    SmolVLATTTFlowMatching,
    SmolVLATTTPolicy,
    _restore_checkpoint_model_fields,
    _validate_checkpoint_keys,
)
from lerobot.policies.smolvla_ttt.sequence import TailPreservingSequenceDataset
from lerobot.policies.smolvla_ttt.smolvlm_with_expert_ttt import SmolVLMWithExpertTTTModel
from lerobot.policies.smolvla_ttt.ttt import TTTMLPLayer
from lerobot.scripts.lerobot_train import _tbptt_segment_loss_weights


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


def test_factory_registers_independent_smolvla_ttt_policy() -> None:
    config = make_policy_config("smolvla_ttt")

    assert isinstance(config, SmolVLATTTConfig)
    assert config.type == "smolvla_ttt"
    assert get_policy_class("smolvla_ttt") is SmolVLATTTPolicy


def test_default_ttt_layers_match_last_four_smolvla_expert_layers() -> None:
    config = SmolVLATTTConfig()

    assert config.resolved_ttt_layer_indices == [12, 13, 14, 15]
    assert config.ttt_num_register_tokens == 16
    assert config.n_action_steps == 1


def test_config_allows_disabling_register_tokens_and_rejects_negative_count() -> None:
    assert SmolVLATTTConfig(ttt_num_register_tokens=0).ttt_num_register_tokens == 0
    with pytest.raises(ValueError, match="ttt_num_register_tokens must be non-negative"):
        SmolVLATTTConfig(ttt_num_register_tokens=-1)


def test_config_rejects_layers_without_a_reduced_expert_layer() -> None:
    with pytest.raises(ValueError, match="no expert exists"):
        SmolVLATTTConfig(num_vlm_layers=16, num_expert_layers=8, ttt_layer_indices=[13])


def test_tail_preserving_sequence_windows_cover_every_frame_once() -> None:
    episode_lengths = [258, 251, 294, 31]
    dataset = _EpisodeDataset(episode_lengths)
    sequences = TailPreservingSequenceDataset(dataset, sequence_length=256, sequence_stride=256)

    covered = []
    for index in range(len(sequences)):
        covered.extend(sample["frame_index"] for sample in sequences[index])

    assert covered == list(range(sum(episode_lengths)))
    assert max(length for _, length in sequences.window_specs) == 256
    assert any(length < 256 for _, length in sequences.window_specs)


def test_gate_is_fixed_during_ttt_only_stage() -> None:
    config = SmolVLATTTConfig(ttt_training_stage="ttt_only")
    layer = TTTMLPLayer(
        dim=8,
        hidden_dim=16,
        effective_gate_init=config.ttt_effective_gate_init,
        gate_trainable=config.trains_gate,
    )

    assert not layer.gate.requires_grad
    torch.testing.assert_close(layer.effective_gate, torch.full((8,), 0.05), rtol=0, atol=1e-7)


def test_fast_state_carries_across_detached_tbptt_segments() -> None:
    torch.manual_seed(1)
    layer = TTTMLPLayer(dim=8, hidden_dim=16, effective_gate_init=0.05, second_order=False)
    inputs = torch.randn(1, 6, 2, 8)

    joint_outputs, joint_state = layer(inputs, update=True, create_graph=False)
    first_outputs, first_state = layer(inputs[:, :3], update=True, create_graph=False)
    second_outputs, split_state = layer(
        inputs[:, 3:], state=first_state.detach(), update=True, create_graph=False
    )

    torch.testing.assert_close(
        torch.cat([first_outputs, second_outputs], dim=1), joint_outputs, rtol=0, atol=0
    )
    for split_tensor, joint_tensor in zip(split_state.tensors(), joint_state.tensors(), strict=True):
        torch.testing.assert_close(split_tensor, joint_tensor, rtol=0, atol=0)
    assert split_state.position.tolist() == joint_state.position.tolist() == [5]


def test_zero_write_gate_skips_fast_weight_mutation_but_advances_position() -> None:
    torch.manual_seed(3)
    layer = TTTMLPLayer(dim=8, hidden_dim=16, effective_gate_init=0.05, second_order=False)
    inputs = torch.randn(1, 3, 2, 8)
    _, skipped = layer(inputs, update=True, write_gate=torch.zeros(1, 3), create_graph=False)
    initial = layer.initial_state(1)
    for skipped_tensor, initial_tensor in zip(skipped.tensors(), initial.tensors(), strict=True):
        torch.testing.assert_close(skipped_tensor, initial_tensor, rtol=0, atol=0)
    assert skipped.position.tolist() == [2]


def test_hindsight_attribution_is_causal_and_counterfactual_reader_is_teacher_detached() -> None:
    full = torch.zeros(1, 4, 4)
    masked = full.clone()
    masked[0, 0, 2] = 2.0
    masked[0, 1, 3] = 1.0
    result = compute_hindsight_attribution(
        full,
        masked,
        episode_lengths=[4],
        event_block_size=1,
    )
    assert result.C[0, 0, 2].item() == 2.0
    assert result.C[0, 0, 0].item() == 0.0
    assert result.C[0, 3].sum().item() == 0.0
    assert build_episode_event_block_mask([2, 3]).shape == (2, 3, 3)

    student_true = torch.randn(2, 5, requires_grad=True)
    student_wrong = torch.randn(2, 5, requires_grad=True)
    teacher_true = torch.randn(2, 5, requires_grad=True)
    teacher_wrong = torch.randn(2, 5, requires_grad=True)
    loss = counterfactual_grounding_loss(
        student_true, student_wrong, teacher_true, teacher_wrong, rho=torch.ones(2)
    )
    loss.backward()
    assert teacher_true.grad is None and teacher_wrong.grad is None
    assert student_true.grad is not None and student_wrong.grad is not None


def test_local_kvb_writer_stops_gradient_into_value_target() -> None:
    key = torch.randn(2, 3, 4, requires_grad=True)
    value = torch.randn(2, 3, 5, requires_grad=True)
    prediction = torch.randn(2, 3, 5, requires_grad=True)
    loss = local_kvb_loss(key, key, value, prediction, write_gate=torch.ones(2, 3))
    loss.backward()
    assert prediction.grad is not None
    assert value.grad is None


def test_flow_denoising_advances_fast_state_only_once_per_observation() -> None:
    class _PrefixOnlyVLM(torch.nn.Module):
        def forward(self, *, inputs_embeds, **kwargs):
            del kwargs
            return inputs_embeds, {}

    flow = SmolVLATTTFlowMatching.__new__(SmolVLATTTFlowMatching)
    torch.nn.Module.__init__(flow)
    flow.config = SimpleNamespace(
        chunk_size=2,
        max_action_dim=3,
        num_steps=4,
        use_cache=True,
        rtc_config=None,
    )
    flow.rtc_processor = None
    flow.vlm_with_expert = _PrefixOnlyVLM()
    flow.ttt_layers = torch.nn.ModuleDict({"0": TTTMLPLayer(dim=4, hidden_dim=8, second_order=False)})

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, state):
        del self, images, img_masks, lang_tokens, lang_masks
        return (
            torch.zeros(state.shape[0], 1, 4),
            torch.ones(state.shape[0], 1, dtype=torch.bool),
            torch.zeros(state.shape[0], 1, dtype=torch.bool),
        )

    def denoise_step(self, *, x_t, expert_layer_callback, **kwargs):
        del kwargs
        hidden_states = torch.ones(x_t.shape[0], 2, 4)
        expert_layer_callback(0, hidden_states)
        return torch.zeros_like(x_t)

    flow.embed_prefix = MethodType(embed_prefix, flow)
    flow.denoise_step = MethodType(denoise_step, flow)
    state = torch.zeros(1, 3)
    noise = torch.zeros(1, 2, 3)
    _, fast_states = flow.sample_actions_with_state([], [], None, None, state, noise=noise)

    assert fast_states[0].position.tolist() == [0]


def test_register_tokens_prepend_and_read_actions_without_action_to_register_leak() -> None:
    flow = SmolVLATTTFlowMatching.__new__(SmolVLATTTFlowMatching)
    torch.nn.Module.__init__(flow)
    flow.config = SimpleNamespace(
        chunk_size=3,
        ttt_num_register_tokens=2,
        min_period=0.004,
        max_period=4.0,
    )
    flow.vlm_with_expert = SimpleNamespace(expert_hidden_size=4)
    flow.action_in_proj = torch.nn.Linear(2, 4)
    flow.action_time_mlp_in = torch.nn.Linear(8, 4)
    flow.action_time_mlp_out = torch.nn.Linear(4, 4)
    flow.register_tokens = torch.nn.Parameter(torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]))

    suffix, pad_mask, attention_mask = flow.embed_suffix(
        torch.zeros(1, 3, 2),
        torch.full((1,), 0.5),
    )

    assert suffix.shape == (1, 5, 4)
    assert pad_mask.tolist() == [[True, True, True, True, True]]
    assert attention_mask.tolist() == [[1, 0, 1, 1, 1]]
    assert flow._make_suffix_att_2d_masks(pad_mask, attention_mask).tolist() == [
        [
            [True, True, True, True, True],
            [True, True, True, True, True],
            [False, False, True, False, False],
            [False, False, True, True, False],
            [False, False, True, True, True],
        ]
    ]
    torch.testing.assert_close(suffix[0, :2], flow.register_tokens)
    torch.testing.assert_close(flow._select_action_tokens(suffix), suffix[:, 2:])


def test_zero_register_tokens_preserve_original_action_only_suffix() -> None:
    flow = SmolVLATTTFlowMatching.__new__(SmolVLATTTFlowMatching)
    torch.nn.Module.__init__(flow)
    flow.config = SimpleNamespace(
        chunk_size=3,
        ttt_num_register_tokens=0,
        min_period=0.004,
        max_period=4.0,
    )
    flow.vlm_with_expert = SimpleNamespace(expert_hidden_size=4)
    flow.action_in_proj = torch.nn.Linear(2, 4)
    flow.action_time_mlp_in = torch.nn.Linear(8, 4)
    flow.action_time_mlp_out = torch.nn.Linear(4, 4)
    flow.register_tokens = None

    suffix, pad_mask, attention_mask = flow.embed_suffix(
        torch.zeros(1, 3, 2),
        torch.full((1,), 0.5),
    )

    assert suffix.shape == (1, 3, 4)
    assert pad_mask.tolist() == [[True, True, True]]
    assert attention_mask.tolist() == [[1, 1, 1]]
    assert flow._make_suffix_att_2d_masks(pad_mask, attention_mask).tolist() == [
        [
            [True, False, False],
            [True, True, False],
            [True, True, True],
        ]
    ]
    torch.testing.assert_close(flow._select_action_tokens(suffix), suffix)


def test_checkpoint_restore_uses_source_ttt_model_fields_but_keeps_training_stage() -> None:
    source = SmolVLATTTConfig(
        ttt_hidden_dim=64,
        ttt_base_inner_lr=0.03,
        ttt_effective_gate_init=0.02,
        ttt_rope_theta=5_000.0,
        ttt_second_order=True,
        ttt_start_layer=4,
        ttt_layer_indices=[4, 5],
        ttt_num_register_tokens=3,
        ttt_training_stage="ttt_only",
    )
    target = SmolVLATTTConfig(
        ttt_hidden_dim=32,
        ttt_base_inner_lr=0.2,
        ttt_effective_gate_init=0.1,
        ttt_rope_theta=20_000.0,
        ttt_second_order=False,
        ttt_layer_indices=[1],
        ttt_num_register_tokens=0,
        ttt_training_stage="action_head",
    )
    raw_config = {
        field_name: getattr(source, field_name)
        for field_name in (
            "ttt_hidden_dim",
            "ttt_base_inner_lr",
            "ttt_effective_gate_init",
            "ttt_rope_theta",
            "ttt_second_order",
            "ttt_start_layer",
            "ttt_layer_indices",
            "ttt_num_register_tokens",
        )
    }

    _restore_checkpoint_model_fields(target, source, raw_config)

    for field_name, expected_value in raw_config.items():
        assert getattr(target, field_name) == expected_value
    assert target.ttt_training_stage == "action_head"


def test_ttt_checkpoint_missing_keys_are_always_rejected() -> None:
    missing = ["model.ttt_layers.12.q_proj.weight", "model.register_tokens"]

    _validate_checkpoint_keys(missing, [], source_is_ttt=False, strict=False)
    with pytest.raises(RuntimeError, match="Incompatible SmolVLA checkpoint"):
        _validate_checkpoint_keys(missing, [], source_is_ttt=True, strict=False)
    with pytest.raises(RuntimeError, match="Incompatible SmolVLA checkpoint"):
        _validate_checkpoint_keys(missing, [], source_is_ttt=False, strict=True)


def test_checkpoint_compatibility_never_ignores_backbone_or_unexpected_keys() -> None:
    with pytest.raises(RuntimeError, match="model.action_out_proj.weight"):
        _validate_checkpoint_keys(
            ["model.action_out_proj.weight"],
            [],
            source_is_ttt=False,
            strict=False,
        )
    with pytest.raises(RuntimeError, match="unexpected.weight"):
        _validate_checkpoint_keys(
            [],
            ["unexpected.weight"],
            source_is_ttt=False,
            strict=False,
        )


def test_tbptt_weights_segments_by_valid_actions_instead_of_timestep_count() -> None:
    batch = {
        "action_is_pad": torch.tensor(
            [
                [False, False, False],
                [False, False, False],
                [False, True, True],
                [True, True, True],
            ]
        )
    }

    weights = _tbptt_segment_loss_weights(
        batch,
        sequence_shape=(1, 4),
        segment_length=2,
        weight_by_valid_actions=True,
    )

    assert weights == pytest.approx([6 / 7, 1 / 7])

    timestep_weights = _tbptt_segment_loss_weights(
        batch,
        sequence_shape=(1, 4),
        segment_length=2,
        weight_by_valid_actions=False,
    )
    assert timestep_weights == pytest.approx([0.5, 0.5])


def test_ttt_hook_runs_after_expert_attention_residual_and_before_mlp() -> None:
    class _Attention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.o_proj = torch.nn.Linear(4, 4, bias=False)
            torch.nn.init.zeros_(self.o_proj.weight)

    class _Layer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = _Attention()
            self.post_attention_layernorm = torch.nn.Identity()
            self.mlp = torch.nn.Identity()

    class _TextModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([_Layer()])
            self.norm = torch.nn.Identity()

    model = SmolVLMWithExpertTTTModel.__new__(SmolVLMWithExpertTTTModel)
    torch.nn.Module.__init__(model)
    vlm_text_model = _TextModel()
    vlm_config = SimpleNamespace(text_config=SimpleNamespace(head_dim=4))
    model.vlm = SimpleNamespace(model=SimpleNamespace(text_model=vlm_text_model), config=vlm_config)
    model.lm_expert = _TextModel()
    model.num_vlm_layers = 1
    model.num_expert_layers = 1
    model.attention_mode = "self_attn"
    model.self_attn_every_n_layers = -1
    model.config = vlm_config

    def forward_attn_layer(self, model_layers, inputs_embeds, *args, past_key_values=None, **kwargs):
        del self, model_layers, args, kwargs
        sequence_length = sum(hidden.shape[1] for hidden in inputs_embeds if hidden is not None)
        batch_size = next(hidden.shape[0] for hidden in inputs_embeds if hidden is not None)
        return [torch.zeros(batch_size, sequence_length, 4)], past_key_values

    model.forward_attn_layer = MethodType(forward_attn_layer, model)
    callback_inputs = []

    def callback(layer_index, hidden_states):
        callback_inputs.append((layer_index, hidden_states.clone()))
        return hidden_states * 2

    prefix = torch.ones(1, 1, 4)
    expert = torch.ones(1, 2, 4)
    outputs, _ = model.forward(
        attention_mask=torch.ones(1, 3, 3, dtype=torch.bool),
        position_ids=torch.arange(3).unsqueeze(0),
        past_key_values=None,
        inputs_embeds=[prefix, expert],
        use_cache=False,
        fill_kv_cache=False,
        expert_layer_callback=callback,
    )

    assert callback_inputs[0][0] == 0
    torch.testing.assert_close(callback_inputs[0][1], expert)
    torch.testing.assert_close(outputs[1], expert * 4)


def test_smolvla_ttt_code_does_not_import_sibling_policy_implementations() -> None:
    policy_dir = Path(__file__).parents[3] / "src" / "lerobot" / "policies" / "smolvla_ttt"
    source = "\n".join(path.read_text() for path in policy_dir.glob("*.py"))

    assert "lerobot.policies.smolvla" not in source
    assert "lerobot.policies.pi0_ttt" not in source
    assert "lerobot.policies.pi05_ttt" not in source
