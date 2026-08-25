import json
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
from lerobot.policies.smolvla_ttt.sequence import HD_WRITER_VALID_KEY, TailPreservingSequenceDataset
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


def test_policy_config_serialization_keeps_choice_type(tmp_path: Path) -> None:
    config = SmolVLATTTConfig(device="cpu")
    config._save_pretrained(tmp_path)

    payload = json.loads((tmp_path / "config.json").read_text())
    assert payload["type"] == "smolvla_ttt"


def test_default_ttt_layers_match_last_four_smolvla_expert_layers() -> None:
    config = SmolVLATTTConfig()

    assert config.resolved_ttt_layer_indices == [12, 13, 14, 15]
    assert config.ttt_num_register_tokens == 16
    assert config.n_action_steps == 1
    assert config.hd_counterfactual_margin == 0.0


def test_legacy_null_hd_flags_decode_as_clean_checkpoint() -> None:
    source = SmolVLATTTPolicy._decode_source_config(
        {
            "type": "smolvla_ttt",
            "hd_ttt_enabled": None,
            "hd_learned_write_gate": None,
        }
    )

    assert source.hd_ttt_enabled is False
    assert source.hd_learned_write_gate is False


def test_config_allows_disabling_register_tokens_and_rejects_negative_count() -> None:
    assert SmolVLATTTConfig(ttt_num_register_tokens=0).ttt_num_register_tokens == 0
    with pytest.raises(ValueError, match="ttt_num_register_tokens must be non-negative"):
        SmolVLATTTConfig(ttt_num_register_tokens=-1)


def test_learned_gate_requires_hd_objective() -> None:
    with pytest.raises(ValueError, match="hd_learned_write_gate requires hd_ttt_enabled"):
        SmolVLATTTConfig(hd_learned_write_gate=True)


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


def test_long_horizon_window_cap_is_deterministic_and_episode_balanced() -> None:
    dataset = _EpisodeDataset([100, 35])
    sequences = TailPreservingSequenceDataset(
        dataset,
        sequence_length=16,
        sequence_stride=16,
        max_windows_per_episode=2,
    )

    # Two windows per episode, with the first window retained for causal
    # warm-up and a later evenly spaced window for temporal coverage.
    assert len(sequences.window_specs) == 4
    assert sequences.window_specs[0] == (0, 16)
    assert sequences.window_specs[2] == (100, 16)
    # Capped sampling must retain a complete terminal window rather than the
    # one-frame tail that would be produced by the raw stride offsets.
    assert sequences.window_specs[1] == (84, 16)
    assert sequences.window_specs[3] == (119, 16)


def test_full_history_single_window_rejects_capacity_truncation() -> None:
    dataset = _EpisodeDataset([17])
    with pytest.raises(ValueError, match="full-history replay"):
        TailPreservingSequenceDataset(
            dataset,
            sequence_length=16,
            sequence_stride=16,
            max_windows_per_episode=1,
            history_warmup_length=None,
        )


def test_history_warmup_masks_prefix_but_keeps_it_in_the_recurrent_window() -> None:
    class _ActionDataset(_EpisodeDataset):
        def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
            return {
                "frame_index": index,
                "action": torch.zeros(3, 2),
                "action_is_pad": torch.zeros(3, dtype=torch.bool),
                "hd_write_gate": torch.tensor(1.0),
            }

    dataset = _ActionDataset([20])
    sequences = TailPreservingSequenceDataset(
        dataset,
        sequence_length=4,
        sequence_stride=4,
        history_warmup_length=3,
    )

    # The second target window starts at frame 4 and receives frames 1..4 as
    # [warm-up 1,2,3, target 4]. Warm-up updates state but has no loss.
    samples = sequences[1]
    assert [sample["frame_index"] for sample in samples] == [1, 2, 3, 4, 5, 6, 7]
    assert all(bool(samples[index]["action_is_pad"].all()) for index in range(3))
    assert all(not bool(samples[index]["action_is_pad"].any()) for index in range(3, 7))
    assert all(bool(sample[HD_WRITER_VALID_KEY]) for sample in samples)


def test_none_history_warmup_replays_from_episode_start() -> None:
    dataset = _EpisodeDataset([20])
    sequences = TailPreservingSequenceDataset(
        dataset,
        sequence_length=4,
        sequence_stride=4,
        history_warmup_length=None,
    )
    assert [sample["frame_index"] for sample in sequences[2]] == list(range(12))


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
    assert layer.write_gate_head is None


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


def test_grounding_branches_carry_their_own_state_across_tbptt_segments() -> None:
    """An early blocked write must remain absent in every later segment."""

    class _ReplayState:
        def __init__(self, position: int) -> None:
            self.position = torch.tensor([position], dtype=torch.int64)

        def clone(self, *, detach: bool = False, requires_grad: bool = True):
            del detach, requires_grad
            return _ReplayState(int(self.position.item()))

        def detach(self, requires_grad: bool = True):
            del requires_grad
            return _ReplayState(int(self.position.item()))

    class _ReplayModel:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int, bool, bool]] = []

        def forward_with_state(
            self,
            *args,
            sequence_shape,
            fast_states=None,
            detach_writer=False,
            write_gate=None,
            return_local_loss=False,
            **kwargs,
        ):
            del args, kwargs
            sequence_length = sequence_shape[1]
            previous_position = (
                -1
                if fast_states is None
                else int(next(iter(fast_states.values())).position.item())
            )
            self.calls.append(
                (previous_position, sequence_length, detach_writer, write_gate is None)
            )
            velocity = torch.zeros(sequence_length, 1, 1, requires_grad=True)
            next_states = {
                0: _ReplayState(previous_position + sequence_length),
            }
            if return_local_loss:
                local_loss = torch.zeros(1, sequence_length, requires_grad=True)
                return velocity, next_states, local_loss
            return velocity, next_states

    policy = SmolVLATTTPolicy.__new__(SmolVLATTTPolicy)
    policy.config = SimpleNamespace(
        adapt_to_pi_aloha=False,
        action_feature=SimpleNamespace(shape=(1,)),
        hd_ttt_enabled=True,
        hd_learned_write_gate=False,
        hd_phase_mode="deployment",
    )
    policy.model = _ReplayModel()
    policy.prepare_images = lambda batch: (
        [torch.zeros(2, 1, 1, 1)],
        [torch.ones(2, 1, dtype=torch.bool)],
    )
    policy.prepare_state = lambda batch: torch.zeros(2, 1)
    policy.prepare_action = lambda batch: batch["action"]
    # This test isolates numerical branch-state bookkeeping.  The tensor
    # utilities and grounding gradients are covered by dedicated tests below.
    policy._hd_auxiliary_losses = lambda *args, **kwargs: (
        torch.zeros((), requires_grad=True),
        {},
    )

    def segment_batch() -> dict[str, torch.Tensor]:
        return {
            "action": torch.zeros(2, 1, 1),
            "observation.state": torch.zeros(2, 1),
            "observation.language.tokens": torch.zeros(2, 1, dtype=torch.long),
            "observation.language.attention_mask": torch.ones(2, 1, dtype=torch.bool),
            "action_is_pad": torch.zeros(2, 1, dtype=torch.bool),
            "hd_teacher_velocity": torch.zeros(2, 1, 1),
            "hd_teacher_true_velocity": torch.zeros(2, 1, 1),
            "hd_teacher_wrong_velocity": torch.zeros(2, 1, 1),
            "hd_write_gate": torch.ones(2),
            "hd_counterfactual_write_gate": torch.ones(2),
        }

    grounding_states = {"true": None, "wrong": None}
    noise = torch.zeros(2, 1, 1)
    time = torch.ones(2)
    _, _, main_states = policy.forward_sequence_segment(
        segment_batch(),
        (1, 2),
        noise=noise,
        time=time,
        grounding_states=grounding_states,
    )
    _, _, _ = policy.forward_sequence_segment(
        segment_batch(),
        (1, 2),
        fast_states={index: state.detach() for index, state in main_states.items()},
        noise=noise,
        time=time,
        grounding_states=grounding_states,
    )

    # Each segment performs main, true/all-write, then wrong/intervened replay.
    # Both counterfactual branches must begin segment two at position 1, the
    # state returned by their own first segment, not from a fresh position -1.
    assert policy.model.calls == [
        (-1, 2, False, False),
        (-1, 2, True, True),
        (-1, 2, True, False),
        (1, 2, False, False),
        (1, 2, True, True),
        (1, 2, True, False),
    ]
    assert grounding_states["true"][0].position.tolist() == [3]
    assert grounding_states["wrong"][0].position.tolist() == [3]


def test_frozen_teacher_still_computes_local_fast_weight_updates() -> None:
    """Freezing outer parameters must not disable the deployment-time inner loop."""

    torch.manual_seed(2)
    layer = TTTMLPLayer(dim=8, hidden_dim=16, effective_gate_init=0.05, second_order=False)
    layer.requires_grad_(False)
    inputs = torch.randn(1, 3, 2, 8)

    with torch.no_grad():
        outputs, state = layer(inputs, update=True, create_graph=False)

    assert torch.isfinite(outputs).all()
    assert state.position.tolist() == [2]
    assert all(not parameter.requires_grad for parameter in layer.parameters())


def test_learned_write_gate_is_causal_and_initializes_near_full_write() -> None:
    torch.manual_seed(21)
    layer = TTTMLPLayer(
        dim=8,
        hidden_dim=16,
        second_order=False,
        learned_write_gate=True,
        write_gate_init=0.95,
        write_gate_token_index=2,
    )
    inputs = torch.randn(2, 3, 5, 8, dtype=torch.float32)
    gates = layer.predict_write_gate(inputs)
    assert gates.shape == (2, 3)
    assert torch.all((gates > 0) & (gates < 1))
    torch.testing.assert_close(gates, torch.full_like(gates, 0.95), atol=1e-6, rtol=0)

    # The head reads only token 2. Changing another (future) token cannot
    # alter the local gate, which is the causal input contract used at deploy.
    changed = inputs.clone()
    changed[:, :, 4] += 100.0
    torch.testing.assert_close(layer.predict_write_gate(changed), gates, atol=0, rtol=0)


def test_learned_write_gate_controls_state_updates() -> None:
    torch.manual_seed(22)
    layer = TTTMLPLayer(
        dim=8,
        hidden_dim=16,
        second_order=False,
        learned_write_gate=True,
        write_gate_init=0.5,
        write_gate_token_index=0,
    )
    inputs = torch.randn(1, 2, 3, 8)
    # Force a deterministic local prediction and compare against an explicit
    # zero intervention.  Both runs must advance position, but only the
    # learned-gate run mutates the fast weights.
    with torch.no_grad():
        layer.write_gate_head.weight.zero_()
        layer.write_gate_head.bias.fill_(10.0)
    predicted = layer.predict_write_gate(inputs)
    _, learned_state = layer(
        inputs,
        update=True,
        write_gate=predicted,
        create_graph=False,
    )
    _, skipped_state = layer(
        inputs,
        state=layer.initial_state(1),
        update=True,
        write_gate=torch.zeros(1, 2),
        create_graph=False,
    )
    assert learned_state.position.tolist() == skipped_state.position.tolist() == [1]
    assert any(
        not torch.equal(a, b)
        for a, b in zip(learned_state.tensors(), skipped_state.tensors(), strict=True)
    )


def test_prefix_context_gate_never_reads_action_or_denoising_inputs() -> None:
    torch.manual_seed(23)
    layer = TTTMLPLayer(
        dim=8,
        hidden_dim=16,
        second_order=False,
        learned_write_gate=True,
        write_gate_init=0.5,
        write_gate_context_dim=6,
    )
    with torch.no_grad():
        layer.write_gate_context_head.weight.copy_(torch.arange(6, dtype=torch.float32)[None, :] / 10)
    inputs = torch.randn(2, 3, 5, 8)
    context = torch.randn(2, 3, 6)
    gates = layer.predict_write_gate(inputs, context=context)
    changed_inputs = inputs + 1000.0
    changed_context = context.clone()
    changed_context[:, 1] += 1000.0
    changed_gates = layer.predict_write_gate(changed_inputs, context=changed_context)
    # The action/noise tensor is ignored; only the corresponding prefix row
    # may change when its observation context changes.
    torch.testing.assert_close(changed_gates[:, 0], gates[:, 0], atol=0, rtol=0)
    torch.testing.assert_close(changed_gates[:, 2], gates[:, 2], atol=0, rtol=0)
    assert not torch.equal(changed_gates[:, 1], gates[:, 1])
    with pytest.raises(RuntimeError, match="prefix context is required"):
        layer.predict_write_gate(inputs)


def test_zero_write_gate_skips_fast_weight_mutation_but_advances_position() -> None:
    torch.manual_seed(3)
    layer = TTTMLPLayer(dim=8, hidden_dim=16, effective_gate_init=0.05, second_order=False)
    inputs = torch.randn(1, 3, 2, 8)
    _, skipped = layer(inputs, update=True, write_gate=torch.zeros(1, 3), create_graph=False)
    initial = layer.initial_state(1)
    for skipped_tensor, initial_tensor in zip(skipped.tensors(), initial.tensors(), strict=True):
        torch.testing.assert_close(skipped_tensor, initial_tensor, rtol=0, atol=0)
    assert skipped.position.tolist() == [2]


def test_detach_writer_keeps_query_gradient_but_blocks_writer_gradient() -> None:
    """Reader-only grounding must not update K/V or fast-weight parameters."""

    torch.manual_seed(7)
    layer = TTTMLPLayer(
        dim=8,
        hidden_dim=16,
        effective_gate_init=0.05,
        gate_trainable=True,
        second_order=True,
    )
    inputs = torch.randn(1, 2, 3, 8, requires_grad=True)
    outputs, _ = layer(
        inputs,
        update=True,
        create_graph=True,
        detach_writer=True,
    )
    outputs.square().mean().backward()

    assert inputs.grad is not None
    assert layer.q_proj.weight.grad is not None
    parameters = dict(layer.named_parameters())
    for parameter_name in (
        "k_proj.weight",
        "v_proj.weight",
        "fast_w1_init",
        "fast_b1_init",
        "fast_w2_init",
        "fast_b2_init",
        "log_inner_lr_multiplier",
        "gate",
    ):
        assert parameters[parameter_name].grad is None


def test_ttt_can_return_per_timestep_local_kv_loss() -> None:
    """The opt-in H2L API exposes raw inner losses without changing defaults."""

    torch.manual_seed(8)
    layer = TTTMLPLayer(dim=8, hidden_dim=16, second_order=False)
    inputs = torch.randn(2, 3, 2, 8, requires_grad=True)
    outputs, state, local_loss = layer(
        inputs,
        update=True,
        create_graph=False,
        return_local_loss=True,
    )

    assert outputs.shape == inputs.shape
    assert state.position.tolist() == [2, 2]
    assert local_loss.shape == (2, 3)
    assert torch.isfinite(local_loss).all()
    assert local_loss.requires_grad
    local_loss.mean().backward()
    assert layer.k_proj.weight.grad is not None
    assert layer.v_proj.weight.grad is not None


def test_flow_forward_with_state_forwards_optional_local_loss() -> None:
    """FlowMatching returns one local loss per physical sequence timestep."""

    flow = SmolVLATTTFlowMatching.__new__(SmolVLATTTFlowMatching)
    torch.nn.Module.__init__(flow)
    flow.ttt_layers = torch.nn.ModuleDict(
        {"0": TTTMLPLayer(dim=4, hidden_dim=8, second_order=False)}
    )

    def fake_forward(
        self,
        *args,
        expert_layer_callback=None,
        return_velocity=False,
        **kwargs,
    ):
        del self, args, kwargs, return_velocity
        hidden = torch.randn(2, 3, 4, requires_grad=True)
        expert_layer_callback(0, hidden)
        return torch.zeros(2, 2, 3)

    flow.forward = MethodType(fake_forward, flow)
    losses, fast_states, local_loss = flow.forward_with_state(
        None,
        None,
        None,
        None,
        torch.zeros(2, 1),
        torch.zeros(2, 2, 3),
        torch.zeros(2, 2, 3),
        torch.zeros(2),
        sequence_shape=(1, 2),
        return_velocity=True,
        return_local_loss=True,
    )

    assert losses.shape == (2, 2, 3)
    assert 0 in fast_states
    assert local_loss.shape == (1, 2)
    assert torch.isfinite(local_loss).all()


def test_flow_forward_with_state_auto_injects_one_shared_learned_gate() -> None:
    flow = SmolVLATTTFlowMatching.__new__(SmolVLATTTFlowMatching)
    torch.nn.Module.__init__(flow)
    flow.ttt_layers = torch.nn.ModuleDict(
        {
            "0": TTTMLPLayer(
                dim=4,
                hidden_dim=8,
                second_order=False,
                learned_write_gate=True,
                write_gate_token_index=0,
            ),
            "1": TTTMLPLayer(dim=4, hidden_dim=8, second_order=False),
        }
    )
    flow.write_gate_layer_index = 0

    def fake_forward(
        self,
        *args,
        expert_layer_callback=None,
        return_velocity=False,
        **kwargs,
    ):
        del self, args, kwargs, return_velocity
        hidden = torch.randn(2, 3, 4, requires_grad=True)
        expert_layer_callback(0, hidden)
        expert_layer_callback(1, hidden)
        return torch.zeros(2, 2, 3)

    flow.forward = MethodType(fake_forward, flow)
    losses, _, predicted_gate = flow.forward_with_state(
        None,
        None,
        None,
        None,
        torch.zeros(2, 1),
        torch.zeros(2, 2, 3),
        torch.zeros(2, 2, 3),
        torch.zeros(2),
        sequence_shape=(1, 2),
        return_velocity=True,
        use_learned_write_gate=True,
        return_write_gate=True,
    )
    assert losses.shape == (2, 2, 3)
    assert predicted_gate.shape == (1, 2)
    assert torch.all((predicted_gate > 0) & (predicted_gate < 1))


def test_inference_mode_can_run_a_learned_gate_inner_update() -> None:
    layer = TTTMLPLayer(
        dim=4,
        hidden_dim=8,
        second_order=False,
        learned_write_gate=True,
        write_gate_context_dim=4,
    )
    inputs = torch.randn(1, 2, 3, 4)
    context = torch.randn(1, 2, 4)
    with torch.inference_mode():
        gate = layer.predict_write_gate(inputs, context=context)
        _, state = layer(inputs, update=True, write_gate=gate, create_graph=False)
    assert state.position.tolist() == [1]
    assert all(torch.isfinite(tensor).all() for tensor in state.tensors())


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


def test_checkpoint_restore_preserves_explicit_hd_opt_in_over_clean_ttt_source() -> None:
    source = SmolVLATTTConfig(hd_ttt_enabled=False, hd_learned_write_gate=False)
    target = SmolVLATTTConfig(hd_ttt_enabled=True, hd_learned_write_gate=True)
    raw_config = {
        "type": "smolvla_ttt",
        "hd_ttt_enabled": False,
        "hd_learned_write_gate": False,
    }

    _restore_checkpoint_model_fields(target, source, raw_config)

    assert target.hd_ttt_enabled is True
    assert target.hd_learned_write_gate is True


def test_checkpoint_restore_preserves_explicit_hd_opt_out_over_hd_source() -> None:
    """A clean/ablation config must be able to disable an HD checkpoint."""

    source = SmolVLATTTConfig(
        hd_ttt_enabled=True,
        hd_learned_write_gate=True,
        hd_hca_weight=0.7,
    )
    # This is the config produced when the user supplies
    # ``--policy.hd_ttt_enabled=false --policy.hd_learned_write_gate=false``
    # after the pretrained-path parser has read the source config.
    target = SmolVLATTTConfig(
        hd_ttt_enabled=False,
        hd_learned_write_gate=False,
    )
    raw_config = {
        "type": "smolvla_ttt",
        "hd_ttt_enabled": True,
        "hd_learned_write_gate": True,
        "hd_hca_weight": 0.7,
    }

    _restore_checkpoint_model_fields(target, source, raw_config)

    assert target.hd_ttt_enabled is False
    assert target.hd_learned_write_gate is False
    # The target's HD hyperparameters remain coherent with the disabled path;
    # they must not be silently replaced by source-only values.
    assert target.hd_hca_weight == 1.0


def test_checkpoint_restore_keeps_source_hd_config_when_parser_did_not_override_it() -> None:
    """A source-parsed config (the normal no-override path) remains HD-enabled."""

    source = SmolVLATTTConfig(hd_ttt_enabled=True, hd_learned_write_gate=True)
    target = SmolVLATTTConfig(hd_ttt_enabled=True, hd_learned_write_gate=True)
    raw_config = {
        "type": "smolvla_ttt",
        "hd_ttt_enabled": True,
        "hd_learned_write_gate": True,
    }

    _restore_checkpoint_model_fields(target, source, raw_config)

    assert target.hd_ttt_enabled is True
    assert target.hd_learned_write_gate is True


def test_old_ttt_checkpoint_may_omit_the_optional_gate_but_hd_checkpoint_may_not() -> None:
    missing_gate = [
        "model.ttt_layers.12.write_gate_head.weight",
        "model.ttt_layers.12.write_gate_context_head.weight",
    ]
    _validate_checkpoint_keys(
        missing_gate,
        [],
        source_is_ttt=True,
        strict=False,
        source_has_learned_write_gate=False,
    )
    with pytest.raises(RuntimeError, match="Incompatible SmolVLA checkpoint"):
        _validate_checkpoint_keys(
            missing_gate,
            [],
            source_is_ttt=True,
            strict=False,
            source_has_learned_write_gate=True,
        )


def test_hd_gate_context_head_can_be_dropped_for_explicit_clean_ablation() -> None:
    context_head = ["model.ttt_layers.12.write_gate_context_head.weight"]
    _validate_checkpoint_keys(
        [],
        context_head,
        source_is_ttt=True,
        strict=False,
        source_has_learned_write_gate=True,
        target_has_learned_write_gate=False,
    )
    with pytest.raises(RuntimeError, match="Incompatible SmolVLA checkpoint"):
        _validate_checkpoint_keys(
            [],
            context_head,
            source_is_ttt=True,
            strict=False,
            source_has_learned_write_gate=True,
            target_has_learned_write_gate=True,
        )


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


def test_hd_action_validity_retains_chunk_slot_mask() -> None:
    pad = torch.tensor(
        [[False, False, True], [False, True, True], [False, False, False], [True, True, True]]
    )
    slot_valid = SmolVLATTTPolicy._hd_action_slot_valid_weight(
        {"action_is_pad": pad},
        sequence_shape=(1, 4),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert slot_valid.tolist() == [[[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]]
    step_valid = SmolVLATTTPolicy._hd_valid_step_weight(
        {"action_is_pad": pad},
        sequence_shape=(1, 4),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    torch.testing.assert_close(step_valid, torch.tensor([[2 / 3, 1 / 3, 1.0, 0.0]]))


def test_grounding_slot_reduction_does_not_square_terminal_padding_weight() -> None:
    # The second timestep has one valid slot out of three.  First averaging
    # valid slots and then applying the 1/3 timestep weight gives
    # (1*1 + 3*(1/3)) / (1 + 1/3) = 1.5.  Multiplying the raw slot field by
    # 1/3 before reduction would incorrectly produce 1.2.
    values = torch.tensor([[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]]])
    slot_valid = torch.tensor([[[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]]])
    step_weights = torch.tensor([[1.0, 1.0 / 3.0]])

    reduced = SmolVLATTTPolicy._hd_reduce_grounding_slots(
        values,
        slot_valid,
        step_weights,
    )

    torch.testing.assert_close(reduced, torch.tensor(1.5))


def test_tbptt_writer_mask_keeps_history_only_segment_trainable() -> None:
    batch = {
        "action_is_pad": torch.tensor(
            [[True, True], [False, False], [False, False], [False, False]]
        ),
        "hd_writer_valid": torch.ones(4, dtype=torch.bool),
    }
    weights = _tbptt_segment_loss_weights(
        batch,
        sequence_shape=(1, 4),
        segment_length=2,
        weight_by_valid_actions=True,
        include_writer_valid=True,
    )
    # Both segments contain two physical writer interactions.  Action chunk
    # slots do not multiply the weight of the target segment.
    assert weights == pytest.approx([0.5, 0.5])


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
