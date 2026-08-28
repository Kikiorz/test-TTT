import copy
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset

from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.policies.smolvla_ttt.configuration_smolvla_ttt import SmolVLATTTConfig
from lerobot.policies.smolvla_ttt.modeling_smolvla_ttt import (
    SmolVLATTTFlowMatching,
    SmolVLATTTPolicy,
    _restore_checkpoint_model_fields,
    _validate_checkpoint_keys,
)
from lerobot.policies.smolvla_ttt.sequence import (
    SEQUENCE_ACTIVE_KEY,
    SEQUENCE_EPISODE_INDEX_KEY,
    SEQUENCE_SHAPE_KEY,
    SEQUENCE_VALID_KEY,
    SEQUENCE_WAVE_END_KEY,
    SEQUENCE_WAVE_START_KEY,
    SEQUENCE_WINDOW_ORDINAL_KEY,
    EpisodeSequenceBatchSampler,
    EpisodeWindow,
    TailPreservingSequenceDataset,
    sequence_collate_fn,
)
from lerobot.policies.smolvla_ttt.smolvlm_with_expert_ttt import SmolVLMWithExpertTTTModel
from lerobot.policies.smolvla_ttt.ttt import TTTMLPLayer
from lerobot.scripts.lerobot_train import (
    _broadcast_sequence_policy_state,
    _tbptt_segment_loss_weights,
    update_policy_tbptt,
)


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


class _ToySequencePolicy(torch.nn.Module):
    tbptt_loss_weighting = "valid_actions_per_sequence"

    def __init__(self) -> None:
        super().__init__()
        self.layer = TTTMLPLayer(dim=4, hidden_dim=8, second_order=False)
        self.received_fresh_state: list[bool] = []

    def forward_sequence_segment(
        self,
        batch,
        sequence_shape,
        fast_states=None,
        reduction="sequence",
    ):
        assert reduction == "sequence"
        self.received_fresh_state.append(fast_states is None)
        batch_size, sequence_length = sequence_shape
        valid = batch[SEQUENCE_VALID_KEY].reshape(sequence_shape)
        inputs = batch["inputs"].reshape(batch_size, sequence_length, 1, 4)
        state_in = None if fast_states is None else fast_states.get(0)
        outputs, state_out = self.layer(inputs, state=state_in, update_mask=valid)
        timestep_loss = outputs.square().mean(dim=(2, 3))
        valid_count = valid.sum(dim=1).clamp_min(1)
        per_sequence_loss = (timestep_loss * valid).sum(dim=1) / valid_count
        loss_per_dim = per_sequence_loss[:, None]
        return (
            per_sequence_loss,
            {
                "loss": per_sequence_loss.mean().item(),
                "loss_per_dim_per_sequence": loss_per_dim.detach().cpu().tolist(),
            },
            {0: state_out},
        )


class _GlooAccelerator:
    def __init__(self, rank: int, world_size: int) -> None:
        self.process_index = rank
        self.num_processes = world_size
        self.is_main_process = rank == 0
        self.device = torch.device("cpu")

    def wait_for_everyone(self) -> None:
        torch.distributed.barrier()

    def unwrap_model(self, model, **kwargs):
        del kwargs
        return model

    def reduce(self, tensor, reduction="sum"):
        reduced = tensor.clone()
        torch.distributed.all_reduce(reduced)
        if reduction == "mean":
            reduced /= self.num_processes
        elif reduction != "sum":
            raise ValueError(reduction)
        return reduced

    def autocast(self):
        return nullcontext()

    def backward(self, loss) -> None:
        loss.backward()

    def clip_grad_norm_(self, parameters, max_norm):
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)


class _LocalAccelerator:
    process_index = 0
    num_processes = 1
    is_main_process = True
    device = torch.device("cpu")

    @staticmethod
    def unwrap_model(model, **kwargs):
        del kwargs
        return model

    @staticmethod
    def reduce(tensor, reduction="sum"):
        if reduction not in {"sum", "mean"}:
            raise ValueError(reduction)
        return tensor

    @staticmethod
    def autocast():
        return nullcontext()

    @staticmethod
    def backward(loss) -> None:
        loss.backward()

    @staticmethod
    def clip_grad_norm_(parameters, max_norm):
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)


class _CountingSGD(torch.optim.SGD):
    def __init__(self, parameters, **kwargs) -> None:
        super().__init__(parameters, **kwargs)
        self.step_count = 0

    def step(self, closure=None):
        self.step_count += 1
        return super().step(closure)


class _CountingScheduler:
    def __init__(self) -> None:
        self.step_count = 0

    def step(self) -> None:
        self.step_count += 1


def _distributed_dummy_lane_worker(rank: int, world_size: int, init_file: str) -> None:
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(900 + rank)
        policy = _ToySequencePolicy()
        accelerator = _GlooAccelerator(rank, world_size)
        _broadcast_sequence_policy_state(policy, accelerator)

        optimizer = _CountingSGD(policy.parameters(), lr=0.01)
        scheduler = _CountingScheduler()
        active = rank == 0
        # Local collate lengths may differ across ranks. The all-dummy rank
        # still participates in the one outer update and contributes zeros.
        sequence_length = 3 if active else 2
        batch = {
            "inputs": torch.randn(sequence_length, 1, 4),
            "action_is_pad": torch.zeros(sequence_length, 1, dtype=torch.bool),
            SEQUENCE_VALID_KEY: torch.full((sequence_length,), active, dtype=torch.bool),
        }
        _, output = update_policy_tbptt(
            SimpleNamespace(),
            policy,
            batch,
            sequence_shape=(1, sequence_length),
            segment_length=1,
            optimizer=optimizer,
            grad_clip_norm=10.0,
            accelerator=accelerator,
            lr_scheduler=scheduler,
        )
        assert optimizer.step_count == 1
        assert scheduler.step_count == 1
        assert output["tbptt_segments"] == sequence_length

        flattened = torch.cat([parameter.detach().flatten() for parameter in policy.parameters()])
        replicas = [torch.empty_like(flattened) for _ in range(world_size)]
        torch.distributed.all_gather(replicas, flattened)
        for replica in replicas[1:]:
            torch.testing.assert_close(replica, replicas[0], rtol=0, atol=0)
    finally:
        torch.distributed.destroy_process_group()


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
    assert config.ttt_sequence_state_semantics == "sequence_outer_step_v1"


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


def test_sequence_config_allows_overlap_short_tbptt_tail_and_action_queueing() -> None:
    config = SmolVLATTTConfig(
        sequence_length=255,
        sequence_stride=128,
        tbptt_segment_length=4,
        n_action_steps=10,
    )

    assert config.sequence_stride == 128
    assert config.tbptt_segment_length == 4
    assert config.n_action_steps == 10

    with pytest.raises(ValueError, match="sequence_stride cannot exceed sequence_length"):
        SmolVLATTTConfig(sequence_length=256, sequence_stride=257)
    with pytest.raises(ValueError, match="n_action_steps must be positive"):
        SmolVLATTTConfig(n_action_steps=0)


def test_episode_sampler_globally_shards_independent_windows_and_final_dummy_lanes() -> None:
    episode_lengths = [513, 512, 257, 256, 20, 800]
    sequences = TailPreservingSequenceDataset(
        _EpisodeDataset(episode_lengths), sequence_length=256, sequence_stride=256
    )
    samplers = [
        EpisodeSequenceBatchSampler(
            sequences,
            batch_size=2,
            num_replicas=4,
            rank=rank,
            seed=7,
            shuffle=False,
        )
        for rank in range(4)
    ]

    assert {len(sampler) for sampler in samplers} == {2}
    assert {sampler.steps_per_epoch for sampler in samplers} == {2}
    schedules = [list(sampler) for sampler in samplers]
    assert {len(schedule) for schedule in schedules} == {2}

    real_windows = []
    dummy_windows = 0
    for schedule in schedules:
        for batch in schedule:
            assert len(batch) == 2
            assert all(ref.wave_start and ref.wave_end for ref in batch)
            real_windows.extend(ref.window_index for ref in batch if ref.active)
            dummy_windows += sum(not ref.active for ref in batch)

    assert sorted(real_windows) == list(range(len(sequences)))
    assert dummy_windows == 3


def test_episode_sampler_has_one_outer_step_per_global_window_batch() -> None:
    # Cover-like mixture: 32 four-window episodes and 18 five-window episodes.
    lengths = [1024] * 32 + [1025] * 18
    sequences = TailPreservingSequenceDataset(
        _EpisodeDataset(lengths), sequence_length=256, sequence_stride=256
    )

    for batch_size, expected_steps in ((8, 7), (16, 4)):
        samplers = [
            EpisodeSequenceBatchSampler(
                sequences,
                batch_size=batch_size,
                num_replicas=4,
                rank=rank,
                shuffle=False,
            )
            for rank in range(4)
        ]
        assert {len(sampler) for sampler in samplers} == {expected_steps}
        assert {sampler.steps_per_epoch for sampler in samplers} == {expected_steps}


def test_sequence_collate_right_pads_multiple_trajectories_and_marks_padding() -> None:
    def sample(frame_index: int) -> dict[str, torch.Tensor]:
        return {
            "frame_index": torch.tensor(frame_index),
            "action": torch.full((3, 2), float(frame_index)),
            "action_is_pad": torch.tensor([False, False, True]),
        }

    short_sequence = [sample(10)]
    original_short_mask = short_sequence[0]["action_is_pad"].clone()
    collated = sequence_collate_fn([[sample(0), sample(1), sample(2)], short_sequence])

    assert collated[SEQUENCE_SHAPE_KEY].tolist() == [2, 3]
    assert collated[SEQUENCE_VALID_KEY].tolist() == [True, True, True, True, False, False]
    assert collated["frame_index"].tolist() == [0, 1, 2, 10, 10, 10]
    assert collated["action_is_pad"][:4].tolist() == [
        [False, False, True],
        [False, False, True],
        [False, False, True],
        [False, False, True],
    ]
    assert collated["action_is_pad"][4:].all()
    torch.testing.assert_close(short_sequence[0]["action_is_pad"], original_short_mask)


def test_sequence_collate_fully_masks_dummy_lanes_and_preserves_wave_metadata() -> None:
    def sample(frame_index: int) -> dict[str, torch.Tensor]:
        return {
            "frame_index": torch.tensor(frame_index),
            "action": torch.zeros(2, 3),
        }

    collated = sequence_collate_fn(
        [
            EpisodeWindow([sample(0), sample(1)], 4, 1, True, True, True),
            EpisodeWindow([sample(99), sample(100)], -1, -1, True, True, False),
        ]
    )

    assert collated[SEQUENCE_SHAPE_KEY].tolist() == [2, 2]
    assert collated[SEQUENCE_VALID_KEY].tolist() == [True, True, False, False]
    assert collated[SEQUENCE_ACTIVE_KEY].tolist() == [True, False]
    assert collated[SEQUENCE_EPISODE_INDEX_KEY].tolist() == [4, -1]
    assert collated[SEQUENCE_WINDOW_ORDINAL_KEY].tolist() == [1, -1]
    assert collated[SEQUENCE_WAVE_START_KEY].item()
    assert collated[SEQUENCE_WAVE_END_KEY].item()
    assert collated["action_is_pad"][:2].logical_not().all()
    assert collated["action_is_pad"][2:].all()


def test_sequence_collate_rejects_action_padding_with_the_wrong_shape() -> None:
    sample = {
        "action": torch.zeros(3, 2),
        "action_is_pad": torch.tensor(False),
    }

    with pytest.raises(ValueError, match="action_is_pad must be one-dimensional"):
        sequence_collate_fn([[sample]])


def test_gate_is_learnable_and_initialized_to_configured_near_zero_value_during_ttt_only_stage() -> None:
    config = SmolVLATTTConfig(ttt_training_stage="ttt_only")
    layer = TTTMLPLayer(
        dim=8,
        hidden_dim=16,
        effective_gate_init=config.ttt_effective_gate_init,
        gate_trainable=config.trains_gate,
    )

    assert layer.gate.requires_grad
    torch.testing.assert_close(layer.effective_gate, torch.full((8,), 0.001), rtol=0, atol=1e-7)

    outputs, _ = layer(torch.randn(1, 2, 3, 8), create_graph=False)
    outputs.square().mean().backward()
    assert layer.gate.grad is not None


def test_fast_state_carries_across_detached_tbptt_segments() -> None:
    torch.manual_seed(1)
    layer = TTTMLPLayer(dim=8, hidden_dim=16, effective_gate_init=0.001, second_order=False)
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


def test_each_selected_sequence_cold_starts_and_steps_once_after_internal_tbptt() -> None:
    torch.manual_seed(202)
    policy = _ToySequencePolicy()
    optimizer = _CountingSGD(policy.parameters(), lr=0.01)
    scheduler = _CountingScheduler()
    accelerator = _LocalAccelerator()
    outputs = []

    for sequence_length in (5, 3):
        batch = {
            "inputs": torch.randn(sequence_length, 1, 4),
            "action_is_pad": torch.zeros(sequence_length, 1, dtype=torch.bool),
            SEQUENCE_VALID_KEY: torch.ones(sequence_length, dtype=torch.bool),
        }
        _, output = update_policy_tbptt(
            SimpleNamespace(),
            policy,
            batch,
            sequence_shape=(1, sequence_length),
            segment_length=2,
            optimizer=optimizer,
            grad_clip_norm=10.0,
            accelerator=accelerator,
            lr_scheduler=scheduler,
        )
        outputs.append(output)

    assert optimizer.step_count == 2
    assert scheduler.step_count == 2
    assert [output["tbptt_segments"] for output in outputs] == [3, 2]
    # Each call starts at W0, while later TBPTT segments in that same call carry
    # and detach the numerical state.
    assert policy.received_fresh_state == [True, False, False, True, False]


def test_ttt_fast_state_update_matches_no_grad_inside_outer_inference_mode() -> None:
    torch.manual_seed(11)
    inference_layer = TTTMLPLayer(
        dim=8,
        hidden_dim=16,
        effective_gate_init=0.001,
        second_order=False,
    ).eval()
    reference_layer = copy.deepcopy(inference_layer)
    first_inputs = torch.randn(1, 1, 3, 8)
    second_inputs = torch.randn(1, 1, 3, 8)

    with torch.no_grad():
        reference_first, reference_state = reference_layer(first_inputs)
        reference_second, reference_state = reference_layer(second_inputs, state=reference_state)

    with torch.inference_mode():
        inference_first, inference_state = inference_layer(first_inputs)
        inference_second, inference_state = inference_layer(second_inputs, state=inference_state)

    torch.testing.assert_close(inference_first, reference_first)
    torch.testing.assert_close(inference_second, reference_second)
    for inference_tensor, reference_tensor in zip(
        inference_state.tensors(), reference_state.tensors(), strict=True
    ):
        torch.testing.assert_close(inference_tensor, reference_tensor)
    assert inference_state.position.tolist() == reference_state.position.tolist() == [1]


def test_masked_batched_ttt_matches_independent_trajectory_states_and_outputs() -> None:
    torch.manual_seed(2)
    layer = TTTMLPLayer(dim=8, hidden_dim=16, effective_gate_init=0.001, second_order=False)
    inputs = torch.randn(2, 3, 2, 8)
    update_mask = torch.tensor([[True, True, True], [True, False, False]])

    batched_outputs, batched_state = layer(inputs, update_mask=update_mask)
    first_outputs, first_state = layer(inputs[:1])
    second_outputs, second_state = layer(inputs[1:2, :1])

    torch.testing.assert_close(batched_outputs[:1], first_outputs)
    torch.testing.assert_close(batched_outputs[1:2, :1], second_outputs)
    for batched_tensor, first_tensor, second_tensor in zip(
        batched_state.tensors(), first_state.tensors(), second_state.tensors(), strict=True
    ):
        torch.testing.assert_close(batched_tensor[:1], first_tensor)
        torch.testing.assert_close(batched_tensor[1:2], second_tensor)
    assert batched_state.position.tolist() == [2, 0]


def test_action_padding_token_values_do_not_change_fast_state_or_valid_outputs() -> None:
    torch.manual_seed(27)
    layer = TTTMLPLayer(dim=8, hidden_dim=16, second_order=False)
    inputs = torch.randn(1, 2, 4, 8)
    changed_padding = inputs.clone()
    changed_padding[:, :, 2:] = torch.randn_like(changed_padding[:, :, 2:]) * 1_000
    token_mask = torch.tensor([[[True, True, False, False], [True, True, False, False]]])

    outputs, state = layer(inputs, token_mask=token_mask, create_graph=False)
    changed_outputs, changed_state = layer(
        changed_padding,
        token_mask=token_mask,
        create_graph=False,
    )

    torch.testing.assert_close(changed_outputs[:, :, :2], outputs[:, :, :2], rtol=0, atol=0)
    for changed_tensor, reference_tensor in zip(changed_state.tensors(), state.tensors(), strict=True):
        torch.testing.assert_close(changed_tensor, reference_tensor, rtol=0, atol=0)
    assert changed_state.position.tolist() == state.position.tolist() == [1]


@pytest.mark.parametrize("second_order", [False, True])
def test_masked_batched_ttt_gradient_matches_average_of_independent_trajectories(
    second_order: bool,
) -> None:
    torch.manual_seed(3)
    batched_layer = TTTMLPLayer(dim=8, hidden_dim=16, second_order=second_order)
    independent_layer = copy.deepcopy(batched_layer)
    inputs = torch.randn(2, 3, 2, 8)
    update_mask = torch.tensor([[True, True, True], [True, False, False]])

    batched_outputs, _ = batched_layer(inputs, update_mask=update_mask)
    batched_loss = (batched_outputs[0].square().mean() + batched_outputs[1, :1].square().mean()) / 2
    batched_loss.backward()

    first_outputs, _ = independent_layer(inputs[:1])
    second_outputs, _ = independent_layer(inputs[1:2, :1])
    independent_loss = (first_outputs.square().mean() + second_outputs.square().mean()) / 2
    independent_loss.backward()

    for batched_parameter, independent_parameter in zip(
        batched_layer.parameters(), independent_layer.parameters(), strict=True
    ):
        if batched_parameter.grad is None or independent_parameter.grad is None:
            assert batched_parameter.grad is independent_parameter.grad is None
        else:
            torch.testing.assert_close(
                batched_parameter.grad,
                independent_parameter.grad,
                rtol=1e-5,
                atol=1e-7,
            )


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


def test_register_tokens_prepend_and_exchange_information_with_causal_actions() -> None:
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
            [True, True, True, False, False],
            [True, True, True, True, False],
            [True, True, True, True, True],
        ]
    ]
    torch.testing.assert_close(suffix[0, :2], flow.register_tokens)
    torch.testing.assert_close(flow._select_action_tokens(suffix), suffix[:, 2:])


def test_padded_actions_are_never_suffix_attention_keys_but_queries_remain_finite() -> None:
    flow = SmolVLATTTFlowMatching.__new__(SmolVLATTTFlowMatching)
    torch.nn.Module.__init__(flow)
    flow.config = SimpleNamespace(chunk_size=3, ttt_num_register_tokens=2)
    pad_mask = torch.ones(1, 5, dtype=torch.bool)
    attention_mask = torch.tensor([[1, 0, 1, 1, 1]], dtype=torch.float32)
    action_token_mask = torch.tensor([[True, False, False]])

    mask = flow._make_suffix_att_2d_masks(
        pad_mask,
        attention_mask,
        action_token_mask=action_token_mask,
    )

    # Neither register nor action queries can read padded action keys.
    assert not mask[:, :, 3:].any()
    # Padded action query rows still have valid register/action keys, avoiding
    # all-masked attention rows and the resulting softmax NaNs.
    assert mask[:, 3:, :].any(dim=-1).all()


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


def test_legacy_sequence_marker_can_be_decoded_for_labelled_evaluation() -> None:
    source = SmolVLATTTPolicy._decode_source_config(
        {
            "type": "smolvla_ttt",
            "ttt_sequence_state_semantics": "full_episode_outer_step_v2",
        }
    )

    assert source.ttt_sequence_state_semantics == "full_episode_outer_step_v2"


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


def test_tbptt_per_sequence_weights_preserve_equal_trajectory_weighting() -> None:
    batch = {
        "action_is_pad": torch.tensor(
            [
                [False, False, False],
                [False, False, False],
                [False, True, True],
                [True, True, True],
                [False, False, False],
                [True, True, True],
                [True, True, True],
                [True, True, True],
            ]
        )
    }

    weights = _tbptt_segment_loss_weights(
        batch,
        sequence_shape=(2, 4),
        segment_length=2,
        weight_by_valid_actions=True,
        per_sequence=True,
    )

    torch.testing.assert_close(weights[0], torch.tensor([6 / 7, 1.0]))
    torch.testing.assert_close(weights[1], torch.tensor([1 / 7, 0.0]))


def test_tbptt_weights_explicitly_exclude_invalid_sequence_timesteps() -> None:
    batch = {
        "action_is_pad": torch.zeros(4, 2, dtype=torch.bool),
        SEQUENCE_VALID_KEY: torch.tensor([True, True, False, False]),
    }

    weights = _tbptt_segment_loss_weights(
        batch,
        sequence_shape=(1, 4),
        segment_length=2,
        weight_by_valid_actions=True,
        per_sequence=True,
    )

    torch.testing.assert_close(weights[0], torch.tensor([1.0]))
    torch.testing.assert_close(weights[1], torch.tensor([0.0]))


def test_tbptt_weights_allow_fully_inactive_distributed_dummy_lane() -> None:
    batch = {
        "action_is_pad": torch.tensor(
            [
                [False, False],
                [False, False],
                [True, True],
                [True, True],
            ]
        ),
        SEQUENCE_VALID_KEY: torch.tensor([True, True, False, False]),
    }

    weights = _tbptt_segment_loss_weights(
        batch,
        sequence_shape=(2, 2),
        segment_length=1,
        weight_by_valid_actions=True,
        per_sequence=True,
    )

    torch.testing.assert_close(weights[0], torch.tensor([0.5, 0.0]))
    torch.testing.assert_close(weights[1], torch.tensor([0.5, 0.0]))


def test_distributed_dummy_rank_uses_zero_gradients_and_stays_synchronized(tmp_path) -> None:
    torch.multiprocessing.spawn(
        _distributed_dummy_lane_worker,
        args=(2, str(tmp_path / "gloo_init")),
        nprocs=2,
        join=True,
    )


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
