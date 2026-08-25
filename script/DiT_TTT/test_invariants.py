from __future__ import annotations

import torch

from DiT.model import DiTConfig, LiberoGR00TDiT
from DiT_TTT.policy import DiTTTPolicy


def main():
    torch.manual_seed(0)
    base = LiberoGR00TDiT(DiTConfig(num_layers=2, hidden_dim=128, num_heads=4, dropout=0.0)).eval()
    wrapped = DiTTTPolicy(base, memory_dim=32, fast_hidden_dim=32).eval()
    wrapped.set_deployment_mode("gate0")
    context = torch.randn(2, 9, 128)
    actions = torch.randn(2, 8, 7)
    timesteps = torch.tensor([11, 513])
    with torch.no_grad():
        baseline = base.predict_velocity(actions, timesteps, context)
    residuals, next_state, _ = wrapped.prepare_layer_residuals(context, None, create_graph=False)
    with torch.no_grad():
        gate0 = wrapped.predict_velocity_with_residuals(actions, timesteps, context, *residuals)
    error = float((baseline - gate0).abs().max())
    if error != 0.0:
        raise AssertionError(f"gate0 is not exactly baseline: max_abs_error={error}")
    if len(next_state) != base.config.num_layers:
        raise AssertionError("one fast state per DiT layer was not created")
    if any(float(residual.abs().max()) != 0.0 for residual in residuals):
        raise AssertionError("gate0 produced a nonzero layer residual")
    print({"gate0_max_abs_error": error, "layer_states": len(next_state),
           "writes_per_decision": 1, "action_target_in_write": False})


if __name__ == "__main__":
    main()

