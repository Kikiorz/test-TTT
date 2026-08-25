"""Paper-faithful RoboTTT action-head reconstruction.

NVIDIA has not released the RoboTTT implementation. This package implements
every public algorithmic detail and labels all under-specified choices.
"""

from .layer import FastState, RoboTTTKVBLayer
from .backbone import LiberoRoboTTTBackbone, RoboTTTBackboneConfig
from .policy import (
    LayerFastStates,
    RoboTTTPolicy,
    sample_sequence_action_forcing_taus,
)

__all__ = [
    "FastState",
    "LayerFastStates",
    "LiberoRoboTTTBackbone",
    "RoboTTTBackboneConfig",
    "RoboTTTKVBLayer",
    "RoboTTTPolicy",
    "sample_sequence_action_forcing_taus",
]
