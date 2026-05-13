"""
ElectrochemicalJEPA
===================
A JEPA world model for encoding electrochemical operating conditions
into a latent profile. Implements the Lár AbstractManifold interface.

Electrochemical conditions are EXPERIMENT-SPECIFIC — they vary by
temperature, voltage, current density, and cycle protocol. Unlike
crystal structures, these cannot be pre-cached. ElectrochemicalJEPA
runs live once per experiment and the result is cached in GraphState
for reuse across the composition search loop.

Domain interpretation
---------------------
An electrochemical profile answers: "Under what conditions is this
electrolyte being asked to operate?" — temperature, voltage window,
charge rate (C-rate), cycle count, and impedance measurements.

The same crystal composition may perform very differently at 25°C vs
60°C, at 3V vs 5V, or at 0.1C vs 5C discharge. The electrochemical
embedding captures these operating demands so the cross-attention head
can ask: "Which elemental sites in this crystal structure are responsible
for surviving THIS specific set of operating conditions?"

Production upgrade path
-----------------------
Replace the linear encoder with a physics-informed neural network
trained on Electrochemical Impedance Spectroscopy (EIS) data from
public repositories (e.g., ECDH at NREL, MPContribs battery datasets).
The AbstractManifold interface is unchanged.
"""

import sys
import os
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.interfaces import AbstractManifold
from core.types import ModelType, SignalType

# Electrochemical measurement dimensions
# [voltage_window, current_density, temperature_K, initial_capacity,
#  coulombic_efficiency, impedance_real, impedance_imag, cycle_number,
#  c_rate, state_of_charge, depth_of_discharge, formation_cycles]
MEASUREMENT_DIM = 12


@dataclass
class LatentElectrochemicalState:
    """
    Latent representation of an electrochemical operating condition.

    capacity_retention : float  [0, 1]
        Predicted fraction of initial capacity retained after N cycles.
        1.0 = perfect retention, 0.0 = complete capacity fade.

    latent_vector : List[float]  shape (embed_dim,)
        The dense embedding passed to the cross-attention query projection.
    """
    voltage_window: float         # V
    temperature_K: float          # K
    c_rate: float                 # charge/discharge rate (1C = full charge in 1h)
    capacity_retention: float     # 0–1 predicted retention
    latent_vector: List[float]    # (embed_dim,)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "voltage_window":     self.voltage_window,
            "temperature_K":      self.temperature_K,
            "c_rate":             self.c_rate,
            "capacity_retention": self.capacity_retention,
        }


class ElectrochemicalJEPA(AbstractManifold):
    """
    Electrochemical operating condition encoder implementing AbstractManifold.

    Input  : (1, MEASUREMENT_DIM) tensor of EIS/cycling measurements
    Output : LatentElectrochemicalState with latent_vector (embed_dim,)

    The pooled (embed_dim,) output is intentional — unlike crystal
    structures which have meaningful per-site structure, electrochemical
    conditions are a holistic profile of the operating environment.
    """
    model_type = ModelType.JEPA

    def __init__(self, embed_dim: int = 256):
        self.embed_dim = embed_dim
        self.measurement_dim = MEASUREMENT_DIM

        self.encoder = nn.Sequential(
            nn.Linear(MEASUREMENT_DIM, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, embed_dim),
        )

        # Capacity retention predictor — how well does the electrolyte
        # hold charge under these specific operating conditions?
        self.retention_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        for module in [self.encoder, self.retention_head]:
            module.eval()

    def embed_context(self, raw_observation: torch.Tensor) -> LatentElectrochemicalState:
        """
        Encode electrochemical measurements into a latent condition profile.

        raw_observation : (1, MEASUREMENT_DIM) tensor
            See MEASUREMENT_DIM comment above for field ordering.
        """
        obs = raw_observation.squeeze(0)  # (MEASUREMENT_DIM,)

        with torch.no_grad():
            latent    = self.encoder(raw_observation)           # (1, embed_dim)
            retention = self.retention_head(latent).item()

        return LatentElectrochemicalState(
            voltage_window=float(obs[0].item()),
            temperature_K=float(obs[2].item() * 400 + 250),   # denormalise to K range
            c_rate=float(obs[8].item() * 5),                   # denormalise to C-rate range
            capacity_retention=retention,
            latent_vector=latent.squeeze(0).tolist(),
        )

    def predict_target(
        self,
        context: LatentElectrochemicalState,
        action_vector: Any = None,
    ) -> LatentElectrochemicalState:
        """Electrochemical profiles don't have a natural action space in this PoC."""
        return context

    def entropic_loss(self, predicted_state: Any) -> float:
        if isinstance(predicted_state, LatentElectrochemicalState):
            return 1.0 - predicted_state.capacity_retention
        if isinstance(predicted_state, dict):
            return 1.0 - predicted_state.get("capacity_retention", 0.5)
        return 0.5

    def decode(self, representation: Any) -> Any:
        if isinstance(representation, LatentElectrochemicalState):
            return representation.to_dict()
        return representation

    @property
    def output_signal_type(self) -> SignalType:
        return SignalType.LATENT_EMBEDDING
