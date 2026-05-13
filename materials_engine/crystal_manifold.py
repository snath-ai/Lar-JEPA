"""
CrystalStructureJEPA
====================
A JEPA world model for encoding crystal structure candidates into a
per-site latent manifold, implementing the Lár AbstractManifold interface.

Crystal compositions are STATIC — the crystallographic card of Li₆PS₅Cl
does not change between experiments or between labs. This model runs
once per candidate material to populate a pre-cached crystal library.
At inference time, the graph indexes the library by composition_id —
no JEPA forward pass required.

Per-site embedding (not pooled)
--------------------------------
Rather than collapsing the entire crystal into a single vector, the
encoder produces one embedding per elemental site in the composition
(20 element slots: H, Li, C, N, O, F, Na, Mg, Al, Si, P, S, Cl, K,
Ca, Ti, Mn, Co, Ni, Fe). This gives shape (n_sites=20, embed_dim).

The cross-attention head in the prediction graph then asks:
    "Which elemental sites in this crystal are responsible for
     stability under this specific electrochemical operating condition?"

Production upgrade path
-----------------------
Replace the per-site linear encoder with a Graph Neural Network (GNN)
over the crystal atom-bond graph — e.g. MatterSim, CHGNet, or M3GNet.
The AbstractManifold interface is unchanged. The Lár spine never sees
inside the node.
"""

import sys
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.interfaces import AbstractManifold
from core.types import ModelType, SignalType

# Elements tracked in the composition vector (20 common battery-relevant elements)
ELEMENT_SYMBOLS = [
    "H", "Li", "C", "N", "O", "F", "Na", "Mg", "Al", "Si",
    "P", "S", "Cl", "K", "Ca", "Ti", "Mn", "Co", "Ni", "Fe",
]
N_SITES = len(ELEMENT_SYMBOLS)  # 20


@dataclass
class LatentCrystalState:
    """
    Latent representation of a crystal structure candidate.
    The materials-domain analog of LatentKinematicState.

    site_embeddings : List[List[float]]
        Per-elemental-site latent vectors — shape (N_SITES, embed_dim).
        Preserved un-pooled so the cross-attention head can identify
        which elemental contribution drives the electrochemical outcome.

    thermal_entropy : float  [0, 1]
        Predicted probability of thermal decomposition / runaway.
        This is the entropic_loss analog — the ThermalStabilityRouter
        vetoes any candidate where this exceeds the configured threshold.

    formation_energy : float  [eV/atom]
        DFT-predicted thermodynamic stability. Negative = stable.
        A secondary veto condition: positive formation energy means the
        crystal is thermodynamically unstable at standard conditions.
    """
    composition_id: int
    composition_label: str
    site_embeddings: List[List[float]]   # (N_SITES, embed_dim)
    latent_vector: List[float]           # (embed_dim,) — pooled, for scalar heads
    formation_energy: float              # eV/atom
    thermal_entropy: float               # 0–1
    band_gap: float                      # eV

    def to_dict(self) -> Dict[str, Any]:
        return {
            "composition_id":    self.composition_id,
            "composition_label": self.composition_label,
            "formation_energy":  self.formation_energy,
            "thermal_entropy":   self.thermal_entropy,
            "band_gap":          self.band_gap,
        }


class CrystalStructureJEPA(AbstractManifold):
    """
    Per-site crystal structure encoder implementing AbstractManifold.

    Input  : (1, N_SITES + 6) tensor
               First N_SITES values  — atomic site occupancies (fractional, sum ≤ 1)
               Last 6 values         — normalised lattice parameters a,b,c,α,β,γ

    Output : LatentCrystalState with site_embeddings (N_SITES, embed_dim)

    The N_SITES + 6 decomposition mirrors how crystallographers describe a
    material: what atoms occupy the Wyckoff sites (composition) and what
    shape does the unit cell have (lattice parameters).
    """
    model_type = ModelType.JEPA

    def __init__(self, embed_dim: int = 256):
        self.embed_dim = embed_dim
        self.n_sites = N_SITES

        # Per-site encoder: each element's occupancy + shared lattice context → site embedding
        # Input per site: occupancy scalar (1) + lattice parameters (6) = 7
        self.site_encoder = nn.Sequential(
            nn.Linear(7, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, embed_dim),
        )

        # Global pooling projection: mean of site embeddings → global crystal latent
        self.global_proj = nn.Linear(embed_dim, embed_dim)

        # Formation energy head — predicts thermodynamic stability from global latent
        self.energy_head = nn.Linear(embed_dim, 1)

        # Thermal entropy head — predicts decomposition probability
        self.entropy_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Band gap proxy head
        self.bandgap_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

        for module in [self.site_encoder, self.global_proj,
                       self.energy_head, self.entropy_head, self.bandgap_head]:
            module.eval()

    def embed_context(self, raw_observation: torch.Tensor) -> LatentCrystalState:
        """
        Encode a crystal composition into per-site latent embeddings.

        raw_observation : (1, N_SITES + 6) tensor
        Returns         : LatentCrystalState with site_embeddings (N_SITES, embed_dim)
        """
        obs = raw_observation  # (1, N_SITES + 6)
        occupancies = obs[:, :self.n_sites]      # (1, N_SITES)
        lattice     = obs[:, self.n_sites:]      # (1, 6)

        with torch.no_grad():
            # Encode each site independently with shared lattice context
            site_embs = []
            for i in range(self.n_sites):
                site_input = torch.cat(
                    [occupancies[:, i:i+1], lattice], dim=-1
                )  # (1, 7)
                emb = self.site_encoder(site_input)  # (1, embed_dim)
                site_embs.append(emb)

            # Stack: (1, N_SITES, embed_dim)
            site_tensor = torch.stack(site_embs, dim=1)

            # Pool to global crystal latent
            global_latent = self.global_proj(site_tensor.mean(dim=1))  # (1, embed_dim)

            energy  = self.energy_head(global_latent).item()
            entropy = self.entropy_head(global_latent).item()
            bandgap = self.bandgap_head(global_latent).item()

        return LatentCrystalState(
            composition_id=0,
            composition_label="unknown",
            site_embeddings=site_tensor.squeeze(0).tolist(),   # (N_SITES, embed_dim)
            latent_vector=global_latent.squeeze(0).tolist(),   # (embed_dim,)
            formation_energy=energy * 3.0 - 1.5,              # scale to realistic eV/atom range
            thermal_entropy=entropy,
            band_gap=bandgap,
        )

    def predict_target(
        self,
        context: LatentCrystalState,
        action_vector: Any = None,
    ) -> LatentCrystalState:
        """
        Predict the crystal state after a small compositional perturbation.
        action_vector: change in site occupancies (e.g., increase Li content).
        """
        if action_vector is None:
            return context

        z = torch.tensor(context.latent_vector, dtype=torch.float32)
        a = (
            torch.tensor(action_vector, dtype=torch.float32)
            if not isinstance(action_vector, torch.Tensor)
            else action_vector
        )
        if a.shape != z.shape:
            a = torch.zeros_like(z)

        z_new = z + a * 0.05
        new_entropy = float(torch.sigmoid(z_new.mean()).item())

        return LatentCrystalState(
            composition_id=context.composition_id,
            composition_label=context.composition_label + "_perturbed",
            site_embeddings=context.site_embeddings,
            latent_vector=z_new.tolist(),
            formation_energy=context.formation_energy * 0.95,
            thermal_entropy=new_entropy,
            band_gap=context.band_gap,
        )

    def entropic_loss(self, predicted_state: Any) -> float:
        if isinstance(predicted_state, LatentCrystalState):
            return predicted_state.thermal_entropy
        if isinstance(predicted_state, dict):
            return predicted_state.get("thermal_entropy", 0.5)
        return 0.5

    def decode(self, representation: Any) -> Any:
        if isinstance(representation, LatentCrystalState):
            return representation.to_dict()
        return representation

    @property
    def output_signal_type(self) -> SignalType:
        return SignalType.LATENT_EMBEDDING
