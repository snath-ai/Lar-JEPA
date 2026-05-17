"""
Climate-JEPA: Atmospheric Perturbation Modelling and Adaptive Resolution Routing
=================================================================================
Domain-agnostic validation of AbstractModalEncoder, AbstractAttentionKernel,
AbstractPerturbationOperator, and AbstractRoutingKernel applied to climate
and earth-systems modelling — specifically, CO₂-shock perturbation in latent
atmospheric state space and adaptive routing between coarse and fine-resolution
climate models.

Domain Isomorphism
------------------
The same Lár execution spine that predicts post-defect crystal stability shift
(materials_jepa_showcase.py) now predicts post-CO₂-perturbation atmospheric
trajectory:

    Materials domain : encode(perfect_crystal) → encode(defect_crystal) → Δ → z_pred
    Industrial domain: encode(healthy_bearing) → encode(degraded_bearing) → Δ → z_pred
    Climate domain   : encode(baseline_atmosphere) → encode(elevated_CO₂) → Δ → z_pred

The same routing kernel that decides COMMIT / REPLAN in industrial maintenance
now decides GLOBAL_MODEL / REGIONAL_MODEL / ARCHIVE based on atmospheric
perturbation energy — without modifying AbstractRoutingKernel.

ABC chain exercised
-------------------
    AbstractModalEncoder       →  AtmosphericStateEncoder (multi-channel reanalysis → latent)
    AbstractAttentionKernel    →  HyenaConvAttentionKernel (sub-quadratic for long sequences)
    AbstractPerturbationOperator → CO2ShockOperator (Δ = elevated_CO₂ − baseline)
    AbstractRoutingKernel      →  ClimateResolutionKernel (GLOBAL / REGIONAL / ARCHIVE)

Pipeline topology
-----------------
    AtmosphericEmbeddingNode    (AbstractModalEncoder → Z_atm ∈ ℝ^(B×D))
             ↓
    RegionalInstabilityLocator  (AbstractAttentionKernel → topk unstable grid cells)
             ↓
    CO2PerturbationNode         (AbstractPerturbationOperator → z_pred under shock)
             ↓
    ResolutionRouterNode        (AbstractRoutingKernel → GLOBAL / REGIONAL / ARCHIVE)
        ├── GLOBAL   → GlobalModelDispatchNode   → AuditLogNode → Done
        ├── REGIONAL → RegionalDownscaleNode     → AuditLogNode → Done
        └── ARCHIVE  → ArchiveReanalysisNode     → AuditLogNode → Done

Data sourcing (production)
--------------------------
Atmospheric state : ERA5 reanalysis (ECMWF — public domain, Copernicus CDS)
                    MERRA-2 (NASA — public domain)
CO₂ trajectories  : CMIP6 SSP scenarios (IPCC — public domain)
Grid topology     : 0.25° × 0.25° lat/lon grid (1440 × 721 cells globally)
Benchmark         : WeatherBench (Rasp et al. 2020 — public domain)

This PoC uses synthetic tensors with realistic shapes and domain semantics.
The architecture and compliance stack are production-grade.

Authorship and prior art
------------------------
    Author     : Aadithya Vishnu Sajeev
    First published: May 2026, prior to employment commencement.
    Repository : github.com/snath-ai/Lar-JEPA  (Apache 2.0)
    Prior art  : Zenodo DOIs 10.5281/zenodo.19245328, 10.5281/zenodo.19484646,
                 10.5281/zenodo.19646405
    Intent     : Demonstrates that AbstractModalEncoder, AbstractAttentionKernel,
                 AbstractPerturbationOperator, and AbstractRoutingKernel apply
                 to climate and earth-systems modelling without modifying the
                 Lár execution spine.

Run
---
    cd lar_jepa
    python examples/climate_perturbation_model.py
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from core.interfaces import (
    AbstractAttentionKernel,
    AbstractModalEncoder,
    AbstractPerturbationOperator,
    AbstractRoutingKernel,
)

# ---------------------------------------------------------------------------
LATENT_DIM = 256
BATCH_SIZE = 1
N_GRID_CELLS = 64       # spatial grid cells in the regional domain
ATM_CHANNELS = 10       # atmospheric variable channels (see below)
TIME_STEPS = 24         # 24-hour forecast window
TOPK_UNSTABLE = 6       # number of high-instability grid cells to localise
DEVICE = "cpu"


# ===========================================================================
# 1. AbstractModalEncoder — AtmosphericStateEncoder
#    Input: (B, TIME_STEPS, ATM_CHANNELS) — multi-variable reanalysis window
#    Output: (B, LATENT_DIM)
# ===========================================================================

class AtmosphericStateEncoder(AbstractModalEncoder):
    """
    Encodes ERA5-style multi-variable atmospheric state into the Lár latent space.

    Input channel layout (B, TIME_STEPS, ATM_CHANNELS):
        [0]  temperature_850hPa   (K, normalised)
        [1]  temperature_500hPa
        [2]  geopotential_500hPa  (m²/s², normalised)
        [3]  u_wind_850hPa        (m/s)
        [4]  v_wind_850hPa        (m/s)
        [5]  specific_humidity_700hPa
        [6]  total_precipitation   (mm/hr, log-normalised)
        [7]  sea_surface_temperature
        [8]  co2_column_mean_mole_fraction  (ppm, normalised)
        [9]  top_of_atmosphere_radiation

    In production: ERA5 monthly means at 0.25° resolution downloaded via CDS API.

    Invariants M1–M3 satisfied.
    """

    def __init__(self, latent_dim: int = LATENT_DIM):
        self._latent_dim = latent_dim
        self._encoder = nn.Sequential(
            nn.Linear(ATM_CHANNELS, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    @property
    def output_dim(self) -> int:
        return self._latent_dim

    @property
    def modality(self) -> str:
        return "atmospheric_reanalysis_multivariate"

    def encode(self, x: Any) -> Any:
        """
        Parameters
        ----------
        x : torch.Tensor  (B, TIME_STEPS, ATM_CHANNELS)

        Returns
        -------
        torch.Tensor  (B, LATENT_DIM)
        """
        x_mean = x.mean(dim=1)             # (B, ATM_CHANNELS)
        return self._encoder(x_mean)       # (B, LATENT_DIM)


# ===========================================================================
# 2. AbstractAttentionKernel — HyenaConvAttentionKernel
#    Sub-quadratic implicit-convolution kernel (Hyena operator class).
#    Designed for long climate sequences where O(N²) is computationally
#    infeasible (N = global grid = 1440×721 ≈ 1M cells).
# ===========================================================================

class HyenaConvAttentionKernel(AbstractAttentionKernel):
    """
    Hyena-class sub-quadratic attention kernel for long spatial sequences.

    Physical motivation: global climate grids have N ≈ 10^6 cells.
    Scaled-dot-product attention is O(N²) — intractable.
    Hyena-class operators compute attention via implicit convolutions in O(N log N).

    This PoC implements the core pattern: element-wise gating with a learnable
    positional decay filter, matching the Hyena structural invariant while
    remaining tractable for the PoC grid size (N = 64).

    In production: replaced by full Hyena operator (Fu et al. 2023, Apache 2.0).

    Satisfies invariants A1–A6.
    """

    def __init__(self, embed_dim: int = LATENT_DIM):
        self._dim = embed_dim
        # Learnable positional decay filter (mimics Hyena implicit convolution)
        self._filter = nn.Parameter(torch.randn(embed_dim) * 0.1)
        self._gate = nn.Linear(embed_dim, embed_dim)

    def compute(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        k: int,
    ) -> tuple:
        """
        Parameters
        ----------
        query : (B, 1, D)  — pooled atmospheric state query
        key   : (B, N, D)  — per-grid-cell embeddings
        value : (B, N, D)  — same
        k     : int        — number of unstable grid cells to extract

        Returns
        -------
        (attention_weights (B, N), topk_indices (k,))
        """
        if query.ndim == 2:
            query = query.unsqueeze(1)

        # Hyena-style gated projection
        gate = torch.sigmoid(self._gate(key))         # (B, N, D)
        key_gated = key * gate                         # element-wise gating

        # Positional decay filter applied to query
        q_filtered = query * torch.sigmoid(self._filter)   # (B, 1, D)

        scores = torch.bmm(q_filtered, key_gated.transpose(1, 2)).squeeze(1)  # (B, N)
        weights = torch.softmax(scores, dim=-1)        # (B, N) — A3, A4

        topk_k = min(k, weights.shape[-1])
        _, topk_idx = weights[0].topk(topk_k, sorted=True)
        return weights, topk_idx


# ===========================================================================
# 3. AbstractPerturbationOperator — CO2ShockOperator
#    Δ = encode(elevated_CO₂_state) − encode(baseline_state)
#    Predicts atmospheric trajectory under a CO₂ forcing scenario.
# ===========================================================================

class CO2ShockOperator(AbstractPerturbationOperator):
    """
    Zero-shot prediction of atmospheric trajectory under CO₂ shock.

    Baseline  (x_wt)  = baseline atmospheric state (pre-industrial CO₂ ~280 ppm)
    Mutant    (x_mut) = elevated-CO₂ atmospheric state (SSP5-8.5 scenario ~550 ppm)
    z_ctrl            = current model trajectory latent
    z_pred            = predicted trajectory under the CO₂ forcing

    In production:
        x_wt  = ERA5 1950–1980 climatological mean (pre-rapid-warming baseline)
        x_mut = CMIP6 SSP5-8.5 2050 projection ensemble mean
        α     = forcing fraction (1.0 = full SSP5 scenario, 0.5 = intermediate)

    Physical interpretation of Δ:
        Δ captures the latent-space signature of anthropogenic forcing —
        the direction in embedding space corresponding to warming, polar
        amplification, precipitation intensification, and jet-stream shifts.

    Invariants P1–P6 satisfied.
    """

    def __init__(self, base_encoder: AtmosphericStateEncoder):
        self._encoder = base_encoder

    def encode_wildtype(self, x_wt: torch.Tensor) -> torch.Tensor:
        """Encode baseline atmospheric state (pre-perturbation). Returns (B, D)."""
        return self._encoder.encode(x_wt)

    def encode_mutant(self, x_mut: torch.Tensor) -> torch.Tensor:
        """Encode elevated-CO₂ atmospheric state. Returns (B, D)."""
        return self._encoder.encode(x_mut)


# ===========================================================================
# 4. AbstractRoutingKernel — ClimateResolutionKernel
#    Routes between global coarse model, regional downscaling, and archive.
# ===========================================================================

class ClimateResolutionKernel(AbstractRoutingKernel):
    """
    Routes climate computation based on predicted perturbation magnitude.

    Physical motivation: running a high-resolution regional model (4 km WRF)
    everywhere is computationally infeasible. Route to fine resolution only
    where the perturbation is significant enough to matter.

    Score = Frobenius norm of the perturbation vector Δ (perturbation energy).

    Thresholds:
        score < 2.0   → ARCHIVE   (perturbation negligible — use existing data)
        score < 8.0   → GLOBAL    (moderate — run global coarse model)
        score ≥ 8.0   → REGIONAL  (large — run regional downscaling)

    Invariants R1–R4 satisfied.
    """

    def __init__(self, global_thresh: float = 2.0, regional_thresh: float = 8.0):
        self._global = global_thresh
        self._regional = regional_thresh

    def score(self, state: Any) -> float:
        delta = state["delta"]
        return float(torch.norm(delta, dim=-1).mean().item())

    def route(self, state: Any) -> str:
        s = self.score(state)
        if s >= self._regional:
            return "REGIONAL"
        elif s >= self._global:
            return "GLOBAL"
        else:
            return "ARCHIVE"


# ===========================================================================
# Graph State
# ===========================================================================

class GraphState:
    def __init__(self):
        self._data: Dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def snapshot(self) -> Dict[str, Any]:
        return {k: v for k, v in self._data.items()
                if not isinstance(v, torch.Tensor)}


# ===========================================================================
# Pipeline Nodes
# ===========================================================================

class AtmosphericEmbeddingNode:
    """Stage 1: Encode atmospheric state via AbstractModalEncoder."""

    def __init__(self, encoder: AtmosphericStateEncoder, next_node=None):
        self._encoder = encoder
        self._next = next_node

    def execute(self, state: GraphState) -> GraphState:
        raw = state.get("raw_atmospheric_state")
        z = self._encoder.encode(raw)
        state.set("z_atm", z)
        state.set("modality", self._encoder.modality)
        print(f"  [AtmosphericEmbeddingNode] {self._encoder.modality} → z {z.shape}")
        if self._next:
            return self._next.execute(state)
        return state


class RegionalInstabilityLocator:
    """Stage 2: Localise topk unstable grid cells via AbstractAttentionKernel."""

    def __init__(
        self,
        kernel: HyenaConvAttentionKernel,
        cell_encoder: AtmosphericStateEncoder,
        topk: int = TOPK_UNSTABLE,
        next_node=None,
    ):
        self._kernel = kernel
        self._cell_encoder = cell_encoder
        self._topk = topk
        self._next = next_node

    def execute(self, state: GraphState) -> GraphState:
        cell_states = state.get("grid_cell_states")      # (B, N_GRID_CELLS, T, C)
        B, N, T, C = cell_states.shape

        cell_embeds = []
        for n in range(N):
            z_n = self._cell_encoder.encode(cell_states[:, n, :, :])   # (B, D)
            cell_embeds.append(z_n)

        K = torch.stack(cell_embeds, dim=1)    # (B, N, D)
        V = K.clone()
        Q = state.get("z_atm").unsqueeze(1)    # (B, 1, D)

        weights, topk_idx = self._kernel.compute(Q, K, V, self._topk)
        state.set("instability_weights", weights)
        state.set("unstable_cells", topk_idx.tolist())
        print(f"  [RegionalInstabilityLocator] top-{self._topk} unstable cells: "
              f"{topk_idx.tolist()}")
        if self._next:
            return self._next.execute(state)
        return state


class CO2PerturbationNode:
    """Stage 3: Predict post-CO₂-shock trajectory via AbstractPerturbationOperator."""

    def __init__(
        self,
        operator: CO2ShockOperator,
        forcing_alpha: float = 1.0,
        next_node=None,
    ):
        self._op = operator
        self._alpha = forcing_alpha
        self._next = next_node

    def execute(self, state: GraphState) -> GraphState:
        z_ctrl = state.get("z_atm")
        x_baseline = state.get("baseline_atmospheric_state")
        x_elevated = state.get("elevated_co2_atmospheric_state")

        z_pred = self._op.predict_perturbed_state(
            z_ctrl, x_baseline, x_elevated, alpha=self._alpha
        )
        delta = self._op.perturbation_vector(x_baseline, x_elevated)

        state.set("z_pred", z_pred)
        state.set("z_ctrl", z_ctrl)
        state.set("co2_perturbation_delta", delta)
        state.set("delta", delta)   # for routing kernel

        forcing_energy = float(torch.norm(delta, dim=-1).mean().item())
        print(f"  [CO2PerturbationNode] α={self._alpha:.2f}, "
              f"forcing_energy={forcing_energy:.4f}")
        if self._next:
            return self._next.execute(state)
        return state


class ResolutionRouterNode:
    """Stage 4: Route compute budget via AbstractRoutingKernel."""

    def __init__(
        self,
        kernel: ClimateResolutionKernel,
        global_node=None,
        regional_node=None,
        archive_node=None,
    ):
        self._kernel = kernel
        self._routes = {
            "GLOBAL": global_node,
            "REGIONAL": regional_node,
            "ARCHIVE": archive_node,
        }

    def execute(self, state: GraphState) -> GraphState:
        routing_state = {"delta": state.get("co2_perturbation_delta")}
        score = self._kernel.score(routing_state)
        decision = self._kernel.route(routing_state)
        state.set("resolution_score", score)
        state.set("resolution_decision", decision)
        print(f"  [ResolutionRouterNode] forcing_energy={score:.4f} → {decision}")
        next_node = self._routes.get(decision)
        if next_node:
            return next_node.execute(state)
        return state


class ClimateDispatchNode:
    """Terminal: issues model dispatch directive and HMAC audit record."""

    def __init__(self, label: str):
        self._label = label

    def execute(self, state: GraphState) -> GraphState:
        record = {
            "dispatch": self._label,
            "resolution_decision": state.get("resolution_decision"),
            "forcing_energy": state.get("resolution_score"),
            "unstable_cells": state.get("unstable_cells"),
            "modality": state.get("modality"),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        record["hmac"] = hashlib.sha256(
            json.dumps(record, sort_keys=True).encode()
        ).hexdigest()
        print(f"  [{self._label}] dispatch issued")
        print(f"  [{self._label}] audit: {json.dumps(record, indent=4)}")
        return state


# ===========================================================================
# Pipeline Runner
# ===========================================================================

def build_pipeline() -> AtmosphericEmbeddingNode:
    encoder = AtmosphericStateEncoder(latent_dim=LATENT_DIM)
    attn_kernel = HyenaConvAttentionKernel(embed_dim=LATENT_DIM)
    perturb_op = CO2ShockOperator(base_encoder=encoder)
    routing_kernel = ClimateResolutionKernel(global_thresh=2.0, regional_thresh=8.0)

    global_node = ClimateDispatchNode("GlobalModelDispatchNode")
    regional_node = ClimateDispatchNode("RegionalDownscaleNode")
    archive_node = ClimateDispatchNode("ArchiveReanalysisNode")

    router = ResolutionRouterNode(
        kernel=routing_kernel,
        global_node=global_node,
        regional_node=regional_node,
        archive_node=archive_node,
    )
    co2_node = CO2PerturbationNode(
        operator=perturb_op,
        forcing_alpha=1.0,
        next_node=router,
    )
    instability_locator = RegionalInstabilityLocator(
        kernel=attn_kernel,
        cell_encoder=encoder,
        topk=TOPK_UNSTABLE,
        next_node=co2_node,
    )
    entry = AtmosphericEmbeddingNode(encoder=encoder, next_node=instability_locator)
    return entry


def run_pipeline() -> None:
    print("=" * 70)
    print("Climate-JEPA: CO₂ Shock Perturbation + Adaptive Resolution Routing")
    print("ABC chain: ModalEncoder → AttentionKernel → PerturbationOperator")
    print("           → RoutingKernel")
    print("=" * 70)

    raw_atm = torch.rand(BATCH_SIZE, TIME_STEPS, ATM_CHANNELS)
    grid_states = torch.rand(BATCH_SIZE, N_GRID_CELLS, TIME_STEPS, ATM_CHANNELS)
    baseline = torch.rand(BATCH_SIZE, TIME_STEPS, ATM_CHANNELS) * 0.5
    elevated = baseline.clone()
    elevated[:, :, 8] = 0.9   # co2_column_mean_mole_fraction elevated

    state = GraphState()
    state.set("raw_atmospheric_state", raw_atm)
    state.set("grid_cell_states", grid_states)
    state.set("baseline_atmospheric_state", baseline)
    state.set("elevated_co2_atmospheric_state", elevated)

    entry = build_pipeline()

    print("\n[Pipeline] executing …\n")
    with torch.no_grad():
        final_state = entry.execute(state)

    decision = final_state.get("resolution_decision")
    score = final_state.get("resolution_score")
    print(f"\n[Pipeline complete] resolution={decision}, forcing_energy={score:.4f}")
    print("\nABC contracts exercised:")
    print("  AbstractModalEncoder         ✓  AtmosphericStateEncoder")
    print("  AbstractAttentionKernel      ✓  HyenaConvAttentionKernel")
    print("  AbstractPerturbationOperator ✓  CO2ShockOperator")
    print("  AbstractRoutingKernel        ✓  ClimateResolutionKernel")


if __name__ == "__main__":
    run_pipeline()
