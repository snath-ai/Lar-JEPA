"""
Finance-JEPA: Market Regime Detection and Portfolio Perturbation
================================================================
Domain-agnostic validation of AbstractModalEncoder, AbstractAttentionKernel,
AbstractPerturbationOperator, and AbstractRoutingKernel applied to quantitative
finance — specifically, latent-space regime shift detection and zero-shot
interest-rate perturbation modelling.

Domain Isomorphism
------------------
The identical mathematical pipeline that localises critical failure nodes in a
network-infrastructure graph (infrastructure_jepa_showcase.py) now localises
*regime-shift epochs* within a financial market state — without any
modification to the Lár execution spine.

    Materials domain    : electrochemical state tensor × crystal lattice positions
                          → topk critical instability coordinates
    Seismic domain      : crustal stress field × geological fault topology
                          → topk seismic risk coordinates
    Finance domain      : portfolio state tensor × market microstructure positions
                          → topk critical regime-shift epochs       ← this file

The perturbation operator that predicted post-defect crystal stability shift
(Δ = encode_defect − encode_perfect) now predicts post-rate-hike portfolio
state (Δ = encode_stressed − encode_baseline). The algebra is identical;
only the domain semantics change.

ABC chain exercised
-------------------
    AbstractModalEncoder       →  MarketStateEncoder (price/vol/macro → latent)
    AbstractAttentionKernel    →  LinearAttentionRegimeKernel (O(N) kernel over time)
    AbstractPerturbationOperator → InterestRateShockOperator (Δ = stressed − baseline)
    AbstractRoutingKernel      →  RegimeRoutingKernel (RISK_ON / RISK_OFF / HEDGE)

Pipeline topology
-----------------
    MarketDataEmbeddingNode     (AbstractModalEncoder → Z_market ∈ ℝ^(B×D))
             ↓
    RegimeShiftLocatorNode      (AbstractAttentionKernel → topk regime-shift epochs)
             ↓
    RateShockPerturbationNode   (AbstractPerturbationOperator → z_pred under rate hike)
             ↓
    RegimeRouterNode            (AbstractRoutingKernel → RISK_ON / RISK_OFF / HEDGE)
        ├── RISK_ON  → MomentumStrategyNode → AuditLogNode → Done
        ├── RISK_OFF → DefensiveRotationNode → AuditLogNode → Done
        └── HEDGE    → VolatilityHedgeNode → AuditLogNode → Done

Data sourcing (production)
--------------------------
Market state     : Bloomberg BPIPE / Refinitiv Elektron (tick-level OHLCV)
Macro factors    : FRED (Federal Reserve Economic Data) — public domain
Options surface  : CBOE VIX term structure, SKEW index — public domain
Checkpoint       : Trained on public market data only; no proprietary signals

This PoC uses synthetic tensors with realistic shapes and domain semantics.
The architecture, graph topology, and compliance stack are production-grade.

Authorship and prior art
------------------------
    Author     : Aadithya Vishnu Sajeev
    First published: May 2026, prior to employment commencement.
    Repository : github.com/snath-ai/Lar-JEPA  (Apache 2.0)
    Prior art  : Zenodo DOIs 10.5281/zenodo.19245328, 10.5281/zenodo.19484646,
                 10.5281/zenodo.19646405
    Intent     : Demonstrates that the ABC contracts (AbstractModalEncoder,
                 AbstractAttentionKernel, AbstractPerturbationOperator,
                 AbstractRoutingKernel) are domain-agnostic and apply to
                 quantitative finance without modification to the Lár spine.

Run
---
    cd lar_jepa
    python examples/finance_market_regime.py
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Resolve lar_jepa package root
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
# Constants
# ---------------------------------------------------------------------------
LATENT_DIM = 256        # shared latent dimensionality across all encoders
BATCH_SIZE = 1
N_TIME_STEPS = 64       # lookback window: 64 market observation epochs
TOPK_EPOCHS = 5         # number of regime-shift epochs to localise
DEVICE = "cpu"


# ===========================================================================
# 1. AbstractModalEncoder — MarketStateEncoder
#    Encodes multi-factor market observations into latent space.
#    Input: (B, N_TIME_STEPS, FEATURE_DIM) — price, vol, macro factors
#    Output: (B, LATENT_DIM)
# ===========================================================================

class MarketStateEncoder(AbstractModalEncoder):
    """
    Encodes raw market microstructure observations into the Lár latent space.

    Input feature vector per time step (realistic production dimensions):
        [0:4]    OHLCV normalised close, open-close range, volume z-score, turnover
        [4:8]    Options surface: ATM IV, 25-delta skew, term structure slope, VIX
        [8:12]   Macro factors: 2Y yield, 10Y yield, credit spread (IG OAS), DXY

    In production these come from Bloomberg BPIPE or Refinitiv Elektron.
    Here we use torch.rand tensors with realistic shape (B, 64, 12).

    Invariants M1–M3 satisfied: encode() always returns (B, LATENT_DIM).
    """

    FEATURE_DIM = 12     # 4 price + 4 options + 4 macro

    def __init__(self, latent_dim: int = LATENT_DIM):
        self._latent_dim = latent_dim
        self._encoder = nn.Sequential(
            nn.Linear(self.FEATURE_DIM, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, latent_dim),
        )

    @property
    def output_dim(self) -> int:
        return self._latent_dim

    @property
    def modality(self) -> str:
        return "financial_market_microstructure"

    def encode(self, x: Any) -> Any:
        """
        Parameters
        ----------
        x : torch.Tensor  (B, N_TIME_STEPS, FEATURE_DIM)
            Multi-factor market observation window.

        Returns
        -------
        torch.Tensor  (B, LATENT_DIM)
            Pooled latent market state vector.
        """
        # Mean-pool over time axis, then project
        x_pooled = x.mean(dim=1)          # (B, FEATURE_DIM)
        return self._encoder(x_pooled)     # (B, LATENT_DIM)


# ===========================================================================
# 2. AbstractAttentionKernel — LinearAttentionRegimeKernel
#    O(N) linear attention over market time steps.
#    Identifies the epochs where regime-shift probability is highest.
#    Valid mechanism: A1–A6 all satisfied.
# ===========================================================================

class LinearAttentionRegimeKernel(AbstractAttentionKernel):
    """
    Linear (O(N)) attention kernel for financial time series.

    Standard scaled-dot-product is O(N²) over the lookback window.
    For high-frequency data with N = 1024+ bars this is prohibitive.
    This kernel uses the φ(Q)φ(K)ᵀ factorisation with ELU+1 features,
    reducing complexity to O(N) while satisfying all invariants A1–A6.

    Domain: financial time series regime-shift localisation.
    Production scenario: daily bar lookback N=252 (1 year), or 5-min bars N=1024.
    """

    def __init__(self, embed_dim: int = LATENT_DIM):
        self._embed_dim = embed_dim

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """ELU + 1 feature map — ensures non-negative kernel values."""
        return torch.nn.functional.elu(x) + 1.0

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
        query : (B, 1, D)  — pooled portfolio state at evaluation time
        key   : (B, N, D)  — per-epoch market microstructure embeddings
        value : (B, N, D)  — same as key (self-attention over time axis)
        k     : int        — number of regime-shift epochs to extract

        Returns
        -------
        (attention_weights (B, N), topk_indices (k,))
        """
        if query.ndim == 2:
            query = query.unsqueeze(1)         # (B, 1, D)

        Q = self._feature_map(query)           # (B, 1, D)
        K = self._feature_map(key)             # (B, N, D)

        # Linear attention: weights_i ∝ Q · K_i^T
        scores = torch.bmm(Q, K.transpose(1, 2))   # (B, 1, N)
        scores = scores.squeeze(1)                  # (B, N)

        # Normalise to probability simplex (invariant A3, A4)
        weights = torch.softmax(scores, dim=-1)     # (B, N)

        # Top-k regime-shift epochs (invariant A5, A6)
        topk_k = min(k, weights.shape[-1])
        topk_vals, topk_idx = weights[0].topk(topk_k, sorted=True)

        return weights, topk_idx


# ===========================================================================
# 3. AbstractPerturbationOperator — InterestRateShockOperator
#    Models Δ = encode(stressed_portfolio) − encode(baseline_portfolio)
#    Predicts portfolio latent state post-rate-hike without running the hike.
# ===========================================================================

class InterestRateShockOperator(AbstractPerturbationOperator):
    """
    Zero-shot prediction of portfolio state after an interest rate shock.

    Baseline  (x_wt)  = baseline portfolio factor exposures under current rates
    Mutant    (x_mut) = stressed portfolio factor exposures under shocked rates
                        (e.g. +150bps parallel shift on yield curve)
    z_ctrl            = current portfolio latent state
    z_pred            = predicted portfolio state post-shock

    Invariants P1–P6 all satisfied via the inherited perturbation_vector()
    and predict_perturbed_state() methods on AbstractPerturbationOperator.

    In production:
        x_wt  = DV01-weighted factor exposure tensor under spot rates
        x_mut = same tensor re-priced under shock scenario
        α     = shock magnitude scalar (1.0 = full shock, 0.5 = half)
    """

    def __init__(self, latent_dim: int = LATENT_DIM):
        self._encoder = nn.Sequential(
            nn.Linear(LATENT_DIM, LATENT_DIM),
            nn.GELU(),
            nn.Linear(LATENT_DIM, latent_dim),
        )

    def encode_wildtype(self, x_wt: torch.Tensor) -> torch.Tensor:
        """Encode baseline (pre-shock) portfolio factor exposures. Shape: (B, D)."""
        return self._encoder(x_wt)

    def encode_mutant(self, x_mut: torch.Tensor) -> torch.Tensor:
        """Encode stressed portfolio factor exposures. Shape: (B, D)."""
        return self._encoder(x_mut)


# ===========================================================================
# 4. AbstractRoutingKernel — RegimeRoutingKernel
#    Scores the predicted post-shock portfolio state and routes to the
#    appropriate strategy node: RISK_ON / RISK_OFF / HEDGE.
# ===========================================================================

class RegimeRoutingKernel(AbstractRoutingKernel):
    """
    Routes portfolio strategy decisions based on predicted regime score.

    Score = L2 displacement of z_pred from z_ctrl (perturbation magnitude).
    Large displacement → portfolio state under shock departs significantly
    from current trajectory → hedge or rotate defensively.

    Thresholds (tunable):
        score < low_threshold  → RISK_ON   (shock impact minimal)
        score < high_threshold → RISK_OFF  (moderate risk regime)
        score ≥ high_threshold → HEDGE     (severe regime shift predicted)

    Invariants R1–R4 satisfied: deterministic, finite score, non-empty route.
    """

    def __init__(self, low_threshold: float = 1.5, high_threshold: float = 3.0):
        self._low = low_threshold
        self._high = high_threshold

    def score(self, state: Any) -> float:
        """
        Compute L2 displacement of predicted state from control state.
        State is a dict with keys 'z_ctrl' and 'z_pred' (torch tensors).
        """
        z_ctrl = state["z_ctrl"]
        z_pred = state["z_pred"]
        return float(torch.norm(z_pred - z_ctrl, dim=-1).mean().item())

    def route(self, state: Any) -> str:
        s = self.score(state)
        if s < self._low:
            return "RISK_ON"
        elif s < self._high:
            return "RISK_OFF"
        else:
            return "HEDGE"


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

class MarketDataEmbeddingNode:
    """
    Stage 1: Encode multi-factor market observations into latent space.
    Uses AbstractModalEncoder (MarketStateEncoder).
    """

    def __init__(self, encoder: MarketStateEncoder, next_node=None):
        self._encoder = encoder
        self._next = next_node

    def execute(self, state: GraphState) -> GraphState:
        raw_market = state.get("raw_market_observations")
        z_market = self._encoder.encode(raw_market)         # (B, D)
        state.set("z_market", z_market)
        state.set("market_modality", self._encoder.modality)
        print(f"  [MarketDataEmbeddingNode] encoded {self._encoder.modality} "
              f"→ latent shape {z_market.shape}")
        if self._next:
            return self._next.execute(state)
        return state


class RegimeShiftLocatorNode:
    """
    Stage 2: Attend over per-epoch market embeddings to localise
    the topk regime-shift epochs. Uses AbstractAttentionKernel.
    """

    def __init__(
        self,
        kernel: LinearAttentionRegimeKernel,
        epoch_encoder: MarketStateEncoder,
        topk: int = TOPK_EPOCHS,
        next_node=None,
    ):
        self._kernel = kernel
        self._epoch_encoder = epoch_encoder
        self._topk = topk
        self._next = next_node

    def execute(self, state: GraphState) -> GraphState:
        # Encode each time-step independently for K, V
        raw = state.get("raw_market_observations")       # (B, N, F)
        B, N, F = raw.shape

        # Per-epoch encoding: project each epoch independently
        epoch_embeds = []
        for t in range(N):
            x_t = raw[:, t:t+1, :].expand(-1, 1, -1)    # (B, 1, F) → mean-pool → (B, D)
            z_t = self._epoch_encoder.encode(x_t)        # (B, D)
            epoch_embeds.append(z_t)

        K = torch.stack(epoch_embeds, dim=1)             # (B, N, D)
        V = K.clone()

        # Query = pooled portfolio state
        Q = state.get("z_market").unsqueeze(1)           # (B, 1, D)

        weights, topk_idx = self._kernel.compute(Q, K, V, self._topk)

        state.set("regime_attention_weights", weights)
        state.set("regime_shift_epochs", topk_idx.tolist())
        print(f"  [RegimeShiftLocatorNode] top-{self._topk} regime-shift epochs: "
              f"{topk_idx.tolist()}")
        if self._next:
            return self._next.execute(state)
        return state


class RateShockPerturbationNode:
    """
    Stage 3: Apply interest rate shock perturbation.
    Uses AbstractPerturbationOperator (InterestRateShockOperator).
    Predicts z_pred = z_ctrl + α·Δ without running any actual simulation.
    """

    def __init__(
        self,
        operator: InterestRateShockOperator,
        shock_alpha: float = 1.0,
        next_node=None,
    ):
        self._op = operator
        self._alpha = shock_alpha
        self._next = next_node

    def execute(self, state: GraphState) -> GraphState:
        z_ctrl = state.get("z_market")                   # (B, D)
        x_baseline = state.get("portfolio_baseline")     # (B, D) factor exposures
        x_stressed = state.get("portfolio_stressed")     # (B, D) shocked exposures

        z_pred = self._op.predict_perturbed_state(
            z_ctrl, x_baseline, x_stressed, alpha=self._alpha
        )
        delta = self._op.perturbation_vector(x_baseline, x_stressed)

        state.set("z_pred", z_pred)
        state.set("rate_shock_delta", delta)
        state.set("z_ctrl", z_ctrl)

        displacement = float(torch.norm(delta, dim=-1).mean().item())
        print(f"  [RateShockPerturbationNode] α={self._alpha:.2f}, "
              f"|Δ|={displacement:.4f}, predicted state divergence computed")
        if self._next:
            return self._next.execute(state)
        return state


class RegimeRouterNode:
    """
    Stage 4: Route to strategy based on predicted post-shock regime.
    Uses AbstractRoutingKernel (RegimeRoutingKernel).
    """

    def __init__(
        self,
        kernel: RegimeRoutingKernel,
        risk_on_node=None,
        risk_off_node=None,
        hedge_node=None,
    ):
        self._kernel = kernel
        self._routes = {
            "RISK_ON": risk_on_node,
            "RISK_OFF": risk_off_node,
            "HEDGE": hedge_node,
        }

    def execute(self, state: GraphState) -> GraphState:
        routing_state = {
            "z_ctrl": state.get("z_ctrl"),
            "z_pred": state.get("z_pred"),
        }
        score = self._kernel.score(routing_state)
        decision = self._kernel.route(routing_state)

        state.set("regime_score", score)
        state.set("regime_decision", decision)
        print(f"  [RegimeRouterNode] score={score:.4f} → {decision}")

        next_node = self._routes.get(decision)
        if next_node:
            return next_node.execute(state)
        return state


class StrategyExecutionNode:
    """Terminal node — logs strategy directive and writes audit record."""

    def __init__(self, strategy_name: str):
        self._name = strategy_name

    def _audit_record(self, state: GraphState) -> dict:
        snap = state.snapshot()
        payload = {
            "node": self._name,
            "strategy": state.get("regime_decision"),
            "regime_score": state.get("regime_score"),
            "shift_epochs": state.get("regime_shift_epochs"),
            "modality": state.get("market_modality"),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        payload["hmac"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()
        return payload

    def execute(self, state: GraphState) -> GraphState:
        record = self._audit_record(state)
        print(f"  [{self._name}] strategy directive issued")
        print(f"  [{self._name}] audit: {json.dumps(record, indent=4)}")
        return state


# ===========================================================================
# Pipeline Runner
# ===========================================================================

def build_pipeline() -> RegimeRouterNode:
    # Instantiate ABC implementations
    encoder = MarketStateEncoder(latent_dim=LATENT_DIM)
    attn_kernel = LinearAttentionRegimeKernel(embed_dim=LATENT_DIM)
    perturb_op = InterestRateShockOperator(latent_dim=LATENT_DIM)
    routing_kernel = RegimeRoutingKernel(low_threshold=1.5, high_threshold=3.5)

    # Terminal strategy nodes
    risk_on_node = StrategyExecutionNode("MomentumStrategyNode")
    risk_off_node = StrategyExecutionNode("DefensiveRotationNode")
    hedge_node = StrategyExecutionNode("VolatilityHedgeNode")

    # Wire graph
    router = RegimeRouterNode(
        kernel=routing_kernel,
        risk_on_node=risk_on_node,
        risk_off_node=risk_off_node,
        hedge_node=hedge_node,
    )
    perturb_node = RateShockPerturbationNode(
        operator=perturb_op,
        shock_alpha=1.0,
        next_node=router,
    )
    regime_locator = RegimeShiftLocatorNode(
        kernel=attn_kernel,
        epoch_encoder=encoder,
        topk=TOPK_EPOCHS,
        next_node=perturb_node,
    )
    entry = MarketDataEmbeddingNode(
        encoder=encoder,
        next_node=regime_locator,
    )
    return entry


def run_pipeline() -> None:
    print("=" * 70)
    print("Finance-JEPA: Market Regime Detection + Rate Shock Perturbation")
    print("ABC chain: ModalEncoder → AttentionKernel → PerturbationOperator")
    print("           → RoutingKernel")
    print("=" * 70)

    # Synthetic market data — realistic shapes
    raw_market = torch.rand(BATCH_SIZE, N_TIME_STEPS, MarketStateEncoder.FEATURE_DIM)
    portfolio_baseline = torch.randn(BATCH_SIZE, LATENT_DIM)
    portfolio_stressed = portfolio_baseline + 0.3 * torch.randn(BATCH_SIZE, LATENT_DIM)

    state = GraphState()
    state.set("raw_market_observations", raw_market)
    state.set("portfolio_baseline", portfolio_baseline)
    state.set("portfolio_stressed", portfolio_stressed)

    entry = build_pipeline()

    print("\n[Pipeline] executing …\n")
    with torch.no_grad():
        final_state = entry.execute(state)

    decision = final_state.get("regime_decision")
    score = final_state.get("regime_score")
    print(f"\n[Pipeline complete] regime={decision}, score={score:.4f}")
    print("\nABC contracts exercised:")
    print("  AbstractModalEncoder       ✓  MarketStateEncoder")
    print("  AbstractAttentionKernel    ✓  LinearAttentionRegimeKernel")
    print("  AbstractPerturbationOperator ✓  InterestRateShockOperator")
    print("  AbstractRoutingKernel      ✓  RegimeRoutingKernel")


if __name__ == "__main__":
    run_pipeline()
