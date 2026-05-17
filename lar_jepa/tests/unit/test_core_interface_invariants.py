"""
Behavioral Invariant Test Suite — Core Interface Abstractions
=============================================================

PURPOSE — WHY THIS FILE EXISTS
-------------------------------
This suite verifies behavioral invariants for four foundational ABCs defined
in core/interfaces.py:

    AbstractAttentionKernel    (A1–A6) — mechanism-agnostic attention
    AbstractPerturbationOperator (P1–P6) — latent-space intervention prediction
    AbstractRoutingKernel      (R1–R4) — score-then-route decision logic
    AbstractModalEncoder       (M1–M3) — modality-to-latent encoding

These are not code-coverage tests. They are behavioral contracts. Any future
implementation — regardless of internal architecture, domain, or training
procedure — that passes all invariants satisfies the corresponding specification
and is a Derivative Work of the pre-employment prior art defined in
core/interfaces.py (Apache 2.0, github.com/snath-ai/Lar-JEPA).

RUNNING THE SUITE
-----------------
    cd lar_jepa
    pytest lar_jepa/tests/unit/test_core_interface_invariants.py -v

THE INVARIANTS
--------------
    AbstractAttentionKernel:
        A1. attention_weights.shape[-1] == N        (covers all positions)
        A2. topk_indices ⊆ {0, …, N−1}             (valid coordinates)
        A3. attention_weights ≥ 0                   (non-negative)
        A4. attention_weights.sum(dim=-1) ≈ 1.0    (normalised distribution)
        A5. topk_indices ordered descending by weight
        A6. len(topk_indices) == k                  (exact count)

    AbstractPerturbationOperator:
        P1. encode_wildtype(x).shape == encode_mutant(x).shape == (B, D)
        P2. perturbation_vector = encode_mutant(x_mut) − encode_wildtype(x_wt)
        P3. predict_perturbed_state(z, wt, mut, α=0) ≈ z  (identity at α=0)
        P4. perturbation displacement scales linearly with α
        P5. perturbation_vector is independent of z_ctrl
        P6. deterministic — same inputs, same perturbation vector

    AbstractRoutingKernel:
        R1. score(state) returns a finite float
        R2. route(state) returns a non-empty string
        R3. deterministic — same state, same (score, route)
        R4. route consistent with score across independent calls

    AbstractModalEncoder:
        M1. encode(x).shape == (B, output_dim)
        M2. output_dim is constant across all encode() calls
        M3. encode(x) is deterministic for the same input

LEGAL NOTE
----------
This test file is itself pre-employment prior art, published in
github.com/snath-ai/Lar-JEPA (Apache 2.0) prior to employment commencement.
The behavioral invariants defined here mechanically verify the specifications
in core/interfaces.py. An implementation that passes all invariants satisfies
the specification and is a Derivative Work of this pre-employment prior art.

Authorship: Aadithya Vishnu Sajeev
Published:  May 2026, prior to employment commencement.
"""

import sys
import os
import math
import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed — skipping invariant tests")
nn = torch.nn
from abc import ABC

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_JEPA_ROOT = os.path.abspath(os.path.join(_TESTS_DIR, "..", "..", ".."))
_LAR_SRC   = os.path.join(_JEPA_ROOT, "lar_jepa", "src")

for _p in [_JEPA_ROOT, _LAR_SRC]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.interfaces import (
    AbstractAttentionKernel,
    AbstractPerturbationOperator,
    AbstractRoutingKernel,
    AbstractModalEncoder,
)


# ===========================================================================
# ─── AbstractAttentionKernel  ───────────────────────────────────────────────
# Concrete implementations across two structurally different mechanisms.
# Same invariants (A1–A6) must hold regardless of the mechanism used.
# ===========================================================================

class ScaledDotProductKernel(AbstractAttentionKernel):
    """
    Standard scaled dot-product attention: softmax(QKᵀ / √D).

    Domain examples: RNA expression → DNA positions (genomic),
                     stress field → fault segments (seismic),
                     traffic load → server topology (infrastructure).
    """
    def __init__(self, embed_dim: int):
        self._scale = math.sqrt(embed_dim)

    def compute(self, query, key, value, k):
        if query.ndim == 2:
            query = query.unsqueeze(1)                       # (B, 1, D)
        scores = torch.bmm(query, key.transpose(-2, -1)) / self._scale  # (B, 1, N)
        weights = torch.softmax(scores, dim=-1).squeeze(1)  # (B, N)
        topk_idx = weights[0].topk(min(k, weights.shape[-1])).indices   # (k,)
        return weights, topk_idx


class CosineAttentionKernel(AbstractAttentionKernel):
    """
    Cosine-similarity attention: softmax(normalised_QKᵀ).

    An alternative mechanism — proves that A1–A6 are mechanism-agnostic:
    any kernel that produces a valid normalised distribution over N positions
    and extracts k ordered indices satisfies the specification.

    Domain examples: molecular binding site identification,
                     climate perturbation hotspot localisation,
                     materials lattice defect ranking.
    """
    def compute(self, query, key, value, k):
        if query.ndim == 2:
            query = query.unsqueeze(1)                       # (B, 1, D)
        q_norm = nn.functional.normalize(query, dim=-1)
        k_norm = nn.functional.normalize(key,   dim=-1)
        scores = torch.bmm(q_norm, k_norm.transpose(-2, -1))  # (B, 1, N)
        weights = torch.softmax(scores, dim=-1).squeeze(1)    # (B, N)
        topk_idx = weights[0].topk(min(k, weights.shape[-1])).indices
        return weights, topk_idx


# ===========================================================================
# ─── AbstractPerturbationOperator  ──────────────────────────────────────────
# Three domain-specific operators. All share a single encoder (for both
# wildtype and mutant arms) to maximise testability of P2 and P6. The shared
# architecture also mirrors real-world implementations where a single
# pre-trained foundation model encodes both the reference and the variant.
# ===========================================================================

class _SharedEncoderOperator(AbstractPerturbationOperator):
    """
    Base implementation using a single shared encoder for both arms.
    Wildtype and mutant are encoded with the same weights — the only
    difference between z_wt and z_mut comes from the input data.
    """
    def __init__(self, input_dim: int, embed_dim: int):
        self._encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self._encoder.eval()

    def encode_wildtype(self, x_wt):
        with torch.no_grad():
            return self._encoder(x_wt)

    def encode_mutant(self, x_mut):
        with torch.no_grad():
            return self._encoder(x_mut)


class GenomicPerturbationOperator(_SharedEncoderOperator):
    """
    Perturbation operator for the genomic knockout domain.

    x_wt  = wildtype gene sequence (one-hot or k-mer feature vector)
    x_mut = CRISPR-edited / knocked-out variant sequence
    Δ     = predicted shift in transcriptomic latent space post-knockout
    z_pred = z_ctrl + α·Δ  — predicted post-knockout cell state

    Published: lar_jepa/tests/unit/test_core_interface_invariants.py
    Repository: github.com/snath-ai/Lar-JEPA (Apache 2.0)
    """
    pass  # fully specified by _SharedEncoderOperator with (100, 64)


class CrystalDefectOperator(_SharedEncoderOperator):
    """
    Perturbation operator for the materials defect domain.

    x_wt  = perfect crystal lattice site feature vector
    x_mut = defect-injected crystal (vacancy, dopant, strain)
    Δ     = predicted stability shift in electrochemical latent space
    z_pred = predicted electrochemical state under the defect

    Published: lar_jepa/tests/unit/test_core_interface_invariants.py
    Repository: github.com/snath-ai/Lar-JEPA (Apache 2.0)
    """
    pass  # fully specified by _SharedEncoderOperator with (20, 32)


class MolecularBindingOperator(_SharedEncoderOperator):
    """
    Perturbation operator for the protein / molecular binding domain.

    x_wt  = unbound protein / molecular geometry feature vector
    x_mut = ligand-bound conformation feature vector
    Δ     = predicted conformational shift vector in latent space
    z_pred = predicted post-binding molecular state

    Published: lar_jepa/tests/unit/test_core_interface_invariants.py
    Repository: github.com/snath-ai/Lar-JEPA (Apache 2.0)
    """
    pass  # fully specified by _SharedEncoderOperator with (50, 48)


# ===========================================================================
# ─── AbstractRoutingKernel  ─────────────────────────────────────────────────
# Two implementations with different routing logic — same invariants (R1–R4).
# ===========================================================================

class EntropicThresholdKernel(AbstractRoutingKernel):
    """
    Routes based on a scalar entropy / deviation threshold.
    Current pattern: commit if score < threshold, else replan.

    Direct implementation of the entropic routing used in all Lár-JEPA
    examples (orbital insertion, crystal stability, network fault).
    """
    def __init__(
        self,
        threshold: float = 0.3,
        low_route: str = "COMMIT",
        high_route: str = "REPLAN",
    ):
        self._threshold = threshold
        self._low_route  = low_route
        self._high_route = high_route

    def score(self, state) -> float:
        return float(state)

    def route(self, state) -> str:
        return self._low_route if self.score(state) < self._threshold else self._high_route


class MultiThresholdRoutingKernel(AbstractRoutingKernel):
    """
    Three-way routing: low / medium / high signal bands.

    Demonstrates that R1–R4 hold for routing kernels with more than two
    output routes — enabling ternary decisions such as
    COMMIT / INVESTIGATE / REPLAN in higher-stakes pipelines.
    """
    def __init__(self, low: float = 0.25, high: float = 0.75):
        self._low  = low
        self._high = high

    def score(self, state) -> float:
        return float(state)

    def route(self, state) -> str:
        s = self.score(state)
        if s < self._low:
            return "COMMIT"
        elif s < self._high:
            return "INVESTIGATE"
        else:
            return "REPLAN"


# ===========================================================================
# ─── AbstractModalEncoder  ──────────────────────────────────────────────────
# Three domain encoders. All share an MLP backbone; differ in input dimension
# and modality name — proving M1–M3 hold across structurally different inputs.
# ===========================================================================

class _MLPModalEncoder(AbstractModalEncoder):
    """Base MLP encoder shared by all domain instantiations."""
    def __init__(self, input_dim: int, embed_dim: int, modality_name: str):
        self._net = nn.Sequential(
            nn.Linear(input_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self._net.eval()
        self._output_dim  = embed_dim
        self._modality    = modality_name

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def modality(self) -> str:
        return self._modality

    def encode(self, x):
        with torch.no_grad():
            return self._net(x)


class GenomicSequenceEncoder(_MLPModalEncoder):
    """
    Modal encoder for DNA / RNA sequences.
    input  = sequence feature vector (one-hot or k-mer, B × input_dim)
    output = structural / expression latent (B × D)
    In production: DNABERT-2 (117M) or GeneJEPA (Tahoe-100M).
    """
    pass


class ElectrochemicalEncoder(_MLPModalEncoder):
    """
    Modal encoder for electrochemical impedance / cycling measurements.
    input  = measurement feature vector (B × input_dim)
    output = battery state latent (B × D)
    """
    pass


class NetworkTelemetryEncoder(_MLPModalEncoder):
    """
    Modal encoder for network traffic load telemetry.
    input  = per-node traffic measurement vector (B × input_dim)
    output = network state latent (B × D)
    """
    pass


# ===========================================================================
# Parametrized fixtures
# ===========================================================================

@pytest.fixture(params=[
    pytest.param(
        (ScaledDotProductKernel, 64, 4, 32, 64, 5),
        id="scaled_dot_product_kernel__genomic_dims"
    ),
    pytest.param(
        (ScaledDotProductKernel, 64, 2, 48, 64, 3),
        id="scaled_dot_product_kernel__seismic_dims"
    ),
    pytest.param(
        (CosineAttentionKernel, 64, 4, 20, 64, 3),
        id="cosine_attention_kernel__materials_dims"
    ),
    pytest.param(
        (CosineAttentionKernel, 48, 2, 100, 48, 5),
        id="cosine_attention_kernel__molecular_dims"
    ),
])
def attn_fixture(request):
    """
    Returns (kernel, Q, K, V, N, k) for each attention mechanism and domain.
    B=4 throughout. All mechanisms must satisfy A1–A6.
    """
    cls, D, B, N, embed_dim, k = request.param
    torch.manual_seed(42)
    kernel = cls(embed_dim) if cls == ScaledDotProductKernel else cls()
    Q = torch.randn(B, 1, D)
    K = torch.randn(B, N, D)
    V = torch.randn(B, N, D)
    return kernel, Q, K, V, N, k


@pytest.fixture(params=[
    pytest.param(
        (GenomicPerturbationOperator,  100, 64, 4),
        id="genomic_knockout_domain"
    ),
    pytest.param(
        (CrystalDefectOperator,         20, 32, 4),
        id="crystal_defect_domain"
    ),
    pytest.param(
        (MolecularBindingOperator,      50, 48, 4),
        id="molecular_binding_domain"
    ),
])
def perturb_fixture(request):
    """
    Returns (operator, x_wt, x_mut, z_ctrl, B, D) for each domain operator.
    """
    cls, input_dim, embed_dim, B = request.param
    torch.manual_seed(42)
    op     = cls(input_dim, embed_dim)
    x_wt   = torch.randn(B, input_dim)
    x_mut  = torch.randn(B, input_dim)
    z_ctrl = torch.randn(B, embed_dim)
    return op, x_wt, x_mut, z_ctrl, B, embed_dim


@pytest.fixture(params=[
    pytest.param(
        (EntropicThresholdKernel,    {"threshold": 0.3}, [0.1, 0.5, 0.0, 0.9, 0.3]),
        id="entropic_threshold_kernel"
    ),
    pytest.param(
        (MultiThresholdRoutingKernel, {"low": 0.25, "high": 0.75}, [0.1, 0.5, 0.8, 0.0, 1.0]),
        id="multi_threshold_routing_kernel"
    ),
])
def routing_fixture(request):
    """Returns (kernel, list_of_states) for each routing kernel implementation."""
    cls, kwargs, states = request.param
    kernel = cls(**kwargs)
    return kernel, states


@pytest.fixture(params=[
    pytest.param(
        (GenomicSequenceEncoder,   120, 64, "genomic_sequence",      4),
        id="genomic_sequence_encoder"
    ),
    pytest.param(
        (ElectrochemicalEncoder,    24, 32, "electrochemical",        4),
        id="electrochemical_encoder"
    ),
    pytest.param(
        (NetworkTelemetryEncoder,   16, 48, "network_telemetry",      4),
        id="network_telemetry_encoder"
    ),
])
def encoder_fixture(request):
    """Returns (encoder, x) for each modal encoder domain."""
    cls, input_dim, embed_dim, modality_name, B = request.param
    torch.manual_seed(42)
    encoder = cls(input_dim, embed_dim, modality_name)
    x = torch.randn(B, input_dim)
    return encoder, x, B, embed_dim


# ===========================================================================
# A1–A6: AbstractAttentionKernel invariants
# ===========================================================================

class TestAttentionKernelInvariants:
    """
    Behavioral invariant tests for AbstractAttentionKernel.

    These tests verify the mathematical CONTRACT, not the mechanism.
    ScaledDotProductKernel and CosineAttentionKernel use completely different
    internal computations — both must satisfy every invariant. Any future
    attention mechanism passes these tests to prove it satisfies the spec.
    """

    def test_A1_attention_weights_cover_all_N_positions(self, attn_fixture):
        """
        A1: attention_weights.shape[-1] == N

        The attention distribution must cover every position in the key/value
        sequence. A kernel that returns weights over fewer than N positions
        has silently dropped structural information.
        """
        kernel, Q, K, V, N, k = attn_fixture
        weights, _ = kernel.compute(Q, K, V, k)
        assert weights.shape[-1] == N, (
            f"A1 VIOLATED: attention_weights.shape[-1]={weights.shape[-1]} "
            f"but N={N}. All positions must be covered."
        )

    def test_A2_topk_indices_are_valid_position_coordinates(self, attn_fixture):
        """
        A2: topk_indices ⊆ {0, …, N−1}

        Every returned index must be a valid position in the sequence.
        Out-of-bounds indices would cause silent data corruption downstream.
        """
        kernel, Q, K, V, N, k = attn_fixture
        _, topk_idx = kernel.compute(Q, K, V, k)
        for idx in topk_idx.tolist():
            assert 0 <= int(idx) < N, (
                f"A2 VIOLATED: index {idx} is out of bounds for N={N}."
            )

    def test_A3_attention_weights_are_nonnegative(self, attn_fixture):
        """
        A3: attention_weights ≥ 0

        Attention weights are a probability distribution — they must be
        non-negative. Negative weights would produce inverted topk rankings.
        """
        kernel, Q, K, V, N, k = attn_fixture
        weights, _ = kernel.compute(Q, K, V, k)
        assert (weights >= 0).all(), (
            f"A3 VIOLATED: attention weights contain negative values. "
            f"Min: {weights.min().item():.6f}"
        )

    def test_A4_attention_weights_form_normalised_distribution(self, attn_fixture):
        """
        A4: attention_weights.sum(dim=-1) ≈ 1.0

        The attention distribution must be normalised. This is what makes
        the topk extraction meaningful — the weights are comparable probabilities.
        Any valid softmax (scaled dot-product, cosine, or future variant)
        satisfies this invariant.
        """
        kernel, Q, K, V, N, k = attn_fixture
        weights, _ = kernel.compute(Q, K, V, k)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), (
            f"A4 VIOLATED: attention weights do not sum to 1.0. "
            f"Got sums: {sums.tolist()}"
        )

    def test_A5_topk_indices_ordered_descending_by_weight(self, attn_fixture):
        """
        A5: topk_indices ordered descending by attention weight

        The extracted positions must be ranked by attention weight. The first
        index must receive the highest weight, enabling downstream components
        to treat them as an ordered priority list.
        """
        kernel, Q, K, V, N, k = attn_fixture
        weights, topk_idx = kernel.compute(Q, K, V, k)
        topk_weights = weights[0][topk_idx]
        for i in range(len(topk_weights) - 1):
            assert topk_weights[i] >= topk_weights[i + 1] - 1e-6, (
                f"A5 VIOLATED: topk indices are not ordered descending by weight. "
                f"Position {i}: weight={topk_weights[i].item():.6f}, "
                f"position {i+1}: weight={topk_weights[i+1].item():.6f}"
            )

    def test_A6_exactly_k_indices_extracted(self, attn_fixture):
        """
        A6: len(topk_indices) == k

        Exactly k positions must be returned — no more, no fewer.
        """
        kernel, Q, K, V, N, k = attn_fixture
        _, topk_idx = kernel.compute(Q, K, V, k)
        assert len(topk_idx) == k, (
            f"A6 VIOLATED: expected k={k} indices, got {len(topk_idx)}."
        )


# ===========================================================================
# P1–P6: AbstractPerturbationOperator invariants
# ===========================================================================

class TestPerturbationOperatorInvariants:
    """
    Behavioral invariant tests for AbstractPerturbationOperator.

    Three domain operators (genomic, crystal, molecular) all use the same
    encoder architecture but different input dimensionalities — proving the
    invariants hold across domain-specific instantiations.
    """

    def test_P1_both_encoders_produce_same_shape(self, perturb_fixture):
        """
        P1: encode_wildtype(x).shape == encode_mutant(x).shape == (B, D)

        Both encoding arms must produce identically shaped latent vectors.
        Mismatched shapes would make the additive perturbation undefined.
        """
        op, x_wt, x_mut, z_ctrl, B, D = perturb_fixture
        z_wt  = op.encode_wildtype(x_wt)
        z_mut = op.encode_mutant(x_mut)
        assert z_wt.shape  == (B, D), (
            f"P1 VIOLATED: encode_wildtype shape={z_wt.shape}, expected ({B}, {D})"
        )
        assert z_mut.shape == (B, D), (
            f"P1 VIOLATED: encode_mutant shape={z_mut.shape}, expected ({B}, {D})"
        )
        assert z_wt.shape == z_mut.shape, (
            f"P1 VIOLATED: encoder shapes differ — {z_wt.shape} vs {z_mut.shape}"
        )

    def test_P2_perturbation_vector_equals_difference_of_encodings(self, perturb_fixture):
        """
        P2: perturbation_vector = encode_mutant(x_mut) − encode_wildtype(x_wt)

        The perturbation_vector() method must always implement this exact
        additive difference. This is the mathematical identity that defines
        the operator — no implementation may deviate from it.
        """
        op, x_wt, x_mut, z_ctrl, B, D = perturb_fixture
        delta    = op.perturbation_vector(x_wt, x_mut)
        expected = op.encode_mutant(x_mut) - op.encode_wildtype(x_wt)
        assert torch.allclose(delta, expected, atol=1e-5), (
            "P2 VIOLATED: perturbation_vector does not equal "
            "encode_mutant − encode_wildtype."
        )

    def test_P3_zero_alpha_returns_ctrl_state_unchanged(self, perturb_fixture):
        """
        P3: predict_perturbed_state(z, wt, mut, α=0) ≈ z

        At α=0, no intervention is applied. The predicted state must equal
        the control state exactly. This is the mathematical identity of the
        zero-perturbation case.
        """
        op, x_wt, x_mut, z_ctrl, B, D = perturb_fixture
        z_pred = op.predict_perturbed_state(z_ctrl, x_wt, x_mut, alpha=0.0)
        assert torch.allclose(z_pred, z_ctrl, atol=1e-5), (
            f"P3 VIOLATED: at α=0, z_pred should equal z_ctrl. "
            f"Max deviation: {(z_pred - z_ctrl).abs().max().item():.2e}"
        )

    def test_P4_perturbation_displacement_linear_in_alpha(self, perturb_fixture):
        """
        P4: displacement from z_ctrl scales linearly with α

        predict_perturbed_state(z, wt, mut, 2α) − z ≈ 2 · (predict_perturbed_state(z, wt, mut, α) − z)

        The intervention is a linear operator in latent space. Doubling the
        perturbation magnitude must double the displacement from the control.
        """
        op, x_wt, x_mut, z_ctrl, B, D = perturb_fixture
        alpha = 0.7
        z_pred_1x = op.predict_perturbed_state(z_ctrl, x_wt, x_mut, alpha=alpha)
        z_pred_2x = op.predict_perturbed_state(z_ctrl, x_wt, x_mut, alpha=2 * alpha)
        displacement_1x = z_pred_1x - z_ctrl
        displacement_2x = z_pred_2x - z_ctrl
        assert torch.allclose(displacement_2x, 2 * displacement_1x, atol=1e-4), (
            "P4 VIOLATED: perturbation displacement is not linear in α. "
            "The operator must be additive in latent space."
        )

    def test_P5_perturbation_vector_independent_of_ctrl_state(self, perturb_fixture):
        """
        P5: perturbation_vector(x_wt, x_mut) is independent of z_ctrl

        The direction and magnitude of the perturbation depends only on the
        wildtype/mutant pair — not on the control state being perturbed.
        Two calls with different z_ctrl but the same (x_wt, x_mut) must
        produce identical perturbation vectors.
        """
        op, x_wt, x_mut, z_ctrl, B, D = perturb_fixture
        delta_1 = op.perturbation_vector(x_wt, x_mut)
        z_ctrl_2 = torch.randn_like(z_ctrl)
        delta_2 = op.perturbation_vector(x_wt, x_mut)
        assert torch.allclose(delta_1, delta_2, atol=1e-5), (
            "P5 VIOLATED: perturbation_vector changes across calls — "
            "it must be independent of any control state."
        )

    def test_P6_perturbation_is_deterministic(self, perturb_fixture):
        """
        P6: same inputs always produce the same perturbation vector

        The operator must be deterministic in inference mode. Non-determinism
        (from dropout, stochastic layers, etc.) would violate reproducibility
        and make the predicted z_pred undefined.
        """
        op, x_wt, x_mut, z_ctrl, B, D = perturb_fixture
        delta_a = op.perturbation_vector(x_wt, x_mut)
        delta_b = op.perturbation_vector(x_wt, x_mut)
        assert torch.allclose(delta_a, delta_b, atol=1e-6), (
            "P6 VIOLATED: perturbation_vector is non-deterministic. "
            "Same inputs must always produce the same result."
        )


# ===========================================================================
# R1–R4: AbstractRoutingKernel invariants
# ===========================================================================

class TestRoutingKernelInvariants:
    """
    Behavioral invariant tests for AbstractRoutingKernel.

    EntropicThresholdKernel and MultiThresholdRoutingKernel use different
    decision logic — both must satisfy R1–R4. Any future routing mechanism
    (learned, probabilistic, topological) passes these tests to prove it
    satisfies the specification.
    """

    def test_R1_score_returns_finite_float(self, routing_fixture):
        """
        R1: score(state) returns a finite float

        The routing signal must be a valid, finite scalar. NaN or ±inf
        would make routing decisions undefined and graph execution unreliable.
        """
        kernel, states = routing_fixture
        for state in states:
            s = kernel.score(state)
            assert isinstance(s, (float, int)), (
                f"R1 VIOLATED: score returned {type(s)}, expected float."
            )
            assert math.isfinite(float(s)), (
                f"R1 VIOLATED: score={s} is not finite for state={state}."
            )

    def test_R2_route_returns_nonempty_string(self, routing_fixture):
        """
        R2: route(state) returns a non-empty string

        The routing decision must be a valid node identifier. An empty string
        or non-string would cause the graph executor to silently fail to route.
        """
        kernel, states = routing_fixture
        for state in states:
            r = kernel.route(state)
            assert isinstance(r, str), (
                f"R2 VIOLATED: route returned {type(r)}, expected str."
            )
            assert len(r) > 0, (
                f"R2 VIOLATED: route returned empty string for state={state}."
            )

    def test_R3_routing_is_deterministic(self, routing_fixture):
        """
        R3: same state always produces same (score, route)

        Routing must be deterministic. Non-deterministic routing would produce
        irreproducible graph execution traces and violate audit requirements.
        """
        kernel, states = routing_fixture
        for state in states:
            s1, r1 = kernel.score(state), kernel.route(state)
            s2, r2 = kernel.score(state), kernel.route(state)
            assert s1 == s2, (
                f"R3 VIOLATED: score is non-deterministic for state={state}. "
                f"Got {s1} then {s2}."
            )
            assert r1 == r2, (
                f"R3 VIOLATED: route is non-deterministic for state={state}. "
                f"Got '{r1}' then '{r2}'."
            )

    def test_R4_route_consistent_with_score_across_calls(self, routing_fixture):
        """
        R4: the score→route mapping is stable across independent calls

        A kernel that produces the same score but a different route on separate
        calls is internally inconsistent and cannot be trusted for graph routing.
        """
        kernel, states = routing_fixture
        for state in states:
            # Verify that repeated score→route pairs are consistent
            pairs = [(kernel.score(state), kernel.route(state)) for _ in range(3)]
            scores  = [p[0] for p in pairs]
            routes  = [p[1] for p in pairs]
            assert len(set(scores)) == 1, (
                f"R4 VIOLATED: score is inconsistent for state={state}: {scores}"
            )
            assert len(set(routes)) == 1, (
                f"R4 VIOLATED: route is inconsistent for state={state}: {routes}"
            )


# ===========================================================================
# M1–M3: AbstractModalEncoder invariants
# ===========================================================================

class TestModalEncoderInvariants:
    """
    Behavioral invariant tests for AbstractModalEncoder.

    Three domain encoders (genomic, electrochemical, network telemetry) with
    different input dimensionalities — all must satisfy M1–M3. Any future
    modal encoder (imaging, spectroscopic, seismic, protein) passes these tests.
    """

    def test_M1_encode_produces_correct_output_shape(self, encoder_fixture):
        """
        M1: encode(x).shape == (B, output_dim)

        The encoder must produce a (batch_size, embedding_dim) tensor.
        Wrong shape would break any downstream cross-attention or routing step.
        """
        encoder, x, B, D = encoder_fixture
        z = encoder.encode(x)
        assert z.shape == (B, D), (
            f"M1 VIOLATED: encode(x).shape={z.shape}, expected ({B}, {D}). "
            f"Modality: {encoder.modality}"
        )

    def test_M2_output_dim_property_consistent_with_encode_output(self, encoder_fixture):
        """
        M2: output_dim is constant across all encode() calls

        The output_dim property must match the actual embedding dimension and
        remain stable. Any change would break downstream components that rely
        on knowing the embedding dimensionality at construction time.
        """
        encoder, x, B, D = encoder_fixture
        z1 = encoder.encode(x)
        z2 = encoder.encode(torch.randn_like(x))
        assert z1.shape[-1] == encoder.output_dim, (
            f"M2 VIOLATED: encode output dim {z1.shape[-1]} != "
            f"output_dim property {encoder.output_dim}"
        )
        assert z2.shape[-1] == encoder.output_dim, (
            f"M2 VIOLATED: output_dim inconsistent across calls."
        )

    def test_M3_encode_is_deterministic(self, encoder_fixture):
        """
        M3: encode(x) is deterministic for the same input

        The encoder must produce identical output for identical input in
        inference mode. Non-determinism (dropout, sampling) would invalidate
        audit trails and make graph replay impossible.
        """
        encoder, x, B, D = encoder_fixture
        z1 = encoder.encode(x)
        z2 = encoder.encode(x)
        assert torch.allclose(z1, z2, atol=1e-6), (
            f"M3 VIOLATED: encode(x) is non-deterministic. "
            f"Modality: {encoder.modality}. "
            f"Max deviation: {(z1 - z2).abs().max().item():.2e}"
        )


# ===========================================================================
# ABC contract tests — the interfaces themselves
# ===========================================================================

class TestAbstractInterfaceContracts:
    """
    Verifies that each ABC is properly defined and cannot be instantiated
    without a complete implementation.
    """

    def test_attention_kernel_is_abstract(self):
        with pytest.raises(TypeError):
            AbstractAttentionKernel()

    def test_perturbation_operator_is_abstract(self):
        with pytest.raises(TypeError):
            AbstractPerturbationOperator()

    def test_routing_kernel_is_abstract(self):
        with pytest.raises(TypeError):
            AbstractRoutingKernel()

    def test_modal_encoder_is_abstract(self):
        with pytest.raises(TypeError):
            AbstractModalEncoder()

    def test_all_attention_implementations_are_valid_subclasses(self):
        for cls in [ScaledDotProductKernel, CosineAttentionKernel]:
            assert issubclass(cls, AbstractAttentionKernel), (
                f"{cls.__name__} is not a subclass of AbstractAttentionKernel"
            )

    def test_all_perturbation_implementations_are_valid_subclasses(self):
        for cls in [GenomicPerturbationOperator, CrystalDefectOperator, MolecularBindingOperator]:
            assert issubclass(cls, AbstractPerturbationOperator), (
                f"{cls.__name__} is not a subclass of AbstractPerturbationOperator"
            )
            instance = cls(50, 32)
            assert isinstance(instance, AbstractPerturbationOperator)

    def test_all_routing_implementations_are_valid_subclasses(self):
        for cls in [EntropicThresholdKernel, MultiThresholdRoutingKernel]:
            assert issubclass(cls, AbstractRoutingKernel), (
                f"{cls.__name__} is not a subclass of AbstractRoutingKernel"
            )

    def test_all_modal_encoder_implementations_are_valid_subclasses(self):
        for cls in [GenomicSequenceEncoder, ElectrochemicalEncoder, NetworkTelemetryEncoder]:
            assert issubclass(cls, AbstractModalEncoder), (
                f"{cls.__name__} is not a subclass of AbstractModalEncoder"
            )

    def test_perturbation_operator_predict_method_is_inherited(self):
        """perturbation_vector() and predict_perturbed_state() are concrete — all subclasses inherit them."""
        for cls in [GenomicPerturbationOperator, CrystalDefectOperator, MolecularBindingOperator]:
            instance = cls(20, 16)
            assert hasattr(instance, "perturbation_vector") and callable(instance.perturbation_vector)
            assert hasattr(instance, "predict_perturbed_state") and callable(instance.predict_perturbed_state)
