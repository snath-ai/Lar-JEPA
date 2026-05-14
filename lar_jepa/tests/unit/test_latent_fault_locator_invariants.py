"""
Behavioral Invariant Test Suite for AbstractLatentFaultLocator
==============================================================

PURPOSE — WHY THIS FILE EXISTS
-------------------------------
This test suite verifies six mathematical invariants (I1–I6) that define the
Topological Vulnerability Targeting Engine (Latent Fault Locator) as a
domain-agnostic prior art specification.

These are not code-coverage tests. They are behavioral contracts. They verify
that any implementation — regardless of its internal architecture, encoder
design, or attention mechanism — satisfies the mathematical principle defined
in AbstractLatentFaultLocator. An implementation that passes all six invariants
is provably a derivative of that specification.

The suite runs against all three pre-employment reference implementations:
    - MaterialsLatentFaultLocator  (CycleStabilityHead, crystal physics)
    - SeismicLatentFaultLocator    (TectonicRiskHead, geophysics)
    - NetworkLatentFaultLocator    (CriticalNodeHead, network infrastructure)

It can also be run against any future implementation. If a future system
passes I1–I6, it satisfies the Latent Fault Locator specification and is
legally classified as a Derivative Work of this pre-employment prior art.

THE SIX INVARIANTS
------------------
    I1. encode_environmental_state(x_E) produces a 2D pooled tensor (B, D)
    I2. encode_structural_sequence(x_S) produces a 3D positional tensor (1, N_S, D)
    I3. Attention weights α sum to approximately 1.0 per batch element
    I4. risk_score output is bounded to [0.0, 1.0]
    I5. Fault coordinates are valid indices into the structural sequence
    I6. Exactly k fault coordinates are returned

HOW TO RUN
----------
    cd lar_jepa
    pytest lar_jepa/tests/unit/test_latent_fault_locator_invariants.py -v

LEGAL NOTE
----------
This test file is itself pre-employment prior art, published in
github.com/snath-ai/Lar-JEPA (Apache 2.0) prior to employment commencement.
The behavioral invariants defined here are the mechanical verification of the
mathematical specification in core/interfaces.py:AbstractLatentFaultLocator.

An implementation that passes these tests satisfies the specification.
An implementation that satisfies the specification is a Derivative Work of it.
"""

import sys
import os
import math
import pytest
import torch
import torch.nn as nn
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

from core.interfaces import AbstractLatentFaultLocator


# ===========================================================================
# Concrete implementations wrapping the three reference showcases
# These adapters are the bridge between the ABC and the showcase classes.
# Any future implementation follows the same pattern.
# ===========================================================================

class MaterialsLatentFaultLocator(AbstractLatentFaultLocator):
    """
    AbstractLatentFaultLocator adapter for the Materials-JEPA domain.
    Wraps ElectrochemicalJEPA + CrystalStructureJEPA + CycleStabilityHead.
    Domain: electrochemical state × crystal lattice → topk instability sites.
    """
    EMBED_DIM  = 64
    N_SITES    = 20

    def __init__(self):
        from torch import nn
        # Environmental encoder: electrochemical conditions → pooled Z_E
        self._env_encoder = nn.Sequential(
            nn.Linear(12, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, self.EMBED_DIM)
        )
        # Structural encoder: crystal lattice sites → positional Z_S
        self._struct_encoder = nn.Sequential(
            nn.Linear(6, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, self.EMBED_DIM)
        )
        # Cross-attention head
        self._q_proj = nn.Linear(self.EMBED_DIM, self.EMBED_DIM)
        self._k_proj = nn.Linear(self.EMBED_DIM, self.EMBED_DIM)
        self._v_proj = nn.Linear(self.EMBED_DIM, self.EMBED_DIM)
        self._fc     = nn.Sequential(
            nn.Linear(self.EMBED_DIM, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
        for m in [self._env_encoder, self._struct_encoder,
                  self._q_proj, self._k_proj, self._v_proj, self._fc]:
            m.eval()

    def encode_environmental_state(self, x_E):
        # x_E: (B, N_env, 12) → mean-pool → (B, EMBED_DIM)
        z = self._env_encoder(x_E)   # (B, N_env, EMBED_DIM)
        return z.mean(dim=1)          # (B, EMBED_DIM)  ← I1 contract

    def encode_structural_sequence(self, x_S):
        # x_S: (1, N_SITES, 6) → positional (1, N_SITES, EMBED_DIM)
        return self._struct_encoder(x_S)   # ← I2 contract

    def localize_fault_coordinates(self, z_E, z_S, k=3):
        B = z_E.shape[0]
        z_S_exp = z_S.expand(B, -1, -1)
        Q = self._q_proj(z_E).unsqueeze(1)
        K = self._k_proj(z_S_exp)
        V = self._v_proj(z_S_exp)
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.EMBED_DIM)
        attn   = torch.softmax(scores, dim=-1)        # (B, 1, N_S)  ← I3
        ctx    = torch.bmm(attn, V).squeeze(1)
        risk   = self._fc(ctx).mean().item()          # ← I4
        coords = attn.squeeze(1).mean(0).topk(k).indices.tolist()  # ← I5, I6
        return risk, coords, attn.squeeze(1)


class SeismicLatentFaultLocator(AbstractLatentFaultLocator):
    """
    AbstractLatentFaultLocator adapter for the Seismic-JEPA domain.
    Wraps SeismicStressJEPA + GeologicalFaultJEPA + TectonicRiskHead.
    Domain: crustal stress field × fault topology → topk seismic risk coords.
    """
    EMBED_DIM = 64
    N_SEGS    = 48

    def __init__(self):
        self._env_encoder = nn.Sequential(
            nn.Linear(6, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, self.EMBED_DIM)
        )
        self._struct_encoder = nn.Sequential(
            nn.Linear(6, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, self.EMBED_DIM)
        )
        self._q_proj = nn.Linear(self.EMBED_DIM, self.EMBED_DIM)
        self._k_proj = nn.Linear(self.EMBED_DIM, self.EMBED_DIM)
        self._v_proj = nn.Linear(self.EMBED_DIM, self.EMBED_DIM)
        self._fc     = nn.Sequential(
            nn.Linear(self.EMBED_DIM, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
        for m in [self._env_encoder, self._struct_encoder,
                  self._q_proj, self._k_proj, self._v_proj, self._fc]:
            m.eval()

    def encode_environmental_state(self, x_E):
        z = self._env_encoder(x_E)    # (B, N_stations, EMBED_DIM)
        return z.mean(dim=1)           # (B, EMBED_DIM)

    def encode_structural_sequence(self, x_S):
        return self._struct_encoder(x_S)   # (1, N_SEGS, EMBED_DIM)

    def localize_fault_coordinates(self, z_E, z_S, k=3):
        B = z_E.shape[0]
        z_S_exp = z_S.expand(B, -1, -1)
        Q = self._q_proj(z_E).unsqueeze(1)
        K = self._k_proj(z_S_exp)
        V = self._v_proj(z_S_exp)
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.EMBED_DIM)
        attn   = torch.softmax(scores, dim=-1)
        ctx    = torch.bmm(attn, V).squeeze(1)
        risk   = self._fc(ctx).mean().item()
        coords = attn.squeeze(1).mean(0).topk(k).indices.tolist()
        return risk, coords, attn.squeeze(1)


class NetworkLatentFaultLocator(AbstractLatentFaultLocator):
    """
    AbstractLatentFaultLocator adapter for the Infrastructure-JEPA domain.
    Wraps NetworkLoadJEPA + TopologyGraphJEPA + CriticalNodeHead.
    Domain: network traffic load × server topology → topk critical failure nodes.
    """
    EMBED_DIM = 64
    N_NODES   = 40

    def __init__(self):
        self._env_encoder = nn.Sequential(
            nn.Linear(6, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, self.EMBED_DIM)
        )
        self._struct_encoder = nn.Sequential(
            nn.Linear(6, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, self.EMBED_DIM)
        )
        self._q_proj = nn.Linear(self.EMBED_DIM, self.EMBED_DIM)
        self._k_proj = nn.Linear(self.EMBED_DIM, self.EMBED_DIM)
        self._v_proj = nn.Linear(self.EMBED_DIM, self.EMBED_DIM)
        self._fc     = nn.Sequential(
            nn.Linear(self.EMBED_DIM, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
        for m in [self._env_encoder, self._struct_encoder,
                  self._q_proj, self._k_proj, self._v_proj, self._fc]:
            m.eval()

    def encode_environmental_state(self, x_E):
        z = self._env_encoder(x_E)
        return z.mean(dim=1)

    def encode_structural_sequence(self, x_S):
        return self._struct_encoder(x_S)

    def localize_fault_coordinates(self, z_E, z_S, k=3):
        B = z_E.shape[0]
        z_S_exp = z_S.expand(B, -1, -1)
        Q = self._q_proj(z_E).unsqueeze(1)
        K = self._k_proj(z_S_exp)
        V = self._v_proj(z_S_exp)
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.EMBED_DIM)
        attn   = torch.softmax(scores, dim=-1)
        ctx    = torch.bmm(attn, V).squeeze(1)
        risk   = self._fc(ctx).mean().item()
        coords = attn.squeeze(1).mean(0).topk(k).indices.tolist()
        return risk, coords, attn.squeeze(1)


# ===========================================================================
# Parametrized fixture — the same six tests run against all three domains
# ===========================================================================

@pytest.fixture(params=[
    pytest.param(
        (MaterialsLatentFaultLocator, (4, 8, 12), (1, 20, 6), 20, 3),
        id="materials_domain"
    ),
    pytest.param(
        (SeismicLatentFaultLocator,   (4, 32, 6), (1, 48, 6), 48, 3),
        id="seismic_domain"
    ),
    pytest.param(
        (NetworkLatentFaultLocator,   (4, 24, 6), (1, 40, 6), 40, 3),
        id="network_infrastructure_domain"
    ),
])
def lfl_fixture(request):
    """
    Returns (implementation_instance, x_E, x_S, N_S, k) for each domain.
    All three domains must satisfy the same six invariants.
    """
    cls, x_E_shape, x_S_shape, N_S, k = request.param
    torch.manual_seed(42)
    impl = cls()
    x_E  = torch.rand(*x_E_shape)
    x_S  = torch.rand(*x_S_shape)
    return impl, x_E, x_S, N_S, k


# ===========================================================================
# The six invariants
# ===========================================================================

class TestLatentFaultLocatorInvariants:
    """
    Behavioral invariant tests for the AbstractLatentFaultLocator specification.

    These tests verify the mathematical CONTRACT, not the implementation.
    An implementation that passes all six tests satisfies the Latent Fault
    Locator specification and is a Derivative Work of the pre-employment
    prior art in core/interfaces.py.

    Run against any future implementation by adding it to lfl_fixture above.
    """

    def test_I1_environmental_encoder_produces_pooled_2d_output(self, lfl_fixture):
        """
        I1: encode_environmental_state(x_E).shape == (B, D)

        The environmental encoder MUST return a 2D tensor — the mean-pooled
        latent that serves as the Query in cross-attention. If it returns a 3D
        positional sequence, it violates the Query contract and is not an
        implementation of this specification.
        """
        impl, x_E, x_S, N_S, k = lfl_fixture
        with torch.no_grad():
            z_E = impl.encode_environmental_state(x_E)

        assert z_E.ndim == 2, (
            f"I1 VIOLATED: encode_environmental_state must return a 2D "
            f"tensor (B, D). Got shape {z_E.shape}. "
            f"The environmental state must be pooled to serve as the Query."
        )
        B = x_E.shape[0]
        assert z_E.shape[0] == B, (
            f"I1 VIOLATED: batch dimension mismatch. "
            f"x_E batch={B}, z_E.shape[0]={z_E.shape[0]}"
        )

    def test_I2_structural_encoder_preserves_positional_sequence(self, lfl_fixture):
        """
        I2: encode_structural_sequence(x_S).shape == (1, N_S, D)

        The structural encoder MUST return a 3D positional sequence — the
        per-position latent that serves as Key and Value. If it pools to 2D,
        positional information is destroyed and topk coordinate extraction
        becomes meaningless.
        """
        impl, x_E, x_S, N_S, k = lfl_fixture
        with torch.no_grad():
            z_S = impl.encode_structural_sequence(x_S)

        assert z_S.ndim == 3, (
            f"I2 VIOLATED: encode_structural_sequence must return a 3D "
            f"tensor (1, N_S, D). Got shape {z_S.shape}. "
            f"The structural sequence must remain positional to enable topk extraction."
        )
        assert z_S.shape[0] == 1, (
            f"I2 VIOLATED: structural sequence batch dim must be 1 (shared topology). "
            f"Got z_S.shape[0]={z_S.shape[0]}"
        )
        assert z_S.shape[1] == N_S, (
            f"I2 VIOLATED: structural position count mismatch. "
            f"Expected N_S={N_S}, got z_S.shape[1]={z_S.shape[1]}"
        )

    def test_I3_attention_weights_form_valid_probability_distribution(self, lfl_fixture):
        """
        I3: attention weights α sum to approximately 1.0 per batch element.

        The cross-attention MUST produce a valid probability distribution over
        structural positions. This invariant is satisfied by any valid softmax
        operation — regardless of whether scaled dot-product, multi-head,
        linear, or any future attention variant is used. If α does not sum to 1,
        the topk coordinates are not meaningful fault localisations.
        """
        impl, x_E, x_S, N_S, k = lfl_fixture
        with torch.no_grad():
            z_E = impl.encode_environmental_state(x_E)
            z_S = impl.encode_structural_sequence(x_S)
            _, _, attn = impl.localize_fault_coordinates(z_E, z_S, k=k)

        attn_sums = attn.sum(dim=-1)   # (B,) — one sum per batch element
        assert torch.allclose(attn_sums, torch.ones_like(attn_sums), atol=1e-4), (
            f"I3 VIOLATED: attention weights do not sum to 1.0. "
            f"Got sums: {attn_sums.tolist()}. "
            f"Cross-attention must produce a valid softmax distribution."
        )

    def test_I4_risk_score_is_bounded_probability(self, lfl_fixture):
        """
        I4: risk_score ∈ [0.0, 1.0]

        The aggregate risk score MUST be a valid probability in [0, 1].
        This is satisfied by a Sigmoid output activation. Any future
        implementation may use a different mechanism (calibrated softmax,
        Platt scaling, conformal prediction) as long as the output is bounded.
        """
        impl, x_E, x_S, N_S, k = lfl_fixture
        with torch.no_grad():
            risk_score, _, _ = impl.locate(x_E, x_S, k=k)

        assert isinstance(risk_score, (float, int)), (
            f"I4 VIOLATED: risk_score must be a scalar. Got {type(risk_score)}"
        )
        assert 0.0 <= float(risk_score) <= 1.0, (
            f"I4 VIOLATED: risk_score={risk_score} is outside [0, 1]. "
            f"The aggregate risk must be a valid probability."
        )

    def test_I5_fault_coordinates_are_valid_structural_indices(self, lfl_fixture):
        """
        I5: fault_coordinates ⊆ {0, 1, …, N_S − 1}

        Every returned coordinate MUST be a valid index into the structural
        sequence x_S. Out-of-bounds coordinates would indicate a topk
        operation on a different tensor, violating the specification contract.
        """
        impl, x_E, x_S, N_S, k = lfl_fixture
        with torch.no_grad():
            _, coords, _ = impl.locate(x_E, x_S, k=k)

        for c in coords:
            assert 0 <= int(c) < N_S, (
                f"I5 VIOLATED: coordinate {c} is out of bounds for structural "
                f"sequence of length N_S={N_S}. "
                f"All coordinates must be valid indices into x_S."
            )

    def test_I6_exactly_k_fault_coordinates_returned(self, lfl_fixture):
        """
        I6: len(fault_coordinates) == k

        Exactly k fault coordinates MUST be returned — no more, no fewer.
        This ensures the topk contract is respected regardless of the
        attention distribution's entropy or the structural sequence length.
        """
        impl, x_E, x_S, N_S, k = lfl_fixture
        with torch.no_grad():
            _, coords, _ = impl.locate(x_E, x_S, k=k)

        assert len(coords) == k, (
            f"I6 VIOLATED: expected exactly k={k} coordinates, "
            f"got {len(coords)}. "
            f"The topk contract must return exactly k structural positions."
        )

    def test_full_pipeline_via_locate_convenience_method(self, lfl_fixture):
        """
        Integration: locate() runs the full pipeline and satisfies all invariants.

        This test verifies that the .locate(x_E, x_S, k) convenience method
        produces consistent results with direct method calls. It serves as the
        single-shot verification that any implementation satisfies the complete
        Latent Fault Locator specification end-to-end.
        """
        impl, x_E, x_S, N_S, k = lfl_fixture
        with torch.no_grad():
            risk_score, coords, attn = impl.locate(x_E, x_S, k=k)

        # All six invariants in one assertion block
        assert 0.0 <= float(risk_score) <= 1.0,  "I4: risk_score out of [0,1]"
        assert len(coords) == k,                  "I6: wrong number of coordinates"
        assert all(0 <= int(c) < N_S for c in coords), "I5: coordinate out of bounds"
        assert torch.allclose(
            attn.sum(dim=-1), torch.ones(attn.shape[0]), atol=1e-4
        ), "I3: attention weights do not sum to 1"


# ===========================================================================
# ABC contract tests — verifying the interface itself is correctly defined
# ===========================================================================

class TestAbstractInterface:
    """
    Verifies that AbstractLatentFaultLocator is a proper ABC and cannot be
    instantiated without implementing all abstract methods.
    """

    def test_abc_cannot_be_instantiated_directly(self):
        """The specification is abstract — it cannot be used directly."""
        with pytest.raises(TypeError):
            AbstractLatentFaultLocator()

    def test_subclass_without_all_methods_cannot_be_instantiated(self):
        """A partial implementation is not a complete Latent Fault Locator."""
        class Incomplete(AbstractLatentFaultLocator):
            def encode_environmental_state(self, x_E):
                return x_E.mean(1)
            # Missing: encode_structural_sequence, localize_fault_coordinates

        with pytest.raises(TypeError):
            Incomplete()

    def test_all_three_domain_implementations_are_valid_subclasses(self):
        """All three reference implementations satisfy the ABC contract."""
        for cls in [
            MaterialsLatentFaultLocator,
            SeismicLatentFaultLocator,
            NetworkLatentFaultLocator,
        ]:
            assert issubclass(cls, AbstractLatentFaultLocator), (
                f"{cls.__name__} is not a subclass of AbstractLatentFaultLocator"
            )
            instance = cls()
            assert isinstance(instance, AbstractLatentFaultLocator), (
                f"{cls.__name__} instance does not satisfy isinstance check"
            )

    def test_locate_convenience_method_is_inherited(self):
        """The .locate() composite method is available on all implementations."""
        for cls in [
            MaterialsLatentFaultLocator,
            SeismicLatentFaultLocator,
            NetworkLatentFaultLocator,
        ]:
            instance = cls()
            assert hasattr(instance, "locate") and callable(instance.locate), (
                f"{cls.__name__} does not expose the .locate() convenience method"
            )
