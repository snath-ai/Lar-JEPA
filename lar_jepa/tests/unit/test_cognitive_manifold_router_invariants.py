"""
Behavioral Invariant Test Suite — AbstractCognitiveNode, AbstractManifold,
AbstractEntropicRouter (ABCs 1, 2, 5)
=============================================================================

PURPOSE — WHY THIS FILE EXISTS
-------------------------------
Of the Lár-JEPA ten-ABC cognitive contract, seven ABCs already have a
dedicated behavioral-invariant test file proving the contract holds across
multiple, structurally different concrete implementations:

    test_latent_fault_locator_invariants.py — ABC 4  (I1–I6)
    test_core_interface_invariants.py        — ABCs 6, 7, 8, 9 (A1–A6, P1–P6, R1–R4, M1–M3)
    test_divergence_router_invariants.py     — ABCs 3, 10 (V1–V6)

The remaining three ABCs — AbstractCognitiveNode (1), AbstractManifold (2),
and AbstractEntropicRouter (5) — were exercised only by implementation-
specific unit tests (test_jepa_components.py: NBodyKinematicsJEPA,
EntropicVetoRouter, CognitiveNodeAdapter) and by UCR's own end-to-end
integration test (examples/powergrid_full_stack.py). Neither proves the
general, cross-implementation CONTRACT the way the invariant files above do
for their ABCs — they prove one concrete class behaves correctly, not that
the interface itself enforces a testable shape on any compliant subclass.

This file closes that gap. It defines CN1–CN3, MF1–MF3, and ER1–ER4 and
verifies them against REAL, already-shipped implementations rather than
fixtures authored for this test:

    AbstractCognitiveNode  -> RealLLMNode   (experiments/heterogeneous_batch_test.py,
                                              sentence-transformers/all-MiniLM-L6-v2)
                              RealJEPANode  (same file, as a plain CognitiveNode)
                              RealGNNNode   (same file, 2-layer message-passing GNN)
    AbstractManifold       -> RealJEPANode  (same file, AbstractManifold subclass)
    AbstractEntropicRouter -> CalibratedEntropicRouter (experiments/lgsl_routing_gate.py)

These are the three real models already proven, in that file's own words, to
run "concurrently... with isolated state" inside a real BatchNode — using
the actual shipped classes (rather than toy stand-ins written only for this
test) is itself part of the proof: the invariants hold for the system as
built, not for a simplified model of it.

RUNNING THE SUITE
-----------------
    cd lar_jepa
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 USE_FLAX=0 \
        pytest lar_jepa/tests/unit/test_cognitive_manifold_router_invariants.py -v

All four of these must be exported at the shell before pytest starts, not
just set via os.environ inside this file: one of pytest's own installed
plugins imports huggingface_hub (and, transitively, transformers) during
its own plugin bootstrap, before this module's top-level code ever runs —
and both huggingface_hub's offline flags and transformers' TF/Flax
backend-availability probe are read once, at that early import time. Set
only here, they come too late: from_pretrained() either falls through to a
network revision check, or probes for a broken local TensorFlow install,
and either one has been observed to hang indefinitely rather than fail
fast (the TF probe hangs inside a native absl mutex — the exact failure
mode heterogeneous_batch_test.py's own USE_TF/USE_FLAX workaround exists to
avoid, which only protects that file's own import, not pytest's earlier
one). Set at the shell, all four at once, the full 19-test suite runs in
under 10s, reproduced twice.

Loading a real pretrained transformer (MiniLM) still makes this file slower
than its siblings (seconds, not milliseconds) — the heavy fixtures are
module-scoped so each real model loads exactly once per test run.

THE INVARIANTS
--------------
    AbstractCognitiveNode (CN1–CN3):
        CN1. Sequencing: encode() -> forward() -> decode() compose end to
             end for any concrete node, each stage consuming the previous
             stage's real output.
        CN2. Stable type declaration: output_signal_type is a fixed
             SignalType, identical across repeated access.
        CN3. Model-type isolation: genuinely different architectures
             (LLM/JEPA/GNN) declare genuinely different model_type values;
             genuinely same-family models (two different LLMs) correctly
             share one.

    AbstractManifold (MF1–MF3):
        MF1. Shared latent dimension: embed_context(x).shape[-1] ==
             predict_target(z, a).shape[-1] for all x — context and target
             exist in the same latent space. (This is the single invariant
             UCR Section 4.1 already states in prose; MF1 is its first
             coded, cross-implementation-tested form.)
        MF2. Determinism: embed_context(x) is deterministic for the same x
             at inference time (no dropout/sampling leaking through).
        MF3. Non-degeneracy: predict_target does not collapse genuinely
             different contexts to the same (or a constant/zero) output —
             the manifold actually discriminates between distinct inputs.

    AbstractEntropicRouter (ER1–ER4):
        ER1. Routing Completeness: evaluate_state() returns exactly one
             member of RouteDecision for every input tested.
        ER2. Calibrated (data-derived) thresholds: a calibration procedure's
             thresholds come from the concrete implementation's own sampled
             data distribution, not a hardcoded constant — re-calibrating
             with a different seed produces genuinely different values.
        ER3. Monotonic ordering: increasing entropic deviation never
             decreases route severity (COMMIT_TRAJECTORY <=
             TRIGGER_REPLAN <= STRUCTURAL_IMPASSE).
        ER4. Determinism: the same predicted_state always yields the same
             RouteDecision.

CROSS-REFERENCE
----------------
UCR main.tex (DOI: 10.5281/zenodo.20278775) is updated alongside this file
to raise the named-invariant count from 33 to 43 (+CN1-3, +MF2-3 coding
MF1, +ER1-4), closing the asymmetry between these three ABCs and the other
seven. See the paper's new subsection, "Closing the Invariant Gap for
AbstractCognitiveNode, AbstractManifold, and AbstractEntropicRouter."

Authorship: Aadithya Vishnu Sajeev
"""

import os
import sys

import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed — skipping invariant tests")
pytest.importorskip("transformers", reason="transformers not installed — skipping invariant tests")

# ---------------------------------------------------------------------------
# Path bootstrap
# Test file: lar_jepa/lar_jepa/tests/unit/test_cognitive_manifold_router_invariants.py
# ../../..   -> lar_jepa top-level (core/, experiments/, spatial_kinematics_engine/, ...)
# ../../src  -> lar_jepa/lar_jepa/src (the embedded lar engine)
# ../../../experiments -> the real model implementations under test
# ---------------------------------------------------------------------------
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Force local-cache-only model loading. Without this, from_pretrained() makes a
# network call to check for updates even when the model is already cached, and
# that call has been observed to hang indefinitely on a degraded connection
# rather than failing fast -- the same category of hang the USE_TF/USE_FLAX
# workaround above exists to avoid.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_JEPA_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_LAR_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", "src"))
_EXPERIMENTS = os.path.join(_JEPA_ROOT, "experiments")

for _p in [_JEPA_ROOT, _LAR_SRC, _EXPERIMENTS]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.types import RouteDecision, ModelType
from heterogeneous_batch_test import (
    RealLLMNode,
    RealJEPANode,
    RealGNNNode,
    RealGenerativeLLMNode,
    build_synthetic_graph,
)
from lgsl_routing_gate import CalibratedEntropicRouter, calibrate


# ===========================================================================
# Module-scoped fixtures — real models, loaded once per test session
# ===========================================================================

@pytest.fixture(scope="module")
def real_llm_node():
    """A real pretrained transformer (sentence-transformers/all-MiniLM-L6-v2)."""
    return RealLLMNode()


@pytest.fixture(scope="module")
def real_jepa_node():
    """A real two-MLP JEPA (context encoder + predictor), AbstractManifold subclass."""
    return RealJEPANode(state_dim=16, latent_dim=8, action_dim=4)


@pytest.fixture(scope="module")
def real_gnn_node():
    """A real two-layer message-passing GNN, pure torch."""
    return RealGNNNode(feat_dim=8, hidden_dim=16)


@pytest.fixture(scope="module")
def calibrated_router(real_jepa_node):
    """
    The real AbstractEntropicRouter implementation, calibrated from the
    JEPA's own null-action behaviour (mean + 2*std / mean + 4*std over 200
    small, near-zero actions) -- exactly as lgsl_routing_gate.py does it.
    """
    context = real_jepa_node.embed_context(torch.randn(16))
    tau_safe, tau_impasse = calibrate(real_jepa_node, context, n=200, seed=7)
    return CalibratedEntropicRouter(tau_safe=tau_safe, tau_impasse=tau_impasse)


@pytest.fixture(params=[
    pytest.param("LLM", id="real_llm_node"),
    pytest.param("JEPA", id="real_jepa_node_as_cognitive_node"),
    pytest.param("GNN", id="real_gnn_node"),
])
def cognitive_node_fixture(request, real_llm_node, real_jepa_node, real_gnn_node):
    """
    Returns (node, raw_input) for each of three structurally different
    AbstractCognitiveNode implementations: a real pretrained transformer,
    a real JEPA (consumed through the base AbstractCognitiveNode interface,
    not its AbstractManifold-specific methods), and a real message-passing GNN.
    """
    if request.param == "LLM":
        return real_llm_node, "Does disagreement between two witnesses carry information?"
    elif request.param == "JEPA":
        return real_jepa_node, torch.randn(16)
    else:
        return real_gnn_node, build_synthetic_graph()


# ===========================================================================
# CN1–CN3: AbstractCognitiveNode invariants
# ===========================================================================

class TestCognitiveNodeInvariants:
    """
    Behavioral invariant tests for AbstractCognitiveNode.

    Three structurally unrelated real implementations -- a pretrained
    transformer, a two-MLP JEPA, and a message-passing GNN -- must all
    satisfy the same encode/forward/decode contract. Any future model
    architecture (diffusion, SSM, or anything in ModelType.FUTURE) proves
    compliance by passing these same tests.
    """

    def test_CN1_encode_forward_decode_sequencing(self, cognitive_node_fixture):
        """
        CN1: encode() -> forward() -> decode() compose end to end.

        Each stage must consume the real output of the previous stage and
        produce a usable GraphState signal (a string, per each of these
        nodes' output_signal_type). A node that breaks this chain cannot be
        routed by the Lár execution spine, which calls exactly this sequence.
        """
        node, raw_input = cognitive_node_fixture
        context = node.encode(raw_input)
        representation = node.forward(context)
        output = node.decode(representation)
        assert isinstance(output, str) and len(output) > 0, (
            f"CN1 VIOLATED: decode() must produce a non-empty GraphState "
            f"signal, got {type(output)}: {output!r}"
        )

    def test_CN2_output_signal_type_is_stable(self, cognitive_node_fixture):
        """
        CN2: output_signal_type is a fixed SignalType, identical across
        repeated access.

        AbstractContextBridge validates cross-modal connections using this
        declared type at graph construction time. A node whose declared
        type changes between accesses would make that validation meaningless.
        """
        node, _ = cognitive_node_fixture
        t1 = node.output_signal_type
        t2 = node.output_signal_type
        assert t1 == t2, (
            f"CN2 VIOLATED: output_signal_type changed between accesses: "
            f"{t1} then {t2}"
        )
        assert hasattr(t1, "name"), (
            f"CN2 VIOLATED: output_signal_type must be a SignalType member, got {t1!r}"
        )

    def test_CN3_model_type_isolation_across_architectures(
        self, real_llm_node, real_jepa_node, real_gnn_node
    ):
        """
        CN3: genuinely different architectures declare genuinely different
        model_type values.

        If model_type were a vestigial constant, the routing-layer
        introspection ModelType exists to support (audit logging,
        PARALLEL_HETEROGENEOUS composition) would be meaningless. This
        checks the inverse failure mode of CN3's companion check below:
        here, three different architectures must NOT collapse to one type.
        """
        types = {real_llm_node.model_type, real_jepa_node.model_type, real_gnn_node.model_type}
        assert types == {ModelType.LLM, ModelType.JEPA, ModelType.GNN}, (
            f"CN3 VIOLATED: expected three distinct model types, got {types}"
        )

    def test_CN3_model_type_correctly_shared_within_architecture_family(self):
        """
        CN3 (companion check): genuinely same-family models correctly SHARE
        one model_type.

        RealLLMNode (MiniLM, embedding-only) and RealGenerativeLLMNode
        (Qwen2.5-0.5B-Instruct, autoregressive generation) are different
        concrete models but the same architecture family -- this is the
        PARALLEL_HOMOGENEOUS composition pattern documented in
        heterogeneous_batch_test.py. model_type must unify them, not split
        them apart for no reason. Checked at the class level: model_type is
        a class attribute, so no model weights need to be loaded for this
        assertion.
        """
        assert RealLLMNode.model_type == RealGenerativeLLMNode.model_type == ModelType.LLM, (
            f"CN3 VIOLATED: two models of the same architecture family must "
            f"share one model_type. Got RealLLMNode={RealLLMNode.model_type}, "
            f"RealGenerativeLLMNode={RealGenerativeLLMNode.model_type}"
        )


# ===========================================================================
# MF1–MF3: AbstractManifold invariants
# ===========================================================================

class TestManifoldInvariants:
    """
    Behavioral invariant tests for AbstractManifold, verified against the
    real RealJEPANode (two-MLP context encoder + predictor).
    """

    def test_MF1_shared_latent_dimension(self, real_jepa_node):
        """
        MF1: embed_context(x).shape[-1] == predict_target(z, a).shape[-1]

        Context and target must exist in the same latent space -- this is
        the invariant UCR Section 4.1 already states in prose. Verified
        here across two different raw observations, not just one.
        """
        obs_a = torch.randn(16)
        obs_b = torch.randn(16)
        ctx_a = real_jepa_node.embed_context(obs_a)
        ctx_b = real_jepa_node.embed_context(obs_b)
        pred = real_jepa_node.predict_target(ctx_a, torch.randn(4))
        assert ctx_a.shape == ctx_b.shape == pred.shape == (8,), (
            f"MF1 VIOLATED: ctx_a.shape={tuple(ctx_a.shape)}, "
            f"ctx_b.shape={tuple(ctx_b.shape)}, pred.shape={tuple(pred.shape)}; "
            f"all three must share one latent dimension."
        )

    def test_MF2_embed_context_is_deterministic(self, real_jepa_node):
        """
        MF2: embed_context(x) is deterministic for the same x at inference
        time.

        The model is in eval() mode -- no dropout, no sampling. Two calls
        with the same observation must produce the exact same latent, or
        downstream prediction and entropic-loss evaluation become
        irreproducible.
        """
        obs = torch.randn(16)
        ctx_1 = real_jepa_node.embed_context(obs)
        ctx_2 = real_jepa_node.embed_context(obs.clone())
        assert torch.allclose(ctx_1, ctx_2), (
            f"MF2 VIOLATED: embed_context is non-deterministic. "
            f"Max deviation: {(ctx_1 - ctx_2).abs().max().item():.2e}"
        )

    def test_MF3_predict_target_is_non_degenerate(self, real_jepa_node):
        """
        MF3: predict_target does not collapse genuinely different contexts
        to the same (or a constant/zero) output.

        A manifold that maps every input to the same latent (or to zero)
        would pass MF1/MF2 trivially while being useless -- this is the
        check that the model is actually discriminating between distinct
        inputs, not just satisfying the shape contract.
        """
        ctx_a = real_jepa_node.embed_context(torch.randn(16))
        ctx_b = real_jepa_node.embed_context(torch.randn(16))
        pred_a = real_jepa_node.predict_target(ctx_a, torch.zeros(4))
        pred_b = real_jepa_node.predict_target(ctx_b, torch.zeros(4))
        assert not torch.allclose(pred_a, pred_b), (
            "MF3 VIOLATED: two different contexts produced the same "
            "predicted target -- the manifold is degenerate."
        )
        assert float(pred_a.norm()) > 1e-6, (
            "MF3 VIOLATED: predicted target collapsed to (near) zero."
        )


# ===========================================================================
# ER1–ER4: AbstractEntropicRouter invariants
# ===========================================================================

class TestEntropicRouterInvariants:
    """
    Behavioral invariant tests for AbstractEntropicRouter, verified against
    the real CalibratedEntropicRouter -- calibrated from a real JEPA's own
    null-action behaviour, the same mean + k*std convention PAV, PERSIST,
    and FLEET all use for their thresholds (not a convention invented for
    this test).
    """

    def test_ER1_routing_completeness(self, calibrated_router):
        """
        ER1: evaluate_state() returns exactly one member of RouteDecision
        for every input.

        Swept across predicted states of very different magnitude (zero,
        small, calibration-scale, and far outside calibration) -- the
        router must never return anything but a valid RouteDecision member.
        """
        test_states = [
            torch.zeros(8),
            torch.randn(8) * 0.1,
            torch.randn(8) * 1.0,
            torch.tensor([8.0, -6.0, 9.0, -7.0, 5.0, -5.0, 6.0, -6.0]),
            torch.randn(8) * 50.0,
        ]
        for state in test_states:
            decision = calibrated_router.evaluate_state(state)
            assert isinstance(decision, RouteDecision), (
                f"ER1 VIOLATED: evaluate_state returned {type(decision)} "
                f"for state={state.tolist()}, expected a RouteDecision member."
            )

    def test_ER2_thresholds_are_data_derived_not_hardcoded(self, real_jepa_node):
        """
        ER2: calibration thresholds come from the concrete implementation's
        own sampled data distribution, not a hardcoded constant.

        Re-running calibrate() with a different seed on the same model must
        produce genuinely different (but still real, ordered) threshold
        values -- proof they're computed from sampled null-action behaviour,
        not a constant baked into the class.
        """
        context = real_jepa_node.embed_context(torch.randn(16))
        tau_safe_a, tau_impasse_a = calibrate(real_jepa_node, context, n=200, seed=7)
        tau_safe_b, tau_impasse_b = calibrate(real_jepa_node, context, n=200, seed=99)
        assert tau_safe_a > 0 and tau_impasse_a > tau_safe_a, (
            f"ER2 VIOLATED: thresholds must be positive and ordered, got "
            f"tau_safe={tau_safe_a}, tau_impasse={tau_impasse_a}"
        )
        assert (tau_safe_a, tau_impasse_a) != (tau_safe_b, tau_impasse_b), (
            "ER2 VIOLATED: two different calibration seeds produced identical "
            "thresholds -- suggests a hardcoded constant, not data-derived calibration."
        )

    def test_ER3_monotonic_ordering_by_severity(self, calibrated_router):
        """
        ER3: increasing entropic deviation never decreases route severity.

        COMMIT_TRAJECTORY <= TRIGGER_REPLAN <= STRUCTURAL_IMPASSE in
        severity. A router that scrambles this ordering would let a
        LARGER deviation commit while a SMALLER one replans or halts --
        an unsafe, non-monotonic gate.
        """
        severity = {
            RouteDecision.COMMIT_TRAJECTORY: 0,
            RouteDecision.TRIGGER_REPLAN: 1,
            RouteDecision.STRUCTURAL_IMPASSE: 2,
        }
        norms = [0.01, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 60.0, 150.0]
        states = [torch.ones(8) * (n / (8 ** 0.5)) for n in norms]
        decisions = [calibrated_router.evaluate_state(s) for s in states]
        sev_seq = [severity[d] for d in decisions]
        assert all(sev_seq[i] <= sev_seq[i + 1] for i in range(len(sev_seq) - 1)), (
            f"ER3 VIOLATED: route severity is not monotonic in deviation magnitude. "
            f"norms={norms} -> {[d.name for d in decisions]}"
        )

    def test_ER4_determinism(self, calibrated_router):
        """
        ER4: the same predicted_state always yields the same RouteDecision.

        Non-determinism here would make audit trails and graph replay
        meaningless -- the same recorded prediction must always route the
        same way.
        """
        state = torch.tensor([1.5, -2.0, 0.3, 4.1, -0.8, 1.2, -3.0, 2.2])
        decisions = [calibrated_router.evaluate_state(state.clone()) for _ in range(3)]
        assert len(set(decisions)) == 1, (
            f"ER4 VIOLATED: identical input produced different decisions: "
            f"{[d.name for d in decisions]}"
        )


# ===========================================================================
# ABC contract tests — the interfaces themselves
# ===========================================================================

class TestAbstractInterfaceContracts:
    """Verifies each ABC cannot be instantiated without a complete implementation."""

    def test_cognitive_node_is_abstract(self):
        from core.interfaces import AbstractCognitiveNode
        with pytest.raises(TypeError):
            AbstractCognitiveNode()

    def test_manifold_is_abstract(self):
        from core.interfaces import AbstractManifold
        with pytest.raises(TypeError):
            AbstractManifold()

    def test_entropic_router_is_abstract(self):
        from core.interfaces import AbstractEntropicRouter
        with pytest.raises(TypeError):
            AbstractEntropicRouter()

    def test_real_implementations_are_valid_subclasses(
        self, real_llm_node, real_jepa_node, real_gnn_node, calibrated_router
    ):
        from core.interfaces import AbstractCognitiveNode, AbstractManifold, AbstractEntropicRouter
        assert isinstance(real_llm_node, AbstractCognitiveNode)
        assert isinstance(real_jepa_node, AbstractCognitiveNode)
        assert isinstance(real_jepa_node, AbstractManifold)
        assert isinstance(real_gnn_node, AbstractCognitiveNode)
        assert isinstance(calibrated_router, AbstractEntropicRouter)
