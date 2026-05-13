"""
Tests for the Lár-JEPA cognitive components:
  - LatentKinematicState (data model + serialisation)
  - NBodyKinematicsJEPA (AbstractManifold implementation)
  - CognitiveNodeAdapter (bridge to Lár BaseNode)
  - EntropicVetoRouter (deterministic trajectory gate)
  - JEPA_DMN_Consolidation_Node (graceful degradation + write + recall)
"""
import sys
import os
from unittest.mock import MagicMock, patch

import pytest

# ------------------------------------------------------------------
# Path resolution
# Test file: lar_jepa/lar_jepa/tests/unit/test_jepa_components.py
# ../../..  → lar_jepa top-level (core/, spatial_kinematics_engine/, dmn_integration/)
# ../../src → lar_jepa/lar_jepa/src  (the embedded lar engine)
# ------------------------------------------------------------------
_HERE = os.path.dirname(__file__)
_JEPA_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_LAR_SRC   = os.path.abspath(os.path.join(_HERE, "..", "..", "src"))

for _p in [_JEPA_ROOT, _LAR_SRC]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lar.node import BaseNode
from lar.state import GraphState

from core.types import RouteDecision, ModelType, SignalType
from core.interfaces import AbstractCognitiveNode, AbstractManifold
from core.adapter import CognitiveNodeAdapter
from spatial_kinematics_engine.jepa_manifold import (
    LatentKinematicState,
    NBodyKinematicsJEPA,
)
from spatial_kinematics_engine.lar_trajectory_router import EntropicVetoRouter
from dmn_integration.consolidation_node import JEPA_DMN_Consolidation_Node


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_state(state_kw: float = 0.05) -> LatentKinematicState:
    return LatentKinematicState(
        timestep=0,
        n_body_tensors=[0.1, 0.2, 0.3, 0.4],
        spatial_mesh={"node_distances": [0.1, 0.2, 0.3, 0.4]},
        collision_entropy=state_kw,
    )


class _StubCogNode(AbstractCognitiveNode):
    """Minimal concrete AbstractCognitiveNode for adapter tests."""
    model_type = ModelType.CLASSICAL

    def encode(self, input_signal):
        return f"ctx:{input_signal}"

    def forward(self, context):
        return f"out:{context}"

    def decode(self, representation):
        return f"decoded:{representation}"

    @property
    def output_signal_type(self):
        return SignalType.STRUCTURED_DATA


class _StubManifold(AbstractManifold):
    """Minimal concrete AbstractManifold for adapter action_key tests."""

    def embed_context(self, raw_observation):
        return _make_state(0.1)

    def predict_target(self, context, action_vector):
        return _make_state(0.2)

    def entropic_loss(self, predicted_state):
        return 0.2

    def decode(self, representation):
        return {"result": "ok"}


# ==================================================================
# LatentKinematicState
# ==================================================================

class TestLatentKinematicState:
    def test_to_dict_contains_required_keys(self):
        state = _make_state(0.05)
        d = state.to_dict()
        for key in ("timestep", "n_body_tensors", "spatial_mesh",
                    "collision_entropy", "tau_deformation_coefficient",
                    "deformation_probability"):
            assert key in d, f"Missing key: {key}"

    def test_to_dict_values_match_fields(self):
        state = _make_state(0.4)
        d = state.to_dict()
        assert d["timestep"] == 0
        assert d["collision_entropy"] == pytest.approx(0.4)
        assert d["tau_deformation_coefficient"] == pytest.approx(0.95)

    def test_deformation_probability_formula(self):
        state = _make_state(0.4)
        expected = 0.4 * 0.95
        assert state.calculate_deformation() == pytest.approx(expected)
        assert state.to_dict()["deformation_probability"] == pytest.approx(expected)

    def test_to_dict_is_json_serialisable(self):
        import json
        d = _make_state(0.1).to_dict()
        # Should not raise
        json.dumps(d)


# ==================================================================
# NBodyKinematicsJEPA
# ==================================================================

class TestNBodyKinematicsJEPA:
    def setup_method(self):
        self.jepa = NBodyKinematicsJEPA(entropy_threshold=0.5)

    def test_embed_context_returns_latent_state(self):
        result = self.jepa.embed_context("raw_telemetry_data")
        assert isinstance(result, LatentKinematicState)

    def test_embed_context_sets_zero_timestep(self):
        result = self.jepa.embed_context("telemetry")
        assert result.timestep == 0

    def test_predict_target_increments_timestep(self):
        context = _make_state(0.05)
        next_state = self.jepa.predict_target(context, [1.0, 0.0, 0.0, 0.0])
        assert next_state.timestep == context.timestep + 1

    def test_predict_target_none_action_uses_zero(self):
        context = _make_state(0.05)
        # Should not raise — falls back to zero-action vector
        result = self.jepa.predict_target(context, None)
        assert isinstance(result, LatentKinematicState)

    def test_predict_target_returns_latent_state(self):
        context = _make_state(0.05)
        result = self.jepa.predict_target(context, [0.5, 0.5, 0.5, 0.5])
        assert isinstance(result, LatentKinematicState)

    def test_entropic_loss_from_latent_state(self):
        state = _make_state(0.7)
        loss = self.jepa.entropic_loss(state)
        assert loss == pytest.approx(0.7)

    def test_entropic_loss_from_dict(self):
        d = {"collision_entropy": 0.33}
        loss = self.jepa.entropic_loss(d)
        assert loss == pytest.approx(0.33)

    def test_entropic_loss_from_dict_missing_key_defaults(self):
        d = {}
        loss = self.jepa.entropic_loss(d)
        assert loss == pytest.approx(0.5)

    def test_entropic_loss_from_other_type_returns_float(self):
        loss = self.jepa.entropic_loss("some_string")
        assert isinstance(loss, float)

    def test_decode_latent_state_returns_dict(self):
        state = _make_state(0.1)
        result = self.jepa.decode(state)
        assert isinstance(result, dict)
        assert "collision_entropy" in result

    def test_decode_non_latent_passthrough(self):
        payload = {"already": "a dict"}
        result = self.jepa.decode(payload)
        assert result is payload

    def test_model_type_is_jepa(self):
        assert self.jepa.model_type == ModelType.JEPA

    def test_output_signal_type_is_latent_embedding(self):
        assert self.jepa.output_signal_type == SignalType.LATENT_EMBEDDING

    def test_encode_delegates_to_embed_context(self):
        result = self.jepa.encode("obs")
        assert isinstance(result, LatentKinematicState)

    def test_forward_delegates_to_predict_target(self):
        context = _make_state(0.05)
        result = self.jepa.forward(context)
        assert isinstance(result, LatentKinematicState)


# ==================================================================
# CognitiveNodeAdapter
# ==================================================================

class TestCognitiveNodeAdapter:
    def test_init_rejects_non_cognitive_node(self):
        with pytest.raises(ValueError, match="AbstractCognitiveNode"):
            CognitiveNodeAdapter(
                cognitive_node="not_a_node",
                input_key="x",
                output_key="y",
            )

    def test_init_rejects_empty_input_key(self):
        with pytest.raises(ValueError, match="input_key"):
            CognitiveNodeAdapter(
                cognitive_node=_StubCogNode(),
                input_key="",
                output_key="y",
            )

    def test_init_rejects_empty_output_key(self):
        with pytest.raises(ValueError, match="output_key"):
            CognitiveNodeAdapter(
                cognitive_node=_StubCogNode(),
                input_key="x",
                output_key="",
            )

    def test_execute_reads_input_and_writes_output(self):
        adapter = CognitiveNodeAdapter(
            cognitive_node=_StubCogNode(),
            input_key="my_input",
            output_key="my_output",
            next_node=None,
        )
        state = GraphState({"my_input": "hello"})
        returned = adapter.execute(state)

        assert state.get("my_output") is not None
        assert "decoded:" in state.get("my_output")
        assert returned is None

    def test_execute_missing_input_key_sets_error(self):
        adapter = CognitiveNodeAdapter(
            cognitive_node=_StubCogNode(),
            input_key="missing_key",
            output_key="out",
            next_node=None,
        )
        state = GraphState({})
        adapter.execute(state)

        assert state.get("last_error") is not None
        assert "missing_key" in state.get("last_error")

    def test_execute_returns_next_node(self):
        mock_next = MagicMock(spec=BaseNode)
        adapter = CognitiveNodeAdapter(
            cognitive_node=_StubCogNode(),
            input_key="x",
            output_key="y",
            next_node=mock_next,
        )
        state = GraphState({"x": "value"})
        returned = adapter.execute(state)
        assert returned is mock_next

    def test_execute_writes_signal_type_metadata(self):
        adapter = CognitiveNodeAdapter(
            cognitive_node=_StubCogNode(),
            input_key="x",
            output_key="y",
        )
        state = GraphState({"x": "value"})
        adapter.execute(state)
        assert state.get("__signal_type_y") is not None

    def test_execute_uses_predict_target_for_manifold_with_action_key(self):
        manifold = _StubManifold()
        adapter = CognitiveNodeAdapter(
            cognitive_node=manifold,
            input_key="obs",
            output_key="prediction",
            action_key="action",
        )
        state = GraphState({"obs": "raw", "action": [1.0, 0.0, 0.0]})
        adapter.execute(state)
        assert state.get("prediction") == {"result": "ok"}

    def test_execute_uses_forward_for_manifold_without_action_key(self):
        manifold = _StubManifold()
        adapter = CognitiveNodeAdapter(
            cognitive_node=manifold,
            input_key="obs",
            output_key="prediction",
        )
        state = GraphState({"obs": "raw"})
        adapter.execute(state)
        assert state.get("prediction") == {"result": "ok"}

    def test_model_type_exposed_on_adapter(self):
        adapter = CognitiveNodeAdapter(
            cognitive_node=_StubCogNode(),
            input_key="x",
            output_key="y",
        )
        assert adapter.model_type == ModelType.CLASSICAL


# ==================================================================
# EntropicVetoRouter
# ==================================================================

class TestEntropicVetoRouter:
    def test_commit_when_entropy_below_threshold(self):
        router = EntropicVetoRouter(entropy_threshold=0.85)
        state = _make_state(0.1)
        assert router.evaluate_state(state) == RouteDecision.COMMIT_TRAJECTORY

    def test_replan_when_entropy_above_threshold(self):
        router = EntropicVetoRouter(entropy_threshold=0.85)
        state = _make_state(0.9)
        assert router.evaluate_state(state) == RouteDecision.TRIGGER_REPLAN

    def test_commit_exactly_at_threshold_boundary(self):
        router = EntropicVetoRouter(entropy_threshold=0.5)
        state = _make_state(0.5)
        # 0.5 is NOT > 0.5, so COMMIT
        assert router.evaluate_state(state) == RouteDecision.COMMIT_TRAJECTORY

    def test_replan_just_above_threshold(self):
        router = EntropicVetoRouter(entropy_threshold=0.5)
        state = _make_state(0.51)
        assert router.evaluate_state(state) == RouteDecision.TRIGGER_REPLAN


# ==================================================================
# JEPA_DMN_Consolidation_Node — graceful degradation
# ==================================================================

class TestConsolidationNodeDegradation:
    """Tests the no-hippocampus degraded mode (ChromaDB unavailable)."""

    def _make_node_without_hippo(self):
        node = JEPA_DMN_Consolidation_Node.__new__(JEPA_DMN_Consolidation_Node)
        node._hippocampus = None
        return node

    def test_write_returns_false_without_hippocampus(self):
        node = self._make_node_without_hippo()
        result = node.write_trajectory_heuristic({
            "domain": "spatial",
            "outcome": "committed",
            "entropic_loss": 0.05,
        })
        assert result is False

    def test_recall_returns_empty_string_without_hippocampus(self):
        node = self._make_node_without_hippo()
        result = node.recall_heuristics("orbital insertion")
        assert result == ""

    def test_extract_heuristic_legacy_delegates_write(self):
        node = self._make_node_without_hippo()
        # Legacy method — should return False when no hippocampus
        result = node.extract_heuristic_from_trajectory({"domain": "test"})
        assert result is False

    def test_extract_heuristic_non_dict_wraps_and_returns_false(self):
        node = self._make_node_without_hippo()
        result = node.extract_heuristic_from_trajectory("a plain string")
        assert result is False


class TestConsolidationNodeWithHippocampus:
    """Tests the write/recall paths with a mocked hippocampus."""

    def _make_node_with_mock_hippo(self):
        node = JEPA_DMN_Consolidation_Node.__new__(JEPA_DMN_Consolidation_Node)
        mock_hippo = MagicMock()
        mock_hippo._generate_embedding.return_value = None  # journal-only path
        node._hippocampus = mock_hippo
        return node, mock_hippo

    def test_write_calls_save_memory(self):
        node, mock_hippo = self._make_node_with_mock_hippo()
        result = node.write_trajectory_heuristic({
            "domain": "spatial_kinematics",
            "outcome": "committed",
            "entropic_loss": 0.049,
            "action": [1.0, 0.0],
        })
        assert result is True
        mock_hippo.save_memory.assert_called_once()

    def test_write_uses_provided_embedding(self):
        node, mock_hippo = self._make_node_with_mock_hippo()
        embedding = [0.1, 0.2, 0.3]
        node.write_trajectory_heuristic(
            {"domain": "test", "outcome": "committed", "entropic_loss": 0.1},
            embedding=embedding,
        )
        call_args = mock_hippo.save_memory.call_args
        assert call_args[0][1] == embedding

    def test_write_includes_domain_in_summary(self):
        node, mock_hippo = self._make_node_with_mock_hippo()
        node.write_trajectory_heuristic({
            "domain": "orbital_mechanics",
            "outcome": "committed",
            "entropic_loss": 0.05,
        })
        summary = mock_hippo.save_memory.call_args[0][0]
        assert "orbital_mechanics" in summary

    def test_write_catches_save_exception_returns_false(self):
        node, mock_hippo = self._make_node_with_mock_hippo()
        mock_hippo.save_memory.side_effect = RuntimeError("ChromaDB down")
        mock_hippo._generate_embedding.return_value = [0.1]  # force save_memory path
        result = node.write_trajectory_heuristic({"domain": "test", "entropic_loss": 0.1})
        assert result is False

    def test_recall_calls_hippocampus_recall(self):
        node, mock_hippo = self._make_node_with_mock_hippo()
        mock_hippo.recall.return_value = "- Prior heuristic from Cycle N"
        result = node.recall_heuristics("orbital insertion", max_results=3)
        mock_hippo.recall.assert_called_once_with(query="orbital insertion", max_memories=3)
        assert "Prior heuristic" in result

    def test_recall_catches_exception_returns_empty(self):
        node, mock_hippo = self._make_node_with_mock_hippo()
        mock_hippo.recall.side_effect = RuntimeError("network error")
        result = node.recall_heuristics("query")
        assert result == ""
