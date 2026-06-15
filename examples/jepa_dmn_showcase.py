"""
Lár-JEPA × DMN Showcase
========================
Demonstrates the full cognitive stack:

    AdaptiveNode (Lár v2.1.0)
        → CognitiveNodeAdapter (wraps NBodyKinematicsJEPA)
        → EntropicRouterNode (COMMIT / REPLAN / IMPASSE)
        → WriteHeuristicNode (ingests trajectory into DMN Tier 1)
        → RecallNode (reads prior heuristics via DMN recall())

Architecture:

    Lár graph engine        — deterministic DAG executor
    lar_jepa interfaces     — AbstractManifold, AbstractContextBridge
    NBodyKinematicsJEPA     — reference JEPA world model (spatial forecasting)
    JEPA_DMN_Consolidation_Node — bridges JEPA to any AbstractDMN implementation
                                  (uses in-memory fallback when no DMN provided)

Run:
    cd lar_jepa
    python examples/jepa_dmn_showcase.py

No GPU, no cloud APIs required. All stubs produce deterministic output.
"""

import sys
import os

# ---------------------------------------------------------------------------
# Path bootstrap — add the embedded lar engine and lar_jepa root to sys.path
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LAR_SRC = os.path.join(_ROOT, "lar_jepa", "src")
if _LAR_SRC not in sys.path:
    sys.path.insert(0, _LAR_SRC)

from lar import (
    GraphState,
    GraphExecutor,
    BaseNode,
    RouterNode,
    AddValueNode,
    AuditLogger,
)
from lar.node import ToolNode

sys.path.insert(0, _ROOT)
from core.adapter import CognitiveNodeAdapter
from core.types import RouteDecision
from spatial_kinematics_engine.jepa_manifold import NBodyKinematicsJEPA
from dmn_integration.consolidation_node import JEPA_DMN_Consolidation_Node


# ---------------------------------------------------------------------------
# 1.  DMN bridge — uses in-memory fallback for this demo.
#     Wire a concrete AbstractDMN subclass for persistent storage:
#         from my_domain.dmn import MyDomainDMN
#         consolidation = JEPA_DMN_Consolidation_Node(dmn=MyDomainDMN())
# ---------------------------------------------------------------------------
consolidation = JEPA_DMN_Consolidation_Node()


# ---------------------------------------------------------------------------
# 2.  Recall node — reads prior JEPA heuristics from DMN at planning start
# ---------------------------------------------------------------------------
class RecallHeuristicsNode(BaseNode):
    """
    Queries the DMN Tier 2 semantic memory for prior committed trajectories
    relevant to the current planning context. Writes results to state["prior_heuristics"].
    """
    def __init__(self, bridge: JEPA_DMN_Consolidation_Node, next_node=None):
        self.bridge = bridge
        self.next_node = next_node

    def execute(self, state: GraphState):
        query = state.get("planning_context", "spatial trajectory planning")
        heuristics = self.bridge.recall_heuristics(query, max_results=3)
        state.set("prior_heuristics", heuristics or "(no prior heuristics)")
        print(f"\n[RecallNode] Prior heuristics:\n  {state.get('prior_heuristics')}")
        return self.next_node


# ---------------------------------------------------------------------------
# 3.  Entropic router — routes on JEPA predicted state entropy
# ---------------------------------------------------------------------------
class EntropicRouterNode(BaseNode):
    """
    Reads the JEPA predicted state from state["jepa_prediction"] and routes:

        collision_entropy < threshold  →  "commit"
        threshold ≤ entropy < 0.9     →  "replan"
        entropy ≥ 0.9                 →  "impasse"
    """
    def __init__(self, jepa: NBodyKinematicsJEPA, commit_node, replan_node, impasse_node):
        self.jepa = jepa
        self.commit_node = commit_node
        self.replan_node = replan_node
        self.impasse_node = impasse_node

    def execute(self, state: GraphState):
        predicted = state.get("jepa_prediction")
        entropy = self.jepa.entropic_loss(predicted) if predicted else 1.0
        state.set("entropic_loss", entropy)

        if entropy < self.jepa.entropy_threshold:
            decision = RouteDecision.COMMIT_TRAJECTORY
        elif entropy < 0.9:
            decision = RouteDecision.TRIGGER_REPLAN
        else:
            decision = RouteDecision.STRUCTURAL_IMPASSE

        state.set("route_decision", decision.value)
        print(f"\n[EntropicRouter] entropy={entropy:.4f} → {decision.name}")
        return {
            RouteDecision.COMMIT_TRAJECTORY: self.commit_node,
            RouteDecision.TRIGGER_REPLAN:    self.replan_node,
            RouteDecision.STRUCTURAL_IMPASSE: self.impasse_node,
        }[decision]


# ---------------------------------------------------------------------------
# 4.  Consolidation node — ingests committed trajectory into DMN Tier 1
# ---------------------------------------------------------------------------
class WriteHeuristicNode(BaseNode):
    """
    After COMMIT_TRAJECTORY, ingests the successful trajectory into the
    DMN Tier 1 episodic queue so future planning cycles can retrieve it
    as warm context via recall().
    """
    def __init__(self, bridge: JEPA_DMN_Consolidation_Node, next_node=None):
        self.bridge = bridge
        self.next_node = next_node

    def execute(self, state: GraphState):
        trajectory_log = {
            "domain":          "spatial_kinematics",
            "action":          state.get("action_vector"),
            "predicted_state": str(state.get("jepa_prediction")),
            "entropic_loss":   state.get("entropic_loss", 0.0),
            "outcome":         "committed",
            "metadata":        {"scenario": state.get("scenario", "default")},
        }
        ok = self.bridge.write_trajectory_heuristic(trajectory_log)
        state.set("heuristic_persisted", ok)
        print(f"\n[WriteHeuristic] Trajectory written to DMN: {ok}")
        return self.next_node


# ---------------------------------------------------------------------------
# 5.  Terminal nodes
# ---------------------------------------------------------------------------
class ReplanNode(BaseNode):
    def __init__(self, next_node=None):
        self.next_node = next_node

    def execute(self, state: GraphState):
        print("\n[ReplanNode] Entropy too high — replanning with modified action vector.")
        current_action = state.get("action_vector", [0.1, 0.2, 0.3, 0.4])
        state.set("action_vector", [a * 0.5 for a in current_action])
        state.set("replan_triggered", True)
        return self.next_node


class ImpasSeNode(BaseNode):
    def __init__(self):
        self.next_node = None

    def execute(self, state: GraphState):
        print("\n[ImpasSeNode] STRUCTURAL IMPASSE — no valid trajectory found. Halting.")
        state.set("outcome", "impasse")
        return None


# ---------------------------------------------------------------------------
# 6.  Wire the graph
# ---------------------------------------------------------------------------
def build_graph(jepa: NBodyKinematicsJEPA):
    # Terminal nodes
    done       = AddValueNode(key="outcome", value="committed", next_node=None)
    impasse    = ImpasSeNode()

    # Write committed heuristic to DMN, then mark done
    write_node = WriteHeuristicNode(bridge=consolidation, next_node=done)

    # Replan — reduce action magnitude and mark for retry
    replan_node = ReplanNode(next_node=None)   # in a real system, loops back to JEPA

    # Entropic router — forward reference, set next_node after construction
    router = EntropicRouterNode(
        jepa=jepa,
        commit_node=write_node,
        replan_node=replan_node,
        impasse_node=impasse,
    )

    # JEPA adapter — wraps the AbstractManifold for use inside the Lár graph
    jepa_adapter = CognitiveNodeAdapter(
        cognitive_node=jepa,
        input_key="raw_telemetry",
        output_key="jepa_prediction",
        next_node=router,
        action_key="action_vector",
    )

    # Recall prior heuristics from DMN before planning
    recall_node = RecallHeuristicsNode(bridge=consolidation, next_node=jepa_adapter)

    return recall_node   # entry point


# ---------------------------------------------------------------------------
# 7.  Run two planning cycles to demonstrate memory persistence
# ---------------------------------------------------------------------------
def run_scenario(label: str, telemetry: dict, action_vector: list, jepa: NBodyKinematicsJEPA):
    print(f"\n{'='*60}")
    print(f"  SCENARIO: {label}")
    print(f"{'='*60}")

    initial = {
        "planning_context": f"N-body spatial forecasting — {label}",
        "scenario":         label,
        "raw_telemetry":    telemetry,
        "action_vector":    action_vector,
    }

    entry = build_graph(jepa)
    executor = GraphExecutor()
    steps = list(executor.run_step_by_step(entry, initial))

    # Final state is accumulated across all steps
    final_state = steps[-1]["state_before"] if steps else {}
    for step in steps:
        final_state.update(step.get("state_before", {}))

    outcome   = final_state.get("outcome", final_state.get("route_decision", "unknown"))
    persisted = final_state.get("heuristic_persisted", False)
    replan    = final_state.get("replan_triggered", False)

    print(f"\n  ── Result: outcome={outcome} | persisted={persisted} | replan={replan}")
    return final_state


if __name__ == "__main__":
    jepa = NBodyKinematicsJEPA(model_dim=768, entropy_threshold=0.5)

    # Cycle 1: low-entropy scenario — should commit and write to DMN
    run_scenario(
        label="Orbital insertion — stable trajectory",
        telemetry={"bodies": 3, "t": 0, "coords": [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]},
        action_vector=[0.01, 0.02, -0.01, 0.005],
        jepa=jepa,
    )

    # Cycle 2: same scenario — DMN should now recall the heuristic from Cycle 1
    run_scenario(
        label="Orbital insertion — second attempt (warm context)",
        telemetry={"bodies": 3, "t": 1, "coords": [[1.1, 0.1], [0.1, 1.1], [-0.9, 0.1]]},
        action_vector=[0.01, 0.02, -0.01, 0.005],
        jepa=jepa,
    )

    print("\n✅ Showcase complete.")
    print("   Heuristics are stored in-memory for this demo (not persisted).")
    print("   Wire a concrete AbstractDMN subclass for durable storage.\n")
