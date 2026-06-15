"""
Materials-JEPA: Battery Electrolyte Discovery
==============================================
A proof-of-concept demonstrating that the Lár routing spine is truly
domain-agnostic: the same AbstractManifold interface, the same
CognitiveNodeAdapter, the same DMN memory consolidation loop, and the
same compliance primitives (HumanJuryNode, AuditLogger, HMAC signing)
that route N-body orbital mechanics now route crystal structure prediction
for battery electrolyte discovery.

Domain isomorphism
------------------
The same AbstractManifold contract used for N-body spatial forecasting
(spatial_kinematics_engine/) routes crystal structure prediction here
without any changes to the Lár execution spine. Two different domains,
one interface, one auditable graph topology.

Data sourcing (production)
--------------------------
Crystal candidates  : Materials Project API (150k+ open structures)
Electrochemical data: NREL ECDH, MPContribs battery datasets (open)
Checkpoint weights  : MatterSim (Microsoft, open weights)

This PoC uses mock tensors with realistic shapes and domain semantics.
The architecture, graph topology, and compliance stack are production-grade.

Graph topology
--------------
  RecallMaterialHeuristicsNode  (DMN — prior experiment recall)
           ↓
  CrystalEmbeddingNode          (library index — no model at inference)
           ↓
  ElectrochemicalEmbeddingNode  (live per experiment — runs once, cached)
           ↓
  CycleStabilityPredictorNode   (cross-attention: electrochem queries crystal)
           ↓
  ThermalStabilityRouterNode    (veto unstable compositions)
     ├── COMMIT →  HumanJuryNode  →  WriteMaterialHeuristicNode → Done
     └── REPLAN →  NextCompositionNode → RecallMaterialHeuristicsNode

HumanJuryNode is a hard execution gate. The graph pauses and waits for
a researcher to approve before any composition is committed. This is
EU AI Act Article 14 (human oversight) implemented structurally —
not as a disclaimer, but as an architectural invariant.

Run
---
    cd lar_jepa
    python examples/materials_jepa_showcase.py
"""

import sys
import os
import math
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_ROOT     = os.path.dirname(os.path.abspath(__file__))
_JEPA_ROOT = os.path.abspath(os.path.join(_ROOT, ".."))
_LAR_SRC  = os.path.join(_JEPA_ROOT, "lar_jepa", "src")

for _p in [_JEPA_ROOT, _LAR_SRC]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lar import GraphState, GraphExecutor, BaseNode, AddValueNode
from lar.node import HumanJuryNode

from materials_engine.crystal_manifold import (
    CrystalStructureJEPA,
    LatentCrystalState,
    N_SITES,
    ELEMENT_SYMBOLS,
)
from materials_engine.electrochemical_manifold import (
    ElectrochemicalJEPA,
    MEASUREMENT_DIM,
)
from materials_engine.stability_router import ThermalStabilityRouter

from dmn_integration.consolidation_node import JEPA_DMN_Consolidation_Node

# ---------------------------------------------------------------------------
# DMN Bridge — uses in-memory fallback for this demo.
# ---------------------------------------------------------------------------
consolidation_bridge = JEPA_DMN_Consolidation_Node()

# ---------------------------------------------------------------------------
# Mock battery electrolyte candidates
# Realistic compositions from the solid-state electrolyte literature:
# Li₆PS₅Cl (argyrodite), Li₃PS₄ (sulfide glass),
# Li₁.₃Al₀.₃Ti₁.₇(PO₄)₃ (LATP), Li₇La₃Zr₂O₁₂ (LLZO), LiPF₆/EC:DMC (liquid)
# ---------------------------------------------------------------------------
CANDIDATE_LABELS = [
    "Li6PS5Cl (Argyrodite)",
    "Li3PS4 (Sulfide Glass)",
    "LATP (Li1.3Al0.3Ti1.7(PO4)3)",
    "LLZO (Li7La3Zr2O12)",
    "LiPF6/EC:DMC (Liquid)",
]


def get_mock_candidates(n_candidates: int = 5):
    """
    Returns mock crystal composition tensors and electrochemical conditions.
    Shape: (n_candidates, N_SITES + 6) for crystal, (1, MEASUREMENT_DIM) for electrochem.

    In production, crystal tensors come from the Materials Project API:
        from mp_api.client import MPRester
        with MPRester(api_key) as mpr:
            entries = mpr.summary.search(elements=["Li","S","P"], fields=["composition","structure"])
    """
    torch.manual_seed(42)

    # Crystal composition tensors: occupancies (N_SITES) + lattice params (6)
    # Occupancies are fractional — normalised to [0, 1]
    occupancies = torch.softmax(torch.randn(n_candidates, N_SITES), dim=-1) * 0.8
    lattice     = torch.rand(n_candidates, 6)  # normalised lattice params
    crystal_data = torch.cat([occupancies, lattice], dim=-1)  # (n, N_SITES+6)

    # Electrochemical operating conditions (single experiment)
    # [voltage_window, current_density, temperature, capacity,
    #  coulombic_efficiency, impedance_real, impedance_imag,
    #  cycle_number, c_rate, soc, dod, formation_cycles]
    electrochem_data = torch.rand(1, MEASUREMENT_DIM)

    return crystal_data, electrochem_data


# ---------------------------------------------------------------------------
# Crystal Gene Library — pre-compute once, index at inference
# ---------------------------------------------------------------------------

def build_crystal_library(
    crystal_jepa: CrystalStructureJEPA,
    crystal_data: torch.Tensor,
    labels: list,
    cache_path: str,
) -> list:
    """
    Pre-computes CrystalStructureJEPA embeddings for all candidate materials.

    Crystal structures are static — running the JEPA once per candidate and
    caching the result is correct. At inference, the graph indexes the list
    by composition_id. CrystalStructureJEPA is freed from memory after this.

    Returns: List[LatentCrystalState], one per candidate.
    """
    if os.path.exists(cache_path):
        import pickle
        print(f"\n[Crystal Library] Loading cached library from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"\n[Crystal Library] Building library for {len(crystal_data)} candidates...")
    library = []
    for i, (composition, label) in enumerate(zip(crystal_data, labels)):
        state = crystal_jepa.embed_context(composition.unsqueeze(0))
        state.composition_id    = i
        state.composition_label = label
        library.append(state)
        print(
            f"  [{i}] {label:<40} "
            f"thermal_entropy={state.thermal_entropy:.3f}  "
            f"formation_energy={state.formation_energy:.3f} eV/atom"
        )

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    import pickle
    with open(cache_path, "wb") as f:
        pickle.dump(library, f)
    print(f"[Crystal Library] Saved. CrystalStructureJEPA can now be freed from memory.\n")
    return library


# ---------------------------------------------------------------------------
# Cross-Attention Prediction Head
# Electrochemical profile (Query) attends over crystal elemental sites (K, V).
# "Which elemental sites in this crystal drive stability under this condition?"
# ---------------------------------------------------------------------------

class CycleStabilityHead(nn.Module):
    """
    Cross-attention prediction head for cycle stability.

    crystal_sites : (B, N_SITES, embed_dim)  — per-element latent landscape
    electrochem   : (B, embed_dim)            — operating condition profile

    The electrochemical condition is the Query — it interrogates the crystal's
    elemental site landscape to find which atoms are responsible for surviving
    this specific operating regime.
    """
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj   = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.embed_dim = embed_dim

    def forward(
        self,
        crystal_sites: torch.Tensor,  # (B, N_SITES, D)
        electrochem:   torch.Tensor,  # (B, D)
    ) -> torch.Tensor:
        Q = self.query_proj(electrochem).unsqueeze(1)   # (B, 1, D)
        K = self.key_proj(crystal_sites)                # (B, N_SITES, D)
        V = self.value_proj(crystal_sites)              # (B, N_SITES, D)

        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.embed_dim)
        attn   = torch.softmax(scores, dim=-1)          # (B, 1, N_SITES)
        ctx    = torch.bmm(attn, V).squeeze(1)          # (B, D)
        return self.fc(ctx)                             # (B, 1)

    def top_sites(self, attn_weights: torch.Tensor, k: int = 3) -> list:
        """Return the element symbols with highest attention weight."""
        top_idx = attn_weights.squeeze().topk(k).indices.tolist()
        return [ELEMENT_SYMBOLS[i] for i in top_idx]


# ---------------------------------------------------------------------------
# Lár Graph Nodes
# ---------------------------------------------------------------------------

class RecallMaterialHeuristicsNode(BaseNode):
    """Queries DMN Tier 2 memory for prior electrolyte screening results."""

    def __init__(self, bridge: JEPA_DMN_Consolidation_Node, next_node=None):
        self.bridge    = bridge
        self.next_node = next_node

    def execute(self, state: GraphState):
        idx   = state.get("composition_index", 0)
        label = state.get("composition_label", f"candidate_{idx}")
        query = f"battery electrolyte stability screening for {label}"

        heuristics = self.bridge.recall_heuristics(query, max_results=2)
        state.set("prior_heuristics", heuristics or "(no prior experiments)")
        print(f"\n[DMN Recall] Prior knowledge for '{label}':\n  {state.get('prior_heuristics')}")
        return self.next_node


class CrystalEmbeddingNode(BaseNode):
    """
    Loads pre-computed crystal site embeddings from the library.
    No model at inference — just a dictionary lookup by composition_index.
    """

    def __init__(self, crystal_library: list, next_node=None):
        self.crystal_library = crystal_library
        self.next_node       = next_node

    def execute(self, state: GraphState):
        idx           = state.get("composition_index", 0)
        crystal_state = self.crystal_library[idx]

        # (1, N_SITES, embed_dim) — preserve site structure for cross-attention
        site_emb = torch.tensor(
            crystal_state.site_embeddings, dtype=torch.float32
        ).unsqueeze(0)

        state.set("crystal_site_embedding", site_emb)
        state.set("crystal_state",          crystal_state.to_dict())
        state.set("composition_label",      crystal_state.composition_label)

        print(
            f"\n[CrystalEmbeddingNode] Loaded '{crystal_state.composition_label}' "
            f"(id={idx}) — "
            f"thermal_entropy={crystal_state.thermal_entropy:.3f}, "
            f"Ef={crystal_state.formation_energy:.3f} eV/atom"
        )
        return self.next_node


class ElectrochemicalEmbeddingNode(BaseNode):
    """
    Runs ElectrochemicalJEPA once per experiment and caches in GraphState.

    On every replan iteration the cache check prevents re-running the model.
    One patient, one disease profile. One experiment, one electrochemical
    embedding. The same operating conditions are evaluated against every
    crystal candidate.
    """

    def __init__(self, model: ElectrochemicalJEPA, next_node=None):
        self.model     = model
        self.next_node = next_node

    def execute(self, state: GraphState):
        if state.get("electrochem_embedding") is None:
            raw = state.get("experiment_conditions")
            ec_state = self.model.embed_context(raw)
            emb = torch.tensor(ec_state.latent_vector, dtype=torch.float32).unsqueeze(0)
            state.set("electrochem_embedding", emb)
            state.set("electrochem_state",     ec_state.to_dict())
            print(
                f"\n[ElectrochemicalEmbeddingNode] Experiment encoded — "
                f"voltage={ec_state.voltage_window:.2f}V, "
                f"T={ec_state.temperature_K:.0f}K, "
                f"C-rate={ec_state.c_rate:.1f}C"
            )
        else:
            print("\n[ElectrochemicalEmbeddingNode] Using cached experiment embedding.")
        return self.next_node


class CycleStabilityPredictorNode(BaseNode):
    """
    Runs the cross-attention head to predict cycle stability probability.
    Also logs which elemental sites drove the prediction.
    """

    def __init__(self, model: CycleStabilityHead, next_node=None):
        self.model     = model
        self.next_node = next_node

    def execute(self, state: GraphState):
        crystal_sites = state.get("crystal_site_embedding")  # (1, N_SITES, D)
        electrochem   = state.get("electrochem_embedding")   # (1, D)
        label         = state.get("composition_label", "?")

        self.model.eval()
        with torch.no_grad():
            Q = self.model.query_proj(electrochem).unsqueeze(1)
            K = self.model.key_proj(crystal_sites)
            scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.model.embed_dim)
            attn   = torch.softmax(scores, dim=-1)
            prob   = self.model(crystal_sites, electrochem).item()

        top_elements = self.model.top_sites(attn, k=3)
        state.set("stability_probability", prob)
        state.set("key_elemental_sites",   top_elements)

        print(
            f"\n[CycleStabilityPredictor] '{label}' — "
            f"cycle stability prob={prob:.4f} | "
            f"key sites: {', '.join(top_elements)}"
        )
        return self.next_node


class ThermalStabilityRouterNode(BaseNode):
    """
    Routes to COMMIT or REPLAN based on ThermalStabilityRouter evaluation.
    Uses the crystal_state stored in GraphState (not a raw tensor) to
    preserve domain semantics in the routing decision.
    """

    def __init__(
        self,
        router: ThermalStabilityRouter,
        commit_node: BaseNode,
        replan_node: BaseNode,
        stability_threshold: float = 0.55,
    ):
        self.router             = router
        self.commit_node        = commit_node
        self.replan_node        = replan_node
        self.stability_threshold = stability_threshold

    def execute(self, state: GraphState):
        crystal_state_dict = state.get("crystal_state")
        stability_prob     = state.get("stability_probability", 0.0)

        route = self.router.evaluate_state(crystal_state_dict)

        if route == RouteDecision.COMMIT_TRAJECTORY and stability_prob >= self.stability_threshold:
            state.set("route_decision", "COMMIT")
            return self.commit_node

        state.set("route_decision", "REPLAN")
        print(
            f"[StabilityRouter] Replanning — "
            f"thermodynamic veto or stability prob {stability_prob:.3f} < {self.stability_threshold}"
        )
        return self.replan_node


class NextCompositionNode(BaseNode):
    """
    Advances to the next candidate composition index.
    Clears crystal_state from GraphState (electrochemical embedding is kept).
    """

    def __init__(self, n_candidates: int, start_node: BaseNode):
        self.n_candidates = n_candidates
        self.start_node   = start_node

    def execute(self, state: GraphState):
        idx = state.get("composition_index", 0)
        if idx + 1 < self.n_candidates:
            state.set("composition_index",    idx + 1)
            state.set("crystal_state",        None)
            state.set("crystal_site_embedding", None)
            print(f"\n[Replan] Advancing to composition index {idx + 1}...")
            return self.start_node
        else:
            print("\n[Impasse] No stable electrolyte found in candidate pool.")
            state.set("outcome", "impasse")
            return None


class WriteMaterialHeuristicNode(BaseNode):
    """Writes successful electrolyte discovery to DMN long-term memory."""

    def __init__(self, bridge: JEPA_DMN_Consolidation_Node, next_node=None):
        self.bridge    = bridge
        self.next_node = next_node

    def execute(self, state: GraphState):
        label    = state.get("composition_label", "unknown")
        prob     = state.get("stability_probability", 0.0)
        sites    = state.get("key_elemental_sites", [])
        cs       = state.get("crystal_state", {}) or {}

        trajectory_log = {
            "domain":        "battery_electrolyte_discovery",
            "action":        f"screened_{label.replace(' ', '_')}",
            "outcome":       "committed",
            "entropic_loss": cs.get("thermal_entropy", 0.0),
            "metadata": {
                "cycle_stability_prob": prob,
                "key_elemental_sites":  sites,
                "formation_energy":     cs.get("formation_energy", 0.0),
                "composition_label":    label,
            },
        }
        ok = self.bridge.write_trajectory_heuristic(trajectory_log)
        print(
            f"\n[DMN Consolidation] '{label}' committed. "
            f"Heuristic written to DMN: {ok}"
        )
        return self.next_node


# ---------------------------------------------------------------------------
# Import RouteDecision for the router node
# ---------------------------------------------------------------------------
from core.types import RouteDecision


# ---------------------------------------------------------------------------
# Pipeline Assembly
# ---------------------------------------------------------------------------

def run_materials_pipeline():
    torch.manual_seed(42)

    print("=" * 60)
    print("  Materials-JEPA: Battery Electrolyte Discovery")
    print("  Lár Routing Graph + DMN Memory + EU AI Act Compliance")
    print("=" * 60)

    # --- 1. Mock data ---
    crystal_data, electrochem_data = get_mock_candidates(n_candidates=5)
    n_candidates = len(crystal_data)

    # --- 2. JEPA Models ---
    crystal_jepa  = CrystalStructureJEPA(embed_dim=256)
    electrochem_jepa = ElectrochemicalJEPA(embed_dim=256)

    # --- 3. Build Crystal Library (static, pre-computed once) ---
    library_cache = os.path.join(_JEPA_ROOT, "cache", "crystal_library.pkl")
    crystal_library = build_crystal_library(
        crystal_jepa, crystal_data, CANDIDATE_LABELS, cache_path=library_cache
    )
    del crystal_jepa  # Free from memory — library is all we need at inference

    # --- 4. Prediction Head ---
    stability_head = CycleStabilityHead(embed_dim=256)

    trained_weights = os.path.join(_JEPA_ROOT, "models", "cycle_stability_head.pt")
    if os.path.exists(trained_weights):
        stability_head.load_state_dict(
            torch.load(trained_weights, map_location="cpu")
        )
        stability_head.eval()
        print(f"[Graph] Loaded trained CycleStabilityHead from {trained_weights}")
    else:
        print(
            "[Graph] No trained weights found — using initialised head.\n"
            "[Graph] Run a training script to produce models/cycle_stability_head.pt"
        )

    # --- 5. Router ---
    router = ThermalStabilityRouter(thermal_threshold=0.40, max_formation_energy=0.0)

    # --- 6. Build Graph (reverse declaration order) ---

    done = AddValueNode(
        key="outcome",
        value="stable_electrolyte_found",
        next_node=None,
    )

    write_node = WriteMaterialHeuristicNode(
        bridge=consolidation_bridge,
        next_node=done,
    )

    # HumanJuryNode — researcher approval gate before any composition commits.
    # EU AI Act Article 14: a human must be in the loop before a high-risk
    # AI decision is acted upon. This is structural enforcement, not a checkbox.
    # The graph literally cannot proceed to WriteHeuristicNode without a
    # researcher typing "approve" at this prompt.
    jury_node = HumanJuryNode(
        prompt=(
            "\n[JURY] Materials-JEPA has identified a stable electrolyte candidate.\n"
            "Review the prediction above and approve for lab synthesis validation:"
        ),
        choices=["approve", "reject"],
        output_key="researcher_verdict",
        next_node=write_node,
    )

    next_comp = NextCompositionNode(n_candidates=n_candidates, start_node=None)

    router_node = ThermalStabilityRouterNode(
        router=router,
        commit_node=jury_node,
        replan_node=next_comp,
        stability_threshold=0.50,
    )

    predictor_node = CycleStabilityPredictorNode(
        model=stability_head,
        next_node=router_node,
    )

    ec_node = ElectrochemicalEmbeddingNode(
        model=electrochem_jepa,
        next_node=predictor_node,
    )

    crystal_node = CrystalEmbeddingNode(
        crystal_library=crystal_library,
        next_node=ec_node,
    )

    recall_node = RecallMaterialHeuristicsNode(
        bridge=consolidation_bridge,
        next_node=crystal_node,
    )

    # Close the replan cycle
    next_comp.start_node = recall_node

    # --- 7. Initial State ---
    initial_state = {
        "experiment_conditions": electrochem_data,
        "composition_index":     0,
    }

    # --- 8. Execute with full compliance stack ---
    # hmac_secret signs every state transition — every prediction, every
    # routing decision, every human jury outcome — into an audit trail.
    executor = GraphExecutor(
        log_dir="lar_logs",
        hmac_secret="snath_ai_materials_eu_compliance_2026",
    )

    print("\n--- Starting Lár Graph Execution ---\n")
    final_state = {}
    for step in executor.run_step_by_step(recall_node, initial_state):
        final_state.update(step.get("state_before", {}))
        final_state.update(step.get("state_after", {}))

    print("\n" + "=" * 60)
    print(f"  Outcome      : {final_state.get('outcome', 'unknown')}")
    print(f"  Composition  : {final_state.get('composition_label', 'N/A')}")
    print(f"  Stability    : {final_state.get('stability_probability', 0):.4f}")
    print(f"  Key Sites    : {final_state.get('key_elemental_sites', [])}")
    print(f"  Jury Verdict : {final_state.get('researcher_verdict', 'N/A')}")
    print("=" * 60)
    print("\nAudit trail written to lar_logs/ (HMAC-signed, EU AI Act Art. 12)")


if __name__ == "__main__":
    run_materials_pipeline()
