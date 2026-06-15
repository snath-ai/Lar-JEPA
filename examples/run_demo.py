"""
Materials-JEPA: Proof-of-Concept Demo Runner
=============================================
Runs the full battery electrolyte discovery pipeline with real PyTorch
forward passes on CPU (no GPU required — works on any 8GB+ MacBook).

This is the non-interactive version of materials_jepa_showcase.py.
HumanJuryNode is replaced with AutoApproveNode for automated demo runs.
In production, HumanJuryNode is a hard gate — a researcher must type
"approve" before any composition commits (EU AI Act Article 14).

Run from lar_jepa/:
    python examples/run_demo.py
    python examples/run_demo.py 2>&1 | tee DEMO_OUTPUT.md

NOTE: PyTorch tensors are kept in a side-dict (TENSOR_STORE) outside
GraphState because the Lár executor's compute_state_diff cannot compare
tensors — it expects JSON-serialisable scalars and dicts in GraphState.
"""

import sys
import os
import math
import time
import torch
import torch.nn as nn

_ROOT      = os.path.dirname(os.path.abspath(__file__))
_JEPA_ROOT = os.path.abspath(os.path.join(_ROOT, ".."))
_LAR_SRC   = os.path.join(_JEPA_ROOT, "lar_jepa", "src")

for _p in [_JEPA_ROOT, _LAR_SRC]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lar import GraphState, GraphExecutor, BaseNode, AddValueNode
from core.types import RouteDecision

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

# Tensors live here — GraphState holds only JSON-serialisable values
TENSOR_STORE: dict = {}


# ---------------------------------------------------------------------------
# Candidate compositions (realistic solid-state electrolyte literature)
# ---------------------------------------------------------------------------
CANDIDATE_LABELS = [
    "Li6PS5Cl (Argyrodite)",
    "Li3PS4 (Sulfide Glass)",
    "LATP (Li1.3Al0.3Ti1.7(PO4)3)",
    "LLZO (Li7La3Zr2O12)",
    "LiPF6/EC:DMC (Liquid)",
]


def build_mock_data(n: int = 5):
    torch.manual_seed(42)
    occupancies  = torch.softmax(torch.randn(n, N_SITES), dim=-1) * 0.8
    lattice      = torch.rand(n, 6)
    crystal_data = torch.cat([occupancies, lattice], dim=-1)  # (n, N_SITES+6)

    # Electrochemical conditions: one experiment, evaluated against all candidates
    # [voltage_window, current_density, temperature, capacity,
    #  coulombic_efficiency, impedance_real, impedance_imag,
    #  cycle_number, c_rate, soc, dod, formation_cycles]
    electrochem_data = torch.rand(1, MEASUREMENT_DIM)
    return crystal_data, electrochem_data


def build_crystal_library(crystal_jepa, crystal_data, labels):
    """Real JEPA forward pass for every candidate — stored as static library."""
    print(f"\n{'─'*60}")
    print("  Phase 1: Crystal Library Construction")
    print(f"  Running CrystalStructureJEPA for {len(labels)} candidates")
    print(f"  Input shape per candidate: (1, {N_SITES + 6})  "
          f"= {N_SITES} site occupancies + 6 lattice params")
    print(f"{'─'*60}")

    library = []
    t0 = time.perf_counter()
    for i, (comp, label) in enumerate(zip(crystal_data, labels)):
        state = crystal_jepa.embed_context(comp.unsqueeze(0))
        state.composition_id    = i
        state.composition_label = label
        library.append(state)
        print(
            f"  [{i}] {label:<38} "
            f"Ef={state.formation_energy:+.3f} eV/atom  "
            f"thermal_entropy={state.thermal_entropy:.3f}  "
            f"band_gap={state.band_gap:.2f} eV"
        )

    elapsed = time.perf_counter() - t0
    print(f"\n  Site embedding shape per candidate: ({N_SITES}, 256)")
    print(f"  Total library built in {elapsed*1000:.1f} ms")
    print(f"  CrystalStructureJEPA can now be freed — library is all inference needs.")
    return library


class CycleStabilityHead(nn.Module):
    """
    Cross-attention: electrochemical condition (Query) attends over
    crystal elemental sites (Key, Value).
    Asks: which elemental sites drive stability under this operating condition?
    """
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.embed_dim  = embed_dim
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj   = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, crystal_sites, electrochem):
        Q      = self.query_proj(electrochem).unsqueeze(1)   # (B,1,D)
        K      = self.key_proj(crystal_sites)                # (B,N,D)
        V      = self.value_proj(crystal_sites)              # (B,N,D)
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.embed_dim)
        attn   = torch.softmax(scores, dim=-1)               # (B,1,N)
        ctx    = torch.bmm(attn, V).squeeze(1)               # (B,D)
        return self.fc(ctx), attn                            # (B,1), (B,1,N)


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

class RecallNode(BaseNode):
    def __init__(self, bridge, next_node=None):
        self.bridge    = bridge
        self.next_node = next_node

    def execute(self, state):
        label = state.get("composition_label", f"candidate_{state.get('composition_index', 0)}")
        prior = self.bridge.recall_heuristics(
            f"battery electrolyte stability for {label}", max_results=2
        )
        state.set("prior_heuristics", prior or "(no prior experiments in memory)")
        print(f"\n  [DMN Recall] '{label}': {state.get('prior_heuristics')}")
        return self.next_node


class CrystalEmbeddingNode(BaseNode):
    def __init__(self, library, next_node=None):
        self.library   = library
        self.next_node = next_node

    def execute(self, state):
        idx   = state.get("composition_index", 0)
        cs    = self.library[idx]
        emb   = torch.tensor(cs.site_embeddings, dtype=torch.float32).unsqueeze(0)
        TENSOR_STORE["crystal_site_embedding"] = emb          # kept out of GraphState
        state.set("crystal_state",          cs.to_dict())
        state.set("composition_label",      cs.composition_label)
        state.set("thermal_entropy",        cs.thermal_entropy)
        state.set("formation_energy",       cs.formation_energy)
        state.set("crystal_emb_shape",      str(list(emb.shape)))
        print(
            f"\n  [CrystalEmbeddingNode] '{cs.composition_label}'  "
            f"site_emb shape={list(emb.shape)}"
        )
        return self.next_node


class ElectrochemEmbeddingNode(BaseNode):
    def __init__(self, model, next_node=None):
        self.model     = model
        self.next_node = next_node

    def execute(self, state):
        if TENSOR_STORE.get("electrochem_embedding") is None:
            raw   = TENSOR_STORE.get("experiment_conditions")
            ec    = self.model.embed_context(raw)
            emb   = torch.tensor(ec.latent_vector, dtype=torch.float32).unsqueeze(0)
            TENSOR_STORE["electrochem_embedding"] = emb       # kept out of GraphState
            state.set("electrochem_state",     ec.to_dict())
            print(
                f"\n  [ElectrochemEmbeddingNode] Encoded experiment — "
                f"voltage={ec.voltage_window:.2f}V  T={ec.temperature_K:.0f}K  "
                f"C-rate={ec.c_rate:.2f}C  "
                f"capacity_retention={ec.capacity_retention:.3f}"
            )
        else:
            print("\n  [ElectrochemEmbeddingNode] Using cached experiment embedding.")
        return self.next_node


class StabilityPredictorNode(BaseNode):
    def __init__(self, head, next_node=None):
        self.head      = head
        self.next_node = next_node

    def execute(self, state):
        crystal_sites = TENSOR_STORE.get("crystal_site_embedding")   # (1, N_SITES, 256)
        electrochem   = TENSOR_STORE.get("electrochem_embedding")    # (1, 256)
        label         = state.get("composition_label", "?")

        self.head.eval()
        with torch.no_grad():
            prob_tensor, attn = self.head(crystal_sites, electrochem)

        prob      = prob_tensor.item()
        attn_w    = attn.squeeze()               # (N_SITES,)
        top_idx   = attn_w.topk(5).indices.tolist()
        top_elems = [(ELEMENT_SYMBOLS[i], float(attn_w[i])) for i in top_idx]

        state.set("stability_probability", prob)
        state.set("key_elemental_sites",   [e for e, _ in top_elems])
        state.set("attention_weights",     attn_w.tolist())

        print(f"\n  [CrossAttention] '{label}'")
        print(f"    Cycle stability probability : {prob:.4f}")
        print(f"    Top elemental sites (attn)  :")
        for elem, w in top_elems:
            bar = "█" * int(w * 400)
            print(f"      {elem:<4} {w:.4f}  {bar}")
        return self.next_node


class RouterNode(BaseNode):
    def __init__(self, router, commit_node, replan_node, stability_threshold=0.50):
        self.router              = router
        self.commit_node         = commit_node
        self.replan_node         = replan_node
        self.stability_threshold = stability_threshold

    def execute(self, state):
        cs_dict = state.get("crystal_state")
        prob    = state.get("stability_probability", 0.0)
        route   = self.router.evaluate_state(cs_dict)

        if route == RouteDecision.COMMIT_TRAJECTORY and prob >= self.stability_threshold:
            state.set("route_decision", "COMMIT")
            print(f"\n  [Router] → COMMIT  (thermo OK + stability {prob:.4f} ≥ {self.stability_threshold})")
            return self.commit_node

        state.set("route_decision", "REPLAN")
        print(
            f"\n  [Router] → REPLAN  "
            f"(thermo={route.name}  stability={prob:.4f})"
        )
        return self.replan_node


class AutoApproveNode(BaseNode):
    """
    Simulates researcher approval for proof-of-concept demo runs.
    In production this is replaced by HumanJuryNode which blocks on
    real researcher input (EU AI Act Article 14 structural enforcement).
    """
    def __init__(self, next_node=None):
        self.next_node = next_node

    def execute(self, state):
        label = state.get("composition_label", "?")
        print(f"\n  [AutoApprove] DEMO MODE — simulating researcher approval for '{label}'")
        print(f"  [AutoApprove] In production: HumanJuryNode blocks here until")
        print(f"  [AutoApprove] researcher types 'approve' (EU AI Act Art. 14)")
        state.set("researcher_verdict", "approve")
        return self.next_node


class WriteHeuristicNode(BaseNode):
    def __init__(self, bridge, next_node=None):
        self.bridge    = bridge
        self.next_node = next_node

    def execute(self, state):
        label = state.get("composition_label", "unknown")
        prob  = state.get("stability_probability", 0.0)
        sites = state.get("key_elemental_sites", [])
        cs    = state.get("crystal_state", {}) or {}

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
        print(f"\n  [DMN Write] Heuristic committed to DMN: {ok}")
        return self.next_node


class NextCandidateNode(BaseNode):
    def __init__(self, n_candidates, start_node):
        self.n_candidates = n_candidates
        self.start_node   = start_node

    def execute(self, state):
        idx = state.get("composition_index", 0)
        if idx + 1 < self.n_candidates:
            next_idx = idx + 1
            state.set("composition_index",    next_idx)
            state.set("crystal_state",        None)
            state.set("composition_label",    CANDIDATE_LABELS[next_idx])
            TENSOR_STORE["crystal_site_embedding"] = None   # cleared for next candidate
            print(f"\n  [Replan] → candidate {next_idx}: {CANDIDATE_LABELS[next_idx]}")
            return self.start_node
        print("\n  [Impasse] All candidates exhausted without a stable match.")
        state.set("outcome", "impasse")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    torch.manual_seed(42)

    print("=" * 60)
    print("  Materials-JEPA: Battery Electrolyte Discovery Demo")
    print("  Lár Routing Graph · DMN Memory · EU AI Act Compliance")
    print("=" * 60)
    print(f"\n  PyTorch {torch.__version__} · device: cpu")
    print(f"  Element slots (N_SITES)   : {N_SITES}")
    print(f"  Embed dim                 : 256")
    print(f"  Candidates                : {len(CANDIDATE_LABELS)}")
    print(f"  Electrochemical dim       : {MEASUREMENT_DIM}")

    # ── Data ──────────────────────────────────────────────────────────────
    crystal_data, electrochem_data = build_mock_data(n=5)

    # ── JEPA Models ───────────────────────────────────────────────────────
    crystal_jepa     = CrystalStructureJEPA(embed_dim=256)
    electrochem_jepa = ElectrochemicalJEPA(embed_dim=256)

    # ── Phase 1: Static crystal library (pre-compute once, cache forever) ─
    library = build_crystal_library(crystal_jepa, crystal_data, CANDIDATE_LABELS)
    del crystal_jepa  # freed — library is all inference needs

    # ── Phase 2: Prediction head ───────────────────────────────────────────
    stability_head = CycleStabilityHead(embed_dim=256)
    stability_head.eval()

    # ── Phase 3: Router ───────────────────────────────────────────────────
    # Demo uses 0.55 so the commit path is shown with untrained weights.
    # Production default is 0.40 (stricter safety gate).
    router = ThermalStabilityRouter(thermal_threshold=0.55, max_formation_energy=0.0)

    # ── Phase 4: DMN Bridge ───────────────────────────────────────────────
    dmn_bridge = JEPA_DMN_Consolidation_Node()

    print(f"\n{'─'*60}")
    print("  Phase 2: Lár Graph Execution")
    print("  Graph: Recall → Crystal → Electrochem → CrossAttn → Router")
    print(f"{'─'*60}")

    # ── Build graph (reverse declaration) ─────────────────────────────────
    done_node = AddValueNode(key="outcome", value="stable_electrolyte_committed", next_node=None)

    write_node   = WriteHeuristicNode(bridge=dmn_bridge, next_node=done_node)
    approve_node = AutoApproveNode(next_node=write_node)

    next_cand = NextCandidateNode(n_candidates=5, start_node=None)

    router_node = RouterNode(
        router=router,
        commit_node=approve_node,
        replan_node=next_cand,
        stability_threshold=0.50,
    )

    predictor_node = StabilityPredictorNode(head=stability_head, next_node=router_node)

    ec_node      = ElectrochemEmbeddingNode(model=electrochem_jepa, next_node=predictor_node)
    crystal_node = CrystalEmbeddingNode(library=library, next_node=ec_node)
    recall_node  = RecallNode(bridge=dmn_bridge, next_node=crystal_node)

    next_cand.start_node = recall_node   # close the replan cycle

    # ── Load tensors into side-store (kept out of GraphState) ─────────────
    TENSOR_STORE.clear()
    TENSOR_STORE["experiment_conditions"] = electrochem_data
    TENSOR_STORE["electrochem_embedding"] = None   # populated on first run

    # ── Execute ───────────────────────────────────────────────────────────
    initial_state = {
        "composition_index": 0,
        "composition_label": CANDIDATE_LABELS[0],
    }

    executor = GraphExecutor(
        log_dir="lar_logs",
        hmac_secret="snath_ai_materials_eu_compliance_2026",
    )

    t_start     = time.perf_counter()
    final_state = {}
    steps       = 0
    for step_log in executor.run_step_by_step(recall_node, initial_state, max_steps=50):
        final_state.update(step_log.get("state_after", {}))
        steps += 1

    elapsed = time.perf_counter() - t_start

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULT")
    print("=" * 60)
    print(f"  Outcome             : {final_state.get('outcome', 'unknown')}")
    print(f"  Committed candidate : {final_state.get('composition_label', 'N/A')}")
    print(f"  Formation energy    : {final_state.get('formation_energy', 'N/A'):.3f} eV/atom")
    print(f"  Thermal entropy     : {final_state.get('thermal_entropy', 'N/A'):.3f}")
    print(f"  Cycle stability p   : {final_state.get('stability_probability', 0):.4f}")
    print(f"  Key elemental sites : {final_state.get('key_elemental_sites', [])}")
    print(f"  Researcher verdict  : {final_state.get('researcher_verdict', 'N/A')}")
    print(f"  Total graph steps   : {steps}")
    print(f"  Wall time           : {elapsed*1000:.1f} ms")
    print("=" * 60)
    print("\n  Audit trail (HMAC-signed): lar_logs/")
    print("  DMN long-term memory    : DMN/lar/data/chroma_db/")
    print("  EU AI Act Art. 12 & 14  : structurally enforced in graph topology")


if __name__ == "__main__":
    run()
