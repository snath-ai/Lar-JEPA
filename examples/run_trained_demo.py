"""
Materials-JEPA: Demo with Trained JEPA Encoder
===============================================
Identical pipeline to run_demo.py, but the crystal library is built
using the *trained* CrystalJEPA encoder (models/crystal_jepa_encoder.pt).

Run train_crystal_jepa.py first, then this.

    python examples/train_crystal_jepa.py   # ~60s, saves models/crystal_jepa_encoder.pt
    python examples/run_trained_demo.py     # uses real JEPA embeddings

What changed vs run_demo.py
---------------------------
  - CrystalStructureJEPA (mock linear encoder, untrained) → CrystalJEPA (trained)
  - embed_dim: 256 (random) → 64 (trained, 97% loss reduction)
  - Site embeddings are now learned representations, not random projections
  - The stability predictor and cross-attention head are still randomly initialised
    (they need downstream training on real electrochemical labels)

The Lár graph topology, DMN connection, and compliance stack are unchanged.
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

from materials_engine.crystal_jepa_model import CrystalJEPA, N_SITES
from materials_engine.crystal_manifold import LatentCrystalState, ELEMENT_SYMBOLS
from materials_engine.stability_router import ThermalStabilityRouter
from dmn_integration.consolidation_node import JEPA_DMN_Consolidation_Node

EMBED_DIM = 64   # matches trained model

CANDIDATE_LABELS = [
    "Li6PS5Cl (Argyrodite)",
    "Li3PS4 (Sulfide Glass)",
    "LATP (Li1.3Al0.3Ti1.7(PO4)3)",
    "LLZO (Li7La3Zr2O12)",
    "LiPF6/EC:DMC (Liquid)",
]

TENSOR_STORE: dict = {}


# ---------------------------------------------------------------------------
# Build crystal library using the TRAINED JEPA encoder
# ---------------------------------------------------------------------------

def build_library_with_trained_jepa(
    jepa: CrystalJEPA,
    crystal_data: torch.Tensor,   # (n, N_SITES + 6)
    labels: list,
) -> list:
    """
    Uses the trained JEPA context encoder (no masking at inference) to produce
    per-site embeddings. These are learned representations, not random projections.
    """
    print(f"\n{'─'*60}")
    print("  Phase 1: Crystal Library — Trained JEPA Encoder")
    print(f"  Encoder: CrystalJEPA context encoder (embed_dim={EMBED_DIM})")
    print(f"  Weights: models/crystal_jepa_encoder.pt  (97% JEPA loss reduction)")
    print(f"{'─'*60}")

    library = []
    t0 = time.perf_counter()
    for i, (comp, label) in enumerate(zip(crystal_data, labels)):
        occ = comp[:N_SITES].unsqueeze(0)          # (1, N_SITES)
        lat = comp[N_SITES:].unsqueeze(0)          # (1, 6)

        emb          = jepa.encode(occ, lat)       # (1, N_SITES, embed_dim)
        global_latent = emb.mean(dim=1)            # (1, embed_dim)

        # Derive thermal proxy from embedding — higher mean activation = more energetic
        activation      = torch.sigmoid(global_latent.mean()).item()
        thermal_entropy = activation * 0.6         # scale to [0, 0.6] range
        # Formation energy proxy: negative is stable
        formation_energy = -1.5 + torch.sigmoid(emb.norm(dim=-1).mean()).item() * 0.8
        band_gap         = abs(float(global_latent.std().item()))

        state = LatentCrystalState(
            composition_id=i,
            composition_label=label,
            site_embeddings=emb.squeeze(0).tolist(),    # (N_SITES, embed_dim)
            latent_vector=global_latent.squeeze(0).tolist(),
            formation_energy=formation_energy,
            thermal_entropy=thermal_entropy,
            band_gap=band_gap,
        )
        library.append(state)
        print(
            f"  [{i}] {label:<38} "
            f"Ef={formation_energy:+.3f} eV/atom  "
            f"thermal_entropy={thermal_entropy:.3f}  "
            f"emb_norm={emb.norm(dim=-1).mean().item():.1f}"
        )

    print(f"\n  Site embedding shape: ({N_SITES}, {EMBED_DIM})  ← learned, not random")
    print(f"  Library built in {(time.perf_counter()-t0)*1000:.1f} ms")
    return library


# ---------------------------------------------------------------------------
# Small electrochemical encoder (embed_dim=64 to match JEPA)
# ---------------------------------------------------------------------------

class SmallElectrochemEncoder(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, embed_dim),
        )
        self.retention_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.net(x)
        r = self.retention_head(z)
        return z, r


# ---------------------------------------------------------------------------
# Cross-attention head (embed_dim=64)
# ---------------------------------------------------------------------------

class CycleStabilityHead(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.embed_dim  = embed_dim
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj   = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, crystal_sites, electrochem):
        Q      = self.query_proj(electrochem).unsqueeze(1)
        K      = self.key_proj(crystal_sites)
        V      = self.value_proj(crystal_sites)
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.embed_dim)
        attn   = torch.softmax(scores, dim=-1)
        ctx    = torch.bmm(attn, V).squeeze(1)
        return self.fc(ctx), attn


# ---------------------------------------------------------------------------
# Graph nodes (same logic as run_demo.py, adapted for trained encoder dims)
# ---------------------------------------------------------------------------

class RecallNode(BaseNode):
    def __init__(self, bridge, next_node=None):
        self.bridge, self.next_node = bridge, next_node

    def execute(self, state):
        label = state.get("composition_label", "candidate")
        prior = self.bridge.recall_heuristics(
            f"battery electrolyte stability for {label}", max_results=2
        )
        state.set("prior_heuristics", prior or "(no prior experiments)")
        print(f"\n  [DMN Recall] '{label}':\n    {state.get('prior_heuristics')}")
        return self.next_node


class CrystalEmbeddingNode(BaseNode):
    def __init__(self, library, next_node=None):
        self.library, self.next_node = library, next_node

    def execute(self, state):
        idx = state.get("composition_index", 0)
        cs  = self.library[idx]
        emb = torch.tensor(cs.site_embeddings, dtype=torch.float32).unsqueeze(0)
        TENSOR_STORE["crystal_site_embedding"] = emb
        state.set("crystal_state",     cs.to_dict())
        state.set("composition_label", cs.composition_label)
        state.set("thermal_entropy",   cs.thermal_entropy)
        state.set("formation_energy",  cs.formation_energy)
        state.set("emb_shape",         str(list(emb.shape)))
        print(
            f"\n  [Crystal] '{cs.composition_label}'  "
            f"emb={list(emb.shape)}  "
            f"norm={emb.norm(dim=-1).mean().item():.1f}"
        )
        return self.next_node


class ElectrochemNode(BaseNode):
    def __init__(self, model, next_node=None):
        self.model, self.next_node = model, next_node

    def execute(self, state):
        if TENSOR_STORE.get("electrochem_embedding") is None:
            raw = TENSOR_STORE.get("experiment_conditions")
            self.model.eval()
            with torch.no_grad():
                emb, ret = self.model(raw)
            TENSOR_STORE["electrochem_embedding"] = emb
            state.set("capacity_retention", ret.item())
            print(
                f"\n  [Electrochem] Encoded experiment — "
                f"capacity_retention={ret.item():.3f}  "
                f"emb shape={list(emb.shape)}"
            )
        else:
            print("\n  [Electrochem] Using cached embedding.")
        return self.next_node


class PredictorNode(BaseNode):
    def __init__(self, head, next_node=None):
        self.head, self.next_node = head, next_node

    def execute(self, state):
        crystal_sites = TENSOR_STORE.get("crystal_site_embedding")
        electrochem   = TENSOR_STORE.get("electrochem_embedding")
        label         = state.get("composition_label", "?")

        self.head.eval()
        with torch.no_grad():
            prob_t, attn = self.head(crystal_sites, electrochem)

        prob      = prob_t.item()
        attn_w    = attn.squeeze()
        top_idx   = attn_w.topk(5).indices.tolist()
        top_elems = [(ELEMENT_SYMBOLS[i], float(attn_w[i])) for i in top_idx]

        state.set("stability_probability", prob)
        state.set("key_elemental_sites",   [e for e, _ in top_elems])

        print(f"\n  [CrossAttention] '{label}'")
        print(f"    Stability probability : {prob:.4f}")
        print(f"    Top sites by attention:")
        for elem, w in top_elems:
            bar = "█" * int(w * 500)
            print(f"      {elem:<4} {w:.4f}  {bar}")
        return self.next_node


class RouterNode(BaseNode):
    def __init__(self, router, commit_node, replan_node, threshold=0.50):
        self.router      = router
        self.commit_node = commit_node
        self.replan_node = replan_node
        self.threshold   = threshold

    def execute(self, state):
        route = self.router.evaluate_state(state.get("crystal_state"))
        prob  = state.get("stability_probability", 0.0)
        if route == RouteDecision.COMMIT_TRAJECTORY and prob >= self.threshold:
            state.set("route_decision", "COMMIT")
            return self.commit_node
        state.set("route_decision", "REPLAN")
        print(f"\n  [Router] → REPLAN  route={route.name}  prob={prob:.4f}")
        return self.replan_node


class AutoApproveNode(BaseNode):
    def __init__(self, next_node=None):
        self.next_node = next_node

    def execute(self, state):
        label = state.get("composition_label", "?")
        print(f"\n  [AutoApprove] Simulated researcher approval: '{label}'")
        print(f"  [AutoApprove] In production: HumanJuryNode blocks for real approval")
        state.set("researcher_verdict", "approve")
        return self.next_node


class WriteHeuristicNode(BaseNode):
    def __init__(self, bridge, next_node=None):
        self.bridge, self.next_node = bridge, next_node

    def execute(self, state):
        label = state.get("composition_label", "?")
        cs    = state.get("crystal_state", {}) or {}
        ok = self.bridge.write_trajectory_heuristic({
            "domain":        "battery_electrolyte_discovery",
            "action":        f"screened_{label.replace(' ', '_')}",
            "outcome":       "committed",
            "entropic_loss": cs.get("thermal_entropy", 0.0),
            "metadata": {
                "cycle_stability_prob": state.get("stability_probability", 0.0),
                "key_elemental_sites":  state.get("key_elemental_sites", []),
                "formation_energy":     cs.get("formation_energy", 0.0),
                "composition_label":    label,
                "jepa_encoder":         "crystal_jepa_encoder.pt",
            },
        })
        print(f"\n  [DMN Write] Heuristic committed to Hippocampus: {ok}")
        return self.next_node


class NextCandidateNode(BaseNode):
    def __init__(self, n_candidates, start_node):
        self.n_candidates = n_candidates
        self.start_node   = start_node

    def execute(self, state):
        idx = state.get("composition_index", 0)
        if idx + 1 < self.n_candidates:
            nxt = idx + 1
            state.set("composition_index",    nxt)
            state.set("crystal_state",        None)
            state.set("composition_label",    CANDIDATE_LABELS[nxt])
            TENSOR_STORE["crystal_site_embedding"] = None
            print(f"\n  [Replan] → candidate {nxt}: {CANDIDATE_LABELS[nxt]}")
            return self.start_node
        print("\n  [Impasse] All candidates exhausted.")
        state.set("outcome", "impasse")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    torch.manual_seed(42)

    encoder_path = os.path.join(_JEPA_ROOT, "models", "crystal_jepa_encoder.pt")
    if not os.path.exists(encoder_path):
        print(f"[ERROR] Trained encoder not found at {encoder_path}")
        print("Run: python examples/train_crystal_jepa.py")
        sys.exit(1)

    print("=" * 60)
    print("  Materials-JEPA: Trained JEPA Encoder Demo")
    print("=" * 60)
    print(f"\n  Loading trained CrystalJEPA from {encoder_path}")

    jepa = CrystalJEPA(embed_dim=EMBED_DIM)
    jepa.context_encoder.load_state_dict(
        torch.load(encoder_path, map_location="cpu")
    )
    jepa.context_encoder.eval()
    print(f"  CrystalJEPA loaded. Parameters: "
          f"{sum(p.numel() for p in jepa.context_encoder.parameters()):,}")

    # Mock crystal data (same seed as training demo)
    torch.manual_seed(42)
    occ  = torch.softmax(torch.randn(5, N_SITES), dim=-1) * 0.8
    lat  = torch.rand(5, 6)
    crystal_data = torch.cat([occ, lat], dim=-1)            # (5, N_SITES+6)
    electrochem_data = torch.rand(1, 12)

    library = build_library_with_trained_jepa(jepa, crystal_data, CANDIDATE_LABELS)
    del jepa

    # Downstream heads (randomly initialised — need real electrochemical labels to train)
    ec_encoder     = SmallElectrochemEncoder(embed_dim=EMBED_DIM)
    stability_head = CycleStabilityHead(embed_dim=EMBED_DIM)
    router         = ThermalStabilityRouter(thermal_threshold=0.55, max_formation_energy=0.0)

    _chroma = os.path.join(_JEPA_ROOT, "DMN", "lar", "data", "chroma_db")
    _dreams = os.path.join(_JEPA_ROOT, "DMN", "lar", "memory", "dreams.json")
    dmn     = JEPA_DMN_Consolidation_Node(chroma_path=_chroma, dreams_path=_dreams)

    print(f"\n{'─'*60}")
    print("  Phase 2: Lár Graph Execution (trained JEPA site embeddings)")
    print(f"{'─'*60}")

    done_node    = AddValueNode(key="outcome", value="stable_electrolyte_committed", next_node=None)
    write_node   = WriteHeuristicNode(bridge=dmn, next_node=done_node)
    approve_node = AutoApproveNode(next_node=write_node)
    next_cand    = NextCandidateNode(n_candidates=5, start_node=None)
    # 0.25 threshold because CycleStabilityHead is untrained (needs electrochemical labels).
    # With trained head, this rises back to 0.50+.
    router_node  = RouterNode(router, commit_node=approve_node, replan_node=next_cand, threshold=0.25)
    pred_node    = PredictorNode(head=stability_head, next_node=router_node)
    ec_node      = ElectrochemNode(model=ec_encoder, next_node=pred_node)
    crystal_node = CrystalEmbeddingNode(library=library, next_node=ec_node)
    recall_node  = RecallNode(bridge=dmn, next_node=crystal_node)
    next_cand.start_node = recall_node

    TENSOR_STORE.clear()
    TENSOR_STORE["experiment_conditions"] = electrochem_data
    TENSOR_STORE["electrochem_embedding"] = None

    executor = GraphExecutor(
        log_dir="lar_logs",
        hmac_secret="snath_ai_materials_eu_compliance_2026",
    )

    t0          = time.perf_counter()
    final_state = {}
    steps       = 0
    for step_log in executor.run_step_by_step(recall_node, {"composition_index": 0, "composition_label": CANDIDATE_LABELS[0]}, max_steps=50):
        final_state.update(step_log.get("state_after", {}))
        steps += 1

    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 60)
    print("  RESULT")
    print("=" * 60)
    print(f"  Outcome             : {final_state.get('outcome', 'unknown')}")
    print(f"  Committed candidate : {final_state.get('composition_label', 'N/A')}")
    print(f"  Formation energy    : {final_state.get('formation_energy', 0):.3f} eV/atom")
    print(f"  Thermal entropy     : {final_state.get('thermal_entropy', 0):.3f}")
    print(f"  Cycle stability p   : {final_state.get('stability_probability', 0):.4f}")
    print(f"  Key elemental sites : {final_state.get('key_elemental_sites', [])}")
    print(f"  Researcher verdict  : {final_state.get('researcher_verdict', 'N/A')}")
    print(f"  JEPA encoder        : trained, 97% loss reduction")
    print(f"  Graph steps         : {steps}")
    print(f"  Wall time           : {elapsed*1000:.1f} ms")
    print("=" * 60)
    print("\n  Audit trail (HMAC-signed): lar_logs/")


if __name__ == "__main__":
    run()
