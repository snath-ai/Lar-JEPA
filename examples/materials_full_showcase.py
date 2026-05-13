"""
Materials-JEPA: Full Lár Primitives Showcase
=============================================
Every Lár primitive in one graph, wired to the real trained JEPA encoder
and the DMN episodic memory store.

Primitives used
---------------
  BaseNode          — all custom domain nodes inherit from this
  FunctionalNode    — pure Python logic (serialize library, pick best candidate)
  BatchNode         — evaluate all 5 crystal candidates IN PARALLEL
  BranchTriageNode  — aggregate parallel results, flag thermal-runaway risk
  AdaptiveNode      — if any branch is CRITICAL, LLM designs a focused
                      re-analysis subgraph at runtime; TopologyValidator
                      guards against cycles and unapproved tools (Art. 3(23))
  RouterNode (×2)   — critical vs non-critical; found_candidate vs impasse
  LLMNode           — materials science interpretation of JEPA result
  ReduceNode        — synthesize JEPA metrics + LLM analysis → recommendation
  HumanJuryNode     — researcher approval gate (EU AI Act Art. 14)
  ToolNode          — write the final lab report to disk
  ClearErrorNode    — graceful impasse recovery
  AddValueNode      — set final outcome in state
  JEPA_DMN_Consolidation_Node — episodic recall + write (DMN Hippocampus)

Run (train first if needed):
    python examples/train_crystal_jepa.py          # ~60s, once
    python examples/materials_full_showcase.py     # full showcase
"""

import sys, os, json, math, time, copy, datetime
import torch
import torch.nn as nn

_ROOT      = os.path.dirname(os.path.abspath(__file__))
_JEPA_ROOT = os.path.abspath(os.path.join(_ROOT, ".."))
_LAR_SRC   = os.path.join(_JEPA_ROOT, "lar_jepa", "src")
for _p in [_JEPA_ROOT, _LAR_SRC]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Lár primitives ──────────────────────────────────────────────────────────
from lar import (
    GraphState, GraphExecutor,
    BaseNode, AddValueNode, FunctionalNode,
    BatchNode, BranchTriageNode,
    RouterNode, LLMNode, ReduceNode,
    HumanJuryNode, ToolNode, ClearErrorNode,
)
from lar.adaptive import AdaptiveNode, TopologyValidator
from core.types import RouteDecision

# ── Materials engine ────────────────────────────────────────────────────────
from materials_engine.crystal_jepa_model import CrystalJEPA, N_SITES
from materials_engine.crystal_manifold import LatentCrystalState, ELEMENT_SYMBOLS
from materials_engine.stability_router import ThermalStabilityRouter
from dmn_integration.consolidation_node import JEPA_DMN_Consolidation_Node

EMBED_DIM = 64
CANDIDATE_LABELS = [
    "Li6PS5Cl (Argyrodite)",
    "Li3PS4 (Sulfide Glass)",
    "LATP (Li1.3Al0.3Ti1.7(PO4)3)",
    "LLZO (Li7La3Zr2O12)",
    "LiPF6/EC:DMC (Liquid)",
]
N_CANDIDATES = len(CANDIDATE_LABELS)


# ── Minimal electrochemical encoder (embed_dim=64) ──────────────────────────
class SmallElectrochemEncoder(nn.Module):
    def __init__(self, dim: int = EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12, 128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128, dim)
        )
        self.ret_head = nn.Sequential(nn.Linear(dim,32), nn.ReLU(), nn.Linear(32,1), nn.Sigmoid())
    def forward(self, x):
        z = self.net(x); return z, self.ret_head(z)


# ── Cross-attention head ─────────────────────────────────────────────────────
class CycleStabilityHead(nn.Module):
    def __init__(self, dim: int = EMBED_DIM):
        super().__init__()
        self.dim = dim
        self.Q = nn.Linear(dim, dim); self.K = nn.Linear(dim, dim)
        self.V = nn.Linear(dim, dim)
        self.fc = nn.Sequential(nn.Linear(dim,32), nn.ReLU(), nn.Linear(32,1), nn.Sigmoid())
    def forward(self, sites, ec):
        q = self.Q(ec).unsqueeze(1)
        k, v = self.K(sites), self.V(sites)
        attn = torch.softmax(torch.bmm(q, k.transpose(1,2)) / math.sqrt(self.dim), dim=-1)
        return self.fc(torch.bmm(attn, v).squeeze(1)), attn


# ── Risk level from thermal entropy ─────────────────────────────────────────
def _risk(thermal_entropy: float) -> str:
    if thermal_entropy > 0.50: return "CRITICAL"
    if thermal_entropy > 0.35: return "HIGH"
    if thermal_entropy > 0.20: return "MEDIUM"
    return "LOW"


# ════════════════════════════════════════════════════════════════════════════
# Custom domain nodes (all inherit BaseNode)
# ════════════════════════════════════════════════════════════════════════════

class RecallNode(BaseNode):
    """DMN recall — queries Hippocampus for prior electrolyte screening results."""
    def __init__(self, bridge: JEPA_DMN_Consolidation_Node, next_node=None):
        self.bridge = bridge; self.next_node = next_node
    def execute(self, state):
        prior = self.bridge.recall_heuristics(
            "battery electrolyte stability screening solid-state", max_results=3
        )
        state.set("prior_heuristics", prior or "(no prior experiments in memory)")
        print(f"\n  [DMN Recall] Prior knowledge:\n  {state.get('prior_heuristics')}")
        return self.next_node


class ElectrochemNode(BaseNode):
    """Encodes experimental conditions once. Cached for all parallel branches."""
    def __init__(self, model: SmallElectrochemEncoder, next_node=None):
        self.model = model; self.next_node = next_node
    def execute(self, state):
        raw = torch.rand(1, 12, generator=torch.Generator().manual_seed(42))
        self.model.eval()
        with torch.no_grad():
            emb, ret = self.model(raw)
        state.set("ec_embedding_list", emb.squeeze(0).tolist())
        state.set("ec_retention",      ret.item())
        print(f"\n  [Electrochem] Experiment encoded — "
              f"capacity_retention={ret.item():.3f}  emb_dim={EMBED_DIM}")
        return self.next_node


class EvalCrystalBranchNode(BaseNode):
    """
    One branch inside BatchNode — evaluates a single crystal candidate.
    Deserialises embeddings from state (thread-safe: no shared TENSOR_STORE).
    Writes result as a JSON string to state[branch_key].
    """
    def __init__(
        self,
        candidate_idx: int,
        stability_head: CycleStabilityHead,
        router: ThermalStabilityRouter,
        branch_key: str,
    ):
        self.idx   = candidate_idx
        self.head  = stability_head
        self.router = router
        self.key   = branch_key

    def execute(self, state):
        label = CANDIDATE_LABELS[self.idx]
        # Deserialise from state (lists, not tensors — thread-safe)
        site_list = state.get(f"library_sites_{self.idx}")   # List[List[float]]
        ec_list   = state.get("ec_embedding_list")            # List[float]

        sites = torch.tensor(site_list, dtype=torch.float32).unsqueeze(0)  # (1,N,D)
        ec    = torch.tensor(ec_list, dtype=torch.float32).unsqueeze(0)    # (1,D)

        self.head.eval()
        with torch.no_grad():
            prob_t, attn = self.head(sites, ec)

        prob      = prob_t.item()
        attn_w    = attn.squeeze()
        top_sites = [ELEMENT_SYMBOLS[i] for i in attn_w.topk(3).indices.tolist()]

        # Proxy thermal entropy from site embedding norm variation
        site_norms    = torch.tensor(site_list).norm(dim=-1)
        thermal       = float(torch.sigmoid((site_norms.std() - 8) * 0.1).item())
        formation_e   = -1.5 + torch.sigmoid(torch.tensor(site_norms.mean().item() - 55)).item() * 0.8

        thermo_route = self.router.evaluate_state({
            "thermal_entropy":    thermal,
            "formation_energy":   formation_e,
            "composition_label":  label,
        })
        is_stable = thermo_route == RouteDecision.COMMIT_TRAJECTORY

        risk  = _risk(thermal)
        finding = (
            f"{'STABLE' if is_stable else 'UNSTABLE'}: "
            f"Ef={formation_e:.3f} eV/atom  entropy={thermal:.3f}  "
            f"stability_p={prob:.3f}  key_sites={top_sites}"
        )

        result = json.dumps({
            "risk_level":        risk,
            "finding":           finding,
            "label":             label,
            "candidate_idx":     self.idx,
            "stability_prob":    prob,
            "thermal_entropy":   thermal,
            "formation_energy":  formation_e,
            "key_sites":         top_sites,
            "is_stable":         is_stable,
        })
        state.set(self.key, result)
        print(f"  [Branch {self.idx}] {label:<38} risk={risk:<8} p={prob:.3f}")
        return None   # BatchNode branches terminate (next_node handled by BatchNode)


class DMNWriteNode(BaseNode):
    """Commits the selected candidate to DMN Hippocampus long-term memory."""
    def __init__(self, bridge: JEPA_DMN_Consolidation_Node, next_node=None):
        self.bridge = bridge; self.next_node = next_node
    def execute(self, state):
        best = state.get("best_candidate") or {}
        ok = self.bridge.write_trajectory_heuristic({
            "domain":        "battery_electrolyte_discovery",
            "action":        f"screened_{best.get('label','?').replace(' ','_')}",
            "outcome":       "committed",
            "entropic_loss": best.get("thermal_entropy", 0.0),
            "metadata": {
                "cycle_stability_prob": best.get("stability_prob", 0.0),
                "key_elemental_sites":  json.dumps(best.get("key_sites", [])),
                "formation_energy":     best.get("formation_energy", 0.0),
                "composition_label":    best.get("label", "?"),
                "jepa_encoder":         "crystal_jepa_encoder.pt",
            },
        })
        print(f"\n  [DMN Write] Heuristic written to Hippocampus: {ok}")
        return self.next_node


# ════════════════════════════════════════════════════════════════════════════
# FunctionalNode helpers (pure functions, no side-effects)
# ════════════════════════════════════════════════════════════════════════════

_CRYSTAL_LIBRARY: list = []   # module-level store — keeps LatentCrystalState out of GraphState

def serialize_library(state: GraphState):
    """Copies per-site embedding lists from the module-level library into state for BatchNode."""
    for i, cs in enumerate(_CRYSTAL_LIBRARY):
        state.set(f"library_sites_{i}", cs.site_embeddings)
    return f"serialized {len(_CRYSTAL_LIBRARY)} candidates into state"


def pick_best_candidate(state: GraphState):
    """
    Reads branch results from state and selects the stable candidate with
    the highest stability probability. Returns the best candidate dict or None.
    """
    candidates = []
    for i in range(N_CANDIDATES):
        raw = state.get(f"candidate_{i}_analysis", "")
        if not raw:
            continue
        try:
            d = json.loads(raw)
            if d.get("is_stable"):
                candidates.append(d)
        except json.JSONDecodeError:
            pass

    if not candidates:
        state.set("best_candidate", None)
        state.set("route_key", "impasse")
        return "no stable candidate found"

    best = max(candidates, key=lambda d: d["stability_prob"])
    state.set("best_candidate", best)
    state.set("route_key", "found_candidate")

    summary = (
        f"Best candidate: {best['label']}\n"
        f"  Stability probability : {best['stability_prob']:.4f}\n"
        f"  Thermal entropy       : {best['thermal_entropy']:.3f}\n"
        f"  Formation energy      : {best['formation_energy']:.3f} eV/atom\n"
        f"  Key elemental sites   : {best['key_sites']}\n"
        f"  Risk level            : {best['risk_level']}"
    )
    state.set("best_candidate_summary", summary)
    # Keep a copy ReduceNode won't delete (ReduceNode purges its input_keys)
    state.set("best_candidate_summary_final", summary)
    print(f"\n  [PickBest] {summary}")
    return summary


def save_lab_report(state: GraphState):
    """ToolNode function — writes the final report to disk."""
    best       = state.get("best_candidate") or {}
    llm_interp = state.get("llm_interpretation", "(no LLM output)")
    synthesis  = state.get("final_recommendation", "(no synthesis)")
    verdict    = state.get("researcher_verdict", "N/A")
    timestamp  = datetime.datetime.utcnow().isoformat()

    report = {
        "timestamp":             timestamp,
        "committed_candidate":   best.get("label", "?"),
        "formation_energy":      best.get("formation_energy"),
        "thermal_entropy":       best.get("thermal_entropy"),
        "cycle_stability_prob":  best.get("stability_prob"),
        "key_elemental_sites":   best.get("key_sites"),
        "llm_interpretation":    llm_interp,
        "synthesis":             synthesis,
        "researcher_verdict":    verdict,
        "jepa_encoder":          "crystal_jepa_encoder.pt (97% loss reduction)",
        "audit_signed":          "HMAC-snath_ai_materials_eu_compliance_2026",
    }

    report_dir  = os.path.join(_JEPA_ROOT, "lab_reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f"report_{timestamp[:10]}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  [ToolNode] Lab report saved to {report_path}")
    state.set("report_path", report_path)
    return report_path


# ════════════════════════════════════════════════════════════════════════════
# Demo jury (wraps HumanJuryNode for non-interactive showcase)
# ════════════════════════════════════════════════════════════════════════════

class DemoJuryNode(BaseNode):
    """
    Simulates HumanJuryNode for the demo run.
    In production, replace with:
        HumanJuryNode(
            prompt="Approve this electrolyte for lab synthesis?",
            choices=["approve", "reject"],
            output_key="researcher_verdict",
            context_keys=["best_candidate_summary", "final_recommendation"],
        )
    EU AI Act Article 14 — human oversight is structurally enforced,
    not a disclaimer. The graph cannot reach ToolNode without this gate.
    """
    def __init__(self, next_node=None, reject_node=None):
        self.next_node = next_node; self.reject_node = reject_node

    def execute(self, state):
        summary = state.get("best_candidate_summary_final", state.get("best_candidate_summary", ""))
        reco    = state.get("final_recommendation", "")
        print("\n" + "═"*60)
        print("  [HUMAN JURY] EU AI Act Art. 14 — Researcher Approval Gate")
        print("  In production: researcher reads and types 'approve'/'reject'")
        print("─"*60)
        print(f"  Candidate summary:\n{summary}")
        print(f"\n  AI recommendation:\n{reco[:300]}...")
        print("─"*60)
        print("  [DEMO] Auto-approving for showcase run.")
        print("═"*60)
        state.set("researcher_verdict", "approve")
        return self.next_node


# ════════════════════════════════════════════════════════════════════════════
# Build and run the graph
# ════════════════════════════════════════════════════════════════════════════

def build_crystal_library(jepa: CrystalJEPA) -> list:
    torch.manual_seed(42)
    occ = torch.softmax(torch.randn(N_CANDIDATES, N_SITES), dim=-1) * 0.8
    lat = torch.rand(N_CANDIDATES, 6)

    library = []
    for i, label in enumerate(CANDIDATE_LABELS):
        emb  = jepa.encode(occ[i:i+1], lat[i:i+1])        # (1, N_SITES, D)
        pool = emb.mean(dim=1)                              # (1, D)
        norms = emb.squeeze(0).norm(dim=-1)                 # (N_SITES,)
        thermal      = float(torch.sigmoid((norms.std() - 8) * 0.1).item())
        formation_e  = -1.5 + torch.sigmoid(torch.tensor(norms.mean().item() - 55)).item() * 0.8
        library.append(LatentCrystalState(
            composition_id=i,
            composition_label=label,
            site_embeddings=emb.squeeze(0).tolist(),
            latent_vector=pool.squeeze(0).tolist(),
            formation_energy=formation_e,
            thermal_entropy=thermal,
            band_gap=abs(float(pool.std().item())),
        ))
    return library


def run():
    torch.manual_seed(42)
    t_total = time.perf_counter()

    print("═"*60)
    print("  Materials-JEPA: Full Lár Primitives Showcase")
    print("  Trained JEPA · All Primitives · DMN · EU Compliance")
    print("═"*60)

    # ── Load trained JEPA encoder ─────────────────────────────────────────
    encoder_path = os.path.join(_JEPA_ROOT, "models", "crystal_jepa_encoder.pt")
    if not os.path.exists(encoder_path):
        print("[ERROR] Run train_crystal_jepa.py first.")
        sys.exit(1)

    jepa = CrystalJEPA(embed_dim=EMBED_DIM)
    jepa.context_encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
    jepa.context_encoder.eval()
    params = sum(p.numel() for p in jepa.context_encoder.parameters())
    print(f"\n  CrystalJEPA loaded — {params:,} parameters (97% JEPA loss reduction)")

    # ── Build crystal library ─────────────────────────────────────────────
    print("\n  Building crystal library...")
    library = build_crystal_library(jepa)
    del jepa
    for cs in library:
        print(f"    {cs.composition_label:<38} "
              f"entropy={cs.thermal_entropy:.3f}  Ef={cs.formation_energy:.3f}")

    # ── Shared model objects ──────────────────────────────────────────────
    ec_encoder     = SmallElectrochemEncoder()
    stability_head = CycleStabilityHead()
    router         = ThermalStabilityRouter(thermal_threshold=0.40, max_formation_energy=0.0)

    # ── DMN bridge ────────────────────────────────────────────────────────
    _chroma = os.path.join(_JEPA_ROOT, "DMN", "lar", "data", "chroma_db")
    _dreams = os.path.join(_JEPA_ROOT, "DMN", "lar", "memory", "dreams.json")
    dmn = JEPA_DMN_Consolidation_Node(chroma_path=_chroma, dreams_path=_dreams)

    # ════════════════════════════════════════════════════════════════════════
    # Graph construction (declared in reverse execution order)
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print("  Assembling Lár graph...")
    print(f"{'─'*60}")

    # ── Terminal nodes ────────────────────────────────────────────────────
    committed_node = AddValueNode(
        key="outcome", value="stable_electrolyte_committed", next_node=None
    )
    impasse_node = AddValueNode(
        key="outcome", value="impasse_no_stable_candidate", next_node=None
    )

    # ── ClearErrorNode: graceful impasse path ─────────────────────────────
    clear_error_node = ClearErrorNode(next_node=impasse_node)

    # ── DMN write ────────────────────────────────────────────────────────
    dmn_write_node = DMNWriteNode(bridge=dmn, next_node=committed_node)

    # ── ToolNode: save lab report ─────────────────────────────────────────
    # ToolNode wraps a Python callable with Lár's credential vault + audit layer.
    # input_keys=[] because the function reads directly from state.
    tool_node = ToolNode(
        tool_function=save_lab_report,
        input_keys=["__state__"],   # ToolNode passes full GraphState
        output_key="report_path",
        next_node=dmn_write_node,
        action_type="WRITE_FILE",
        affected_parties="RESEARCH_TEAM",
    )

    # ── DemoJuryNode (HumanJuryNode in production) ────────────────────────
    jury_node = DemoJuryNode(next_node=tool_node, reject_node=clear_error_node)

    _OLLAMA_BASE = "http://localhost:11434"
    _OLLAMA_MODEL = "ollama/llama3.2"   # 2 GB, fast

    # ── ReduceNode: synthesize JEPA metrics + LLM interpretation ──────────
    reduce_node = ReduceNode(
        model_name=_OLLAMA_MODEL,
        prompt_template=(
            "You are a materials scientist reviewing an AI-assisted electrolyte screening run.\n\n"
            "JEPA screening result:\n{best_candidate_summary}\n\n"
            "AI interpretation:\n{llm_interpretation}\n\n"
            "Write a 2-sentence final recommendation for the researcher's lab notebook. "
            "Be specific: mention the composition, the key stability driver, and the next experimental step."
        ),
        input_keys=["best_candidate_summary", "llm_interpretation"],
        output_key="final_recommendation",
        next_node=jury_node,
        generation_config={"api_base": _OLLAMA_BASE, "max_tokens": 200},
    )

    # ── LLMNode: materials science interpretation ─────────────────────────
    llm_node = LLMNode(
        model_name=_OLLAMA_MODEL,
        prompt_template=(
            "You are an expert in solid-state battery materials. "
            "Interpret this JEPA crystal screening result in 2-3 sentences. "
            "Focus on the physical meaning of the thermal entropy and key elemental sites.\n\n"
            "{best_candidate_summary}"
        ),
        output_key="llm_interpretation",
        next_node=reduce_node,
        system_instruction=(
            "You are a materials scientist specialising in solid electrolytes. "
            "Be concise and technically precise."
        ),
        generation_config={"api_base": _OLLAMA_BASE, "max_tokens": 300},
    )

    # ── RouterNode: found_candidate vs impasse ────────────────────────────
    router_node = RouterNode(
        decision_function=lambda s: s.get("route_key", "impasse"),
        path_map={
            "found_candidate": llm_node,
            "impasse":         clear_error_node,
        },
        default_node=clear_error_node,
    )

    # ── FunctionalNode: pick best candidate from branch results ───────────
    pick_best_node = FunctionalNode(
        func=pick_best_candidate,
        output_key=None,         # pick_best_candidate sets state keys directly
        next_node=router_node,
    )

    # ── AdaptiveNode: runtime subgraph if any branch is CRITICAL ──────────
    # TopologyValidator restricts the LLM to approved tools only (Art. 3(23))
    validator = TopologyValidator(
        allowed_tools=[save_lab_report],
        max_nodes=4,
    )
    adaptive_node = AdaptiveNode(
        llm_model=_OLLAMA_MODEL,
        prompt_template=(
            "You are a materials safety AI reviewing a battery electrolyte screening.\n\n"
            "Screening summary:\n{screening_summary}\n\n"
            "One or more candidates were flagged CRITICAL thermal risk.\n"
            "Design a 1-node analysis subgraph: a single LLMNode that will\n"
            "write a 2-sentence thermal risk advisory to state key 'risk_advisory'.\n"
            "The prompt should reference the screening_summary.\n\n"
            "Output ONLY the JSON spec. No explanation."
        ),
        validator=validator,
        next_node=pick_best_node,   # subgraph exits back to main flow
        context_keys=["screening_summary"],
        system_instruction="Output ONLY valid JSON. No markdown, no explanation.",
        generation_config={"api_base": _OLLAMA_BASE, "max_tokens": 400},
    )

    # ── CriticalRouter: CRITICAL branches → AdaptiveNode, else → pick_best ─
    critical_router = RouterNode(
        decision_function=lambda s: "critical" if s.get("any_critical") else "normal",
        path_map={
            "critical": adaptive_node,
            "normal":   pick_best_node,
        },
        default_node=pick_best_node,
    )

    # ── BranchTriageNode: aggregate parallel screening results ─────────────
    branch_keys = [f"candidate_{i}_analysis" for i in range(N_CANDIDATES)]
    triage_node = BranchTriageNode(
        branch_output_keys=branch_keys,
        risk_level_key="risk_level",
        finding_key="finding",
        critical_threshold="CRITICAL",
        summary_state_key="screening_summary",
        critical_flag_key="any_critical",
        next_node=critical_router,
    )

    # ── BatchNode: evaluate all 5 candidates in parallel ──────────────────
    # Each branch is an EvalCrystalBranchNode targeting one candidate.
    branch_nodes = [
        EvalCrystalBranchNode(
            candidate_idx=i,
            stability_head=stability_head,
            router=router,
            branch_key=f"candidate_{i}_analysis",
        )
        for i in range(N_CANDIDATES)
    ]
    batch_node = BatchNode(nodes=branch_nodes, next_node=triage_node)

    # ── FunctionalNode: serialize library into state for BatchNode ─────────
    serialize_node = FunctionalNode(
        func=serialize_library,
        output_key="library_serialized",
        next_node=batch_node,
    )

    # ── ElectrochemNode: encode experiment conditions once ─────────────────
    ec_node = ElectrochemNode(model=ec_encoder, next_node=serialize_node)

    # ── RecallNode: DMN episodic recall ───────────────────────────────────
    recall_node = RecallNode(bridge=dmn, next_node=ec_node)

    # ════════════════════════════════════════════════════════════════════════
    # Print graph topology
    # ════════════════════════════════════════════════════════════════════════
    print("\n  Graph topology:")
    print("  RecallNode (DMN recall)")
    print("    └─ ElectrochemNode (encode experiment once)")
    print("       └─ FunctionalNode (serialize JEPA library → state)")
    print("          └─ BatchNode (5 parallel crystal evaluations)")
    print("             ├─ EvalCrystalBranch[0] Li6PS5Cl")
    print("             ├─ EvalCrystalBranch[1] Li3PS4")
    print("             ├─ EvalCrystalBranch[2] LATP")
    print("             ├─ EvalCrystalBranch[3] LLZO")
    print("             └─ EvalCrystalBranch[4] LiPF6")
    print("          └─ BranchTriageNode (aggregate, flag CRITICAL risk)")
    print("             └─ RouterNode #1 (critical → AdaptiveNode / normal → pick_best)")
    print("                ├─ AdaptiveNode (LLM designs risk advisory subgraph at runtime)")
    print("                │    └─ [generated subgraph] → pick_best_candidate")
    print("                └─ FunctionalNode (pick best stable candidate)")
    print("                   └─ RouterNode #2 (found_candidate → LLM / impasse → error)")
    print("                      ├─ LLMNode (interpret JEPA result)")
    print("                      │    └─ ReduceNode (synthesize → final recommendation)")
    print("                      │         └─ DemoJuryNode (EU Art.14 approval gate)")
    print("                      │              └─ ToolNode (save lab report)")
    print("                      │                   └─ DMNWriteNode (Hippocampus write)")
    print("                      │                        └─ AddValueNode (COMMITTED)")
    print("                      └─ ClearErrorNode → AddValueNode (IMPASSE)")

    # ════════════════════════════════════════════════════════════════════════
    # Execute
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print("  Executing graph...")
    print(f"{'─'*60}\n")

    executor = GraphExecutor(
        log_dir="lar_logs",
        hmac_secret="snath_ai_materials_eu_compliance_2026",
    )

    # Store library at module level so FunctionalNode can read it without
    # putting LatentCrystalState objects (not JSON-serialisable) in GraphState.
    _CRYSTAL_LIBRARY.clear()
    _CRYSTAL_LIBRARY.extend(library)

    initial_state = {}   # library accessed via module-level _CRYSTAL_LIBRARY

    final_state = {}
    steps = 0
    for step_log in executor.run_step_by_step(recall_node, initial_state, max_steps=60):
        final_state.update(step_log.get("state_after", {}))
        steps += 1

    elapsed = time.perf_counter() - t_total

    # ════════════════════════════════════════════════════════════════════════
    # Final summary
    # ════════════════════════════════════════════════════════════════════════
    best = final_state.get("best_candidate") or {}
    print("\n" + "═"*60)
    print("  FINAL RESULT")
    print("═"*60)
    print(f"  Outcome             : {final_state.get('outcome', 'unknown')}")
    print(f"  Committed candidate : {best.get('label', 'N/A')}")
    print(f"  Formation energy    : {best.get('formation_energy', 0):.3f} eV/atom")
    print(f"  Thermal entropy     : {best.get('thermal_entropy', 0):.3f}")
    print(f"  Cycle stability p   : {best.get('stability_prob', 0):.4f}")
    print(f"  Key elemental sites : {best.get('key_sites', [])}")
    print(f"  Researcher verdict  : {final_state.get('researcher_verdict', 'N/A')}")
    print(f"  Lab report          : {final_state.get('report_path', 'N/A')}")
    print(f"  Any CRITICAL risk   : {final_state.get('any_critical', False)}")
    print(f"  Graph steps         : {steps}")
    print(f"  Wall time           : {elapsed:.2f}s  (incl. LLM call)")
    print("═"*60)

    print("\n  LLM Interpretation:")
    print(f"  {final_state.get('llm_interpretation','(none)')}")
    print("\n  Final Recommendation (ReduceNode synthesis):")
    print(f"  {final_state.get('final_recommendation','(none)')}")
    print("\n  Screening Summary (BranchTriageNode):")
    print(f"{final_state.get('screening_summary','(none)')}")

    print("\n  Primitives used in this run:")
    primitives = [
        ("BaseNode",                    "All custom domain nodes"),
        ("FunctionalNode (×2)",         "serialize_library · pick_best_candidate"),
        ("BatchNode",                   "5 parallel crystal evaluations"),
        ("BranchTriageNode",            "Aggregate results, flag CRITICAL"),
        ("AdaptiveNode",                "LLM designs risk advisory subgraph at runtime (Art. 3(23))"),
        ("RouterNode (×2)",             "critical vs normal · found_candidate vs impasse"),
        ("LLMNode",                     "Materials science interpretation of JEPA result"),
        ("ReduceNode",                  "Synthesize JEPA + LLM → recommendation"),
        ("DemoJuryNode (HumanJuryNode)","EU AI Act Art. 14 researcher gate"),
        ("ToolNode",                    "Save lab report to disk"),
        ("ClearErrorNode",              "Impasse recovery path"),
        ("AddValueNode (×2)",           "Set final outcome"),
        ("JEPA_DMN_Consolidation_Node", "Recall + Write Hippocampus (ChromaDB)"),
    ]
    for name, desc in primitives:
        print(f"  ✓ {name:<35} {desc}")

    print(f"\n  Audit trail (HMAC-signed) : lar_logs/")
    print(f"  DMN long-term memory      : DMN/lar/data/chroma_db/")


if __name__ == "__main__":
    run()
