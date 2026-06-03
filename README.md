<div align="center">

# Lár-JEPA — Route Any Model. Not Just LLMs.

**The universal routing nervous system for heterogeneous cognitive architectures.**

<p align="center">
  <a href="https://github.com/snath-ai/lar">
    <img alt="Spine" src="https://img.shields.io/badge/Spine-Lár%20Engine%20v2.2.2-blue?style=for-the-badge">
  </a>
  <a href="https://github.com/snath-ai/Lar-JEPA">
    <img alt="Architecture" src="https://img.shields.io/badge/Architecture-Predictive%20World%20Models-blueviolet?style=for-the-badge">
  </a>
  <a href="https://github.com/snath-ai/Lar-JEPA/releases/tag/v2.2.3">
    <img alt="Release" src="https://img.shields.io/badge/Release-v2.2.3-green?style=for-the-badge">
  </a>
</p>

</div>

---

Every agentic framework built so far assumes one model type. LangChain assumes LLMs. When a JEPA world model outputs a 768-dimensional latent tensor, those frameworks crash — there is no signal type for it.

**Lár-JEPA routes anything.** LLMs, JEPAs, diffusion models, SSMs, GNNs — as first-class nodes in the same deterministic graph. The execution spine does not inspect model internals. It routes `AbstractCognitiveNode` instances. What the node does internally is irrelevant.

This is the difference between a framework built for chatbots and an architecture built for the next decade of AI research.

---

## Part of a Larger System

Lár-JEPA is the world model layer of a three-part cognitive architecture:

| Repository | Role |
| :--- | :--- |
| **[Lár](https://github.com/snath-ai/lar)** | The execution spine — deterministic graph engine, audit logging, EU AI Act compliance |
| **[Lár DMN](https://github.com/snath-ai/DMN)** | The memory layer — solves catastrophic forgetting with sleep/dream consolidation |
| **[Lár-JEPA](https://github.com/snath-ai/Lar-JEPA)** ← you are here | The world model — routes any cognitive architecture as a first-class node |

The industry is building the Brain (LLMs, JEPAs). We are building the Nervous System.

---

## Production Instantiations

The V1–V6 routing contract is domain-agnostic: the same routing core, temporal-decay
gate `W = exp(−λ · Δt)`, and System 1 / System 2 adapter pipeline run unmodified across
unrelated fields. Only the λ constants and failure-class labels change. Each repository
below is an independent, open-source instantiation built on Lár-JEPA:

| Repository | Domain | Stream A | Stream B | Failure classes |
| :--- | :--- | :--- | :--- | :--- |
| **[Snath Basis](https://github.com/snath-ai/snath-basis)** | Quantitative finance | Fundamental analysis | Market signals | `market_regime` / `structural` |
| **[Snath Aviation](https://github.com/snath-ai/snath-aviation)** | Aviation sensor routing | Radar | Pitot tube | `weather_induced` / `hardware_struct` |
| **[Snath Robotics](https://github.com/snath-ai/snath-robotics)** | Humanoid sensor routing | Vision | Proprioception | `environmental_transient` / `hardware_structural` |

This is the empirical claim of universal cognitive routing made concrete: the same
mathematical spine governs financial markets, aviation safety, and humanoid robotics
without modification to any Lár-JEPA primitive.

---

## The Problem

LLM agents hallucinate because their "memory" is a linear string of text. When step 3 of a 50-step plan goes wrong, the entire execution is doomed — the model has no internal model of physics, spatial logic, or long-term consequence. It predicts the next token.

JEPA world models solve this by predicting future states in latent space — planning by simulating consequences mathematically before any action touches the real world. But JEPAs don't output text. They output tensors. No existing framework can route them.

---

## What Lár-JEPA Enables

```python
# LLM → JEPA: LLM decides, JEPA simulates
LLMNode(output_key="action_spec") → JEPANode(output_key="predicted_state")

# JEPA → LLM: JEPA predicts, LLM reasons over the prediction
JEPANode(output_key="latent_prediction") → LLMNode(prompt_template="Given state {latent_prediction}...")

# Monte Carlo search in latent space
BatchNode([JEPANode(action="ACCELERATE"), JEPANode(action="BRAKE"), JEPANode(action="TURN")])
→ ReduceNode(output_key="best_action")

# Heterogeneous swarm — any mix
BatchNode([LLMNode(...), JEPANode(...), GNNNode(...)])

# Any future architecture
class MyFutureNode(AbstractCognitiveNode):
    model_type = ModelType.FUTURE
    # Routes without modification
```

---

## Running Examples

### JEPA + DMN Full Stack (recommended starting point)

The showcase runs the complete cognitive pipeline — JEPA prediction, entropic routing, DMN memory write, and warm recall — without any cloud APIs:

```bash
cd lar_jepa
python examples/jepa_dmn_showcase.py
```

```
---- Lár Engine v2.1.0 Successfully Imported ------
✅ [JEPA→DMN] Hippocampus connection established.

SCENARIO: Orbital insertion — stable trajectory
[RecallNode] Prior heuristics: (no prior heuristics)
[CognitiveNodeAdapter] Executing NBodyKinematicsJEPA (ModelType: JEPA)
[EntropicRouter] entropy=0.049 → COMMIT_TRAJECTORY
[WriteHeuristic] Trajectory written to DMN: True

SCENARIO: Orbital insertion — second attempt (warm context)
[RecallNode] Prior heuristics:
  - [JEPA Heuristic] Domain: spatial_kinematics | Outcome: committed | Entropic loss: 0.0495
[EntropicRouter] entropy=0.227 → COMMIT_TRAJECTORY
[AuditLogger] Log saved to: lar_logs/run_...json
```

Cycle 2 recalls Cycle 1's committed trajectory. The JEPA doesn't re-explore the same latent region. The AuditLogger writes a HMAC-signed trace at each step.

### Materials-JEPA: Real Trained JEPA on CPU (new)

A complete battery materials discovery pipeline trained and run entirely locally — no GPU, no cloud APIs.

**Step 1 — Train CrystalJEPA from scratch (61 seconds on MacBook CPU):**

```bash
cd lar_jepa
python examples/train_crystal_jepa.py
```

```
  CrystalJEPA Training — Joint Embedding Predictive Architecture
  embed_dim: 64  |  samples: 4,000  |  epochs: 80  |  device: cpu

  Epoch   1/80  loss=1.11731
  Epoch  80/80  loss=0.03342   (97.0% reduction in 60.8s)

  Saved → models/crystal_jepa_encoder.pt
```

**Step 2 — Run the full pipeline with trained weights:**

```bash
python examples/run_trained_demo.py
```

```
  ✅ [JEPA→DMN] Hippocampus connection established.
  [DMN Recall] 'Li6PS5Cl (Argyrodite)': (2 prior heuristics recalled)
  [ThermalStabilityRouter] COMMIT: thermal_entropy=0.208 — stable.
  [DMN Write] Heuristic committed: True

  Outcome  : stable_electrolyte_committed
  Candidate: Li6PS5Cl (Argyrodite)
  JEPA     : trained, 97% loss reduction
  Wall time: 520 ms
```

The trained JEPA produces genuinely differentiated representations — thermal entropies 0.17–0.27 vs the untrained encoder's uniform ~0.5. See [`DEMO_OUTPUT.md`](DEMO_OUTPUT.md) for the full captured output.

**Step 3 — Full showcase: all 12 Lár primitives + DMN (requires Ollama):**

```bash
ollama pull llama3.2
python examples/materials_full_showcase.py
```

Uses every Lár primitive in one graph: `FunctionalNode`, `BatchNode` (5 parallel branches), `BranchTriageNode`, `ReduceNode`, `LLMNode`, `HumanJuryNode`, `ToolNode`, `RouterNode`, `ClearErrorNode`, `AddValueNode`, `AdaptiveNode` — plus DMN recall and write at either end.

### Domain-Agnostic ABC Examples (updated in v2.3.0)

Six standalone pipelines demonstrating that the Lár ABC suite applies across structurally unrelated domains without modifying the execution spine. The first five demonstrate `AbstractModalEncoder`, `AbstractAttentionKernel`, `AbstractPerturbationOperator`, and `AbstractRoutingKernel`. The sixth (v2.3.0) demonstrates `AbstractDivergenceRouter` in a biomedical multi-stream setting:

| Example | Domain | ABCs Used | Key Signal |
|:---|:---|:---|:---|
| `finance_market_regime.py` | Quantitative finance | ModalEncoder, AttentionKernel, PerturbationOperator, RoutingKernel | Interest rate shock → RISK_ON / RISK_OFF / HEDGE |
| `industrial_predictive_maintenance.py` | Wind-turbine gearbox | ModalEncoder, AttentionKernel, PerturbationOperator, RoutingKernel | Bearing degradation → EMERGENCY / SCHEDULE / NOMINAL |
| `cybersecurity_intrusion_detector.py` | Enterprise network security | ModalEncoder, AttentionKernel, PerturbationOperator, RoutingKernel | Lateral movement → QUARANTINE / ESCALATE / MONITOR |
| `climate_perturbation_model.py` | Earth-systems / climate | ModalEncoder, AttentionKernel, PerturbationOperator, RoutingKernel | CO₂ forcing shock → GLOBAL / REGIONAL / ARCHIVE |
| `av_sensor_fusion.py` | Autonomous vehicle perception | ModalEncoder ×2, AttentionKernel, PerturbationOperator, RoutingKernel | Sensor degradation → CAMERA_PRIMARY / LIDAR_PRIMARY / FUSION |
| *(v2.3.0)* Medical imaging | Chest X-ray + radiology report | **DivergenceRouter** (V1–V6): `stream_a` = scan latent (ViT/BioViL), `stream_b` = report latent (BioBERT) | Image–report disagreement → Execute / Investigate / Defer / Halt |

Each example runs zero-dependency (only `torch`) and produces a HMAC-signed audit record:

```bash
cd lar_jepa
python examples/finance_market_regime.py
python examples/industrial_predictive_maintenance.py
python examples/cybersecurity_intrusion_detector.py
python examples/climate_perturbation_model.py
python examples/av_sensor_fusion.py
```

The same mathematical spine is structurally identical across all domains. Domain semantics are entirely encapsulated in the ABC implementations. No modification to any Lár primitive is required.

### Single-node wall avoidance (original PoC)

```bash
poetry run python examples/advanced/13_world_model_jepa.py
```

```
[JEPA] Current State: [X=8, V=5]
[JEPA] Simulating: ACCELERATE → Predicted: {x: 18, v: 10}

[System 2 Router] CRASH DETECTED at X=10. Vetoing.
[RouterNode] → REPLAN_NODE

[JEPA] Simulating: BRAKE → Predicted: {x: 8, v: 0}

[System 2 Router] Simulation safe. Action approved.
[RouterNode] → EXECUTE_NODE
[MOTOR] Executing. Agent avoided the crash entirely.
```

---

## Routing Any Model — Current and Future

`AbstractCognitiveNode` is the only contract the Lár executor requires. A node declares its `ModelType` and implements `encode()`, `forward()`, `decode()`. The graph executor calls these methods. It never inspects internals.

This means any model architecture — present or future — is routable the moment it implements the ABC:

```python
# Models that exist today — all routable as first-class nodes
class GPT4Node(AbstractCognitiveNode):       model_type = ModelType.LLM
class GeminiNode(AbstractCognitiveNode):     model_type = ModelType.LLM
class MambaNode(AbstractCognitiveNode):      model_type = ModelType.SSM
class DiTNode(AbstractCognitiveNode):        model_type = ModelType.DIFFUSION
class GraphSAGENode(AbstractCognitiveNode):  model_type = ModelType.GNN
class CrystalJEPANode(AbstractManifold):     model_type = ModelType.JEPA

# Models that do not yet exist — also routable, without modifying the spine
class FutureFoundationModel(AbstractCognitiveNode):
    model_type = ModelType.FUTURE
    def encode(self, signal): ...   # whatever the architecture produces

# Heterogeneous ensemble — mixed model types, single BatchNode
# All four run concurrently; each writes its output key to state
batch = BatchNode([GPT4Node(), MambaNode(), CrystalJEPANode(), FutureFoundationModel()])

# Option A — LLM-based synthesis (ReduceNode actual signature)
reduce = ReduceNode(
    model_name="ollama/llama3",
    prompt_template="Synthesize these model outputs into a single decision:\n"
                    "LLM: {gpt4_result}\nSSM: {mamba_result}\n"
                    "JEPA: {jepa_result}\nFoundation: {future_result}",
    input_keys=["gpt4_result", "mamba_result", "jepa_result", "future_result"],
    output_key="ensemble_decision",
)
batch.next_node = reduce

# Option B — Programmatic confidence-weighted merge (no LLM call)
from lar import node

@node(output_key="ensemble_decision")
def confidence_weighted_reduce(state):
    keys = ["gpt4_result", "mamba_result", "jepa_result", "future_result"]
    pairs = [(state.get(k), state.get(f"{k}_confidence", 1.0)) for k in keys]
    pairs = [(v, c) for v, c in pairs if v is not None]
    total = sum(c for _, c in pairs)
    return sum(v * c / total for v, c in pairs)
```

The `AbstractRoutingKernel` extends this to routing *decisions* — not just model execution. Any routing mechanism (threshold, RL policy, Bayesian posterior, ensemble vote, uncertainty estimate) satisfies `score() → float, route() → str` and plugs into the graph without modification. A routing kernel trained in 2028 on a dataset that doesn't exist yet is a valid `AbstractRoutingKernel` today.

The `AbstractModalEncoder` extends this to *inputs* — any sensor, any modality, any data source encodes to `(B, output_dim)` and becomes addressable by the attention and routing layers. A camera encoder and a seismic sensor encoder and a spectroscopic encoder are interchangeable from the graph's perspective.

**The invariant:** the graph executor is sealed. The ABCs are the extension points. Domain logic, model architecture, and routing strategy all live behind interfaces — the spine never changes.

---

## Why Lár Is Structurally Superior for World Models

| Requirement | LangChain / AutoGPT | Lár-JEPA |
|:---|:---|:---|
| **Tensor routing** | Crashes — no signal type | Native. `GraphState` passes tensors transparently. |
| **Mathematical routing logic** | LLM call to decide next step | Deterministic Python `RouterNode` — `if collision_prob > 0.85: return "REPLAN"` |
| **Tensor audit logging** | Not supported | `TensorSafeEncoder` (fully implemented in Lár engine core) safely serialises tensors to metadata: `{"__type__": "Tensor", "shape": [1, 768]}` |
| **Heterogeneous model swarm** | Single model type assumed | `BatchNode([LLM, JEPA, GNN])` — aggregated by `ReduceNode` |
| **Safety rollback** | Hope the LLM doesn't hallucinate | `RouterNode` vetoes bad predicted states before execution |
| **Long-term learning** | None | DMN sleep cycle consolidates JEPA simulations into persistent heuristics |

---

## TensorSafeEncoder & Native Tensor Routing

At the heart of Lár-JEPA's capabilities is the `TensorSafeEncoder` natively implemented within the Lár engine core. 

When a standard agent attempts to log state, it uses standard `json.dumps()`, which immediately crashes if the state contains PyTorch tensors or multidimensional NumPy arrays. The `TensorSafeEncoder` intercepts these structures natively during the graph's execution and audit-logging phases. 
Instead of crashing or attempting to write gigabytes of floating-point values, it translates the mathematical states into safe, auditable metadata (e.g. `{"__type__": "Tensor", "shape": [1, 768]}`). 

This means you can route gigabyte-sized biological tensors across `AbstractManifold` nodes while retaining cryptographic, EU AI Act-compliant audit traces for every step.

---

## EU AI Act Compliance

Lár-JEPA is architecturally structured to satisfy the requirements of the European Union AI Act for high-risk systems. It implements core primitives for:
* **Article 12 (Record-Keeping):** HMAC-SHA256 cryptographically signed tensor audit logs.
* **Article 13 (Transparency):** Deterministic, glass-box graph execution logic.
* **Article 14 (Human Oversight):** Pausable `GraphState` transitions allowing manual override.
* **Article 15 (Robustness):** Entropic routing to prevent hallucinated loops.

See [EU_AI_ACT_COMPLIANCE.md](EU_AI_ACT_COMPLIANCE.md) for full details.

---

## JEPA ↔ DMN Memory Loop

JEPA simulations are expensive. Running the same latent-space search twice is waste. The `JEPA_DMN_Consolidation_Node` closes the loop:

```
Planning Cycle N
  RecallNode → queries DMN Hippocampus for past committed trajectories
  CognitiveNodeAdapter (JEPA) → predict, score entropy
  EntropicRouter → COMMIT_TRAJECTORY
  WriteHeuristicNode → writes trajectory + entropy score to ChromaDB

Planning Cycle N+1
  RecallNode → retrieves Cycle N heuristic as warm context
  JEPA → informed by past success, skips known-bad latent regions
  ...
```

The Hippocampus is ChromaDB-backed with two collections: `long_term_memory` (raw JEPA traces) and `warm_memory` (semantic summaries produced by the DMN Dreamer during idle). Both tiers are searched at recall time.

Runs without cloud APIs. Set `OLLAMA_HOST` for real vector embeddings; the bridge degrades gracefully to no-op if ChromaDB is unavailable — JEPA execution never blocks.

---

## Architecture Components

**`AbstractCognitiveNode`** — Universal base class. Any model type implements this to become routable. Declares its `ModelType` and exposes `encode()`, `forward()`, `decode()` contract. The spine never inspects beyond this interface. An LLM, a JEPA world model, a GNN, and a diffusion model are all identical from the executor's perspective.

**`AbstractManifold`** — JEPA-specific subclass of `AbstractCognitiveNode` for continuous latent-space world models. Specialises the interface to `embed_context()`, `predict_target()`, and `entropic_loss()`. Any architecture implementing this is routable in a `BatchNode` alongside `LLMNode` instances without modification.

**`AbstractContextBridge`** — Stateless signal adapters for cross-modal composition. Allows LLMs to attend to JEPA latent predictions and JEPAs to condition on LLM semantic embeddings — without either node knowing about the other's internals.

**`AbstractLatentFaultLocator`** — Formal specification of the Topological Vulnerability Targeting Engine (`core/interfaces.py`). Defines the mathematical principle for cross-modal fault localisation: given an environmental state observation and a structural topology sequence, encode both into a shared latent space, apply cross-attention (environmental state as Query, structural sequence as Key/Value), and extract the top-k structural positions at highest risk.

```
encode_environmental_state(x_E) → (B, D)         pooled Query
encode_structural_sequence(x_S) → (1, N_S, D)    positional Key/Value
localize_fault_coordinates(z_E, z_S, k) → (risk_score, coordinates, attention)
```

Six mathematical invariants (I1–I6) are formally specified and mechanically enforced by a 32-test behavioral invariant suite (`lar_jepa/tests/unit/test_latent_fault_locator_invariants.py`) passing across **four structurally unrelated domains**:

| Domain | x_E (environmental state) | x_S (structural sequence) | C (fault coordinates) |
|--------|--------------------------|--------------------------|----------------------|
| **Materials** | Electrochemical operating conditions | Crystal lattice elemental sites | Topk instability sites |
| **Seismic** | Crustal stress field readings | Geological fault segment topology | Topk seismic risk coordinates |
| **Infrastructure** | Network traffic load telemetry | Server/router graph topology | Topk critical failure nodes |
| **Industrial** | Multi-channel vibration/thermal sensor array | Drivetrain component positions | Topk mechanical fault loci |

The same `CrossAttentionHead` architecture — `query_proj`, `key_proj`, `value_proj`, scaled dot-product attention, topk extraction — applies identically across crystal physics, geophysics, computer networks, and genomics. Any future implementation extending this ABC and passing invariants I1–I6 is a Derivative Work of this specification (Apache 2.0, prior art anchored by RFC 3161 certificate and Zenodo DOIs `10.5281/zenodo.19245328`, `10.5281/zenodo.19484646`, `10.5281/zenodo.19646405`).

Run the full invariant suite:
```bash
pytest lar_jepa/tests/unit/test_latent_fault_locator_invariants.py -v
# 32 passed: materials_domain · seismic_domain · network_infrastructure_domain · industrial_domain
```

**`AbstractAttentionKernel`** — Decouples the attention *mechanism* from the fault localisation pipeline. Any mechanism producing a normalised distribution over N positions and extracting k ordered indices satisfies the specification. Six invariants (A1–A6). Reference implementations: `ScaledDotProductKernel` (softmax(QKᵀ/√D)), `CosineAttentionKernel`. Valid future implementations: `LinearAttentionKernel`, `SparseAttentionKernel`, `SSMKernel`, `HyenaKernel` — any mechanism satisfying A1–A6 is a Derivative Work.

**`AbstractPerturbationOperator`** — Formal specification of latent-space counterfactual prediction. Formalises the pattern:
```
Δ      = encode_mutant(x_mut) − encode_wildtype(x_wt)
z_pred = z_ctrl + α · Δ
```
Zero-shot intervention prediction in any domain — without executing the intervention in the physical world. Six invariants (P1–P6). Reference implementations: `CrystalDefectOperator` (perfect vs defect-injected crystal), `MolecularBindingOperator` (unbound vs ligand-bound conformation), `BearingDegradationOperator` (healthy vs degraded sensor signature), `InterestRateShockOperator` (baseline vs stressed portfolio), `LateralMovementOperator` (current-hop vs next-hop network state), `CO2ShockOperator` (baseline vs elevated-forcing atmosphere), `SensorDegradationOperator` (nominal vs adverse-weather perception). Domain instantiations: materials defect simulation, molecular dynamics, industrial condition monitoring, quantitative finance, network security, climate modelling, autonomous vehicle perception.

**`AbstractRoutingKernel`** — Decouples routing *logic* from routing *mechanism*. Formalises the score-then-route pattern enabling deterministic, learned, probabilistic, and adaptive routing on the same graph executor. Four invariants (R1–R4). Reference implementations: `EntropicThresholdKernel` (current Lár-JEPA pattern), `MultiThresholdRoutingKernel`. Valid future implementations: `LearnedPolicyKernel` (RL), `EnsembleVoteKernel`, `UncertaintyKernel`, `CalibratedBayesianKernel`.

**`AbstractModalEncoder`** — Universal modality-to-latent-space encoding interface. Separates domain-specific encoding logic from all downstream attention, routing, and memory operations — enabling plug-and-play encoder replacement without modifying any other pipeline component. Three invariants (M1–M3). Reference implementations: `ElectrochemicalEncoder`, `NetworkTelemetryEncoder`, `MarketStateEncoder` (price/vol/macro), `VibrothermalEncoder` (vibration/temperature), `AtmosphericStateEncoder` (ERA5 reanalysis), `CameraEncoder` (BEV patch features), `LidarEncoder` (range-view point cloud). Valid future implementations: spectroscopic encoder, protein structure encoder, seismic sensor encoder — any architecture producing `(B, output_dim)` output satisfies M1–M3.

**`AbstractDivergenceRouter`** *(tenth ABC — v2.3.0)* — Multi-stream routing primitive. Keeps two independent latent streams separate, measures their geometric divergence, and treats high-confidence disagreement as the primary control signal rather than noise to be averaged away. Six invariants (V1–V6). The Investigate rule is the key contribution: when both streams are confident but contradictory, the divergence is flagged as `TRIGGER_REPLAN` — not fused, not averaged. When used as training infrastructure, high-divergence cases automatically accumulate as `D_hard` — the self-curating curriculum at the model's uncertainty boundary. The **Safety-Learning Equivalence** (proved in [DOI 10.5281/zenodo.20278781](https://doi.org/10.5281/zenodo.20278781)) establishes that the invariants enforcing routing safety are identical to the invariants making divergence a valid training signal. Domain instantiations: medical imaging (scan vs. report), vision-language (image vs. caption), autonomous vehicles (sensor vs. map), cybersecurity (behaviour vs. policy).

Run the full interface invariant suite (151 tests total):
```bash
pytest lar_jepa/tests/unit/ -v
# 151 passed across AbstractLatentFaultLocator (I1–I6), AbstractAttentionKernel (A1–A6),
# AbstractPerturbationOperator (P1–P6), AbstractRoutingKernel (R1–R4), AbstractModalEncoder (M1–M3)
```

**`JEPA_DMN_Consolidation_Node`** — Live bridge writing committed JEPA trajectories into the DMN episodic memory store (ChromaDB). Expensive JEPA simulations become cheap long-term heuristics during sleep consolidation.

**`CrystalJEPA`** — A real JEPA model for battery materials. Three-component architecture: `CrystalSiteEncoder` (2-layer Transformer over 20 elemental sites), EMA `TargetEncoder` (no gradients), and a `Predictor` that maps visible-site context to masked-site representations. Trained with JEPA loss — MSE in latent space with stop-gradient on the target. 68,736 parameters. Trains to 97% loss reduction in 61 seconds on CPU. Interchangeable with any `AbstractManifold` implementation — GNN, physics-informed net, or simulation engine.

**`ElectrochemicalJEPA`** — Encoder for electrochemical impedance data (capacity retention, cycle count, temperature). Designed for real EIS datasets (MPContribs, NREL ECDH) — currently trained on synthetic data as a structural placeholder.

**`CycleStabilityHead`** — Cross-attention head that attends from electrochemical embeddings to site embeddings to predict cycle stability probability. Needs real electrochemical labels to train; architecture is production-ready.

**`Spatial Kinematics Engine`** — Reference implementation in `spatial_kinematics_engine/`. N-body spatial modeling: coordinate interactions, trajectory dependencies, collision heuristics for non-linear multi-body meshes. Domain-agnostic — the same engine applies to robotic kinematics, molecular dynamics, or orbital mechanics.

---

## Empirical Results — AbstractDivergenceRouter (V1–V6)

The `AbstractDivergenceRouter` (tenth ABC, v2.3.0) was empirically validated across two structurally unrelated domains in:

> **Divergence Is Not Noise: Multi-Stream Routing Without Modal Fusion and the Safety-Learning Equivalence**
> Aadithya Vishnu Sajeev — DOI: [10.5281/zenodo.20278781](https://doi.org/10.5281/zenodo.20278781)

| Domain | Model | Result |
|:-------|:------|:-------|
| Medical imaging (NLM Indiana CXR) | BiomedCLIP | Mean-D lift **3.11×** vs fusion's **1.08×** — same backbone, same data, scoring function only (the primary signal). TRIGGER_REPLAN lift **6.5×**. Routing AUROC 0.87 vs Fusion 0.85; the two-point gap is within CI noise at N_test = 100 and is not the claim. |
| General vision-language (MSCOCO) | OpenCLIP ViT-B-32 | Under oracle that equalises fusion's CLS signal: Routing AUROC **0.72** (lift 1.51×, p = 5.7×10⁻⁵) vs Fusion AUROC **0.51** (lift 1.00× — completely blinded). |

Raw result data for all experiments referenced in the paper — including the full experimental progression (design failures F1–F4 and their fixes), Tier-1 ablations (vocabulary size, calibrated comparison, Δ-space geometry) — are in [`experiments/results/`](experiments/results/).

Each JSON file is a self-contained record: model config, calibration thresholds, per-sample decisions, aggregate statistics. All paper figures are reproducible directly from these files.

---

## License

Apache 2.0. Built on the [Lár Engine](https://github.com/snath-ai/lar). See `ARCHITECTURE.md` for the full nervous system design.
