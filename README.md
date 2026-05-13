<div align="center">

# Lár-JEPA — Route Any Model. Not Just LLMs.

**The universal routing nervous system for heterogeneous cognitive architectures.**

<p align="center">
  <a href="https://github.com/snath-ai/lar">
    <img alt="Spine" src="https://img.shields.io/badge/Spine-Lár%20Engine%20v2.1.0-blue?style=for-the-badge">
  </a>
  <a href="https://github.com/snath-ai/Lar-JEPA">
    <img alt="Architecture" src="https://img.shields.io/badge/Architecture-Predictive%20World%20Models-blueviolet?style=for-the-badge">
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

## Why Lár Is Structurally Superior for World Models

| Requirement | LangChain / AutoGPT | Lár-JEPA |
|:---|:---|:---|
| **Tensor routing** | Crashes — no signal type | Native. `GraphState` passes tensors transparently. |
| **Mathematical routing logic** | LLM call to decide next step | Deterministic Python `RouterNode` — `if collision_prob > 0.85: return "REPLAN"` |
| **Tensor audit logging** | Not supported | `TensorSafeEncoder` (now fully implemented in Lár v2.1.0 engine core) safely serialises tensors to metadata: `{"__type__": "Tensor", "shape": [1, 768]}` |
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

**`AbstractCognitiveNode`** — Universal base class. Any model type implements this to become routable. Declares its `ModelType` and exposes a `execute(state)` interface. The spine never inspects beyond this contract.

**`AbstractManifold`** — JEPA-specific subclass for continuous latent-space world models. Handles tensor state transport, prediction logging, and safety rollback hooks.

**`AbstractContextBridge`** — Stateless signal adapters for cross-modal composition. Allows LLMs to attend to JEPA latent predictions and JEPAs to condition on LLM semantic embeddings — without either node knowing about the other's internals.

**`JEPA_DMN_Consolidation_Node`** — Live bridge writing committed JEPA trajectories into the DMN episodic memory store (ChromaDB). Expensive JEPA simulations become cheap long-term heuristics during sleep consolidation.

**`CrystalJEPA`** — A real JEPA model for battery materials. Three-component architecture: `CrystalSiteEncoder` (2-layer Transformer over 20 elemental sites), EMA `TargetEncoder` (no gradients), and a `Predictor` that maps visible-site context to masked-site representations. Trained with JEPA loss — MSE in latent space with stop-gradient on the target. 68,736 parameters. Trains to 97% loss reduction in 61 seconds on CPU. Interchangeable with any `AbstractManifold` implementation — GNN, physics-informed net, or simulation engine.

**`ElectrochemicalJEPA`** — Encoder for electrochemical impedance data (capacity retention, cycle count, temperature). Designed for real EIS datasets (MPContribs, NREL ECDH) — currently trained on synthetic data as a structural placeholder.

**`CycleStabilityHead`** — Cross-attention head that attends from electrochemical embeddings to site embeddings to predict cycle stability probability. Needs real electrochemical labels to train; architecture is production-ready.

**`Spatial Kinematics Engine`** — Reference implementation in `spatial_kinematics_engine/`. N-body spatial modeling: coordinate interactions, trajectory dependencies, collision heuristics for non-linear multi-body meshes. Domain-agnostic — the same engine applies to robotic kinematics, molecular dynamics, or orbital mechanics.

---

## License

Apache 2.0. Built on the [Lár Engine](https://github.com/snath-ai/lar). See `ARCHITECTURE.md` for the full nervous system design.
