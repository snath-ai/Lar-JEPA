# Lár-JEPA: The Cognitive Nervous System

## What Lár-JEPA Is

Lár-JEPA is **not** an orchestration framework for a specific model type.

It is a **universal cognitive routing nervous system**: a deterministic execution
spine that routes signals between heterogeneous model types — large language
models (LLMs), Joint-Embedding Predictive Architectures (JEPAs), and any
cognitive architecture that follows — as first-class, equally routable nodes
within the same directed acyclic graph (DAG) executor.

The execution spine does not know or care what is inside a node. It routes
signals between them. This makes the nervous system model-agnostic by
construction, and forward-compatible by definition.

---

## The Three Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                    COGNITIVE NODES                               │
│  LLMNode │ JEPANode │ DiffusionNode │ SSMNode │ GNNNode │ ...   │
│       (Any AbstractCognitiveNode implementation)                 │
├─────────────────────────────────────────────────────────────────┤
│                  SIGNAL BRIDGE LAYER                             │
│         AbstractContextBridge implementations                    │
│   (cross-modal signal format adaptation between node types)      │
├─────────────────────────────────────────────────────────────────┤
│                  ROUTING SPINE (Lár DAG)                         │
│  GraphExecutor │ RouterNode │ BatchNode │ ReduceNode │ ...        │
│     (deterministic, model-agnostic, HMAC-signed audit log)       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    Default Mode Network (DMN)
              (episodic → semantic → procedural memory)
```

### 1. The Routing Spine (Lár DAG)

The `GraphExecutor` runs one node at a time. A `RouterNode` is a pure Python
function that reads `GraphState` and returns a string routing decision. A
`BatchNode` fans out across N nodes concurrently in isolated `GraphState`
copies and merges results on fan-in.

The spine never inspects node internals. It operates entirely on `GraphState`
fields — which may carry text, latent tensors, LGSL specifications, graphs,
images, or any other signal type.

### 2. The Cognitive Nodes

Every model — LLM, JEPA, diffusion, SSM, or anything else — implements
`AbstractCognitiveNode`:

```python
class AbstractCognitiveNode(ABC):
    model_type: ModelType          # LLM | JEPA | DIFFUSION | SSM | FUTURE | ...

    def encode(self, input_signal) -> Any   # Signal → internal context
    def forward(self, context) -> Any       # Internal inference pass
    def decode(self, representation) -> Any # Internal output → GraphState signal
    def output_signal_type -> SignalType    # Declares what this node writes
```

This is the neuron specification. No node implementation touches the routing
spine. No routing spine implementation touches a node.

### 3. The Bridge Layer

`AbstractContextBridge` adapts one node's output `SignalType` to another
node's expected `encode()` input — enabling cross-modal composition without
either node being aware of the other's internals.

```python
class AbstractContextBridge(ABC):
    source_signal_type: SignalType  # What this bridge consumes
    target_signal_type: SignalType  # What this bridge produces
    def bridge(source_output, target_node_type) -> Any
```

---

## The Nine Abstract Base Classes

`core/interfaces.py` defines nine ABCs that together form the **complete cognitive
contract** of the Lár-JEPA system. Every concrete implementation is a Derivative Work
under Apache 2.0. Every invariant is machine-checkable, domain-agnostic, and
timestamped in the Zenodo prior-art chain.

---

### 1. `AbstractCognitiveNode`

The universal routable node specification. Every model — LLM, JEPA, diffusion, SSM,
or any future architecture — implements this interface.

```python
class AbstractCognitiveNode(ABC):
    model_type: ModelType          # LLM | JEPA | DIFFUSION | SSM | FUTURE | ...

    def encode(self, input_signal) -> Any      # Signal → internal context
    def forward(self, context) -> Any          # Internal inference pass
    def decode(self, representation) -> Any    # Internal output → GraphState signal
    def output_signal_type(self) -> SignalType # Declares what this node writes
```

**Domain examples:** `GridCognitiveNode` (power-grid), `GPT4Node` (language),
`MambaNode` (SSM), `CrystalJEPANode` (materials).

---

### 2. `AbstractManifold`

The JEPA world-model contract. Any model that embeds context, predicts target latents,
and produces an entropic loss implements this ABC.

```python
class AbstractManifold(ABC):
    def embed_context(self, x_context) -> torch.Tensor    # Context → latent
    def predict_target(self, z_context, action=None) -> torch.Tensor  # Predict next state
    def entropic_loss(self, z_predicted, z_target) -> torch.Tensor    # VICReg-style loss
```

**Invariant:** `embed_context` and `predict_target` must operate in the same latent
space — i.e. `embed_context(x).shape[-1] == predict_target(z, a).shape[-1]`.

**Domain examples:** `GridCascadeJEPA` (power-grid cascade prediction),
`CrystalJEPA` (materials phase prediction), `GeneJEPA` (expression prediction).

---

### 3. `AbstractContextBridge`

A stateless signal adaptor. Converts one node's output `SignalType` to another node's
expected input format without either node being aware of the other.

```python
class AbstractContextBridge(ABC):
    source_signal_type: SignalType  # What this bridge consumes
    target_signal_type: SignalType  # What this bridge produces

    def bridge(self, source_output, target_node_type) -> Any
```

**Invariant:** `bridge()` is a pure function — no side effects, no state mutations.

**Domain examples:** `SensorTopologyBridge` (grid sensor→topology), seismic→structural,
market microstructure→latent embedding.

---

### 4. `AbstractLatentFaultLocator`

Cross-attention fault localisation. Identifies the positions in a structural sequence
most causally responsible for an observed anomaly in an environmental signal. The
seminal ABC in the prior-art chain.

```python
class AbstractLatentFaultLocator(ABC):
    def encode_environmental_state(self, x_E)                   # I1: (B, D) pooled
    def encode_structural_sequence(self, x_S)                   # I2: (1, N_S, D) positional
    def localize_fault_coordinates(self, z_env, z_struct, k=3)  # I3–I6: topk indices
```

**Invariants:**
- **I1**: `encode_environmental_state(x_E)` → `(B, D)` pooled representation
- **I2**: `encode_structural_sequence(x_S)` → `(1, N_S, D)` positional representation
- **I3**: attention weights over structural positions → `(B, N_S)` probability distribution
- **I4**: risk score ∈ [0, 1]
- **I5**: topk fault coordinates → `list[int]` of length k
- **I6**: coordinates are valid indices into the structural sequence

**Domain isomorphisms:**

| Domain | x_E (environmental signal) | x_S (structural sequence) | Fault coordinates |
|--------|---------------------------|--------------------------|-------------------|
| Power grid | sensor telemetry (voltage, current) | bus-line topology sequence | transmission line indices |
| Materials | characterisation profile (XRD/spectroscopy) | crystal lattice graph | defect sites |
| Seismology | subsurface pressure / strain state | fault geology sequence | rupture loci |
| Cybersecurity | runtime syscall / memory profile | binary instruction sequence | vulnerable offsets |
| Finance | volatility regime / microstructure | order book sequence | regime shift points |
| Infrastructure | environmental sensor array | structural element sequence | degradation loci |

---

### 5. `AbstractEntropicRouter`

The entropy gate that sits downstream of any JEPA world-model prediction. Evaluates
the entropic loss of a predicted latent state and returns a `RouteDecision` enum value
that controls graph branching.

```python
class AbstractEntropicRouter(ABC):
    def evaluate_state(self, entropic_loss: float, **context) -> RouteDecision
```

**`RouteDecision` enum:**
```python
class RouteDecision(Enum):
    COMMIT_TRAJECTORY   # Prediction confidence sufficient — proceed
    TRIGGER_REPLAN      # Moderate uncertainty — request replanning
    STRUCTURAL_IMPASSE  # High uncertainty — halt, escalate
```

**Domain examples:** `GridEntropicRouter` (power-grid), climate resolution router,
market regime confirmation gate.

---

### 6. `AbstractAttentionKernel`

A pluggable attention mechanism. Decouples the attention algorithm (full quadratic,
linear, sparse-window, Hyena convolution, SSM/Mamba-class) from the locator or
routing component that uses it.

```python
class AbstractAttentionKernel(ABC):
    def compute(
        self,
        query: torch.Tensor,   # (B, D_q)
        key: torch.Tensor,     # (B, N, D_k)
        value: torch.Tensor,   # (B, N, D_v)
        k: int = 3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Returns: (attention_weights (B, N), topk_indices (B, k))
```

**Invariants:**
- **A1**: `attention_weights` ∈ [0, 1] and sum to 1 over N
- **A2**: `topk_indices` are valid indices into the N dimension of key/value
- **A3**: deterministic for identical inputs (no dropout at inference)
- **A4**: complexity class declared in docstring (O(N²), O(N), O(N log N), etc.)
- **A5**: numerically stable (no raw softmax overflow on long sequences)
- **A6**: `k ≤ N`

**Complexity variants in the example suite:**
| Kernel | Complexity | Use case |
|--------|-----------|----------|
| `LinearAttentionGridKernel` | O(N) | Power-grid (long topology sequences) |
| `CosineAttentionFaultKernel` | O(N²) | Industrial (short sensor windows) |
| `SparseWindowAttentionKernel` | O(N·W) | Cybersecurity (sliding packet windows) |
| `HyenaConvAttentionKernel` | O(N log N) | Climate (planetary-scale sequences) |
| `SSMAttentionKernel` | O(N) causal | Autonomous vehicles (real-time causal streams) |

---

### 7. `AbstractPerturbationOperator`

Latent-space counterfactual reasoning. Computes the delta between a baseline state
and a perturbed state entirely in latent space, then predicts the outcome of the
perturbation without executing it in the real world.

```python
class AbstractPerturbationOperator(ABC):
    def encode_wildtype(self, x_baseline) -> torch.Tensor    # Baseline → latent (P1)
    def encode_mutant(self, x_perturbed) -> torch.Tensor     # Perturbed → latent (P2)
    def perturbation_vector(
        self,
        z_baseline: torch.Tensor,
        z_perturbed: torch.Tensor,
    ) -> torch.Tensor                                         # Δz = z_p − z_b  (P3)
    def predict_perturbed_state(
        self,
        z_baseline: torch.Tensor,
        perturbation: torch.Tensor,
    ) -> torch.Tensor                                         # ẑ_p (P4–P6)
```

**Invariants:**
- **P1**: `encode_wildtype` and `encode_mutant` map to the same latent space
- **P2**: `perturbation_vector = encode_mutant − encode_wildtype` (additive in latent space)
- **P3**: `perturbation_vector` shape matches latent embedding shape
- **P4**: `predict_perturbed_state(z_b, Δz)` is differentiable w.r.t. Δz
- **P5**: when Δz = 0, `predict_perturbed_state` ≈ `encode_wildtype` (identity baseline)
- **P6**: operator does not access real-world state — purely latent counterfactual

> **Terminology note:** `encode_wildtype` / `encode_mutant` are the ABC method names
> inherited from perturbation theory (Schrödinger, 1926). "Wildtype" = unperturbed
> baseline; "mutant" = post-perturbation state. These terms are pure mathematics —
> domain-agnostic across finance (interest-rate shock), grid (line trip), climate
> (CO₂ forcing), and security (lateral movement).

**Domain examples:**

| Operator | x_baseline | x_perturbed | Perturbation |
|----------|-----------|-------------|--------------|
| `LineTripOperator` | normal grid state | post-trip grid state | transmission line removal |
| `InterestRateShockOperator` | pre-hike market state | post-hike state | Fed rate delta |
| `BearingDegradationOperator` | healthy bearing signal | degraded bearing | mechanical wear delta |
| `LateralMovementOperator` | clean network telemetry | compromised telemetry | attacker δ |
| `CO2ShockOperator` | pre-industrial atmosphere | elevated CO₂ state | forcing delta |
| `SensorDegradationOperator` | clean sensor reading | rain/fog-degraded reading | SNR delta |

---

### 8. `AbstractRoutingKernel`

Score-then-route. Computes a continuous score over a graph state, then maps that
score to a discrete routing string consumed by a `RouterNode`.

```python
class AbstractRoutingKernel(ABC):
    def score(self, state: GraphState) -> float    # R1–R3
    def route(self, state: GraphState) -> str      # R4
```

**Invariants:**
- **R1**: `score()` ∈ ℝ (unbounded; callers normalise as needed)
- **R2**: `score()` is deterministic for identical state
- **R3**: `score()` does not mutate state
- **R4**: `route()` returns a string key present in the downstream `RouterNode` path_map

**Domain examples:**
| Kernel | Routing decisions |
|--------|-----------------|
| `GridActionKernel` | `ISOLATE_FAULT` / `REBALANCE_LOAD` / `MONITOR` |
| `RegimeRoutingKernel` | `BULL_MOMENTUM` / `BEAR_MOMENTUM` / `SIDEWAYS` / `CRISIS` |
| `MaintenanceRoutingKernel` | `EMERGENCY` / `SCHEDULE_MAINTENANCE` / `NOMINAL` |
| `ThreatRoutingKernel` | `QUARANTINE` / `ESCALATE` / `MONITOR` |
| `ClimateResolutionKernel` | `GLOBAL_INTERVENTION` / `REGIONAL_ADAPTATION` / `ARCHIVE` |
| `SensorTrustKernel` | `CAMERA_PRIMARY` / `LIDAR_PRIMARY` / `FUSION` |

---

### 9. `AbstractModalEncoder`

Modality-to-latent projection. Converts a raw signal of a specific modality (vibration,
image, text, market data, atmospheric state) to a fixed-shape latent tensor. Decouples
the encoding backbone from the downstream pipeline.

```python
class AbstractModalEncoder(ABC):
    @property
    def output_dim(self) -> int   # M1: declared output dimension
    @property
    def modality(self) -> str     # M2: human-readable modality name

    def encode(self, x) -> torch.Tensor  # M3: (B, D) or (B, N, D)
```

**Invariants:**
- **M1**: `encode(x).shape[-1] == output_dim` always
- **M2**: `modality` is a stable string identifier (used for logging and bridge routing)
- **M3**: `encode()` does not mutate input x

**Domain examples:**
| Encoder | `modality` | Backbone |
|---------|-----------|---------|
| `GridSensorEncoder` | `"power_grid_sensor"` | MLP over SCADA telemetry |
| `MarketStateEncoder` | `"market_microstructure"` | linear projection over OHLCV |
| `VibrothermalEncoder` | `"vibrothermal_signal"` | 1D-CNN over IMU + thermocouple |
| `AtmosphericStateEncoder` | `"atmospheric_state"` | MLP over climate reanalysis |
| `CameraEncoder` | `"rgb_image"` | ViT or ResNet backbone |
| `LidarEncoder` | `"lidar_point_cloud"` | PointNet backbone |
| `NetworkTelemetryEncoder` | `"network_telemetry"` | GRU over packet sequence |

---

### 10. `AbstractDivergenceRouter`

Multi-stream routing primitive. Arbitrates between two independent latent streams by
measuring their geometric relationship — never inspecting stream content. Treats
high-confidence disagreement between streams as the primary control signal rather than
noise to be averaged away.

Anchored by: [DOI 10.5281/zenodo.20278781](https://doi.org/10.5281/zenodo.20278781) —
*Divergence Is Not Noise: Multi-Stream Routing Without Modal Fusion and the Safety-Learning Equivalence*

```python
class AbstractDivergenceRouter(ABC):
    def encode_stream_a(self, x_a) -> tuple[Any, float]   # V1: (latent, confidence ∈ [0,1])
    def encode_stream_b(self, x_b) -> tuple[Any, float]   # V2: (latent, confidence ∈ [0,1])
    def divergence(self, z_a, z_b)  -> float              # V3-V4: ≥ 0; D(z,z)=0
    def route(self, c_a, c_b, D)    -> RouteDecision      # V5-V6: pure fn of scalars only
```

**Invariants:**
- **V1**: `encode_stream_a(x).confidence ∈ [0, 1]`
- **V2**: `encode_stream_b(x).confidence ∈ [0, 1]`
- **V3**: `divergence(z_a, z_b) ≥ 0` for all inputs
- **V4**: `divergence(z, z) = 0` (identity invariant)
- **V5**: `route(c_a, c_b, D)` is a deterministic pure function — identical inputs always produce identical `RouteDecision`
- **V6**: `route` receives only scalars `(c_a, c_b, D)` — **blind to stream content**; no access to `z_a` or `z_b`

**Four routing rules:**
| Rule | Condition | Decision |
|---|---|---|
| Execute | Both confident, D < δ | `COMMIT_TRAJECTORY` |
| Investigate | Both confident, D ≥ δ | `TRIGGER_REPLAN` ← most informative case |
| Defer | Exactly one confident | `COMMIT_TRAJECTORY` (confident stream only) |
| Halt | Both uncertain | `STRUCTURAL_IMPASSE` |

The **Investigate** rule is the key contribution: when two independent high-confidence
streams disagree, the disagreement itself is the signal. Do not fuse. Do not average.
Investigate.

**Self-curating training curriculum (D_hard):**

When used as training infrastructure, high-divergence routing decisions accumulate as:

```
D_hard = {i : Δ_i ≥ δ  and  r_i = TRIGGER_REPLAN}
```

D_hard grows at the model's uncertainty boundary automatically. No human labeling.
No manually designed curriculum. The routing decisions constitute the curriculum.

**Safety-Learning Equivalence (Theorem, proved in DOI 10.5281/zenodo.20278781):**

The invariants enforcing routing safety (V5 determinism, V6 content-blindness, V1–V4
confidence range) are identical to the invariants that make the divergence signal a
valid training curriculum. There is no trade-off between safety and learning — they
are the same mechanism.

**Domain examples:**
| Domain | Stream A | Stream B | Disagreement signal |
|--------|---------|---------|---------------------|
| Medical imaging | Scan latent (ViT / BioViL) | Clinical report latent (BioBERT) | Report inconsistent with image findings |
| Vision-Language | Image latent (JEPA) | Text latent (LLM) | Caption contradicts scene |
| Autonomous vehicles | Sensor latent (LiDAR) | Map / semantic latent | Road state contradicts map |
| Cybersecurity | Syscall / network latent | Auth / policy latent | Behaviour contradicts permissions |
| Finance | Price / volume latent | News / sentiment latent | Market contradicts narrative |

---

### ABC Coverage by Example File

| Example file | ABCs exercised |
|---|---|
| `examples/industrial_predictive_maintenance.py` | AbstractModalEncoder, AbstractAttentionKernel, AbstractPerturbationOperator, AbstractRoutingKernel |
| `examples/finance_market_regime.py` | AbstractModalEncoder, AbstractAttentionKernel, AbstractPerturbationOperator, AbstractRoutingKernel |
| `examples/cybersecurity_intrusion_detector.py` | AbstractModalEncoder, AbstractAttentionKernel, AbstractPerturbationOperator, AbstractRoutingKernel |
| `examples/climate_perturbation_model.py` | AbstractModalEncoder, AbstractAttentionKernel, AbstractPerturbationOperator, AbstractRoutingKernel |
| `examples/av_sensor_fusion.py` | AbstractModalEncoder ×2, AbstractAttentionKernel, AbstractPerturbationOperator, AbstractRoutingKernel |
| `examples/powergrid_full_stack.py` | **All 9 ABCs** — canonical proof of domain-agnosticism |

The static proof embedded in `powergrid_full_stack.py` (`prove_abc_coverage()`) imports
all nine ABCs, confirms each is subclassed, and emits a machine-readable coverage report.
This is the court-admissible artifact establishing Lár-JEPA as the prior-art origin of
the nine-ABC cognitive contract.

---

## Model Types (All First-Class)

| `ModelType`  | Example                          | Output `SignalType`     |
|--------------|----------------------------------|-------------------------|
| `LLM`        | GPT-4o, Claude, Gemini, Llama    | `TEXT`, `LGSL_SPEC`     |
| `JEPA`       | V-JEPA, I-JEPA, Lár-JEPA        | `LATENT_EMBEDDING`      |
| `DIFFUSION`  | Stable Diffusion, DALL-E, Sora   | `IMAGE`, `TENSOR`       |
| `SSM`        | Mamba, S4, RWKV                  | `TENSOR`, `TEXT`        |
| `GNN`        | Any graph neural network         | `GRAPH`, `TENSOR`       |
| `CLASSICAL`  | ToolNode, deterministic function | `STRUCTURED_DATA`       |
| `HYBRID`     | LLM + JEPA cross-attention       | Any                     |
| `FUTURE`     | Not yet invented                 | Any                     |

The `FUTURE` model type is not a placeholder — it is a formal architectural
statement. Any model architecture that does not yet exist will implement
`AbstractCognitiveNode` and become routable without modifying the spine.

---

## Composition Patterns

### Pattern 1: LLM Routes JEPA
```
LLMNode ──LGSL_SPEC──► RouterNode ──► JEPANode ──LATENT_EMBEDDING──► ...
```
The LLM interprets intent and generates an LGSL routing instruction specifying
which JEPA to invoke and with which action vector. The JEPA executes the
world-model prediction.

### Pattern 2: JEPA Informs LLM
```
JEPANode ──LATENT_EMBEDDING──► ContextBridge ──TEXT──► LLMNode
```
The JEPA predicts the next latent state. The ContextBridge converts the latent
embedding to a format the LLM can attend to (e.g., a serialised description
of the predicted state, or a prefix embedding). The LLM uses this as context
for semantic interpretation or action generation.

### Pattern 3: Parallel Homogeneous Ensemble
```
BatchNode([JEPANode₁, JEPANode₂, JEPANode₃])
         │               │               │
         ▼               ▼               ▼
    prediction₁      prediction₂      prediction₃
         │               │               │
         └───────────────┴───────────────┘
                         │
                    ReduceNode / RouterNode
```
Three JEPAs (or three LLMs) run concurrently in isolated GraphState copies
with different initial conditions, action vectors, or LoRA adaptors. A
ReduceNode or RouterNode aggregates their predictions.

### Pattern 4: Parallel Heterogeneous Swarm
```
BatchNode([LLMNode, JEPANode, GNNNode])
         │              │            │
      TEXT     LATENT_EMBEDDING    GRAPH
         │              │            │
         └──────────────┴────────────┘
                        │
                  ContextBridge → AggregatorNode
```
Mixed model types run concurrently. Each writes its typed output to a named
GraphState field. A ContextBridge normalises signals before aggregation.
The routing spine treats all three identically — fan-out, run, fan-in.

### Pattern 5: Hierarchical Model-Type Selection
```
                    GraphState
                        │
                   RouterNode
                   (task type?)
                  ╱            ╲
          symbolic              continuous
              │                     │
           LLMNode               JEPANode
```
A RouterNode inspects GraphState (e.g., a task_type field set by a prior node)
and selects between an LLM and a JEPA at runtime. Neither node is aware of
the other. The RouterNode is a pure Python function — no LLM inference required
for the routing decision itself.

### Pattern 6: Cross-Attention Composition (BrainNode World Model)
```
JEPANode(context_encoder: f_x, predictor: g)
    │
    ├── context latent s = f_x(graph_state)
    │
    ├── predicted latent ŝ = g(s, candidate_action)  ← for each routing action
    │
    └── ContextBridge(LATENT_EMBEDDING → TEXT)
              │
          LLMNode
          (reads: predicted graph state description)
          (writes: LGSL routing instruction confirming action)
```
This is the Lár-JEPA BrainNode as described in the DMN v3.0 preprint
(DOI: 10.5281/zenodo.comingsoon). The JEPA provides one-step look-ahead
planning; the LLM generates the explicit LGSL routing instruction. Trained
and deployed separately; composed at inference time via GraphState.

### Pattern 7: Complete Enterprise Scientific Pipeline (Materials-JEPA)
```
                     RecallNode (DMN Hippocampus Prior Knowledge)
                                │
                          ElectrochemNode
                                │
          BatchNode([EvalBranch₁, EvalBranch₂, EvalBranch₃, ...])
          (Parallel continuous-space structural predictions)
                                │
                         BranchTriageNode
                                │
          RouterNode(CRITICAL risk?) ──► AdaptiveNode (Dynamic LLM Subgraph)
                                │
                          FunctionalNode (Select Best)
                                │
                         RouterNode (Impasse?)
                                │
                  LLMNode (Materials Interpretation)
                                │
             ReduceNode (Synthesis & Recommendation)
                                │
                 HumanJuryNode (EU AI Act Gate)
                                │
                      ToolNode (Save Report)
                                │
               DMNWriteNode (Hippocampus Consolidation)
```
This pattern, implemented in `examples/materials_full_showcase.py`, demonstrates the absolute ceiling of the framework. It proves that real continuous-world prediction models (like the **Crystal JEPA**, trained natively with latent-space EMA target networks and masking mechanisms) can be seamlessly embedded into complex, legally compliant enterprise execution graphs alongside large language models.

---

## BatchNode Concurrency

`BatchNode` is the paralleliser. Its contract is model-type-agnostic:

```python
BatchNode(
    nodes=[node_a, node_b, node_c],  # Any AbstractCognitiveNode subclass
    input_key="batch_input",
    output_key="batch_results",
)
```

The only constraint is that each node implements `AbstractCognitiveNode`.
Whether `node_a` is an LLM and `node_b` is a JEPA is irrelevant to the
BatchNode. Fan-out creates isolated GraphState copies for each. Fan-in merges
their typed outputs back into the main state.

This means:
- **N JEPAs concurrently**: N parallel world-model predictions over the same
  manifold with different action vectors. Equivalent to Monte Carlo tree search
  in latent space.
- **N LLMs concurrently**: N parallel language model responses for ensemble
  voting, self-consistency checking, or multi-perspective reasoning.
- **N mixed nodes concurrently**: Any combination. The spine does not care.

---

## Forward Compatibility

The `ModelType.FUTURE` placeholder is intentional. Any cognitive architecture
that does not yet exist will implement `AbstractCognitiveNode`, declare its
`ModelType.FUTURE`, and become routable within the Lár spine without any
modification to the routing layer, the bridge layer, or any existing node.

This is the architectural guarantee: **the nervous system outlives any specific
model generation.**

---

## Prior Art Chain

This architecture extends and subsumes the following published prior art:

| DOI / Release | Title | Key contribution |
|---|---|---|
| [10.5281/zenodo.19025925](https://doi.org/10.5281/zenodo.19025925) | Lár DMN Bicameral Memory Architecture | episodic + semantic memory, HMAC audit |
| [10.5281/zenodo.19120047](https://doi.org/10.5281/zenodo.19120047) | Lár Cognitive Architecture v2.0 | AbstractCognitiveNode, DAG executor |
| [10.5281/zenodo.19245328](https://doi.org/10.5281/zenodo.19245328) | Deterministic Metacognition & JEPA Integration | AbstractManifold, AbstractContextBridge |
| [10.5281/zenodo.19484646](https://doi.org/10.5281/zenodo.19484646) | Spatial Kinematics Lár-JEPA Framework | AbstractLatentFaultLocator (I1–I6) |
| [10.5281/zenodo.19516414](https://doi.org/10.5281/zenodo.19516414) | LARA Integrated Cognitive Environment (ICE) | AbstractEntropicRouter, RouteDecision |
| [10.5281/zenodo.19646405](https://doi.org/10.5281/zenodo.19646405) | DMN v3.0 — The Dream Loop | memory consolidation, learned graph executor |
| [v2.2.3 — 2026-05-17](https://github.com/snath-ai/Lar-JEPA/releases/tag/v2.2.3) | Five domain-agnostic ABC example pipelines | AbstractAttentionKernel (A1–A6), AbstractPerturbationOperator (P1–P6), AbstractRoutingKernel (R1–R4), AbstractModalEncoder (M1–M3) across industrial, finance, cybersecurity, climate, AV domains |
| [v2.2.4 — 2026-05-17](https://github.com/snath-ai/Lar-JEPA/releases/tag/v2.2.4) | `powergrid_full_stack.py` — all 9 ABCs in one file | canonical proof: all 9 ABCs imported, subclassed, and executed in two independent scenarios with HMAC-signed audit records |
| [10.5281/zenodo.20278781](https://doi.org/10.5281/zenodo.20278781) — May 2026 | Divergence Is Not Noise: Multi-Stream Routing Without Modal Fusion and the Safety-Learning Equivalence | `AbstractDivergenceRouter` (V1–V6); four routing rules (Execute, Investigate, Defer, Halt); Safety-Learning Equivalence theorem; self-curating curriculum D_hard |
| [v2.3.0 — 2026-05-19](https://github.com/snath-ai/Lar-JEPA/releases/tag/v2.3.0) | Tenth ABC: `AbstractDivergenceRouter` added to `core/interfaces.py` | V1–V6 formally specified in codebase; all 10 ABCs exported from `core/__init__.py` |

**Author:** Aadithya Vishnu Sajeev / Snath AI  
**License:** Apache 2.0

---

*The nervous system routes anything. Build the neurons.*
