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

| DOI | Title |
|-----|-------|
| [10.5281/zenodo.19025925](https://doi.org/10.5281/zenodo.19025925) | Lár DMN Bicameral Memory Architecture |
| [10.5281/zenodo.19120047](https://doi.org/10.5281/zenodo.19120047) | Lár Cognitive Architecture v2.0 |
| [10.5281/zenodo.19245328](https://doi.org/10.5281/zenodo.19245328) | Deterministic Metacognition & JEPA Integration |
| [10.5281/zenodo.19484646](https://doi.org/10.5281/zenodo.19484646) | Spatial Kinematics Lár-JEPA Framework |
| [10.5281/zenodo.19516414](https://doi.org/10.5281/zenodo.19516414) | LARA Integrated Cognitive Environment (ICE) |

**Author:** Aadithya Vishnu Sajeev / Snath AI  
**License:** Apache 2.0

---

*The nervous system routes anything. Build the neurons.*
