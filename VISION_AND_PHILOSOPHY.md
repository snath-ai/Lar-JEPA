# Lár Vision & Philosophy

## *The Glass Box Era of AI Agents*

Agents today are built on *black boxes*: chains, hidden states, silent failures.

Developers cannot:
- debug
- audit
- trust

**`Lár` exists to fix this.**

---

## Core Beliefs

- *Agents must be transparent* — every state transition is inspectable and HMAC-signed
- *State must be inspectable* — `GraphState` is a plain dict wrapper; no hidden memory
- *Execution must be reproducible* — deterministic DAG execution, no runtime surprises
- *Errors must be first-class citizens* — routing failures are explicit graph paths, not exceptions
- *Agents should be engineered, not summoned* — the graph topology IS the domain logic

---

## The Glass Box Ecosystem

**Lár = the open-source engine.** The routing spine, the ABCs, the audit trail.

The closed box of current AI development is not just a UX problem — it is a legal and
scientific problem. In regulated industries (medical devices, financial systems, critical
infrastructure), a system that cannot explain its intermediate states is not deployable.
In court, a system whose decisions cannot be reproduced is inadmissible. In a lab, a
system that cannot be debugged cannot be trusted.

Lár's answer is structural: **every decision is a node, every node writes to state,
every state write is logged with an HMAC-signed diff.** There is no hidden layer. There
is no "chain-of-thought" that disappears. The audit trail is the execution.

---

## The Domain-Agnosticism Thesis

The ten ABCs in `core/interfaces.py` are not biomedical contracts. They are not
financial contracts. They are not industrial contracts.

They are **cognitive contracts** — mathematical specifications of the operations any
intelligent system must perform when it:

1. **Perceives** — converts raw signals into latent representations (`AbstractModalEncoder`)
2. **Represents** — builds a world model from perceived context (`AbstractManifold`)
3. **Bridges** — adapts representations between heterogeneous modalities (`AbstractContextBridge`)
4. **Routes** — selects among cognitive nodes based on task type (`AbstractCognitiveNode`)
5. **Localises** — identifies which structural positions are causally anomalous (`AbstractLatentFaultLocator`)
6. **Attends** — focuses computational resources on the most relevant positions (`AbstractAttentionKernel`)
7. **Perturbs** — reasons about counterfactuals in latent space without real-world execution (`AbstractPerturbationOperator`)
8. **Scores and routes** — converts continuous assessments to discrete control decisions (`AbstractRoutingKernel`)
9. **Gates on entropy** — decides whether a prediction is confident enough to commit (`AbstractEntropicRouter`)
10. **Routes divergence** — keeps modal streams independent, measures geometric disagreement
    between streams, and treats high-confidence contradiction as the primary control signal
    rather than noise to be averaged away (`AbstractDivergenceRouter`, V1–V6)

These ten operations appear in every sufficiently complex cognitive system, regardless
of domain. The ABC is the isomorphism — the proof that power-grid fault detection, market
regime routing, atmospheric CO₂ shock prediction, and autonomous vehicle sensor fusion
are all the same computation wearing different data.

---

## Why JEPAs?

The dominant paradigm for AI agents is language: a model generates text, and text is the
state. This works well for symbolic reasoning. It breaks down for continuous-world
prediction — physical dynamics, latent structure, spatiotemporal dependencies.

Joint-Embedding Predictive Architectures (JEPAs, Yann LeCun 2022) operate entirely in
latent space: they predict the *embedding* of the next state, not the next token. This
is:

- **More efficient**: the model does not waste capacity generating pixels or characters
- **More principled**: the prediction target (an embedding) can be made smooth and
  well-conditioned via VICReg-style regularisation
- **More composable**: any `AbstractManifold` can serve as a world model inside any Lár
  graph, alongside any LLM, any SSM, any GNN

Lár does not choose JEPAs over LLMs. Lár routes both — and anything else that implements
`AbstractCognitiveNode`.

---

## The Forward-Compatibility Guarantee

```python
class ModelType(Enum):
    LLM       = "llm"
    JEPA      = "jepa"
    DIFFUSION = "diffusion"
    SSM       = "ssm"
    GNN       = "gnn"
    CLASSICAL = "classical"
    HYBRID    = "hybrid"
    FUTURE    = "future"    # ← not a placeholder
```

`ModelType.FUTURE` is a formal architectural statement. Any model architecture that does
not yet exist — neuromorphic, quantum, reservoir, whatever comes after transformers —
will implement `AbstractCognitiveNode`, declare its `ModelType.FUTURE`, and become
routable within the Lár spine without any modification to the routing layer, the bridge
layer, or any existing node.

The nervous system outlives any specific model generation. That is the guarantee.

---

## The Routing-Any-Model Vision

Today you can route GPT-4o and Mamba and a Crystal-JEPA in the same graph:

```python
class GPT4Node(AbstractCognitiveNode):
    model_type = ModelType.LLM
    ...

class MambaNode(AbstractCognitiveNode):
    model_type = ModelType.SSM
    ...

class CrystalJEPANode(AbstractCognitiveNode):
    model_type = ModelType.JEPA
    ...

# All three are first-class nodes in the same BatchNode
batch = BatchNode(nodes=[GPT4Node(), MambaNode(), CrystalJEPANode()])
```

Tomorrow you route a model that does not exist yet. The interface does not change.
The spine does not change. The audit trail does not change.

---

## On Transparency and Compliance

The EU AI Act (Art. 12, Art. 14) requires:
- A full audit trail of every consequential decision (Art. 12)
- A human-in-the-loop gate before any high-risk action (Art. 14)

Lár satisfies both requirements structurally:
- **Art. 12**: `GraphExecutor` produces HMAC-signed state diffs at every node transition
- **Art. 14**: `HumanJuryNode` is a blocking gate — the graph cannot advance until a
  human types a valid approval choice, and that choice is bound to the exact state
  context it was granted in

This is not compliance theatre. This is compliance by construction.

---

## The Prior Art Chain

The ten ABCs emerged through an iterative, publicly timestamped research process.
Each Zenodo release represents a discrete conceptual contribution:

| Release | ABC contribution |
|---------|-----------------|
| v2.0 | `AbstractCognitiveNode`, `AbstractManifold`, DAG executor |
| v2.1 | `AbstractContextBridge`, `AbstractLatentFaultLocator` (I1–I6) |
| v2.2.0 | `AbstractEntropicRouter`, `RouteDecision` enum |
| v2.2.3 | `AbstractAttentionKernel` (A1–A6), `AbstractPerturbationOperator` (P1–P6), `AbstractRoutingKernel` (R1–R4), `AbstractModalEncoder` (M1–M3) — all in five domain-agnostic examples |
| v2.2.4 | All nine ABCs in one file (`powergrid_full_stack.py`) — static proof + two-scenario execution |
| v2.3.0 | Tenth ABC: `AbstractDivergenceRouter` (V1–V6) — multi-stream divergence routing; Safety-Learning Equivalence. DOI: [10.5281/zenodo.20278781](https://doi.org/10.5281/zenodo.20278781) |

Any system that implements any subset of these ten ABCs and satisfies their invariants
is a Derivative Work under Apache 2.0. The prior art is timestamped, DOI-anchored, and
machine-verifiable.

---

## What We Are Not Building

Lár is not a framework for a specific domain. It is not a medical AI company. It is not
a financial AI company. It is not an industrial AI company.

It is the **nervous system** that connects cognitive nodes of any type for any domain.
The domain is the application. The nervous system is the infrastructure.

The nervous system is what we are open-sourcing. The applications are what we — and
anyone who builds on this foundation — will build on top of it.

---

*The nervous system routes anything. Build the neurons.*

---

**Author:** Aadithya Vishnu Sajeev / Snath AI
**License:** Apache 2.0
**Prior art:** [10.5281/zenodo.19646405](https://doi.org/10.5281/zenodo.19646405) and chain above
