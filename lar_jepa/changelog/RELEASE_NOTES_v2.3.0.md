# v2.3.0 — The Tenth ABC: AbstractDivergenceRouter

**Release date:** 2026-05-19  
**Author:** Aadithya Vishnu Sajeev / Snath AI  
**Anchored by:** [DOI 10.5281/zenodo.20278781](https://doi.org/10.5281/zenodo.20278781)

---

## Summary

This release formally adds `AbstractDivergenceRouter` — the tenth cognitive ABC in
the Lár-JEPA cognitive contract — to `core/interfaces.py`. It also fixes a pre-existing
export gap where five of the nine existing ABCs were defined but not exported from
`core/__init__.py`.

---

## New: `AbstractDivergenceRouter` (V1–V6)

Multi-stream routing primitive. Arbitrates between two independent latent streams by
measuring their geometric relationship. Does not inspect stream content — only confidence
scores and divergence between predictions.

**Invariants:**

| Invariant | Specification |
|---|---|
| V1 | `encode_stream_a(x).confidence ∈ [0, 1]` |
| V2 | `encode_stream_b(x).confidence ∈ [0, 1]` |
| V3 | `divergence(z_a, z_b) ≥ 0` for all inputs |
| V4 | `divergence(z, z) = 0` (identity invariant) |
| V5 | `route(c_a, c_b, D)` is a deterministic pure function |
| V6 | `route` receives only scalars `(c_a, c_b, D)` — blind to stream content |

**Four routing rules:**

| Rule | Condition | Decision |
|---|---|---|
| Execute | Both confident, D < δ | `COMMIT_TRAJECTORY` |
| Investigate | Both confident, D ≥ δ | `TRIGGER_REPLAN` |
| Defer | Exactly one confident | `COMMIT_TRAJECTORY` (confident stream only) |
| Halt | Both uncertain | `STRUCTURAL_IMPASSE` |

**The Investigate rule is the key contribution.** When two independent, high-confidence
streams disagree, that disagreement is the most informative signal the system produces.
The correct response is not fusion. The correct response is investigation.

**Self-curating training curriculum:**

```python
D_hard = {i : delta_i >= delta  and  r_i == TRIGGER_REPLAN}
```

D_hard grows automatically at the model's uncertainty boundary. No human labeling.
No manually designed curriculum. The routing decisions constitute the curriculum.

**Safety-Learning Equivalence** (Theorem 1, proved in DOI 10.5281/zenodo.20278781):

The invariants enforcing routing safety (V5 determinism, V6 content-blindness,
V1–V4 confidence and divergence bounds) are identical to the invariants that make
the divergence signal a valid training curriculum. There is no trade-off between
safety and learning — they are the same mechanism.

**Prior art:**
- Concept: [DOI 10.5281/zenodo.20278781](https://doi.org/10.5281/zenodo.20278781) — May 2026
- Code: this release — v2.3.0, 2026-05-19

---

## Bug Fix: Export Gap in `core/__init__.py`

Five ABCs were defined in `core/interfaces.py` but missing from `core/__init__.py`
exports. Fixed in this release.

**Was (4 exports):**
```python
from .interfaces import (
    AbstractCognitiveNode, AbstractManifold,
    AbstractContextBridge, AbstractEntropicRouter,
)
```

**Now (10 exports):**
```python
from .interfaces import (
    AbstractCognitiveNode, AbstractManifold,
    AbstractContextBridge, AbstractEntropicRouter,
    AbstractLatentFaultLocator,    # was missing
    AbstractAttentionKernel,       # was missing
    AbstractPerturbationOperator,  # was missing
    AbstractRoutingKernel,         # was missing
    AbstractModalEncoder,          # was missing
    AbstractDivergenceRouter,      # new (tenth ABC)
)
```

---

## Files Changed

| File | Change |
|---|---|
| `core/interfaces.py` | Added `AbstractDivergenceRouter` (V1–V6) |
| `core/__init__.py` | Exported all 10 ABCs; fixed 5 missing exports |
| `ARCHITECTURE.md` | Added Section 10; added DAS paper row to Prior Art Chain |
| `VISION_AND_PHILOSOPHY.md` | Updated nine → ten operations; added operation 10 |
| `README.md` | Added `AbstractDivergenceRouter` description; updated ABC count |
| `pyproject.toml` | Version 2.2.0 → 2.3.0 |

---

## Prior Art Chain (cumulative)

| Zenodo DOI | Contribution |
|---|---|
| 10.5281/zenodo.19025925 | Lár DMN: episodic + semantic memory, HMAC audit |
| 10.5281/zenodo.19120047 | AbstractCognitiveNode, DAG executor |
| 10.5281/zenodo.19245328 | AbstractManifold, AbstractContextBridge |
| 10.5281/zenodo.19484646 | AbstractLatentFaultLocator (I1–I6) |
| 10.5281/zenodo.19516414 | AbstractEntropicRouter, RouteDecision |
| 10.5281/zenodo.19646405 | DMN v3.0, Learned Graph Executor |
| 10.5281/zenodo.20278775 | Nine-ABC cognitive contract (UCR paper) |
| 10.5281/zenodo.20278781 | AbstractDivergenceRouter (V1–V6), Safety-Learning Equivalence |
