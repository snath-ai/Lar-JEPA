# Experiment Results — AbstractDivergenceRouter (V1–V6)

Raw result data backing the empirical claims in:

> **Divergence Is Not Noise: Multi-Stream Routing Without Modal Fusion and the Safety-Learning Equivalence**
> Aadithya Vishnu Sajeev, Snath AI, May 2026
> DOI: [10.5281/zenodo.20278781](https://doi.org/10.5281/zenodo.20278781)

All numbers reported in the paper — AUROC, p-values, mean-D lifts, TRIGGER_REPLAN percentages — are reproducible from these files. No post-processing was applied between the raw outputs and the reported figures.

---

## File Index

### Medical Imaging — NLM Indiana CXR / BiomedCLIP

| File | Experiment | Key Result |
|:-----|:-----------|:-----------|
| `experiment_results_c.json` | **C** — TXV DenseNet-121 (image) × BiomedCLIP finding-vector (text). Heterogeneous-backbone design. | AUROC = 0.47. Failed: heterogeneous backbones produce incommensurable confidence scales. Design flaw motivating C2. |
| `experiment_results_c2.json` | **C2** — BiomedCLIP both streams, TXV oracle GT, 18 clinical findings, cal/test split (N=200). | Routing AUROC = **0.8736**, p = 3.1×10⁻¹², TRIGGER_REPLAN lift = **6.5×** (78% contra vs 12% normal). Mean-D lift = 3.11×. |
| `experiment_results_e.json` | **E** — Same model/data as C2. Routing (finding-vector L1) vs Fusion (CLS cosine) on identical backbone. | Routing AUROC = **0.8736** vs Fusion AUROC = **0.8536** (+0.020). Fusion lift = 1.08× vs Routing lift = 3.11×. |

**C2** is the primary medical result (abstract, Table C2, conclusion).
**E** is the routing-vs-fusion ablation (Table E). Both use `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`.

---

### General Vision-Language — MSCOCO / OpenCLIP ViT-B-32

The COCO experiments are a progression: each iteration fixes a design flaw identified in the previous one.

| File | Experiment | Key Result |
|:-----|:-----------|:-----------|
| `experiment_results_f.json` | **F** (F1) — Flickr30k, concept-nearest oracle (original design). | Fusion AUROC = **1.00** (ceiling). Oracle leaked CLS signal — comparison invalid. |
| `experiment_results_f2.json` | **F2** — Flickr30k, image-text hard-negative oracle. Fusion blinded, concept vocab too coarse for Flickr captions. | Routing AUROC = **0.37** (signal inverted). Flickr captions don't reliably name visual concepts by keyword. |
| `experiment_results_f3.json` | **F3** — MSCOCO, hybrid concept oracle (CLIP image templates × keyword text). | Routing AUROC = 0.55, concept-L1 lift = **1.00×** (no signal). Image-only concept vector incommensurable with keyword text vector. |
| `experiment_results_f4.json` | **F4** — MSCOCO, 80-class softmax image × keyword text L1, without oracle equalisation. | Routing AUROC = **0.807** (lift 1.63×) vs Fusion AUROC = **0.831** (lift 1.07×). Routing −0.024 vs fusion — fusion CLS still informative. |
| `experiment_results_f5.json` | **F5** — MSCOCO, 80-class softmax image × keyword text L1, **image-text hard-negative oracle** (fusion blinded by construction). | Routing AUROC = **0.7166** (lift 1.51×, p = 5.7×10⁻⁵) vs Fusion AUROC = **0.508** (lift 1.00×). Primary COCO result. |

**F5** is the primary COCO result (abstract, Table F5, conclusion).

Oracle diagnostic for F5 (stored in `distributions` field): CLS cosine distance normal = 0.7449, contra = 0.7435, gap = 0.001 — fusion effectively blinded. Concept L1 ratio = 5.5× — routing signal intact.

---

### Tier-1 Ablations — Directions 11, 12, 13

| File | Experiment | Key Result |
|:-----|:-----------|:-----------|
| `tier1_full_run.log` | **D11** — Vocabulary size ablation K ∈ {5, 9, 18, 36}, NLM CXR. | AUROC range 0.723–0.770 (non-monotone). Routing is insensitive to vocabulary size; K=5 marginal best. |
| `tier1_full_run.log` | **D12** — Calibrated routing vs calibrated fusion (equal calibration budget, K=18). | Routing AUROC = **0.723** vs Fusion AUROC = **0.771** (gap −0.048). Under equal calibration the AUROC gap reverses; routing D-lift = **1.68×** vs fusion 1.08× — structural signal persists even when raw AUROC is similar. |
| `tier1_full_run.log` | **D13** — Geometry of Δ-space, D_hard enrichment, k-means silhouette. | 122/200 cases in D_hard. GT positive rate 0.787 vs 0.500 overall (**1.57× enrichment**). k=3 silhouette = **0.659** (strong separation). |

---

## Reproducing the Numbers

All statistics (AUROC, Fisher's exact p-values, accuracy) are computed over N=100 balanced test pairs (50 normal, 50 contradiction) drawn from 800-sample pools via oracle construction. The calibration split (100 samples) is held separate from the test split.

Thresholds used in C2/E: `τ_high = 0.65`, `τ_low = 0.1`, `δ = 4.35` (calibrated on the 100-sample calibration set).

Backbone: `hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` (medical), `OpenCLIP ViT-B-32/openai` (COCO).

Ground-truth oracle (medical): TorchXRayVision DenseNet-121 (`densenet121-res224-all`). The oracle constructs pairs only — it never participates in routing.

---

## Citation

```bibtex
@misc{sajeev2026das,
  author       = {Sajeev, Aadithya Vishnu},
  title        = {Divergence Is Not Noise: Multi-Stream Routing Without Modal Fusion
                  and the Safety-Learning Equivalence},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20278781},
  url          = {https://doi.org/10.5281/zenodo.20278781}
}
```
