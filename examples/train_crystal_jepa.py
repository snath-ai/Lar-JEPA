"""
Train CrystalJEPA — Battery Electrolyte Site Embedding
=======================================================
Trains the real JEPA model on synthetic crystal composition data.
Designed to run in < 60 seconds on CPU (Apple M-series MacBook).

What happens during training
----------------------------
Each step:
  1. Sample a batch of random crystal compositions
  2. Randomly split 20 element sites: 12 context (visible), 8 target (masked)
  3. ContextEncoder sees only the 12 visible sites
  4. TargetEncoder (EMA, frozen) sees all 20 sites
  5. Predictor: given pooled context + target site position → predict target embedding
  6. Loss: MSE(predicted embedding, stop_grad(target encoder output))
  7. Backprop through ContextEncoder + Predictor only
  8. EMA update: target ← 0.996 * target + 0.004 * context

After training, save the ContextEncoder weights to:
    models/crystal_jepa_encoder.pt

Run from lar_jepa/:
    python examples/train_crystal_jepa.py
"""

import sys
import os
import time
import random
import torch
import torch.optim as optim

_ROOT      = os.path.dirname(os.path.abspath(__file__))
_JEPA_ROOT = os.path.abspath(os.path.join(_ROOT, ".."))
_LAR_SRC   = os.path.join(_JEPA_ROOT, "lar_jepa", "src")
for _p in [_JEPA_ROOT, _LAR_SRC]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from materials_engine.crystal_jepa_model import CrystalJEPA, N_SITES


# ── Hyperparameters ──────────────────────────────────────────────────────────
EMBED_DIM       = 64
N_TRAIN_SAMPLES = 4_000      # synthetic crystals generated on-the-fly
BATCH_SIZE      = 64
EPOCHS          = 80
LR              = 3e-4
N_CONTEXT       = 12         # visible sites per step
N_TARGET        = 8          # masked sites to predict
EMA_MOMENTUM    = 0.996
SEED            = 42


def generate_batch(batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a batch of random but physically plausible crystal compositions.

    occupancies : (B, N_SITES)  — fractional site occupancies, sum ≤ 1
    lattice     : (B, 6)        — normalised lattice params a,b,c,α,β,γ ∈ [0,1]
    """
    occupancies = torch.softmax(torch.randn(batch_size, N_SITES), dim=-1) * 0.8
    lattice     = torch.rand(batch_size, 6)
    return occupancies, lattice


def sample_context_target(n_sites: int, n_context: int, n_target: int):
    """
    Randomly split site indices into context (visible) and target (masked).
    Ensures no overlap and full coverage (context ∪ target ⊂ {0..n_sites-1}).
    """
    indices = list(range(n_sites))
    random.shuffle(indices)
    context_sites = indices[:n_context]
    target_sites  = indices[n_context : n_context + n_target]
    return context_sites, target_sites


def train():
    torch.manual_seed(SEED)
    random.seed(SEED)

    print("=" * 60)
    print("  CrystalJEPA Training")
    print("  Joint Embedding Predictive Architecture")
    print("  for elemental site representations")
    print("=" * 60)
    print(f"\n  embed_dim      : {EMBED_DIM}")
    print(f"  training samples: {N_TRAIN_SAMPLES:,}")
    print(f"  batch_size     : {BATCH_SIZE}")
    print(f"  epochs         : {EPOCHS}")
    print(f"  context sites  : {N_CONTEXT}/{N_SITES}  (visible)")
    print(f"  target sites   : {N_TARGET}/{N_SITES}  (to predict)")
    print(f"  EMA momentum   : {EMA_MOMENTUM}")
    print(f"  device         : cpu\n")

    model     = CrystalJEPA(embed_dim=EMBED_DIM, ema_momentum=EMA_MOMENTUM)
    optimizer = optim.AdamW(
        list(model.context_encoder.parameters()) +
        list(model.predictor.parameters()) +
        list(model.target_pos_emb.parameters()),
        lr=LR,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    steps_per_epoch = N_TRAIN_SAMPLES // BATCH_SIZE
    t_start = time.perf_counter()

    loss_history = []

    for epoch in range(1, EPOCHS + 1):
        model.context_encoder.train()
        model.predictor.train()
        epoch_loss = 0.0

        for _ in range(steps_per_epoch):
            occupancies, lattice = generate_batch(BATCH_SIZE)

            context_sites, target_sites = sample_context_target(
                N_SITES, N_CONTEXT, N_TARGET
            )

            loss = model(occupancies, lattice, context_sites, target_sites)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            model.update_ema()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / steps_per_epoch
        loss_history.append(avg_loss)

        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            elapsed = time.perf_counter() - t_start
            bar_len = int((1 - avg_loss / loss_history[0]) * 30) if loss_history[0] > 0 else 0
            bar_len = max(0, min(30, bar_len))
            bar = "█" * bar_len + "░" * (30 - bar_len)
            print(
                f"  Epoch {epoch:3d}/{EPOCHS}  "
                f"loss={avg_loss:.5f}  "
                f"[{bar}]  "
                f"{elapsed:.1f}s"
            )

    total_time = time.perf_counter() - t_start
    reduction  = (loss_history[0] - loss_history[-1]) / loss_history[0] * 100

    print(f"\n  Training complete in {total_time:.1f}s")
    print(f"  Loss: {loss_history[0]:.5f} → {loss_history[-1]:.5f}  "
          f"({reduction:.1f}% reduction)")

    # ── Save weights ──────────────────────────────────────────────────────────
    models_dir   = os.path.join(_JEPA_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    encoder_path = os.path.join(models_dir, "crystal_jepa_encoder.pt")
    full_path    = os.path.join(models_dir, "crystal_jepa_full.pt")

    torch.save(model.context_encoder.state_dict(), encoder_path)
    torch.save(model.state_dict(), full_path)

    print(f"\n  Saved context encoder  → {encoder_path}")
    print(f"  Saved full JEPA model  → {full_path}")

    # ── Inference check ───────────────────────────────────────────────────────
    print("\n  Inference check (5 mock candidates):")
    CANDIDATE_LABELS = [
        "Li6PS5Cl (Argyrodite)",
        "Li3PS4 (Sulfide Glass)",
        "LATP",
        "LLZO",
        "LiPF6/EC:DMC",
    ]
    torch.manual_seed(42)
    occ_demo = torch.softmax(torch.randn(5, N_SITES), dim=-1) * 0.8
    lat_demo = torch.rand(5, 6)

    for i, label in enumerate(CANDIDATE_LABELS):
        emb = model.encode(occ_demo[i:i+1], lat_demo[i:i+1])  # (1, N_SITES, D)
        print(
            f"  [{i}] {label:<38} "
            f"emb shape={list(emb.shape)}  "
            f"norm={emb.norm(dim=-1).mean().item():.3f}"
        )

    print("\n  JEPA encoder produces per-site embeddings: (1, 20, 64)")
    print("  These replace the mock linear encoder in crystal_manifold.py")
    print("  at inference time — same AbstractManifold interface, trained weights.")

    return loss_history


if __name__ == "__main__":
    train()
