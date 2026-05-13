"""
CrystalJEPA — Real JEPA Architecture for Crystal Site Embeddings
================================================================
This is the actual Joint Embedding Predictive Architecture applied to
crystal elemental site occupancies.

The three components that make it a JEPA (not just an encoder):

  ContextEncoder  — sees only VISIBLE sites (randomly selected subset)
  TargetEncoder   — EMA copy of ContextEncoder, sees ALL sites, no gradients
  Predictor       — takes context summary + target site position →
                    predicts what TargetEncoder would produce for that site

Loss: MSE(Predictor(ctx, pos_t), stop_grad(TargetEncoder(all_sites)[t]))

This is the same principle as I-JEPA (LeCun et al.) applied to crystal
elemental sites instead of image patches:
  - Image patches → elemental site occupancies
  - Spatial position → element index (periodic table position)
  - Visible patches → visible element slots
  - Masked patches → element slots to predict

The predictor never sees the target input — it must learn to predict the
TARGET ENCODER'S representation of a masked site from the context alone.
This forces the encoder to build rich, transferable site embeddings.

Architecture sizes (designed to train in <30s on CPU):
  embed_dim  = 64
  n_layers   = 2  (TransformerEncoder)
  n_heads    = 4
  predictor  = Linear(128→128, GELU) → Linear(128→64)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


N_SITES = 20  # element slots tracked


class CrystalSiteEncoder(nn.Module):
    """
    Encodes crystal elemental sites into per-site embeddings using
    a small Transformer over the site sequence.

    Each site's input: occupancy scalar (1) + shared lattice parameters (6) = 7 dims.
    The Transformer shares information across sites via self-attention.

    context_mask: boolean (B, N_SITES) — True = THIS SITE IS MASKED (hidden).
    When used as ContextEncoder, masked sites are zeroed before projection
    so they contribute no information to the attention computation.
    """

    def __init__(self, embed_dim: int = 64, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.site_proj = nn.Linear(7, embed_dim)
        self.pos_emb   = nn.Embedding(N_SITES, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(
        self,
        occupancies: torch.Tensor,    # (B, N_SITES)
        lattice:     torch.Tensor,    # (B, 6)
        context_mask: torch.Tensor | None = None,  # (B, N_SITES) bool, True=masked
    ) -> torch.Tensor:                # (B, N_SITES, embed_dim)
        B, N = occupancies.shape
        lattice_exp = lattice.unsqueeze(1).expand(-1, N, -1)   # (B, N, 6)

        occ = occupancies.unsqueeze(-1)                         # (B, N, 1)
        if context_mask is not None:
            occ = occ * (~context_mask).float().unsqueeze(-1)   # zero masked sites

        site_in = torch.cat([occ, lattice_exp], dim=-1)         # (B, N, 7)
        x = self.site_proj(site_in)                             # (B, N, D)
        x = x + self.pos_emb(torch.arange(N, device=x.device)) # add positional emb

        x = self.transformer(x)                                 # (B, N, D)
        return x


class CrystalJEPA(nn.Module):
    """
    Full JEPA model for crystal site embeddings.

    Training loop (called externally):
        1. Sample a batch of crystal compositions
        2. Randomly select context_sites (visible) and target_sites (masked)
        3. ContextEncoder sees only context_sites (mask=True on target_sites)
        4. TargetEncoder sees all sites (no mask) — EMA, no gradients
        5. Predictor takes pooled context + target position → predicted embedding
        6. Loss = mean MSE(predicted, stop_grad(target_encoder_output[target_sites]))
        7. Backprop through ContextEncoder + Predictor only
        8. EMA update: target_encoder = τ * target_encoder + (1-τ) * context_encoder

    The predictor is the key component: it maps from "what I can see about
    this crystal" + "which element site am I predicting" → "what would a
    fully-informed encoder produce for that site?"
    """

    def __init__(self, embed_dim: int = 64, ema_momentum: float = 0.996):
        super().__init__()
        self.embed_dim    = embed_dim
        self.ema_momentum = ema_momentum

        self.context_encoder = CrystalSiteEncoder(embed_dim)
        self.target_encoder  = CrystalSiteEncoder(embed_dim)

        # Target encoder is EMA — copy weights, disable gradients
        self._sync_target_encoder()
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Predictor: context_pool(D) + target_pos_emb(D) → predicted site emb(D)
        self.target_pos_emb = nn.Embedding(N_SITES, embed_dim)
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def _sync_target_encoder(self):
        for p_c, p_t in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            p_t.data.copy_(p_c.data)

    @torch.no_grad()
    def update_ema(self):
        """EMA update: τ * target + (1-τ) * context. Call after each optimizer step."""
        τ = self.ema_momentum
        for p_c, p_t in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            p_t.data.mul_(τ).add_(p_c.data, alpha=1 - τ)

    def forward(
        self,
        occupancies:  torch.Tensor,   # (B, N_SITES)
        lattice:      torch.Tensor,   # (B, 6)
        context_sites: list[int],     # indices of visible sites (e.g. 12 of 20)
        target_sites:  list[int],     # indices of sites to predict (e.g. 8 of 20)
    ) -> torch.Tensor:                # scalar JEPA loss
        B = occupancies.shape[0]
        device = occupancies.device

        # Build mask: True = this site is hidden from context encoder
        context_mask = torch.ones(B, N_SITES, dtype=torch.bool, device=device)
        context_mask[:, context_sites] = False           # context sites are visible

        # 1. Context encoder (receives gradient) — sees only context_sites
        ctx_emb = self.context_encoder(
            occupancies, lattice, context_mask=context_mask
        )                                                # (B, N_SITES, D)

        # Pool over visible sites only
        ctx_visible = ctx_emb[:, context_sites, :]      # (B, K_c, D)
        ctx_pool    = ctx_visible.mean(dim=1)            # (B, D)

        # 2. Target encoder (no gradient) — sees ALL sites
        with torch.no_grad():
            tgt_emb = self.target_encoder(occupancies, lattice)  # (B, N_SITES, D)

        # 3. Predict each target site from context + positional query
        target_idx_tensor = torch.tensor(target_sites, device=device)
        pos_queries = self.target_pos_emb(target_idx_tensor)     # (K_t, D)
        pos_queries = pos_queries.unsqueeze(0).expand(B, -1, -1) # (B, K_t, D)

        ctx_exp = ctx_pool.unsqueeze(1).expand(-1, len(target_sites), -1)  # (B, K_t, D)
        pred_in = torch.cat([ctx_exp, pos_queries], dim=-1)       # (B, K_t, 2D)
        pred    = self.predictor(pred_in)                          # (B, K_t, D)

        # 4. JEPA loss: MSE in latent space (stop-gradient on target)
        tgt     = tgt_emb[:, target_sites, :].detach()            # (B, K_t, D)
        loss    = F.mse_loss(pred, tgt)

        return loss

    def encode(
        self,
        occupancies: torch.Tensor,    # (B, N_SITES)
        lattice:     torch.Tensor,    # (B, 6)
    ) -> torch.Tensor:                # (B, N_SITES, embed_dim)
        """
        Inference: use the FULL context encoder (no masking) to produce
        per-site embeddings. This is what gets stored in the crystal library.
        """
        self.context_encoder.eval()
        with torch.no_grad():
            return self.context_encoder(occupancies, lattice)
