"""The full detector: per-view CNNs -> fusion -> temporal encoder -> logit."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.backbones import build_backbone
from src.models.fusion import GatedFusion, build_fusion
from src.models.temporal import TemporalEncoder


class QKVHalluDetector(nn.Module):
    """Input:  (B, T, V, 3, L, C) token images + (B, T) padding mask
    Output: (B,) logits -- raw, NOT sigmoided (we use BCEWithLogits).

    Q, K and V each pass through their OWN CNN and meet only at the fusion
    module. See src/models/fusion.py for why they are never channel-stacked.
    """

    def __init__(self, cfg, n_views: int):
        super().__init__()
        self.cfg = cfg
        self.n_views = n_views
        self.share_backbone = cfg.model.share_backbone

        if self.share_backbone:
            # One CNN applied to all views. Fewer params, but it assumes Q, K and
            # V share low-level structure -- which is a real assumption, not a
            # free saving, so it is not the default.
            self.backbones = nn.ModuleList([build_backbone(cfg)])
        else:
            self.backbones = nn.ModuleList([build_backbone(cfg) for _ in range(n_views)])

        self.fusion = build_fusion(cfg, n_views)
        self.temporal = TemporalEncoder(
            input_dim=cfg.model.fused_dim,
            conv_layers=cfg.model.conv1d_layers,
            lstm_hidden=cfg.model.lstm_hidden,
            lstm_layers=cfg.model.lstm_layers,
            dropout=cfg.model.dropout,
        )

        head_in = self.temporal.out_dim
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Dropout(cfg.model.dropout),
            nn.Linear(head_in, 1),
        )

    def encode_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """(B, T, V, 3, L, C) -> (B, T, F) fused per-token embeddings."""
        b, t, v, c, h, w = images.shape
        if v != self.n_views:
            raise ValueError(
                f"model was built for {self.n_views} views but got {v}"
            )

        if self.share_backbone:
            # Fold BOTH the token and view axes into the batch: one CNN call for
            # everything. Never loop over tokens or views in Python.
            flat = images.reshape(b * t * v, c, h, w)
            emb = self.backbones[0](flat)                       # (B*T*V, E)
            emb = emb.reshape(b, t, v, -1)
        else:
            # Untied: one CNN per view. Each still sees all B*T tokens at once,
            # so this is 3 big batched calls, not 3*T small ones.
            per_view = []
            for i in range(v):
                flat = images[:, :, i].reshape(b * t, c, h, w)  # (B*T, 3, L, C)
                emb = self.backbones[i](flat)                   # (B*T, E)
                per_view.append(emb.reshape(b, t, -1))
            emb = torch.stack(per_view, dim=2)                  # (B, T, V, E)

        # Fuse per token: fold tokens into the batch so fusion sees (B*T, V, E).
        e = emb.shape[-1]
        fused = self.fusion(emb.reshape(b * t, v, e))           # (B*T, F)
        return fused.reshape(b, t, -1)                          # (B, T, F)

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        combined = self.temporal(self.encode_tokens(images), mask)  # (B, 2H)
        return self.head(combined).squeeze(-1)         # (B,)

    @torch.no_grad()
    def view_gates(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
        """Mean softmax gate weight per view, averaged over real tokens.

        Returns (n_views,) summing to 1, or None if the fusion module has no
        gates. This is the "how much does the model rely on Q vs K vs V" number.
        """
        if not isinstance(self.fusion, GatedFusion):
            return None

        self.encode_tokens(images)
        gates = self.fusion.last_gates                # (B*T, V, 1) or (B*T, V, E)
        if gates is None:
            return None

        b, t = mask.shape
        gates = gates.mean(dim=-1).reshape(b, t, self.n_views)   # (B, T, V)
        flat_mask = mask.unsqueeze(-1).float()
        # Average over real tokens only -- padded tokens have meaningless gates.
        return (gates * flat_mask).sum(dim=(0, 1)) / flat_mask.sum().clamp(min=1)


def build_model(cfg, n_views: int) -> QKVHalluDetector:
    return QKVHalluDetector(cfg, n_views=n_views)
