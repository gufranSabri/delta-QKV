"""The full detector: per-view CNNs -> fusion -> temporal encoder -> logit.

Optionally a SECOND stream (model.stream2.enable) runs alongside the first:
stream 1 treats each token's activation image as (L, D) spatial axes with T
(generated tokens) as the sequence axis; stream 2 transposes so T becomes a
spatial axis instead (D takes T's old role, folded into the batch), letting
a CNN find "this happens over these tokens" structure directly. Stream 2
has no sequence axis left to pool with an LSTM, so it runs backbones ->
fusion only, and its pooled vector is concatenated with stream 1's before
the head. See src/models/backbones/masked.py for how stream 2's variable
per-example token count is handled inside a CNN batch.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.backbones import MaskedPoolBackbone, ScratchCNN, build_backbone
from src.models.fusion import GatedFusion, build_fusion
from src.models.temporal import TemporalEncoder


class QKVHalluDetector(nn.Module):
    """Input:  (B, T, V, 3, L, C) token images + (B, T) padding mask
    (+ stream2: (B, D, V2, 3, L, T) images2 + (B, T) mask2, when enabled)
    Output: (B,) logits -- raw, NOT sigmoided (we use BCEWithLogits).

    Q, K and V each pass through their OWN CNN and meet only at the fusion
    module. See src/models/fusion.py for why they are never channel-stacked.
    """

    def __init__(self, cfg, n_views: int, n_views2: int | None = None):
        super().__init__()
        self.cfg = cfg
        self.n_views = n_views
        self.share_backbone = cfg.model.share_backbone
        self.stream2_enabled = cfg.model.stream2.enable

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
        if self.stream2_enabled:
            self.n_views2 = n_views2
            # config.validate() rejects stream2 + resnet18, so build_backbone
            # only ever returns ScratchCNN instances here.
            backbones2 = (
                [build_backbone(cfg)] if self.share_backbone
                else [build_backbone(cfg) for _ in range(n_views2)]
            )
            self.backbones2 = nn.ModuleList(
                MaskedPoolBackbone(bb) for bb in backbones2
            )
            self.fusion2 = build_fusion(cfg, n_views2)
            # Stream 2 has no sequence axis to pool over (T was consumed as a
            # spatial axis inside the CNN) -- its fusion output IS the final
            # per-example vector, concatenated onto stream 1's pooled vector.
            head_in += cfg.model.fused_dim

        

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

    def encode_stream2(self, images2: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
        """(B, D, V2, 3, L, T) + (B, T) mask -> (B, F) fused per-example vector.

        D (fixed per run) plays the role T played in encode_tokens: it is
        folded into the batch so each substream's CNN sees B*D images at
        once. T is now a spatial axis inside each image and varies per
        example, so the batch-level mask is repeated D times to line up with
        the folded (B*D) axis before reaching MaskedPoolBackbone.
        """
        b, d, v2, c, h, w = images2.shape
        if v2 != self.n_views2:
            raise ValueError(
                f"model was built for {self.n_views2} stream2 substreams but got {v2}"
            )

        flat_mask = mask2.unsqueeze(1).expand(b, d, -1).reshape(b * d, -1)  # (B*D, T)

        if self.share_backbone:
            flat = images2.reshape(b * d * v2, c, h, w)
            emb = self.backbones2[0](flat, flat_mask.repeat_interleave(v2, dim=0))
            emb = emb.reshape(b, d, v2, -1)
        else:
            per_view = []
            for i in range(v2):
                flat = images2[:, :, i].reshape(b * d, c, h, w)   # (B*D, 3, L, T)
                emb = self.backbones2[i](flat, flat_mask)          # (B*D, E)
                per_view.append(emb.reshape(b, d, -1))
            emb = torch.stack(per_view, dim=2)                     # (B, D, V2, E)

        # Fuse per D-slice, then average over D -- D is a fixed extraction
        # geometry axis (extract.n_cols), not a sequence with real/padded
        # positions, so a plain mean is correct (no masking needed here).
        e = emb.shape[-1]
        fused = self.fusion2(emb.reshape(b * d, v2, e))            # (B*D, F)
        fused = fused.reshape(b, d, -1).mean(dim=1)                # (B, F)
        return fused

    def forward(self, *args) -> torch.Tensor:
        if self.stream2_enabled:
            images, images2, mask, mask2 = args
            pooled1 = self.temporal(self.encode_tokens(images), mask)   # (B, 2H)
            pooled2 = self.encode_stream2(images2, mask2)               # (B, F)
            combined = torch.cat([pooled1, pooled2], dim=-1)
        else:
            images, mask = args
            combined = self.temporal(self.encode_tokens(images), mask)  # (B, 2H)

        return self.head(combined).squeeze(-1)         # (B,)

    @torch.no_grad()
    def view_gates(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
        """Mean softmax gate weight per view, averaged over real tokens.

        Returns (n_views,) summing to 1, or None if the fusion module has no
        gates. This is the "how much does the model rely on Q vs K vs V" number.
        Stream 1 only -- stream 2's fusion gates are not currently surfaced here.
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


def build_model(cfg, n_views: int, n_views2: int | None = None) -> QKVHalluDetector:
    return QKVHalluDetector(cfg, n_views=n_views, n_views2=n_views2)
