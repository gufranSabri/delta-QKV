"""Fusion modules: combine the per-view (Q, K, V) embeddings into one.

Every fusion module maps (B, V, E) -> (B, F), where V = number of views used.

WHY FUSION EXISTS AT ALL
------------------------
The obvious alternative -- stacking Q, K and V into a single 9-channel image --
is exactly what we refuse to do. A 9-channel conv lets the very first kernel mix
Q, K and V together, so by layer one there is no longer any such thing as "the Q
representation". That destroys the premise of the method (three distinct views of
what attention is doing) and makes it impossible to ask which view carries the
hallucination signal.

Keeping the views separate through the CNN and combining them HERE means:
  - the combination weights are explicit and inspectable (see GatedFusion.gates),
  - "the model relies mostly on V" becomes a measurable, reportable claim,
  - dropping a view is a config change, not a re-architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class IdentityFusion(nn.Module):
    """Single-view passthrough. Used automatically when len(views) == 1."""

    def __init__(self, embed_dim: int, fused_dim: int, n_views: int = 1):
        super().__init__()
        if n_views != 1:
            raise ValueError("IdentityFusion is only valid for exactly one view")
        self.proj = (
            nn.Identity() if embed_dim == fused_dim else nn.Linear(embed_dim, fused_dim)
        )

    def forward(self, x):                     # (B, 1, E)
        return self.proj(x.squeeze(1))        # (B, F)


class GatedFusion(nn.Module):
    """Learned per-view gates: out = proj( sum_v  g_v * e_v ).

    The gates are produced from the concatenated embeddings and normalised with a
    softmax over the view axis, so they form a distribution over {Q, K, V} that
    reads directly as "how much is the model leaning on each view".

    `gate_mode`:
      - "scalar": one weight per view (V weights).       Maximally interpretable.
      - "vector": one weight per view per dim (V*E).     More expressive, still
                  readable after averaging over the feature axis.

    The gates are input-DEPENDENT (computed per token), so they can also be
    inspected per-token -- e.g. "on hallucinated tokens the model shifts weight
    onto V" would be a genuinely interesting finding, and this is what makes that
    measurable.
    """

    def __init__(self, embed_dim, fused_dim, n_views, gate_mode: str = "scalar"):
        super().__init__()
        if gate_mode not in ("scalar", "vector"):
            raise ValueError(f"gate_mode must be scalar|vector, got {gate_mode!r}")
        self.n_views = n_views
        self.gate_mode = gate_mode

        out_dim = n_views if gate_mode == "scalar" else n_views * embed_dim
        self.gate_net = nn.Sequential(
            nn.LayerNorm(n_views * embed_dim),
            nn.Linear(n_views * embed_dim, out_dim),
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, fused_dim),
        )
        self._last_gates: torch.Tensor | None = None

    def forward(self, x):                          # (B, V, E)
        b, v, e = x.shape
        logits = self.gate_net(x.reshape(b, v * e))

        if self.gate_mode == "scalar":
            gates = torch.softmax(logits, dim=-1).view(b, v, 1)     # (B, V, 1)
        else:
            gates = torch.softmax(logits.view(b, v, e), dim=1)      # (B, V, E)

        # Stash for inspection/logging. Detached: this must never affect grads.
        self._last_gates = gates.detach()

        fused = (gates * x).sum(dim=1)             # (B, E)
        return self.proj(fused)                    # (B, F)

    @property
    def last_gates(self) -> torch.Tensor | None:
        """Gates from the most recent forward, (B, V, 1) or (B, V, E)."""
        return self._last_gates


class ConcatMLPFusion(nn.Module):
    """The honest control: just concatenate and let an MLP sort it out.

    If GatedFusion cannot beat this, the "learned fusion" framing is not earning
    its keep and we should say so.
    """

    def __init__(self, embed_dim, fused_dim, n_views):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(n_views * embed_dim),
            nn.Linear(n_views * embed_dim, fused_dim),
            nn.GELU(),
            nn.Linear(fused_dim, fused_dim),
        )

    def forward(self, x):                     # (B, V, E)
        return self.net(x.flatten(1))         # (B, F)


class BilinearFusion(nn.Module):
    """Low-rank multiplicative fusion over view PAIRS.

    Motivated by the architecture we are probing: an attention score is
    literally q . k, a multiplicative interaction. A purely additive fusion (sum
    or concat) can never represent that. So we give the model an explicit
    low-rank bilinear term for every unordered pair of views:

        pair(i, j) = (U_i e_i) * (U_j e_j)        elementwise product, rank R

    then concatenate the pair terms with the linear (unary) terms and project.
    With V=3 there are 3 pairs (Q-K, Q-V, K-V); Q-K is the theoretically
    interesting one.
    """

    def __init__(self, embed_dim, fused_dim, n_views, rank: int = 64):
        super().__init__()
        self.n_views = n_views
        self.rank = rank
        self.norm = nn.LayerNorm(embed_dim)

        # One projection per view into the interaction space.
        self.view_proj = nn.ModuleList(
            [nn.Linear(embed_dim, rank) for _ in range(n_views)]
        )
        self.pairs = [
            (i, j) for i in range(n_views) for j in range(i + 1, n_views)
        ]

        unary = n_views * embed_dim
        pairwise = len(self.pairs) * rank
        self.out = nn.Sequential(
            nn.LayerNorm(unary + pairwise),
            nn.Linear(unary + pairwise, fused_dim),
            nn.GELU(),
            nn.Linear(fused_dim, fused_dim),
        )

    def forward(self, x):                          # (B, V, E)
        xn = self.norm(x)
        projected = [self.view_proj[i](xn[:, i]) for i in range(self.n_views)]

        terms = [x.flatten(1)]                     # unary (linear) terms
        for i, j in self.pairs:
            terms.append(projected[i] * projected[j])   # multiplicative term

        return self.out(torch.cat(terms, dim=-1))  # (B, F)


class CrossAttnFusion(nn.Module):
    """Treat {e_Q, e_K, e_V} as a 3-token sequence and run one transformer block.

    The most general option: every view can attend to every other. Almost
    certainly overkill for a length-3 sequence, but cheap and worth an ablation
    row. A learned CLS token does the pooling.
    """

    def __init__(self, embed_dim, fused_dim, n_views, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cls = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        # A view-identity embedding: without it the block is permutation-
        # invariant and literally cannot tell Q from V.
        self.view_embed = nn.Parameter(torch.randn(1, n_views, embed_dim) * 0.02)

        self.block = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.proj = nn.Linear(embed_dim, fused_dim)

    def forward(self, x):                          # (B, V, E)
        b = x.shape[0]
        x = x + self.view_embed
        seq = torch.cat([self.cls.expand(b, -1, -1), x], dim=1)  # (B, 1+V, E)
        out = self.block(seq)
        return self.proj(out[:, 0])                # CLS -> (B, F)


FUSIONS = {
    "gated": GatedFusion,
    "concat_mlp": ConcatMLPFusion,
    "bilinear": BilinearFusion,
    "cross_attn": CrossAttnFusion,
}


def build_fusion(cfg, n_views: int) -> nn.Module:
    """Pick a fusion module. A single view bypasses fusion entirely."""
    embed_dim, fused_dim = cfg.model.embed_dim, cfg.model.fused_dim

    if n_views == 1:
        # Nothing to fuse. This is what makes the Q-only / V-only ablations a
        # one-line config change rather than a separate model.
        return IdentityFusion(embed_dim, fused_dim, n_views=1)

    name = cfg.model.fusion
    if name not in FUSIONS:
        raise ValueError(f"unknown fusion {name!r}; valid: {sorted(FUSIONS)}")

    kwargs = {}
    if name == "cross_attn":
        kwargs["dropout"] = cfg.model.dropout
    return FUSIONS[name](embed_dim, fused_dim, n_views, **kwargs)
