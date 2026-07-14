"""Pure tensor math for building per-token Q/K/V images.

This module is deliberately free of any model/HuggingFace dependency so the
image-construction logic can be unit-tested without loading an LLM.

Pipeline for a single view (Q, K or V) of one example:

    raw      (T, L, D)   per-token, per-layer activation vectors
      -> pool_feature_axis  ->  (T, L, C)   C columns, C == L gives a square image
      -> add_delta_channels ->  (T, L, C, 3)

Channel semantics (axis -1):
    0: pooled activation at (token t, layer l)
    1: delta to the PREVIOUS layer:  pooled[l] - pooled[l-1]
    2: delta to the NEXT layer:      pooled[l] - pooled[l+1]

Deltas are SIGNED (not absolute): the sign distinguishes a representation
growing from one shrinking, and abs() would discard that.

Deltas are computed AFTER pooling, on the C pooled columns. Note this is not
the same as pooling a full-D delta (max of differences != difference of maxes);
we take the pooled columns as the canonical representation because they are
exactly what the CNN consumes.
"""

from __future__ import annotations

import torch

# How to reduce the D-axis down to C columns.
POOL_MODES = ("max", "mean", "l2")

# How to fill the delta channels at layers that have no previous/next neighbour.
#
#   zero      -- the delta is 0. Honest: "no delta exists here".
#   replicate -- copy the nearest valid delta (edge replication).
#   wrap      -- circular: layer 0's "previous" is the last layer, and the last
#                layer's "next" is layer 0.
#
# NOTE ON `wrap`: this is a DESIGN CHOICE, not a mathematical necessity, and it
# is not the default. Wrapping conflates two conceptually different quantities:
# "how does this representation change with more processing" (a true adjacent-
# layer delta) and "how far has it drifted from the raw input embedding" (what
# you get when the last layer's `next` wraps around to layer 0). Because the
# embedding output and the final layer are typically very far apart, the wrapped
# rows carry a much larger magnitude than any genuine adjacent-layer delta, and
# a CNN will happily latch onto those two outlier rows. `zero` is the default
# for exactly this reason. `wrap` is kept only so the choice can be ablated.
BOUNDARY_MODES = ("zero", "replicate", "wrap")


def pool_feature_axis(raw: torch.Tensor, n_cols: int, mode: str = "max") -> torch.Tensor:
    """Reduce the feature axis D down to `n_cols` by pooling contiguous chunks.

    Args:
        raw:     (..., D) tensor. Typically (T, L, D).
        n_cols:  number of output columns C. D must be divisible by C.
        mode:    one of POOL_MODES.

    Returns:
        (..., n_cols) tensor.

    Chunk j covers raw[..., j*C : (j+1)*C] where C = D // n_cols.

    For Llama-3-8B the Q view has D=4096 with 32 heads of head_dim 128, so with
    n_cols=32 the contiguous chunks coincide exactly with attention heads and
    column j is "the peak activation of head j". That alignment is a happy
    accident of the architecture, NOT something this function enforces -- under
    GQA the K/V views have D=1024 (8 kv-heads x 128) and pooling those to 32
    columns splits each head across 4 columns. Do not rely on the head
    interpretation without checking D, n_cols and the model config.
    """
    if mode not in POOL_MODES:
        raise ValueError(f"pool mode must be one of {POOL_MODES}, got {mode!r}")

    d = raw.shape[-1]
    if n_cols <= 0:
        raise ValueError(f"n_cols must be positive, got {n_cols}")
    if d % n_cols != 0:
        raise ValueError(
            f"feature dim D={d} is not divisible by n_cols={n_cols}; "
            "pooling would drop or duplicate dimensions"
        )

    chunk = d // n_cols
    chunked = raw.reshape(*raw.shape[:-1], n_cols, chunk)

    if mode == "max":
        return chunked.amax(dim=-1)
    if mode == "mean":
        return chunked.mean(dim=-1)
    # l2: the norm of each chunk. Non-negative by construction, unlike max/mean.
    return chunked.norm(dim=-1)


def add_delta_channels(pooled: torch.Tensor, boundary_mode: str = "zero") -> torch.Tensor:
    """Build the 3-channel image stack from pooled activations.

    Args:
        pooled:        (T, L, C) pooled activations. L is the LAYER axis.
        boundary_mode: one of BOUNDARY_MODES. See module docstring.

    Returns:
        (T, L, C, 3) tensor. Channels: (raw, delta-to-prev, delta-to-next).

    The layer axis is dim 1 and is the axis the deltas run along.
    """
    if boundary_mode not in BOUNDARY_MODES:
        raise ValueError(
            f"boundary_mode must be one of {BOUNDARY_MODES}, got {boundary_mode!r}"
        )
    if pooled.ndim != 3:
        raise ValueError(f"expected (T, L, C), got shape {tuple(pooled.shape)}")

    n_layers = pooled.shape[1]
    if n_layers < 2:
        raise ValueError(
            f"need at least 2 layers to form layer deltas, got L={n_layers}"
        )

    if boundary_mode == "wrap":
        # Circular: prev of layer 0 is layer L-1; next of layer L-1 is layer 0.
        prev = torch.roll(pooled, shifts=1, dims=1)   # prev[l] == pooled[l-1 mod L]
        nxt = torch.roll(pooled, shifts=-1, dims=1)   # nxt[l]  == pooled[l+1 mod L]
        d_prev = pooled - prev
        d_next = pooled - nxt
    else:
        # Interior deltas are identical in every non-wrap mode; only the two
        # boundary rows differ, so compute the interior once and then fill.
        d_prev = torch.zeros_like(pooled)
        d_next = torch.zeros_like(pooled)

        # d_prev[l] = pooled[l] - pooled[l-1], defined for l = 1..L-1
        d_prev[:, 1:] = pooled[:, 1:] - pooled[:, :-1]
        # d_next[l] = pooled[l] - pooled[l+1], defined for l = 0..L-2
        d_next[:, :-1] = pooled[:, :-1] - pooled[:, 1:]

        if boundary_mode == "replicate":
            # Layer 0 has no previous layer: reuse layer 1's backward delta.
            d_prev[:, 0] = d_prev[:, 1]
            # Layer L-1 has no next layer: reuse layer L-2's forward delta.
            d_next[:, -1] = d_next[:, -2]
        # boundary_mode == "zero": the boundary rows keep their zeros.

    return torch.stack([pooled, d_prev, d_next], dim=-1)


def build_view_image(
    raw: torch.Tensor,
    n_cols: int,
    pool_mode: str = "max",
    boundary_mode: str = "zero",
) -> torch.Tensor:
    """raw (T, L, D) -> image (T, L, n_cols, 3) for a single view (Q, K or V)."""
    if raw.ndim != 3:
        raise ValueError(f"expected raw (T, L, D), got shape {tuple(raw.shape)}")
    # Pool in float32: fp16 max is exact, but mean/l2 over 128-element chunks can
    # overflow or lose precision at fp16 range, and deltas are differences of
    # similar magnitudes (catastrophic cancellation). Cast back at save time.
    pooled = pool_feature_axis(raw.float(), n_cols=n_cols, mode=pool_mode)
    return add_delta_channels(pooled, boundary_mode=boundary_mode)


def pool_layer_axis(img: torch.Tensor, n_layers_out: int) -> torch.Tensor:
    """Down-pool the LAYER axis of an image stack to a fixed size.

    Args:
        img:          (T, L, C, 3)
        n_layers_out: target number of layers L_eff.

    Returns:
        (T, L_eff, C, 3)

    Needed only for cross-LLM training: Llama-3-8B has L=32 while Qwen2.5-7B has
    L=28, so their images are 32x32 and 28x28 respectively and a single CNN
    cannot consume both. Pooling the layer axis to a common L_eff makes image
    size a fixed hyperparameter rather than a property of the LLM.

    Uses adaptive max pooling, so L need not be divisible by L_eff. This is a
    no-op when L == n_layers_out, so it is safe to call unconditionally.
    """
    t, n_layers, n_cols, n_chan = img.shape
    if n_layers == n_layers_out:
        return img
    if n_layers < n_layers_out:
        raise ValueError(
            f"cannot pool layer axis up: have L={n_layers}, asked for {n_layers_out}"
        )
    # adaptive_max_pool1d wants (N, C, L_in) and pools the last axis.
    flat = img.permute(0, 2, 3, 1).reshape(t * n_cols * n_chan, 1, n_layers)
    pooled = torch.nn.functional.adaptive_max_pool1d(flat, n_layers_out)
    return pooled.reshape(t, n_cols, n_chan, n_layers_out).permute(0, 3, 1, 2)
