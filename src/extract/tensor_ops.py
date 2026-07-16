"""Pure tensor math for building per-token Q/K/V images.

This module is deliberately free of any model/HuggingFace dependency so the
image-construction logic can be unit-tested without loading an LLM.

Pipeline for a single view (Q, K, V, or the hidden state H) of one example:

    raw      (T, L, D)   per-token, per-layer activation vectors
      -> pool_feature_axis  ->  (T, L, C)   C columns, C == L gives a square image
      -> add_*_channels     ->  (T, L, C, 3)

There are two ways to build the three channels, selected by `extraction_type`:

DELTA  (add_delta_channels), channel semantics (axis -1):
    0: pooled activation at (token t, layer l)
    1: delta to the PREVIOUS layer:  pooled[l] - pooled[l-1]
    2: delta to the NEXT layer:      pooled[l] - pooled[l+1]

  Deltas are SIGNED (not absolute): the sign distinguishes a representation
  growing from one shrinking, and abs() would discard that. Deltas are computed
  AFTER pooling, on the C pooled columns. Note this is not the same as pooling a
  full-D delta (max of differences != difference of maxes); we take the pooled
  columns as the canonical representation because they are exactly what the CNN
  consumes.

TRANSFORMS  (add_transform_channels), channel semantics (axis -1):
    0: raw pooled activation (unchanged)
    1: DWT (Haar, symmetric) magnitude along the LAYER axis
    2: DWT (Sym3, smooth) magnitude along the LAYER axis

  Both transforms run ALONG L, per (token, column), asking how a given pooled
  dimension EVOLVES with depth. DWT is multi-resolution and does NOT assume
  stationarity, so it can flag a localised jump at specific layers without
  smearing it across the spectrum.

  Two wavelet variants capture different aspects:
  - Haar (symmetric): orthogonal, sharp, localized discontinuities
  - Sym3 (smooth): more vanishing moments, smoother transitions, gradual changes
"""

from __future__ import annotations

import torch

# How to reduce the D-axis down to C columns.
#   max   -- peak value in each chunk
#   mean  -- average value in each chunk
#   l2    -- L2 norm (Euclidean length) of each chunk
#   sdk   -- selective dimension keeper: pick the C dimensions with highest entropy
#           (dimensions with highest entropy carry more information)
POOL_MODES = ("max", "mean", "l2", "sdk")

# Which channel-construction to apply after pooling. See module docstring.
EXTRACTION_TYPES = ("delta", "transforms")

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
    """Reduce the feature axis D down to `n_cols` by pooling or selecting dimensions.

    Args:
        raw:     (..., D) tensor. Typically (T, L, D).
        n_cols:  number of output columns C. For contiguous modes (max/mean/l2),
                 D must be divisible by C. For sdk, C can be any value <= D.
        mode:    one of POOL_MODES: "max", "mean", "l2", "sdk".

    Returns:
        (..., n_cols) tensor.

    Modes:
      max/mean/l2: Chunk j covers raw[..., j*C : (j+1)*C] where C = D // n_cols.
                   max pools peak value, mean pools average, l2 pools L2 norm.
      sdk:         Selective Dimension Keeper. Selects the n_cols dimensions
                   with highest Shannon entropy (information content) across the
                   (...,) axes. Entropy measures how much each dimension varies
                   in the batch; high-entropy dims carry more signal.

    For Llama-3-8B the Q view has D=4096 with 32 heads of head_dim 128, so with
    n_cols=32 the contiguous modes' chunks coincide exactly with attention heads
    and column j is "the peak/mean/norm activation of head j". That alignment is
    a happy accident of the architecture, NOT something enforced -- under GQA the
    K/V views have D=1024 and pooling those to 32 columns splits each head across
    4 columns. The sdk mode ignores this chunking and selects purely by entropy.
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
    if mode == "l2":
        # l2: the norm of each chunk. Non-negative by construction, unlike max/mean.
        return chunked.norm(dim=-1)

    # sdk: selective dimension keeper. Pick the n_cols dimensions with highest
    # entropy (treating the D-dim distribution per (T,L) position as a signal).
    # Reshape to (..., D) and compute Shannon entropy per dimension across the
    # (...,) axes, then select the top-entropy dimensions.
    x = raw.reshape(-1, d)                          # (T*L or broader, D)
    # Entropy per dimension: -sum(p * log(p)) where p is the normalized absolute value.
    # Use absolute value so both positive and negative activations contribute.
    x_abs = x.abs()
    x_sum = x_abs.sum(dim=0, keepdim=True)
    x_sum = x_sum.clamp(min=1e-8)                   # avoid division by zero
    p = x_abs / x_sum                               # (T*L, D) normalized probabilities
    entropy = -(p * (p.log() + 1e-8)).sum(dim=0)   # (D,) entropy per dimension
    # Select the n_cols dimensions with highest entropy
    topk_indices = entropy.topk(n_cols, dim=0).indices
    selected = raw[..., topk_indices]               # (..., n_cols)
    return selected


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


def _dwt_sym3_along_layers(pooled: torch.Tensor) -> torch.Tensor:
    """DWT (Sym3, smooth mode) detail magnitude along the LAYER axis of (T, L, C).

    Sym3 has more vanishing moments than Haar, so it captures smoother transitions
    and gradual changes between layers. Uses smooth boundary mode. Prefers PyWavelets;
    falls back to a simple symmetric smoothing in torch if pywt is unavailable.

    Returns (T, L, C), non-negative (magnitudes), aligned to the raw image.
    """
    n_layers = pooled.shape[1]

    try:
        import numpy as np
        import pywt  # optional; see scripts/install.sh

        x = pooled.transpose(1, 2).cpu().numpy()    # (T, C, L)
        max_level = pywt.dwt_max_level(n_layers, pywt.Wavelet("sym3").dec_len)
        max_level = max(1, max_level)
        coeffs = pywt.wavedec(x, "sym3", axis=-1, level=max_level, mode="smooth")
        out = np.zeros_like(x)
        for detail in coeffs[1:]:
            band = np.abs(detail)
            reps = int(np.ceil(n_layers / band.shape[-1]))
            up = np.repeat(band, reps, axis=-1)[..., :n_layers]
            out += up
        return torch.from_numpy(out).to(pooled.dtype).transpose(1, 2)  # (T, L, C)
    except Exception:
        # ---- fallback: symmetric difference (smooth approximation) -----------
        x = pooled.transpose(1, 2)                  # (T, C, L)
        if n_layers < 2:
            return torch.zeros_like(pooled)
        # Symmetric difference: captures local smoothness
        # (a[i+1] - a[i-1]) / 2, with boundary handling
        sym_diff = torch.zeros_like(x)
        sym_diff[:, :, 1:-1] = (x[:, :, 2:] - x[:, :, :-2]) / 2
        sym_diff[:, :, 0] = (x[:, :, 1] - x[:, :, 0])
        sym_diff[:, :, -1] = (x[:, :, -1] - x[:, :, -2])
        return sym_diff.abs().transpose(1, 2)       # (T, L, C)


def _dwt_along_layers(pooled: torch.Tensor) -> torch.Tensor:
    """DWT (Haar, symmetric mode) detail magnitude along the LAYER axis of (T, L, C).

    Multi-resolution: unlike the FFT it does not assume the layer signal is
    stationary, so a localised jump at a few layers shows up locally rather than
    smeared across every frequency bin. Prefers PyWavelets (a real Haar
    multilevel DWT with symmetric boundary extension, reconstructed to length L);
    if pywt is unavailable it falls back to a single-level Haar detail computed
    directly in torch, so extraction never hard-fails on a missing optional dependency.

    Returns (T, L, C), non-negative (magnitudes), aligned to the raw image.
    """
    n_layers = pooled.shape[1]

    try:
        import numpy as np
        import pywt  # optional; see scripts/install.sh

        # pywt works on numpy along an axis. Run a multilevel Haar DWT with
        # symmetric boundary mode along L, then upsample every detail band back
        # to L and sum their magnitudes, so the channel carries multi-scale
        # "how much is this dimension changing here" energy at each layer.
        x = pooled.transpose(1, 2).cpu().numpy()    # (T, C, L)
        max_level = pywt.dwt_max_level(n_layers, pywt.Wavelet("haar").dec_len)
        max_level = max(1, max_level)
        coeffs = pywt.wavedec(x, "haar", axis=-1, level=max_level, mode="symmetric")
        out = np.zeros_like(x)
        # coeffs[0] is the approximation; coeffs[1:] are detail bands, coarse->fine.
        for detail in coeffs[1:]:
            band = np.abs(detail)
            # Nearest-neighbour upsample this band back to L along the last axis.
            reps = int(np.ceil(n_layers / band.shape[-1]))
            up = np.repeat(band, reps, axis=-1)[..., :n_layers]
            out += up
        return torch.from_numpy(out).to(pooled.dtype).transpose(1, 2)  # (T, L, C)
    except Exception:
        # ---- pure-torch Haar fallback (single level) --------------------------
        # Haar detail d[i] = (a[2i] - a[2i+1]) / sqrt(2); upsample each detail
        # coefficient across its two source layers so the channel stays length L.
        x = pooled.transpose(1, 2)                  # (T, C, L)
        if n_layers < 2:
            return torch.zeros_like(pooled)
        even = x[..., 0 : n_layers - n_layers % 2 : 2]
        odd = x[..., 1 : n_layers - n_layers % 2 + 1 : 2]
        detail = (even - odd).abs() / (2 ** 0.5)    # (T, C, floor(L/2))
        up = detail.repeat_interleave(2, dim=-1)    # (T, C, 2*floor(L/2))
        if up.shape[-1] < n_layers:                 # odd L: pad the last layer
            up = torch.cat([up, up[..., -1:]], dim=-1)
        return up.transpose(1, 2)                    # (T, L, C)


def add_transform_channels(pooled: torch.Tensor) -> torch.Tensor:
    """Build the (raw, DWT-Haar, DWT-Sym3) image stack from pooled activations.

    Args:
        pooled: (T, L, C) pooled activations. L is the LAYER axis.

    Returns:
        (T, L, C, 3). Channels: (raw pooled, DWT-Haar magnitude, DWT-Sym3 magnitude),
        the latter two computed ALONG L. See module docstring.
    """
    if pooled.ndim != 3:
        raise ValueError(f"expected (T, L, C), got shape {tuple(pooled.shape)}")
    if pooled.shape[1] < 2:
        raise ValueError(
            f"need at least 2 layers for layer-axis transforms, got L={pooled.shape[1]}"
        )
    dwt_haar = _dwt_along_layers(pooled)
    dwt_sym3 = _dwt_sym3_along_layers(pooled)
    return torch.stack([pooled, dwt_haar, dwt_sym3], dim=-1)


def build_view_image(
    raw: torch.Tensor,
    n_cols: int,
    extraction_type: str = "delta",
    pool_mode: str = "max",
    boundary_mode: str = "zero",
) -> torch.Tensor:
    """raw (T, L, D) -> image (T, L, n_cols, 3) for a single view.

    `extraction_type` (see EXTRACTION_TYPES) selects how the three channels are
    built: `delta` -> (raw, dprev, dnext); `transforms` -> (raw, DWT-Haar, DWT-Sym3).
    The `boundary_mode` argument is only consulted for `delta`.
    """
    if raw.ndim != 3:
        raise ValueError(f"expected raw (T, L, D), got shape {tuple(raw.shape)}")
    if extraction_type not in EXTRACTION_TYPES:
        raise ValueError(
            f"extraction_type must be one of {EXTRACTION_TYPES}, got {extraction_type!r}"
        )
    # Pool in float32: fp16 max is exact, but mean/l2 over 128-element chunks can
    # overflow or lose precision at fp16 range, and deltas are differences of
    # similar magnitudes (catastrophic cancellation). Cast back at save time.
    pooled = pool_feature_axis(raw.float(), n_cols=n_cols, mode=pool_mode)
    if extraction_type == "transforms":
        return add_transform_channels(pooled)
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
