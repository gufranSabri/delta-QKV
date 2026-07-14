"""Render token images to PNG, and report how different the views actually are.

This exists because of a specific risk: the whole architecture assumes Q, K and V
are three MEANINGFULLY DIFFERENT views. If their images turn out to be near-
identical, then giving each its own CNN and fusing them buys nothing, and you
want to discover that by looking at day-one data rather than after a week of
training runs.

The printed correlation matrix is the quantitative version of that check.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.config import Config
from src.extract.run_extraction import parse_meta
from src.utils.logger import get_logger

logger = get_logger(__name__)


def inspect(cfg: Config, idx: int = 0, n_tokens: int = 4, out: str | None = None) -> None:
    root = cfg.example_dir()
    ex_dir = root / f"{idx:05d}"
    tokens_path = ex_dir / "tokens.npy"
    if not tokens_path.exists():
        raise FileNotFoundError(f"no extracted example at {ex_dir}")

    geometry = json.loads((root / "geometry.json").read_text())
    views = geometry["views"]

    images = np.load(tokens_path).astype(np.float32)   # (T, V, L, C, 3)
    meta = parse_meta(ex_dir / "meta.txt")

    print(f"\nexample {idx}  ({ex_dir})")
    print(f"  response: {meta.get('response', '')[:120]}")
    print(f"  gold:     {str(meta.get('gold', ''))[:120]}")
    print(f"  label:    {meta.get('label')}  (1 = hallucinated)")
    print(f"  images:   {images.shape}  (T, views, layers, cols, channels)")
    print(f"  views:    {views}\n")

    # Per-view, per-channel magnitude. Wildly different scales across views is
    # exactly why normalisation is per-view.
    print("  magnitude by view/channel (mean |x|):")
    chan_names = ["raw", "d_prev", "d_next"]
    for v, name in enumerate(views):
        mags = [np.abs(images[:, v, :, :, c]).mean() for c in range(3)]
        print(f"    {name}: " + "  ".join(
            f"{cn}={m:9.4f}" for cn, m in zip(chan_names, mags)
        ))

    # THE check: are the views actually different from one another?
    print("\n  cross-view correlation of the raw channel (flattened):")
    raw = images[:, :, :, :, 0].reshape(images.shape[0], len(views), -1)
    flat = [raw[:, v].ravel() for v in range(len(views))]
    print("        " + "  ".join(f"{n:>6}" for n in views))
    for i, vi in enumerate(views):
        row = []
        for j in range(len(views)):
            r = np.corrcoef(flat[i], flat[j])[0, 1]
            row.append(f"{r:6.3f}")
        print(f"    {vi:>4}  " + "  ".join(row))
    print(
        "\n    (off-diagonal near 1.0 would mean the views are redundant and the\n"
        "     separate-CNN + fusion design is not buying anything.)\n"
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping PNG render")
        return

    n_tokens = min(n_tokens, images.shape[0])
    n_rows = len(views) * 3          # view x channel
    fig, axes = plt.subplots(
        n_rows, n_tokens,
        figsize=(2.2 * n_tokens, 2.0 * n_rows),
        squeeze=False,
    )

    for t in range(n_tokens):
        for v, vname in enumerate(views):
            for c, cname in enumerate(chan_names):
                ax = axes[v * 3 + c][t]
                img = images[t, v, :, :, c]
                # Symmetric colour scale for the signed delta channels, so zero
                # reads as neutral and the sign is visible.
                if c == 0:
                    ax.imshow(img, aspect="auto", cmap="viridis")
                else:
                    lim = np.abs(img).max() or 1.0
                    ax.imshow(img, aspect="auto", cmap="coolwarm", vmin=-lim, vmax=lim)
                ax.set_xticks([])
                ax.set_yticks([])
                if t == 0:
                    ax.set_ylabel(f"{vname}\n{cname}", fontsize=8)
                if v == 0 and c == 0:
                    ax.set_title(f"token {t}", fontsize=9)

    fig.suptitle(
        f"{cfg.dataset.name}/{cfg.llm.alias} example {idx} "
        f"(label={meta.get('label')})  |  rows: layers, cols: pooled features",
        fontsize=10,
    )
    fig.tight_layout()

    dest = Path(out or (ex_dir / "preview.png"))
    fig.savefig(dest, dpi=110)
    print(f"  rendered -> {dest}\n")
