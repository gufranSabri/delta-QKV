"""Paper-figure generation: aggregated Grad-CAM, cross-view correlation,
temporal attention, and a Q-K alignment proxy.

Every function here writes to `<out_dir>/<fig_name>.png` plus a
`<out_dir>/README.txt` explaining how to read the figure and what it does and
does NOT show -- see `_write_readme` for the fixed template every figure uses.

REUSES rather than reimplements:
  - Grad-CAM/Eigen-CAM machinery (`_Recorder`, `_cam_from_activations`,
    `_last_conv_block`, `_set_eval_but_cudnn_rnn_trainable`, `_load_example`,
    `_stream_titles`) from `src.cam`.
  - `load_source` from `src.train`.
  - `compute_stats`, `QKVImageDataset`, `n_images`, `collate` from
    `src.data.dataset`.
  - `TemporalEncoder`'s new `return_weights` param (added in
    `src.models.temporal` alongside this module) for the attention-pool figure.

Two gate-based figures (a view-importance bar chart and per-token gate
evolution) were removed along with the ablation suite's fusion axis: the
main table trains with model.include=[0] (single image stream), under which
build_fusion always returns IdentityFusion, so no checkpoint in this
pipeline ever has real fusion gates to read.

HONESTY NOTE: `qkv_hooks.capture_all` never requests or stores
`output_attentions=True` -- no actual attention-PROBABILITY matrix between
token pairs is ever captured anywhere in this repo. `qk_alignment_proxy`
below is explicitly a PROXY (cosine similarity of pooled Q/K images, no RoPE,
no per-head softmax) and its README says so prominently. The only genuinely
real per-token "attention" figure this repo can honestly produce is the
temporal encoder's own `MaskedAttentionPool` weights (`temporal_attention_profile`).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.cam import (
    _cam_from_activations,
    _last_conv_block,
    _load_example,
    _Recorder,
    _set_eval_but_cudnn_rnn_trainable,
    _stream_titles,
)
from src.config import Config
from src.data.dataset import collate, n_images
from src.extract.run_extraction import parse_meta
from src.models.classifier import build_model
from src.train import load_source
from src.utils.logger import get_logger
from src.utils.seed import pick_device, seed_everything

logger = get_logger(__name__)


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _write_readme(
    out_dir: Path,
    what: str,
    how_to_read: str,
    provenance: dict,
    caveats: str = "None.",
) -> None:
    """Fixed-template interpretation guide, written next to every figure.

    Every figure's README has the same 4 sections in the same order, so a
    reader always finds the same information in the same place regardless of
    which figure they're looking at.
    """
    text = (
        "WHAT THIS SHOWS\n"
        "----------------\n"
        f"{what}\n\n"
        "HOW TO READ IT\n"
        "--------------\n"
        f"{how_to_read}\n\n"
        "PROVENANCE\n"
        "----------\n"
        f"{json.dumps(provenance, indent=2, default=str)}\n\n"
        "CAVEATS\n"
        "-------\n"
        f"{caveats}\n"
    )
    (out_dir / "README.txt").write_text(text, encoding="utf-8")


# ============================================================================
# (a) Aggregate Grad-CAM / Eigen-CAM: hallucinated vs truthful, averaged.
# ============================================================================


def aggregate_gradcam(
    cfg: Config,
    checkpoint: str | Path,
    dataset_name: str,
    n_examples: int = 50,
    method: str = "gradcam",
    out_dir: str | Path = "figs/aggregate_cam",
) -> Path:
    """Average Grad-CAM/Eigen-CAM heatmaps over N hallucinated vs N truthful
    examples, per image stream, in (layer x pooled-column) space.

    Extends src.cam.run_cam's single-example loop into an aggregate: same
    hook/heatmap machinery, looped over many examples per label, averaged
    (post per-token normalization, pre-rendering) within each label group.
    """
    if method not in ("gradcam", "eigencam"):
        raise ValueError(f"method must be 'gradcam' or 'eigencam', got {method!r}")

    seed_everything(cfg.train.seed)
    device = pick_device()

    ckpt_path = Path(checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg.extract.views = ckpt["views"]
    cfg.model.channels = ckpt.get("channels", "default")
    cfg.model.include = ckpt.get("include", None)

    source = load_source(cfg, dataset_name, cfg.llm.alias)
    source.stats = ckpt["stats"]

    labels = source.labels
    hall_idx = [i for i, y in enumerate(labels) if y == 1][:n_examples]
    truth_idx = [i for i, y in enumerate(labels) if y == 0][:n_examples]
    if not hall_idx or not truth_idx:
        raise ValueError(
            f"{dataset_name}: need at least one hallucinated AND one truthful "
            f"example, got {len(hall_idx)} hallucinated / {len(truth_idx)} truthful"
        )

    n_streams = n_images(cfg.model.channels, len(cfg.extract.views), cfg.model.include)
    model = build_model(cfg, n_views=n_streams).to(device)
    model.load_state_dict(ckpt["model"])

    need_grad = method == "gradcam"
    if need_grad:
        _set_eval_but_cudnn_rnn_trainable(model)
    else:
        model.eval()

    backbones = model.backbones if not model.share_backbone else [model.backbones[0]] * n_streams

    def _mean_cam_per_stream(indices: list[int]) -> list[np.ndarray]:
        """(n_streams,) list of (h, w) heatmaps, averaged over tokens AND
        over every example in `indices`."""
        sums = [None] * n_streams
        counts = [0] * n_streams
        for idx in indices:
            item = source[idx]
            batch = collate([item])
            images, _labels, mask, _origins = batch
            images = images.to(device)
            mask = mask.to(device)

            recorders = [_Recorder(_last_conv_block(bb), need_grad) for bb in backbones]
            with torch.set_grad_enabled(need_grad):
                logit = model(images, mask)
                if need_grad:
                    model.zero_grad(set_to_none=True)
                    logit.sum().backward()

            real_tokens = int(mask[0].sum().item())
            for s, rec in enumerate(recorders):
                acts = rec.activations
                grads = rec.gradients if need_grad else None
                cam = _cam_from_activations(acts, grads)[:real_tokens]  # (T, h, w)
                mean_cam = cam.mean(dim=0).detach().cpu().numpy()       # (h, w)
                if sums[s] is None:
                    sums[s] = mean_cam
                else:
                    sums[s] = sums[s] + mean_cam
                counts[s] += 1
                rec.remove()

        return [s / max(c, 1) for s, c in zip(sums, counts)]

    hall_cams = _mean_cam_per_stream(hall_idx)
    truth_cams = _mean_cam_per_stream(truth_idx)

    titles = _stream_titles(
        cfg.extract.views, cfg.extract.extraction_type, cfg.model.channels, cfg.model.include,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "aggregate_cam.png"

    plt = _matplotlib()
    fig, axes = plt.subplots(
        n_streams, 2, figsize=(6, 2.4 * n_streams), squeeze=False,
    )
    for s in range(n_streams):
        for c, (cam, label) in enumerate([(hall_cams[s], "hallucinated (mean)"), (truth_cams[s], "truthful (mean)")]):
            ax = axes[s][c]
            ax.imshow(cam, cmap="jet", aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            if s == 0:
                ax.set_title(label, fontsize=10)
            if c == 0:
                ax.set_ylabel(titles[s], fontsize=9)
    fig.suptitle(
        f"{method}: mean over {len(hall_idx)} hallucinated / {len(truth_idx)} "
        f"truthful examples -- {dataset_name}/{cfg.llm.alias}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(dest, dpi=130)
    plt.close(fig)

    _write_readme(
        out_dir,
        what=(
            f"Grad-CAM/Eigen-CAM heatmaps ({method}) averaged over "
            f"{len(hall_idx)} hallucinated and {len(truth_idx)} truthful "
            f"{dataset_name}/{cfg.llm.alias} examples, one row per image "
            "stream, showing where in (layer x pooled-dimension) space the "
            "detector's CNN attends, on average, when it fires vs. when it "
            "doesn't."
        ),
        how_to_read=(
            "Rows = image streams (labelled on the left, same convention as "
            "src/cam.py's _stream_titles). Columns = hallucinated-mean vs "
            "truthful-mean. Y axis of each heatmap = layer, X axis = pooled "
            "column. Brighter (more red/yellow on the 'jet' colormap) = "
            "the CNN's last conv block activates more strongly there, "
            "averaged over every real (non-padded) token of every example "
            "in that group. A systematic difference between the two columns "
            "in the SAME row is evidence the detector looks at a different "
            "part of that stream's image depending on the label."
        ),
        provenance={
            "checkpoint": str(ckpt_path), "dataset": dataset_name,
            "llm": cfg.llm.alias, "method": method,
            "n_hallucinated": len(hall_idx), "n_truthful": len(truth_idx),
            "views": cfg.extract.views, "channels": cfg.model.channels,
            "include": cfg.model.include,
        },
        caveats=(
            "Each example's per-token CAM is averaged over tokens BEFORE "
            "averaging over examples, so a token-level effect (e.g. only the "
            "very first hallucinated token lights up) can be diluted in this "
            "aggregate view. For a single-example, per-token view, use "
            "`main.py cam` instead."
        ),
    )
    logger.info("wrote %s", dest)
    return dest


# ============================================================================
# (c) Cross-view correlation matrix, aggregated over many examples.
# ============================================================================


def cross_view_correlation_figure(
    cfg: Config,
    dataset_name: str,
    n_examples: int = 100,
    out_dir: str | Path = "figs/cross_view_correlation",
) -> Path:
    """Reformats inspect_images.py's raw-channel correlation-matrix check
    (originally one example, printed to stdout) into a heatmap PNG, averaged
    over n_examples examples' correlation matrices instead of just one.
    """
    root = cfg.example_dir()
    geometry_path = root / "geometry.json"
    if not geometry_path.exists():
        raise FileNotFoundError(f"no extracted data at {root}; run `extract` first")
    geometry = json.loads(geometry_path.read_text())
    views = geometry["views"]

    manifest_path = root / "manifest.jsonl"
    records = []
    with open(manifest_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    records = records[:n_examples]
    if not records:
        raise ValueError(f"no examples in {manifest_path}")

    n_views = len(views)
    corr_sum = np.zeros((n_views, n_views))
    n_used = 0
    for rec in records:
        ex_dir = root / rec["dir"]
        images = np.load(ex_dir / "tokens.npy").astype(np.float32)  # (T, V, L, C, 3)
        raw = images[:, :, :, :, 0].reshape(images.shape[0], n_views, -1)
        flat = [raw[:, v].ravel() for v in range(n_views)]
        for i in range(n_views):
            for j in range(n_views):
                r = np.corrcoef(flat[i], flat[j])[0, 1]
                corr_sum[i, j] += 0.0 if np.isnan(r) else r
        n_used += 1

    corr_mean = corr_sum / max(n_used, 1)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "cross_view_correlation.png"

    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(corr_mean, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(n_views)); ax.set_xticklabels(views)
    ax.set_yticks(range(n_views)); ax.set_yticklabels(views)
    for i in range(n_views):
        for j in range(n_views):
            ax.text(j, i, f"{corr_mean[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    ax.set_title(f"cross-view correlation (raw channel)\n{cfg.dataset.name}/{cfg.llm.alias}, n={n_used}")
    fig.tight_layout()
    fig.savefig(dest, dpi=130)
    plt.close(fig)

    _write_readme(
        out_dir,
        what=(
            f"Pearson correlation of the raw (channel-0) activation values "
            f"between every pair of extracted views ({', '.join(views)}), "
            f"averaged over {n_used} {cfg.dataset.name}/{cfg.llm.alias} examples."
        ),
        how_to_read=(
            "Symmetric matrix, one row/column per view, diagonal is always "
            "1.0. Off-diagonal near 1.0 means two views carry nearly "
            "redundant information; near 0 means they are genuinely "
            "distinct signals. This is the quantitative version of the "
            "check src/inspect_images.py already prints per-example -- the "
            "whole per-view-CNN-plus-fusion architecture only earns its "
            "keep if the off-diagonal values are meaningfully below 1.0."
        ),
        provenance={
            "dataset": dataset_name, "llm": cfg.llm.alias,
            "source": cfg.extract.source, "extraction_type": cfg.extract.extraction_type,
            "n_examples": n_used, "views": views,
        },
    )
    logger.info("wrote %s", dest)
    return dest


# ============================================================================
# (d) Temporal attention profile -- per-token weights from MaskedAttentionPool.
# ============================================================================


def temporal_attention_profile(
    cfg: Config,
    checkpoint: str | Path,
    dataset_name: str,
    idx: int = 0,
    out_dir: str | Path = "figs/temporal_attention",
) -> Path:
    """Per-token attention-pool weight across ONE generated response.

    Real softmax attention weights the model itself computes over TOKENS
    (not between token pairs) -- calls model.temporal directly with
    return_weights=True (added alongside this module), bypassing
    model.forward the same way src.cam bypasses it to reach intermediate
    layers.
    """
    seed_everything(cfg.train.seed)
    device = pick_device()

    ckpt_path = Path(checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    batch, source = _load_example(cfg, ckpt, dataset_name, idx)
    images, labels, mask, origins = batch
    images = images.to(device)
    mask = mask.to(device)

    n_streams = n_images(cfg.model.channels, len(cfg.extract.views), cfg.model.include)
    model = build_model(cfg, n_views=n_streams).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    with torch.no_grad():
        fused = model.encode_tokens(images)
        _pooled, weights = model.temporal(fused, mask, return_weights=True)

    real_tokens = int(mask[0].sum().item())
    w = weights[0, :real_tokens].cpu().numpy()

    rec = source.records[idx]
    meta = parse_meta(source.root / rec["dir"] / "meta.txt")
    response = meta.get("response", "")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "temporal_attention.png"

    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(max(4, 0.35 * real_tokens), 3))
    ax.bar(range(real_tokens), w, color="#DD8452")
    ax.set_xlabel("generated token position")
    ax.set_ylabel("attention-pool weight")
    ax.set_title(
        f"temporal attention -- {origins[0]} idx {idx} "
        f"(label={int(labels.item())}, 1=hallucinated)"
    )
    fig.tight_layout()
    fig.savefig(dest, dpi=130)
    plt.close(fig)

    _write_readme(
        out_dir,
        what=(
            f"Per-token softmax attention weight from the temporal encoder's "
            f"MaskedAttentionPool, for ONE example (idx {idx}) of "
            f"{dataset_name}/{cfg.llm.alias} -- how much each generated "
            "token contributes to the final pooled representation the "
            "classifier head sees."
        ),
        how_to_read=(
            "One bar per generated token, in generation order (left to "
            "right = start to end of the response). Bars sum to 1. A tall "
            "bar at position t means the model's final hallucination "
            "prediction leans heavily on that token's fused embedding."
        ),
        provenance={
            "checkpoint": str(ckpt_path), "dataset": dataset_name,
            "llm": cfg.llm.alias, "idx": idx, "label": int(labels.item()),
            "response": response,
        },
        caveats=(
            "This is a REAL attention weight the model computes (not a "
            "proxy) -- but it is over TOKENS of one response, not between "
            "token PAIRS. This repo never captures actual inter-token "
            "attention probability matrices (no output_attentions=True "
            "anywhere in qkv_hooks.py); see qk_alignment_proxy's caveats for "
            "the explicitly-labeled proxy this repo offers instead."
        ),
    )
    logger.info("wrote %s", dest)
    return dest


# ============================================================================
# (f) Q-K alignment proxy -- explicitly NOT a real attention map.
# ============================================================================


def qk_alignment_proxy(
    cfg: Config,
    dataset_name: str,
    idx: int = 0,
    layer: int | None = None,
    out_dir: str | Path = "figs/qk_alignment_proxy",
) -> Path:
    """Cosine similarity between the pooled Q and pooled K images (per layer,
    or averaged over layers if layer=None) for one example -- an explicit,
    clearly-labeled PROXY for inter-token attention.

    NOT the real attention matrix: real attention applies RoPE to Q/K before
    the dot product and a per-head softmax normalization; this uses the
    pooled, pre-RoPE Q/K images already stored on disk and a plain cosine
    similarity, with no per-head structure. Only meaningful under
    extract.source=qkv (Q and K both exist); raises otherwise.
    """
    if cfg.extract.source != "qkv":
        raise ValueError(
            f"qk_alignment_proxy needs extract.source=qkv (both Q and K), "
            f"got source={cfg.extract.source!r}"
        )
    if "Q" not in cfg.extract.views or "K" not in cfg.extract.views:
        raise ValueError(
            f"qk_alignment_proxy needs both Q and K in extract.views, got {cfg.extract.views}"
        )

    root = cfg.example_dir()
    geometry = json.loads((root / "geometry.json").read_text())
    views = geometry["views"]
    q_idx, k_idx = views.index("Q"), views.index("K")

    ex_dir = root / f"{idx:05d}"
    images = np.load(ex_dir / "tokens.npy").astype(np.float32)  # (T, V, L, C, 3)
    q = images[:, q_idx, :, :, 0]  # (T, L, C) raw channel
    k = images[:, k_idx, :, :, 0]  # (T, L, C)

    t, n_layers, n_cols = q.shape
    q_t = torch.from_numpy(q)
    k_t = torch.from_numpy(k)

    if layer is not None:
        q_t = q_t[:, layer]   # (T, C)
        k_t = k_t[:, layer]   # (T, C)
        sim = F.cosine_similarity(q_t.unsqueeze(1), k_t.unsqueeze(0), dim=-1).numpy()  # (T, T)
        layer_label = f"layer {layer}"
    else:
        q_flat = q_t.reshape(t, -1)   # (T, L*C)
        k_flat = k_t.reshape(t, -1)
        sim = F.cosine_similarity(q_flat.unsqueeze(1), k_flat.unsqueeze(0), dim=-1).numpy()  # (T, T)
        layer_label = "all layers (flattened)"

    meta = parse_meta(ex_dir / "meta.txt")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "qk_alignment_proxy.png"

    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(sim, cmap="viridis", aspect="auto")
    ax.set_xlabel("K token position")
    ax.set_ylabel("Q token position")
    fig.colorbar(im, ax=ax, label="cosine similarity")
    ax.set_title(f"Q-K alignment PROXY ({layer_label})\n{cfg.dataset.name}/{cfg.llm.alias} idx {idx}")
    fig.tight_layout()
    fig.savefig(dest, dpi=130)
    plt.close(fig)

    _write_readme(
        out_dir,
        what=(
            f"Cosine similarity between the pooled Q image and pooled K image "
            f"of every pair of generated tokens ({layer_label}), for example "
            f"{idx} of {cfg.dataset.name}/{cfg.llm.alias} -- a PROXY for how "
            "aligned each token's query and key representations are."
        ),
        how_to_read=(
            "(T, T) heatmap, T = number of generated tokens. Cell (i, j) = "
            "cosine similarity between token i's pooled Q and token j's "
            "pooled K. Brighter = more aligned. If this proxy tracked real "
            "attention, a bright cell (i, j) would suggest token i's "
            "generation was influenced by token j."
        ),
        provenance={
            "dataset": dataset_name, "llm": cfg.llm.alias, "idx": idx,
            "layer": layer if layer is not None else "all (flattened)",
            "response": meta.get("response", ""),
        },
        caveats=(
            "*** THIS IS NOT A REAL ATTENTION MATRIX. ***\n"
            "qkv_hooks.capture_all never captures output_attentions -- no "
            "actual softmax(QK^T/sqrt(d)) attention probability is stored "
            "anywhere in this repo. This figure uses the POOLED, PRE-RoPE "
            "Q/K projections already saved to disk (see src/extract/qkv_hooks.py's "
            "module docstring: Q/K are captured before RoPE is applied) and "
            "a plain per-token cosine similarity, with no per-head structure "
            "and no softmax normalization. Treat this as a rough proxy for "
            "'how aligned are two tokens' query/key representations', not as "
            "the model's actual attention pattern. temporal_attention_profile "
            "is the honest, real per-token attention weight this repo can "
            "produce without new capture code."
        ),
    )
    logger.info("wrote %s", dest)
    return dest
