"""Training loop.

Supports both experimental settings:
  - same-dataset:  train and test on one (dataset, LLM) source.
  - leave-one-dataset-out: train on the union of several sources, test zero-shot
    on a held-out source that the model never saw.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.config import Config
from src.data.dataset import (
    ConcatQKVDataset,
    QKVImageDataset,
    _set_stats,
    collate,
    compute_stats,
    n_channels,
    n_images,
)
from src.models.classifier import build_model
from src.utils.logger import get_logger, setup_logging
from src.utils.metrics import compute_metrics, format_metrics
from src.utils.progress import progress
from src.utils.seed import pick_device, seed_everything
from src.utils.snapshot import snapshot_code
from src.extract.datasets import HAS_TEST_CORPUS, base_name
from src.utils.splits import make_split

logger = get_logger(__name__)


def load_source(cfg: Config, dataset_name: str, llm_alias: str, **kw) -> QKVImageDataset:
    # Mirror Config.example_dir()'s layout, but for an ARBITRARY (dataset, llm)
    # pair -- LODO trains over several sources that differ from cfg.dataset /
    # cfg.llm. The source/extraction_type prefix still comes from cfg, since a run
    # trains on one source+extraction_type at a time.
    root = (
        Path(cfg.data_root)
        / cfg.extract.source
        / cfg.extract.extraction_type
        / dataset_name
        / llm_alias
    )
    return QKVImageDataset(
        root,
        views=cfg.extract.views,
        max_tokens=cfg.extract.max_tokens,
        origin=f"{llm_alias}/{dataset_name}",
        channels=cfg.model.channels,
        include=cfg.model.include,
        stream2_enable=cfg.model.stream2.enable,
        stream2_include=cfg.model.stream2.include,
        **kw,
    )


def build_datasets(
    cfg: Config,
    train_datasets: list[str],
    test_dataset: str | None,
    llm_alias: str,
):
    """Returns (train_sources, test_source_or_None)."""
    sources = [load_source(cfg, name, llm_alias) for name in train_datasets]
    for s in sources:
        pos = np.mean(s.labels)
        logger.info(
            "source %-28s n=%-6d hallucination rate=%.1f%%",
            s.origin, len(s), 100 * pos,
        )

    test_source = None
    if test_dataset and test_dataset not in train_datasets:
        test_source = load_source(cfg, test_dataset, llm_alias)
        logger.info(
            "held-out test source %-14s n=%-6d hallucination rate=%.1f%%",
            test_source.origin, len(test_source), 100 * np.mean(test_source.labels),
        )
    return sources, test_source


def _unpack_batch(batch, device):
    """collate() yields a 4-tuple (stream2 off) or 6-tuple (stream2 on).

    Returns (model_args, labels, origins) where model_args is the exact
    positional-arg tuple QKVHalluDetector.forward expects.
    """
    if len(batch) == 4:
        images, labels, mask, origins = batch
        images = images.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        return (images, mask), labels.to(device, non_blocking=True), origins

    images, images2, labels, mask, mask2, origins = batch
    images = images.to(device, non_blocking=True)
    images2 = images2.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)
    mask2 = mask2.to(device, non_blocking=True)
    return (images, images2, mask, mask2), labels.to(device, non_blocking=True), origins


def run_epoch(model, loader, criterion, device, optimizer=None, desc=""):
    """One pass. Trains if `optimizer` is given, else evaluates.

    Takes no scheduler: the LR schedule is ReduceLROnPlateau, which steps once
    per epoch against the validation metric, so the caller drives it.
    """
    training = optimizer is not None
    model.train(training)

    total_loss, n_batches = 0.0, 0
    all_y, all_p, all_origins = [], [], []

    with torch.set_grad_enabled(training):
        for batch in progress(loader, desc=desc, leave=False, ncols=100):
            model_args, labels, origins = _unpack_batch(batch, device)

            logits = model(*model_args)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # Exploding gradients through a BiLSTM are a classic failure here.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            all_y.extend(labels.detach().cpu().numpy())
            # Metrics need probabilities; the model emits raw logits.
            all_p.extend(torch.sigmoid(logits).detach().cpu().numpy())
            all_origins.extend(origins)

    metrics = compute_metrics(all_y, all_p)
    metrics["loss"] = total_loss / max(n_batches, 1)

    # Per-source breakdown -- the whole point of the LODO setting.
    per_origin = {}
    if len(set(all_origins)) > 1:
        buckets = defaultdict(lambda: ([], []))
        for y, p, o in zip(all_y, all_p, all_origins):
            buckets[o][0].append(y)
            buckets[o][1].append(p)
        per_origin = {o: compute_metrics(ys, ps) for o, (ys, ps) in buckets.items()}

    return metrics, per_origin


def _section(title: str) -> None:
    logger.info("")
    logger.info("--- %s %s", title, "-" * max(0, 58 - len(title)))


def log_run_header(cfg: Config, run_dir, device, train_datasets, test_dataset) -> None:
    """Log what this run actually IS, before anything expensive happens.

    Deliberately reports DERIVED facts (how many images each stream really
    gets, how many channels each carries, which fusion was really built), not
    just echoed config strings: `channels`/`include` mean the image count is
    NOT len(views), and a single image collapses fusion to IdentityFusion
    regardless of model.fusion. Echoing the raw config hides both.
    """
    n_views = len(cfg.extract.views)
    n1 = n_images(cfg.model.channels, n_views, cfg.model.include)
    in_ch = n_channels(cfg.model.channels, n_views)

    _section("run")
    logger.info("run dir      : %s", run_dir)
    logger.info("device       : %s", device)
    logger.info("seed         : %d", cfg.train.seed)

    _section("data")
    logger.info("train sources: %s", train_datasets)
    logger.info("test source  : %s", test_dataset or "(val split)")
    logger.info("llm          : %s (%s)", cfg.llm.alias, cfg.llm.name)
    logger.info("source       : %s / %s", cfg.extract.source, cfg.extract.extraction_type)
    logger.info("views        : %s", cfg.extract.views)
    logger.info("pool         : %s | boundary: %s", cfg.extract.pool, cfg.extract.boundary_mode)
    logger.info("max_tokens   : %s | n_cols: %s | l_eff: %s",
                cfg.extract.max_tokens, cfg.extract.n_cols, cfg.extract.l_eff)
    logger.info("labeling     : %s", cfg.labeling.scheme)

    _section("model")
    logger.info("backbone     : %s (shared=%s, pretrained=%s)",
                cfg.model.backbone, cfg.model.share_backbone,
                cfg.model.pretrained_backbone if cfg.model.backbone == "resnet18" else "n/a")
    logger.info("channels     : %s | include: %s",
                cfg.model.channels, cfg.model.include if cfg.model.include is not None else "all")
    # The line that actually says what the CNNs see. `same` + include=[0] means
    # ONE image of V channels, not V images -- which "views: [Q, K, V]" implies.
    logger.info("stream 1     : %d image(s) x %d channel(s), (L, D) spatial -> temporal encoder",
                n1, in_ch)

    if cfg.model.stream2.enable:
        n2 = n_images(cfg.model.channels, n_views, cfg.model.stream2.include)
        logger.info("stream 2     : %d image(s) x %d channel(s), (L, T) spatial -> masked pool  [include: %s]",
                    n2, in_ch,
                    cfg.model.stream2.include if cfg.model.stream2.include is not None else "all")
        logger.info("             : stream vectors are concatenated before the head")
    else:
        logger.info("stream 2     : disabled")

    # fusion only exists with >1 image to fuse; otherwise it is IdentityFusion.
    fusion_desc = cfg.model.fusion if n1 > 1 else f"identity (only 1 image; {cfg.model.fusion} unused)"
    logger.info("fusion       : %s", fusion_desc)
    logger.info("embed_dim    : %d | fused_dim: %d | dropout: %.3g",
                cfg.model.embed_dim, cfg.model.fused_dim, cfg.model.dropout)
    logger.info("temporal     : conv1d x%d | bilstm hidden=%d x%d layer(s)",
                cfg.model.conv1d_layers, cfg.model.lstm_hidden, cfg.model.lstm_layers)

    _section("train")
    logger.info("epochs       : %d | patience: %d | batch_size: %d",
                cfg.train.epochs, cfg.train.patience, cfg.train.batch_size)
    logger.info("lr           : %.3g | weight_decay: %.3g | backbone_lr_scale: %.3g",
                cfg.train.lr, cfg.train.weight_decay, cfg.train.backbone_lr_scale)
    logger.info("balance_class: %s | val_fraction: %.3g | test_fraction: %.3g",
                cfg.train.balance_classes, cfg.train.val_fraction, cfg.train.test_fraction)


def log_param_counts(model) -> None:
    """Per-component parameter breakdown -- where the capacity actually is."""
    groups = [
        ("backbones (stream 1)", getattr(model, "backbones", None)),
        ("backbones (stream 2)", getattr(model, "backbones2", None)),
        ("fusion (stream 1)", getattr(model, "fusion", None)),
        ("fusion (stream 2)", getattr(model, "fusion2", None)),
        ("temporal encoder", getattr(model, "temporal", None)),
        ("head", getattr(model, "head", None)),
    ]
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for name, mod in groups:
        if mod is None:
            continue
        n = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        logger.info("  %-22s %11s  (%4.1f%%)", name, f"{n:,}", 100 * n / max(total, 1))
    logger.info("  %-22s %11s", "TOTAL trainable", f"{total:,}")


def log_model_repr(model) -> None:
    """The full module tree, exactly as PyTorch sees it.

    Emitted line by line rather than as one blob: the log formatter prefixes
    every record, so a single multi-line message would leave all but the first
    line unprefixed and misaligned.
    """
    for line in repr(model).splitlines():
        logger.info("  %s", line)


def train(
    cfg: Config,
    train_datasets: list[str],
    test_dataset: str | None = None,
    run_name: str | None = None,
) -> dict:
    seed_everything(cfg.train.seed)
    device = pick_device()

    run_dir = Path(cfg.runs_root) / (run_name or default_run_name(cfg, train_datasets, test_dataset))
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_code(run_dir)
    setup_logging(log_file=run_dir / "train.log")

    log_run_header(cfg, run_dir, device, train_datasets, test_dataset)

    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    # ---- data ---------------------------------------------------------
    sources, test_source = build_datasets(cfg, train_datasets, test_dataset, cfg.llm.alias)
    full = ConcatQKVDataset(sources) if len(sources) > 1 else sources[0]

    # Datasets that ship a separate held-out corpus (`<name>_test`) need no slice
    # carved out of the training pool -- their test set is a different corpus.
    # Only single-split datasets (TruthfulQA) do, otherwise we'd be discarding
    # training data to build a test set we already have.
    needs_slice = [d for d in train_datasets if base_name(d)[0] not in HAS_TEST_CORPUS]
    test_fraction = cfg.train.test_fraction if needs_slice else 0.0
    if needs_slice:
        logger.info(
            "%s have no separate test corpus; carving out a %.0f%% stratified test slice",
            needs_slice, 100 * test_fraction,
        )

    train_idx, val_idx, heldout_idx = make_split(
        full.labels,
        val_fraction=cfg.train.val_fraction,
        test_fraction=test_fraction,
        seed=cfg.train.seed,
        cache=run_dir / "split.json",
    )
    logger.info("split        : train %d | val %d | heldout %d",
                len(train_idx), len(val_idx), len(heldout_idx))

    geom = getattr(full, "geometry", None) or getattr(full.datasets[0], "geometry", {})
    logger.info("image size   : %s rows (L) x %s cols (D)",
                geom.get("n_rows", "?"), geom.get("n_cols", "?"))

    _section("normalisation (train split only)")
    # Normalisation statistics come from the TRAIN split only -- computing them
    # over val/test would leak those distributions into the input scaling.
    stats_path = run_dir / "stats.json"
    stats = compute_stats(full, train_idx)
    stats_path.write_text(json.dumps(stats, indent=2))
    for view, s in stats.items():
        logger.info("norm %-3s: mean=%s std=%s",
                    view,
                    [f"{m:+.3f}" for m in s["mean"]],
                    [f"{v:.3f}" for v in s["std"]])

    _set_stats(full, stats)
    if test_source is not None:
        # The held-out set is normalised with the TRAIN statistics. It must be:
        # at inference you do not get to peek at the test distribution.
        test_source.stats = stats

    loader_kw = dict(
        batch_size=cfg.train.batch_size,
        collate_fn=collate,
        num_workers=cfg.train.num_workers,
        pin_memory=device.type == "cuda",
    )
    # drop_last on TRAIN only: BatchNorm1d in TemporalEncoder's conv stack
    # can't compute a variance over a batch of size 1, which a trailing
    # remainder batch hits whenever len(train_idx) % batch_size == 1. Val/test
    # must never drop data -- that would silently shrink the reported metrics.
    train_loader = DataLoader(
        Subset(full, train_idx), shuffle=True, drop_last=True, **loader_kw
    )
    val_loader = DataLoader(Subset(full, val_idx), shuffle=False, **loader_kw)
    test_loader = (
        DataLoader(test_source, shuffle=False, **loader_kw) if test_source else None
    )

    # ---- model --------------------------------------------------------
    # n_views here means "number of CNN streams", which is the number of IMAGES
    # after regrouping -- not len(extract.views). They differ under model.channels.
    n_streams = n_images(cfg.model.channels, len(cfg.extract.views), cfg.model.include)
    n_streams2 = None
    if cfg.model.stream2.enable:
        n_streams2 = n_images(
            cfg.model.channels, len(cfg.extract.views), cfg.model.stream2.include
        )
    model = build_model(cfg, n_views=n_streams, n_views2=n_streams2).to(device)

    _section("architecture")
    log_model_repr(model)

    _section("parameters")
    log_param_counts(model)

    _section("optimisation")
    # Class weighting: hallucination rates are typically far from 50/50, and an
    # unweighted BCE will happily collapse to predicting the majority class.
    pos_weight = None
    if cfg.train.balance_classes:
        y = np.asarray([full.labels[i] for i in train_idx])
        n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
        if n_pos > 0 and n_neg > 0:
            # float32 explicitly: numpy's int division yields float64, which MPS
            # refuses outright and which would silently upcast the loss on CUDA.
            pos_weight = torch.tensor(
                [n_neg / n_pos], dtype=torch.float32, device=device
            )
            logger.info("pos_weight = %.3f (neg=%d, pos=%d)", pos_weight.item(), n_neg, n_pos)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Discriminative LR: the CNN backbones get lr * backbone_lr_scale, everything
    # else (fusion, temporal, head) gets the full lr. A pretrained backbone
    # (scale < 1) wants a gentler LR than the randomly-initialised head, or it
    # gets its ImageNet features wrecked before the head stabilises. scale == 1.0
    # collapses to a single group -- correct for scratch / random-init, where
    # nothing is pretrained. ReduceLROnPlateau scales every group by the same
    # factor, so the backbone:head LR ratio holds for the whole run.
    scale = cfg.train.backbone_lr_scale
    backbone_lr = cfg.train.lr * scale
    if scale == 1.0:
        param_groups = [{"params": model.parameters(), "lr": cfg.train.lr}]
    else:
        backbone_params = list(model.backbones.parameters())
        if cfg.model.stream2.enable:
            # Same reasoning applies to stream 2's backbones -- they are also
            # per-view CNNs, not part of fusion/temporal/head.
            backbone_params += list(model.backbones2.parameters())
        backbone_ids = {id(p) for p in backbone_params}
        other_params = [p for p in model.parameters() if id(p) not in backbone_ids]
        param_groups = [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": other_params, "lr": cfg.train.lr},
        ]
        logger.info(
            "discriminative LR: backbone=%.2e, rest=%.2e (scale=%.3g)",
            backbone_lr, cfg.train.lr, scale,
        )

    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.train.weight_decay)

    # Linear decay: hold the LR flat for the first `lr_decay_start` epochs, then
    # ramp it linearly down to `lr_final_scale` of its initial value by the last
    # epoch. LambdaLR multiplies each group's OWN initial LR by the same factor,
    # so the backbone:head ratio set above holds for the entire run.
    #
    # Stepped once per EPOCH (not per batch), so it is deliberately not handed to
    # run_epoch(). Note the slope is tied to `epochs`: if early stopping fires
    # first, the LR simply never reaches the floor.
    warm = cfg.train.lr_decay_start
    final = cfg.train.lr_final_scale
    total = cfg.train.epochs

    def lr_lambda(epoch: int) -> float:      # epoch is 0-based
        if epoch < warm:
            return 1.0
        # Guard the degenerate case where decay starts on/after the final epoch.
        span = max(1, total - warm)
        progress = min(1.0, (epoch - warm) / span)
        return 1.0 + progress * (final - 1.0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    logger.info(
        "linear LR decay: flat for %d epochs, then -> %.3g x initial by epoch %d",
        warm, final, total,
    )

    # ---- loop ---------------------------------------------------------
    _section("training")
    best_auroc, best_epoch, stale = -1.0, -1, 0
    history = []

    for epoch in range(1, cfg.train.epochs + 1):
        t0 = time.time()
        # No scheduler here: the LR schedule steps per EPOCH, not per batch.
        tr, _ = run_epoch(model, train_loader, criterion, device,
                          optimizer, desc=f"epoch {epoch} train")
        va, va_origins = run_epoch(model, val_loader, criterion, device,
                                   desc=f"epoch {epoch} val")

        lrs = [g["lr"] for g in optimizer.param_groups]
        logger.info("epoch %2d | train loss %.4f AUROC %.4f | val %s | lr %s | %.0fs",
                    epoch, tr["loss"], tr["auroc"], format_metrics(va),
                    " ".join(f"{lr:.2e}" for lr in lrs), time.time() - t0)

        # Linear decay is a pure function of the epoch index -- no metric.
        scheduler.step()

        record = {"epoch": epoch, "train": tr, "val": va, "lr": lrs}

        # Model selection on validation AUROC, never on test.
        if va["auroc"] > best_auroc:
            best_auroc, best_epoch, stale = va["auroc"], epoch, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg.to_dict(),
                    "stats": stats,
                    "views": cfg.extract.views,
                    "channels": cfg.model.channels,
                    "include": cfg.model.include,
                    "stream2_enable": cfg.model.stream2.enable,
                    "stream2_include": cfg.model.stream2.include,
                    "epoch": epoch,
                    "val_auroc": va["auroc"],
                    # So test.py can tell an in-distribution eval (must be
                    # restricted to heldout_idx) from a zero-shot one (evaluate
                    # the whole corpus), and recover the exact held-out rows.
                    "train_datasets": list(train_datasets),
                    "heldout_idx": heldout_idx,
                },
                run_dir / "best.pt",
            )
            logger.info("  new best (val AUROC %.4f) -> saved", best_auroc)
        else:
            stale += 1

        history.append(record)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))

        if stale >= cfg.train.patience:
            logger.info("early stopping: no val improvement for %d epochs", stale)
            break

    # ---- final evaluation with the BEST checkpoint ---------------------
    _section("final evaluation")
    logger.info("loading best checkpoint (epoch %d, val AUROC %.4f)", best_epoch, best_auroc)
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    results = {"best_epoch": best_epoch, "val_auroc": best_auroc}

    va, _ = run_epoch(model, val_loader, criterion, device, desc="final val")
    results["val"] = va
    logger.info("FINAL val  | %s", format_metrics(va))

    if test_loader is not None:
        te, te_origins = run_epoch(model, test_loader, criterion, device, desc="final test")
        results["test"] = te
        results["test_per_origin"] = te_origins
        logger.info("FINAL test | %s   <- ZERO-SHOT on held-out %s",
                    format_metrics(te), test_dataset)

    gates = report_gates(model, val_loader, device, cfg)
    if gates:
        results["view_gates"] = gates

    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    logger.info("results written to %s", run_dir / "results.json")
    return results


@torch.no_grad()
def report_gates(model, loader, device, cfg) -> dict | None:
    """Average the fusion gates over the val set: which view does the model use?

    Only meaningful for the gated fusion; returns None otherwise. This is the
    number that turns "we use Q, K and V" into an actual finding.
    """
    if cfg.model.fusion != "gated" or len(cfg.extract.views) < 2:
        return None

    model.eval()
    totals = torch.zeros(len(cfg.extract.views))
    n = 0
    for batch in loader:
        # view_gates only looks at stream 1. collate() yields either
        # (images, labels, mask, origins) or, with stream2 enabled,
        # (images, images2, labels, mask, mask2, origins) -- mask's position
        # differs, so pick it by the tuple's actual length.
        images = batch[0]
        mask = batch[2] if len(batch) == 4 else batch[3]
        g = model.view_gates(images.to(device), mask.to(device))
        if g is None:
            return None
        totals += g.cpu()
        n += 1

    if n == 0:
        return None
    mean = (totals / n).tolist()
    gates = dict(zip(cfg.extract.views, mean))
    logger.info("view gates (mean softmax weight): %s",
                ", ".join(f"{v}={w:.3f}" for v, w in gates.items()))
    return gates


def default_run_name(cfg: Config, train_datasets: list[str], test_dataset: str | None) -> str:
    tag = "+".join(train_datasets)
    if test_dataset and test_dataset not in train_datasets:
        tag = f"{tag}-to-{test_dataset}"
    views = "".join(cfg.extract.views)
    return f"{cfg.llm.alias}_{tag}_{views}_{cfg.model.fusion}_{cfg.model.backbone}"
