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
from tqdm import tqdm

from src.config import Config
from src.data.dataset import (
    ConcatQKVDataset,
    QKVImageDataset,
    _set_stats,
    collate,
    compute_stats,
)
from src.models.classifier import build_model
from src.utils.logger import get_logger, setup_logging
from src.utils.metrics import compute_metrics, format_metrics
from src.utils.seed import pick_device, seed_everything
from src.utils.splits import make_split

logger = get_logger(__name__)


def load_source(cfg: Config, dataset_name: str, llm_alias: str, **kw) -> QKVImageDataset:
    root = Path(cfg.data_root) / dataset_name / llm_alias
    return QKVImageDataset(
        root,
        views=cfg.extract.views,
        max_tokens=cfg.extract.max_tokens,
        origin=f"{llm_alias}/{dataset_name}",
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


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None, desc=""):
    """One pass. Trains if `optimizer` is given, else evaluates."""
    training = optimizer is not None
    model.train(training)

    total_loss, n_batches = 0.0, 0
    all_y, all_p, all_origins = [], [], []

    with torch.set_grad_enabled(training):
        for images, labels, mask, origins in tqdm(loader, desc=desc, leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            logits = model(images, mask)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # Exploding gradients through a BiLSTM are a classic failure here.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

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
    setup_logging(log_file=run_dir / "train.log")

    logger.info("run dir: %s", run_dir)
    logger.info("device: %s", device)
    logger.info("train sources: %s | test source: %s", train_datasets, test_dataset or "(val split)")
    logger.info("views: %s | fusion: %s | backbone: %s (shared=%s)",
                cfg.extract.views, cfg.model.fusion, cfg.model.backbone, cfg.model.share_backbone)

    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    # ---- data ---------------------------------------------------------
    sources, test_source = build_datasets(cfg, train_datasets, test_dataset, cfg.llm.alias)
    full = ConcatQKVDataset(sources) if len(sources) > 1 else sources[0]

    train_idx, val_idx = make_split(
        full.labels,
        val_fraction=cfg.train.val_fraction,
        seed=cfg.train.seed,
        cache=run_dir / "split.json",
    )
    logger.info("train %d | val %d", len(train_idx), len(val_idx))

    # Normalisation statistics come from the TRAIN split only -- computing them
    # over val/test would leak those distributions into the input scaling.
    stats_path = run_dir / "stats.json"
    stats = compute_stats(full, train_idx)
    stats_path.write_text(json.dumps(stats, indent=2))
    for view, s in stats.items():
        logger.info("norm %s: mean=%s std=%s",
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
    train_loader = DataLoader(Subset(full, train_idx), shuffle=True, **loader_kw)
    val_loader = DataLoader(Subset(full, val_idx), shuffle=False, **loader_kw)
    test_loader = (
        DataLoader(test_source, shuffle=False, **loader_kw) if test_source else None
    )

    # ---- model --------------------------------------------------------
    model = build_model(cfg, n_views=len(cfg.extract.views)).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("model: %s parameters", f"{n_params:,}")

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
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    total_steps = max(1, len(train_loader) * cfg.train.epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.train.lr, total_steps=total_steps, pct_start=0.1
    )

    # ---- loop ---------------------------------------------------------
    best_auroc, best_epoch, stale = -1.0, -1, 0
    history = []

    for epoch in range(1, cfg.train.epochs + 1):
        t0 = time.time()
        tr, _ = run_epoch(model, train_loader, criterion, device,
                          optimizer, scheduler, desc=f"epoch {epoch} train")
        va, va_origins = run_epoch(model, val_loader, criterion, device,
                                   desc=f"epoch {epoch} val")

        logger.info("epoch %2d | train loss %.4f AUROC %.4f | val %s | %.0fs",
                    epoch, tr["loss"], tr["auroc"], format_metrics(va), time.time() - t0)

        record = {"epoch": epoch, "train": tr, "val": va}

        # Model selection on validation AUROC, never on test.
        if va["auroc"] > best_auroc:
            best_auroc, best_epoch, stale = va["auroc"], epoch, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg.to_dict(),
                    "stats": stats,
                    "views": cfg.extract.views,
                    "epoch": epoch,
                    "val_auroc": va["auroc"],
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
    for images, _, mask, _ in loader:
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
