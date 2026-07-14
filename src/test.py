"""Evaluate a saved checkpoint on a dataset."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import collate
from src.models.classifier import build_model
from src.train import load_source, report_gates, run_epoch
from src.utils.logger import get_logger
from src.utils.metrics import format_metrics
from src.utils.seed import pick_device, seed_everything

logger = get_logger(__name__)


def test(cfg: Config, checkpoint: str | Path, dataset_name: str | None = None) -> dict:
    seed_everything(cfg.train.seed)
    device = pick_device()

    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    views = ckpt["views"]
    stats = ckpt["stats"]

    if views != cfg.extract.views:
        logger.warning(
            "checkpoint was trained on views %s but config asks for %s; "
            "using the CHECKPOINT's views (the model's shape depends on them)",
            views, cfg.extract.views,
        )
        cfg.extract.views = views

    name = dataset_name or cfg.dataset.name
    source = load_source(cfg, name, cfg.llm.alias)
    # Normalise with the TRAINING statistics baked into the checkpoint, never
    # with statistics recomputed on the test set.
    source.stats = stats
    logger.info("evaluating on %s (n=%d)", source.origin, len(source))

    loader = DataLoader(
        source,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=cfg.train.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(cfg, n_views=len(views)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    criterion = torch.nn.BCEWithLogitsLoss()
    metrics, per_origin = run_epoch(model, loader, criterion, device, desc="test")

    logger.info("TEST %s | %s", source.origin, format_metrics(metrics))

    out = {"dataset": name, "checkpoint": str(ckpt_path), "metrics": metrics}
    if per_origin:
        out["per_origin"] = per_origin

    gates = report_gates(model, loader, device, cfg)
    if gates:
        out["view_gates"] = gates

    dest = ckpt_path.parent / f"test_{name}.json"
    dest.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s", dest)
    return out
