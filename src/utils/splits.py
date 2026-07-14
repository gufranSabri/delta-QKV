"""Stratified train/val splits, persisted so runs are comparable."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger

logger = get_logger(__name__)


def make_split(
    labels: list[int],
    val_fraction: float = 0.2,
    seed: int = 0,
    cache: Path | None = None,
) -> tuple[list[int], list[int]]:
    """Stratified split of indices into (train, val).

    Stratified because hallucination rates are often far from 50/50; a random
    split could hand the validation set a wildly different positive rate and make
    val AUROC incomparable to train.

    Persisted to `cache` so that repeated runs (and the model-selection decisions
    they drive) all see the same split.
    """
    if cache is not None and cache.exists():
        data = json.loads(cache.read_text())
        if data.get("n") == len(labels) and data.get("seed") == seed:
            logger.info("reusing cached split from %s", cache)
            return data["train"], data["val"]
        logger.warning("cached split at %s is stale (n or seed changed); rebuilding", cache)

    idx = np.arange(len(labels))
    y = np.asarray(labels)

    stratify = y if len(np.unique(y)) > 1 else None
    if stratify is None:
        logger.warning("only one class present; falling back to an unstratified split")

    train, val = train_test_split(
        idx, test_size=val_fraction, random_state=seed, stratify=stratify
    )
    train, val = sorted(train.tolist()), sorted(val.tolist())

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(
            json.dumps({"n": len(labels), "seed": seed, "train": train, "val": val})
        )
        logger.info("saved split to %s", cache)

    return train, val
