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
    test_fraction: float = 0.0,
    seed: int = 0,
    cache: Path | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """Stratified split of indices into (train, val, test).

    Stratified because hallucination rates are often far from 50/50; a random
    split could hand the validation set a wildly different positive rate and make
    val AUROC incomparable to train.

    `test_fraction=0` yields an empty test list, which is the right thing for the
    datasets that ship a *separate* held-out corpus (see datasets.SPLIT_SOURCES):
    there, the test set is a different corpus entirely, not a slice of this one.
    It is non-zero only for datasets with a single upstream split (TruthfulQA),
    where the only honest test set is one we carve out ourselves.

    Persisted to `cache` so that repeated runs (and the model-selection decisions
    they drive) all see the same split.
    """
    key = {"n": len(labels), "seed": seed,
           "val_fraction": val_fraction, "test_fraction": test_fraction}

    if cache is not None and cache.exists():
        data = json.loads(cache.read_text())
        # Compare the full parameterisation, not just (n, seed): a run that
        # changes only test_fraction must not silently reuse a two-way split.
        if all(data.get(k) == v for k, v in key.items()):
            logger.info("reusing cached split from %s", cache)
            return data["train"], data["val"], data.get("test", [])
        logger.warning("cached split at %s is stale; rebuilding", cache)

    idx = np.arange(len(labels))
    y = np.asarray(labels)

    def _stratify(subset_y):
        if len(np.unique(subset_y)) > 1:
            return subset_y
        logger.warning("only one class present; falling back to an unstratified split")
        return None

    test: list[int] = []
    rest = idx
    if test_fraction > 0:
        rest, test_arr = train_test_split(
            idx, test_size=test_fraction, random_state=seed, stratify=_stratify(y)
        )
        test = sorted(test_arr.tolist())

    # val_fraction is expressed w.r.t. the FULL dataset, so rescale it against
    # what's left after the test slice -- otherwise carving out a test set would
    # silently shrink val too.
    rel_val = val_fraction / (1.0 - test_fraction) if test_fraction > 0 else val_fraction
    train_arr, val_arr = train_test_split(
        rest, test_size=rel_val, random_state=seed, stratify=_stratify(y[rest])
    )
    train, val = sorted(train_arr.tolist()), sorted(val_arr.tolist())

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(
            json.dumps({**key, "train": train, "val": val, "test": test})
        )
        logger.info("saved split to %s", cache)

    return train, val, test
