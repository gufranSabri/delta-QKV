#!/usr/bin/env python3
"""Scan runs/*/ and write runs/results.csv: one row per run, its test metrics.

A run's numbers come from its test_<dataset>.json (the held-out test set --
see src/test.py). A run with no test_*.json yet (still training, or `test`
was never run) still gets a row, with every metric cell left empty rather
than being skipped -- so the table always reflects every run directory that
exists, not just the finished ones.

    python scripts/collect_results.py [--runs-root runs] [--out runs/results.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: One row's worth of test.py's `metrics` dict, in table order.
METRIC_COLS = [
    "auroc", "accuracy", "precision", "recall", "f1", "pr_auc", "tpr@5fpr", "n",
]


def _find_test_json(run_dir: Path) -> Path | None:
    """A run may have zero or several test_*.json (one per --dataset tested
    against). Several is rare but possible if `test` was run more than once
    with different --dataset values; take the most recently written one."""
    candidates = sorted(run_dir.glob("test_*.json"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _read_llm_alias(run_dir: Path) -> str:
    config_json = run_dir / "config.json"
    if not config_json.exists():
        return ""
    data = json.loads(config_json.read_text())
    return data.get("llm", {}).get("alias", "")


def collect_row(run_dir: Path) -> dict:
    row = {
        "model": run_dir.name,
        "llm": _read_llm_alias(run_dir),
        "dataset": "",
        **{c: "" for c in METRIC_COLS},
    }

    test_json = _find_test_json(run_dir)
    if test_json is None:
        return row

    data = json.loads(test_json.read_text())
    row["dataset"] = data.get("dataset", "")
    metrics = data.get("metrics", {})
    for col in METRIC_COLS:
        if col in metrics:
            row[col] = metrics[col]
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", default=str(REPO_ROOT / "runs"))
    ap.add_argument("--out", default=None, help="defaults to <runs-root>/results.csv")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    out_path = Path(args.out) if args.out else runs_root / "results.csv"

    run_dirs = sorted(d for d in runs_root.iterdir() if d.is_dir())
    rows = [collect_row(d) for d in run_dirs]

    fieldnames = ["model", "llm", "dataset", *METRIC_COLS]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {out_path} ({len(rows)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
