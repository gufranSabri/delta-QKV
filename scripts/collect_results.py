#!/usr/bin/env python3
"""Scan runs/*/ and write result tables.

A run's numbers come from its test_<dataset>.json (the held-out test set --
see src/test.py). A run with no test_*.json yet (still training, or `test`
was never run) still gets a row in runs/results.csv, with every metric cell
left empty rather than being skipped -- so that table always reflects every
run directory that exists, not just the finished ones.

Three outputs:
    runs/results.csv                 one row per run (unchanged behaviour)
    runs/results_<dataset>.csv       one such file per dataset, all its runs
                                      combined (rows with no test_*.json yet
                                      are excluded -- there is no AUROC to
                                      rank or compare there)
    runs/results_best.csv            one row per dataset: its single
                                      highest-AUROC run

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

FIELDNAMES = ["model", "llm", "dataset", *METRIC_COLS]


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


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", default=str(REPO_ROOT / "runs"))
    ap.add_argument("--out", default=None, help="defaults to <runs-root>/results.csv")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    out_path = Path(args.out) if args.out else runs_root / "results.csv"

    run_dirs = sorted(d for d in runs_root.iterdir() if d.is_dir())
    rows = [collect_row(d) for d in run_dirs]

    _write_csv(out_path, rows)
    print(f"wrote {out_path} ({len(rows)} runs)")

    # Only rows with a real test result (a dataset name and a numeric AUROC)
    # are useful for the per-dataset / best-of tables -- an unfinished run has
    # nothing to combine or rank.
    scored = [r for r in rows if r["dataset"] and isinstance(r["auroc"], float)]

    by_dataset: dict[str, list[dict]] = {}
    for row in scored:
        by_dataset.setdefault(row["dataset"], []).append(row)

    for dataset, dataset_rows in sorted(by_dataset.items()):
        dataset_rows = sorted(dataset_rows, key=lambda r: r["auroc"], reverse=True)
        dataset_path = out_path.parent / f"results_{dataset}.csv"
        _write_csv(dataset_path, dataset_rows)
        print(f"wrote {dataset_path} ({len(dataset_rows)} runs)")

    best_rows = [
        max(dataset_rows, key=lambda r: r["auroc"])
        for _, dataset_rows in sorted(by_dataset.items())
    ]
    best_path = out_path.parent / "results_best.csv"
    _write_csv(best_path, best_rows)
    print(f"wrote {best_path} ({len(best_rows)} datasets)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
