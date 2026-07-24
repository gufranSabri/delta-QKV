"""Evaluate a saved checkpoint on a dataset."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.config import Config
from src.data.dataset import collate, n_images
from src.models.classifier import build_model
from src.train import load_source, report_gates, run_epoch
from src.utils.logger import get_logger
from src.utils.metrics import format_metrics
from src.utils.seed import pick_device, seed_everything

logger = get_logger(__name__)

#: Every run appends here, regardless of which path the repo was invoked through.
#: Hardcoded on purpose: the repo is reachable as both /project/6101771/... and
#: /home/ahmedubc/projects/... (the latter is a symlink to the former), and a
#: REPO_ROOT-derived path would follow whichever one the job happened to use.
#: Pinning it keeps every result in ONE file.
RESULTS_CSV = Path(
    "/home/ahmedubc/projects/aip-lsigal/ahmedubc/delta-QKV/docs/results.csv"
)


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

    # Same story for the channel regrouping: it decides how many CNN streams the
    # model has and how many channels each takes, so the checkpoint's value wins.
    # Checkpoints written before this option existed carry no key -> "default".
    ckpt_channels = ckpt.get("channels", "default")
    if ckpt_channels != cfg.model.channels:
        logger.warning(
            "checkpoint was trained with channels=%r but config asks for %r; "
            "using the CHECKPOINT's (the model's shape depends on it)",
            ckpt_channels, cfg.model.channels,
        )
        cfg.model.channels = ckpt_channels

    # Same story for `include`: it drops images by index, so it changes the
    # number of CNN streams just like `channels` does. Checkpoints written
    # before this option existed carry no key -> keep every image.
    ckpt_include = ckpt.get("include", None)
    if ckpt_include != cfg.model.include:
        logger.warning(
            "checkpoint was trained with include=%r but config asks for %r; "
            "using the CHECKPOINT's (the model's shape depends on it)",
            ckpt_include, cfg.model.include,
        )
        cfg.model.include = ckpt_include

    name = dataset_name or cfg.dataset.name
    name, eval_set = _resolve_eval_target(name, ckpt)

    source = load_source(cfg, name, cfg.llm.alias)
    # Normalise with the TRAINING statistics baked into the checkpoint, never
    # with statistics recomputed on the test set.
    source.stats = stats

    # Restrict to the held-out rows when the model was trained on this corpus.
    # Skipping this is what made every same-dataset score a train-set score.
    eval_data = source
    if eval_set is not None:
        if not eval_set:
            raise ValueError(
                f"checkpoint was trained on {name!r} but carries no held-out indices. "
                "It predates the held-out split; retrain before testing on it."
            )
        eval_data = Subset(source, eval_set)

    logger.info("evaluating on %s (n=%d of %d)", source.origin, len(eval_data), len(source))

    loader = DataLoader(
        eval_data,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=cfg.train.num_workers,
        pin_memory=device.type == "cuda",
    )

    # Number of CNN streams = number of images after regrouping and `include`
    # filtering, not len(views).
    n_streams = n_images(cfg.model.channels, len(views), cfg.model.include)
    model = build_model(cfg, n_views=n_streams).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    criterion = torch.nn.BCEWithLogitsLoss()
    metrics = run_epoch(model, loader, criterion, device, desc="test")

    logger.info("TEST %s | %s", source.origin, format_metrics(metrics))

    out = {"dataset": name, "checkpoint": str(ckpt_path), "metrics": metrics}

    gates = report_gates(model, loader, device, cfg)
    if gates:
        out["view_gates"] = gates

    dest = ckpt_path.parent / f"test_{name}.json"
    dest.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s", dest)

    # ── Save to docs/results.csv ───────────────────────────────────────────
    # Disabled during ablation runs -- results live in each run's test_*.json;
    # re-enable when ablation.sh's sweep is done and the CSV should track again.
    # _save_to_results_csv(cfg, name, metrics, ckpt_path)

    return out


def _resolve_eval_target(name: str, ckpt: dict) -> tuple[str, list[int] | None]:
    """Pick what to actually evaluate on. Returns (corpus_name, row_subset).

    The rule, in order:

    1. We trained on it     -> evaluate the stratified slice held out at train
       time. (Every dataset mirrors HalluShift, which never trains/tests on
       separate corpora.)
    2. We never trained on it -> zero-shot; evaluate it in full.
       (A different dataset than the checkpoint was trained on. Nothing was
       fit on this corpus, so every row is fair game.)

    A `row_subset` of None means "use the whole corpus".
    """
    trained_on = set(ckpt.get("train_datasets", []))

    if name not in trained_on:
        logger.info("%s was not in this checkpoint's training set: zero-shot eval", name)
        return name, None

    logger.info("%s has no separate test corpus; evaluating its held-out slice", name)
    return name, list(ckpt.get("heldout_idx") or [])


KEY_COLS = ["model", "llm", "metric"]
METRIC_NAMES = ["auroc", "accuracy", "precision", "recall", "f1"]
#: Which of the above metrics are written to the shared results.csv. Only AUROC:
#: it is the number the baseline papers report, and it keeps the table readable.
CSV_METRICS = ["auroc"]

DATASET_COLS = {
    "truthfulqa": "TruthfulQA",
    "triviaqa": "TriviaQA",
    "coqa": "CoQA",
}

# The three LLMs HalluShift reports on (hal_detection.py:22-24). All base models.
HALLUSHIFT_LLMS = {
    "llama2_7b": "meta-llama/Llama-2-7b-hf",
    "llama3.1_8b": "meta-llama/Llama-3.1-8B",
    "opt_6.7b": "facebook/opt-6.7b",
}


def _save_to_results_csv(cfg: Config, dataset_name: str, metrics: dict, checkpoint: Path) -> None:
    """Update docs/results.csv in place, keyed on (model, llm, metric).

    The LLM column is load-bearing: ACT-ViT reports one number per
    (LLM, dataset) cell, so without it our rows cannot line up against theirs.

    Rows are upserted -- an existing (model, llm, metric) row has its dataset
    cell overwritten; anything new is appended. Baseline rows are left alone.
    """
    results_csv = RESULTS_CSV
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    run_config: dict = {}
    config_path = checkpoint.parent / "config.json"
    if config_path.exists():
        run_config = json.loads(config_path.read_text())

    run_model = run_config.get("model", {})
    run_extract = run_config.get("extract", {})
    fusion = run_model.get("fusion", cfg.model.fusion)
    views = run_extract.get("views", cfg.extract.views)
    llm = run_config.get("llm", {}).get("alias", cfg.llm.alias)

    # The model name must encode every axis that changes what the run IS, or two
    # different runs collide onto one (model, llm, metric) row and overwrite each
    # other. source/extraction_type change what was extracted; channels changes
    # how it is regrouped for the CNNs. fusion/views round out the model.
    source = run_extract.get("source", cfg.extract.source)
    extraction_type = run_extract.get("extraction_type", cfg.extract.extraction_type)
    channels = run_model.get("channels", cfg.model.channels)
    model_name = (
        f"delta-QKV-{source}-{extraction_type}-{channels}-{fusion}-{''.join(views)}"
    )

    dataset_col = DATASET_COLS.get(dataset_name, dataset_name)

    rows: list[dict] = []
    fieldnames = list(KEY_COLS)
    if results_csv.exists():
        with open(results_csv, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or KEY_COLS)
            rows = list(reader)

    for col in (*KEY_COLS, dataset_col):
        if col not in fieldnames:
            fieldnames.append(col)

    # Only AUROC goes into the shared comparison table -- it is the headline
    # number both baseline papers report per (LLM, dataset) cell. The full metric
    # set (accuracy/precision/recall/f1) still lives in each run's test_*.json.
    for metric_name in CSV_METRICS:
        if metric_name not in metrics:
            continue
        match = next(
            (
                r for r in rows
                if r.get("model") == model_name
                and r.get("llm") == llm
                and r.get("metric") == metric_name
            ),
            None,
        )
        if match is None:
            match = {"model": model_name, "llm": llm, "metric": metric_name}
            rows.append(match)
        # Store as a percentage, matching how both papers report AUROC.
        match[dataset_col] = f"{100 * metrics[metric_name]:.2f}"

    for row in rows:
        for col in fieldnames:
            row.setdefault(col, "-")

    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(
        "updated %s: %s / %s on %s (AUROC %.2f)",
        results_csv, model_name, llm, dataset_col, 100 * metrics["auroc"],
    )
