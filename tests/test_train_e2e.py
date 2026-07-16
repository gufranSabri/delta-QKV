"""End-to-end training tests on a synthetic, learnable dataset.

The plan's step-7 gate: "overfit 50 examples to ~100% train AUROC. If it can't
overfit, the model or the mask is broken."

We plant a signal that is ONLY visible in one view, so these tests also verify
that the per-view CNN + fusion path actually carries information from each
individual view through to the prediction.
"""

import json

import numpy as np
import pytest
import torch

from src.config import Config
from src.train import train

L, C = 8, 8
VIEWS = ["Q", "K", "V"]


def make_learnable_source(root, n=60, signal_view=0, seed=0, n_rows=L, n_cols=C):
    """Synthetic dataset where the label IS recoverable from the images.

    Positive examples get a constant offset added to `signal_view`'s raw channel.
    A working pipeline must reach ~1.0 AUROC; a broken one (dead view, padding
    leak, label misalignment) will sit near 0.5.
    """
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)

    (root / "geometry.json").write_text(json.dumps({
        "views": VIEWS, "n_rows": n_rows, "n_cols": n_cols,
        "geometry": {"n_layers": n_rows},
    }))

    records = []
    for i in range(n):
        label = i % 2
        t = int(rng.integers(3, 9))       # variable length -> exercises padding
        arr = rng.normal(0, 1, size=(t, len(VIEWS), n_rows, n_cols, 3))
        if label == 1:
            arr[:, signal_view, :, :, 0] += 3.0     # the plant

        d = root / f"{i:05d}"
        d.mkdir(exist_ok=True)
        np.save(d / "tokens.npy", arr.astype(np.float16))
        (d / "meta.txt").write_text(
            f"prompt: p{i}\nresponse: r{i}\ngold: g{i}\nscore: 0\nlabel: {label}\n"
        )
        records.append({"idx": i, "dir": d.name, "n_tokens": t,
                        "score": 0.0, "label": label})

    with open(root / "manifest.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return root


def base_cfg(tmp_path, **model_kw) -> Config:
    cfg = Config()
    cfg.data_root = str(tmp_path / "data")
    cfg.runs_root = str(tmp_path / "runs")
    cfg.llm.alias = "fake_llm"
    cfg.dataset.name = "fakeds"
    cfg.extract.views = list(VIEWS)
    cfg.extract.max_tokens = 16

    cfg.model.embed_dim = 32
    cfg.model.fused_dim = 32
    cfg.model.lstm_hidden = 16
    cfg.model.conv1d_layers = 1
    cfg.model.dropout = 0.0
    for k, v in model_kw.items():
        setattr(cfg.model, k, v)

    cfg.train.batch_size = 8
    cfg.train.epochs = 12
    cfg.train.patience = 12
    cfg.train.lr = 3e-3
    cfg.train.num_workers = 0
    cfg.train.val_fraction = 0.25
    cfg.train.seed = 0
    cfg.validate()
    return cfg


@pytest.fixture
def cfg_and_data(tmp_path):
    cfg = base_cfg(tmp_path)
    make_learnable_source(tmp_path / "data" / "fakeds" / "fake_llm", n=64)
    return cfg


def test_pipeline_learns_a_planted_signal(cfg_and_data):
    """THE gate: if this does not reach a high AUROC, something is broken."""
    results = train(cfg_and_data, train_datasets=["fakeds"])
    assert results["val_auroc"] > 0.9, (
        f"val AUROC only {results['val_auroc']:.3f} on a trivially separable "
        "dataset -- the model, the mask, or the label alignment is broken"
    )


@pytest.mark.parametrize("signal_view,view_name", [(0, "Q"), (1, "K"), (2, "V")])
def test_signal_is_learnable_from_any_single_view(tmp_path, signal_view, view_name):
    """Plant the signal in ONE view and confirm the pipeline finds it.

    This proves every view's CNN is wired through to the output. If, say, only
    view 0 ever worked, the Q-planted case would pass and the other two would sit
    at chance -- exactly the kind of silent bug the untied-backbone design could
    hide.
    """
    cfg = base_cfg(tmp_path)
    make_learnable_source(
        tmp_path / "data" / "fakeds" / "fake_llm", n=64, signal_view=signal_view
    )
    results = train(cfg, train_datasets=["fakeds"], run_name=f"sig_{view_name}")
    assert results["val_auroc"] > 0.9, (
        f"signal planted in {view_name} was not learned "
        f"(AUROC {results['val_auroc']:.3f}) -- that view's path may be dead"
    )


def test_single_view_config_trains(tmp_path):
    """views: [V] must work end to end -- the ablation the paper needs."""
    cfg = base_cfg(tmp_path)
    cfg.extract.views = ["V"]
    make_learnable_source(
        tmp_path / "data" / "fakeds" / "fake_llm", n=48, signal_view=2  # V
    )
    results = train(cfg, train_datasets=["fakeds"], run_name="vonly")
    assert results["val_auroc"] > 0.9


@pytest.mark.parametrize("fusion", ["gated", "concat_mlp", "bilinear", "cross_attn"])
def test_every_fusion_trains(tmp_path, fusion):
    cfg = base_cfg(tmp_path, fusion=fusion)
    make_learnable_source(tmp_path / "data" / "fakeds" / "fake_llm", n=48)
    results = train(cfg, train_datasets=["fakeds"], run_name=f"fus_{fusion}")
    assert results["val_auroc"] > 0.8, f"{fusion} failed to learn"


def test_gated_fusion_finds_the_view_holding_the_signal(tmp_path):
    """The interpretability claim, tested.

    Plant the signal ONLY in V. The gated fusion's learned weights should lean
    toward V. If they do not, the "gates tell you which view matters" story does
    not hold and we should not make it.
    """
    cfg = base_cfg(tmp_path, fusion="gated")
    cfg.train.epochs = 20
    make_learnable_source(
        tmp_path / "data" / "fakeds" / "fake_llm", n=96, signal_view=2  # V
    )
    results = train(cfg, train_datasets=["fakeds"], run_name="gates")

    gates = results.get("view_gates")
    assert gates is not None, "gated fusion should report view gates"
    assert set(gates) == {"Q", "K", "V"}
    assert abs(sum(gates.values()) - 1.0) < 1e-3, "gates should sum to 1"

    # The informative view should not be the LEAST used one.
    assert gates["V"] > min(gates["Q"], gates["K"]), (
        f"signal was planted in V but gates are {gates}; the gates are not "
        "tracking which view carries information"
    )


def test_leave_one_dataset_out_runs_and_reports_zero_shot(tmp_path):
    """Train on two sources, evaluate zero-shot on a third never seen."""
    cfg = base_cfg(tmp_path)
    for name in ("ds_a", "ds_b", "ds_held"):
        make_learnable_source(
            tmp_path / "data" / name / "fake_llm", n=48, seed=hash(name) % 1000
        )

    results = train(
        cfg,
        train_datasets=["ds_a", "ds_b"],
        test_dataset="ds_held",
        run_name="lodo",
    )
    assert "test" in results, "held-out evaluation should have run"
    assert results["test"]["n"] == 48
    # The planted signal is identical across sources, so it should transfer.
    assert results["test"]["auroc"] > 0.8


def test_run_artifacts_are_written(cfg_and_data, tmp_path):
    train(cfg_and_data, train_datasets=["fakeds"], run_name="artifacts")
    run_dir = tmp_path / "runs" / "artifacts"

    for name in ("config.json", "split.json", "stats.json",
                 "history.json", "results.json", "best.pt"):
        assert (run_dir / name).exists(), f"missing artifact: {name}"

    # The checkpoint must carry everything needed to reproduce inference:
    # weights, the views it was built for, and the TRAIN normalisation stats.
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    assert set(ckpt) >= {"model", "config", "stats", "views", "val_auroc"}
    assert ckpt["views"] == VIEWS


def test_test_command_reproduces_training_metrics(cfg_and_data, tmp_path):
    """A checkpoint evaluated via src.test must agree with what training saw."""
    from src.test import test as run_test

    train(cfg_and_data, train_datasets=["fakeds"], run_name="reload")
    ckpt = tmp_path / "runs" / "reload" / "best.pt"

    out = run_test(cfg_and_data, ckpt, dataset_name="fakeds")
    # Evaluated over the FULL set (train+val), so it should be at least as good
    # as val alone on a separable problem.
    assert out["metrics"]["auroc"] > 0.9
    assert out["metrics"]["n"] == 64


def test_stats_are_computed_from_train_split_only(cfg_and_data, tmp_path):
    """Guard against val leaking into the normalisation constants."""
    from src.data.dataset import QKVImageDataset, compute_stats
    from src.utils.splits import make_split

    root = tmp_path / "data" / "fakeds" / "fake_llm"
    ds = QKVImageDataset(root, views=VIEWS)
    train_idx, val_idx, _ = make_split(ds.labels, val_fraction=0.25, seed=0)

    from_train = compute_stats(ds, train_idx)
    from_all = compute_stats(ds, list(range(len(ds))))

    # They must differ -- if they were identical, compute_stats would be ignoring
    # its `indices` argument, which is precisely the leak we are guarding against.
    assert from_train["Q"]["mean"] != from_all["Q"]["mean"]
