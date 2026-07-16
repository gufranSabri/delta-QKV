"""Dataset, collation, normalisation and split tests.

Builds a synthetic on-disk dataset so nothing here needs a real LLM.
"""

import json

import numpy as np
import pytest
import torch

from src.data.dataset import (
    ConcatQKVDataset,
    QKVImageDataset,
    collate,
    compute_stats,
    normalize,
)
from src.utils.splits import make_split

L, C = 8, 8
VIEWS = ["Q", "K", "V"]


def make_fake_source(root, n=10, views=VIEWS, n_rows=L, n_cols=C, seed=0, token_counts=None):
    """Write a synthetic extracted dataset to `root`."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)

    (root / "geometry.json").write_text(json.dumps({
        "views": views, "n_rows": n_rows, "n_cols": n_cols,
        "geometry": {"n_layers": n_rows},
    }))

    records = []
    for i in range(n):
        t = token_counts[i] if token_counts else int(rng.integers(3, 8))
        d = root / f"{i:05d}"
        d.mkdir(exist_ok=True)
        arr = rng.normal(size=(t, len(views), n_rows, n_cols, 3)).astype(np.float16)
        np.save(d / "tokens.npy", arr)
        label = i % 2
        (d / "meta.txt").write_text(
            f"prompt: p{i}\nresponse: r{i}\ngold: g{i}\nscore: 0.5\nlabel: {label}\n"
        )
        records.append({"idx": i, "dir": d.name, "n_tokens": t,
                        "score": 0.5, "label": label})

    with open(root / "manifest.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return root


@pytest.fixture
def source(tmp_path):
    return make_fake_source(tmp_path / "triviaqa" / "llama3_8b")


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------

def test_dataset_loads_and_permutes_channels_into_conv_position(source):
    ds = QKVImageDataset(source)
    images, label, origin = ds[0]
    t = ds.records[0]["n_tokens"]
    # (T, V, 3, L, C) -- channel axis moved so the CNN needs no permute.
    assert images.shape == (t, 3, 3, L, C)
    assert label in (0.0, 1.0)
    assert origin


def test_dataset_view_subsetting_slices_the_view_axis(source):
    ds = QKVImageDataset(source, views=["V"])
    images, _, _ = ds[0]
    assert images.shape[1] == 1, "should have exactly one view"
    assert ds.views == ["V"]

    # And it must pick the RIGHT view, not just the first one.
    full = QKVImageDataset(source)
    img_full, _, _ = full[0]
    torch.testing.assert_close(images[:, 0], img_full[:, 2])  # V is index 2


def test_dataset_rejects_a_view_that_was_never_extracted(source):
    ds_path = source
    geom = json.loads((ds_path / "geometry.json").read_text())
    geom["views"] = ["Q"]                       # only Q was extracted
    (ds_path / "geometry.json").write_text(json.dumps(geom))

    with pytest.raises(ValueError, match=r"were not extracted"):
        QKVImageDataset(ds_path, views=["Q", "V"])


def test_dataset_rejects_unlabeled_examples(tmp_path):
    root = make_fake_source(tmp_path / "d" / "m", n=3)
    # Corrupt one label to the "not yet labeled" sentinel.
    lines = (root / "manifest.jsonl").read_text().splitlines()
    rec = json.loads(lines[0]); rec["label"] = -1
    lines[0] = json.dumps(rec)
    (root / "manifest.jsonl").write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError, match="no valid label"):
        QKVImageDataset(root)


def test_dataset_truncates_to_max_tokens(source):
    ds = QKVImageDataset(source, max_tokens=2)
    images, _, _ = ds[0]
    assert images.shape[0] == 2


def test_missing_manifest_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run `python main.py extract`"):
        QKVImageDataset(tmp_path)


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------

def test_normalize_is_per_view_per_channel(source):
    """Two views with wildly different scales must BOTH end up standardised.

    A single global statistic would leave the large-scale view dominating, which
    is exactly the failure this design avoids.
    """
    stats = {
        "Q": {"mean": [10.0, 0.0, 0.0], "std": [2.0, 1.0, 1.0]},
        "K": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
        "V": {"mean": [-5.0, 0.0, 0.0], "std": [4.0, 1.0, 1.0]},
    }
    images = torch.zeros(2, 3, L, C, 3)
    images[:, 0, :, :, 0] = 10.0     # Q raw == its mean -> should become 0
    images[:, 2, :, :, 0] = -5.0     # V raw == its mean -> should become 0

    out = normalize(images, stats, VIEWS)
    assert torch.allclose(out[:, 0, :, :, 0], torch.zeros(1))
    assert torch.allclose(out[:, 2, :, :, 0], torch.zeros(1))


def test_compute_stats_uses_only_the_given_indices(source):
    """Stats must come from the TRAIN split alone -- otherwise the val/test
    distribution leaks into the model's input scaling."""
    ds = QKVImageDataset(source)
    stats = compute_stats(ds, indices=[0, 1, 2])

    assert set(stats) == set(VIEWS)
    for v in VIEWS:
        assert len(stats[v]["mean"]) == 3
        assert len(stats[v]["std"]) == 3
        assert all(s > 0 for s in stats[v]["std"])


def test_compute_stats_reads_raw_values_even_if_stats_already_attached(source):
    """compute_stats must not double-normalise if the dataset already has stats."""
    ds = QKVImageDataset(source)
    raw = compute_stats(ds, indices=list(range(10)))

    ds.stats = raw                       # pretend normalisation is already on
    again = compute_stats(ds, indices=list(range(10)))

    # If it had read normalised data, the means would collapse toward 0.
    for v in VIEWS:
        np.testing.assert_allclose(raw[v]["mean"], again[v]["mean"], rtol=1e-5)


def test_normalized_data_is_roughly_standard(source):
    ds = QKVImageDataset(source)
    stats = compute_stats(ds, indices=list(range(10)))
    ds.stats = stats

    all_imgs = torch.cat([ds[i][0] for i in range(10)], dim=0)  # (sumT, V, 3, L, C)
    # Channel axis is dim 2 after the permute.
    for v in range(3):
        for c in range(3):
            chunk = all_imgs[:, v, c]
            assert abs(chunk.mean().item()) < 0.2
            assert 0.7 < chunk.std().item() < 1.4


# --------------------------------------------------------------------------
# collation
# --------------------------------------------------------------------------

def test_collate_pads_and_masks_correctly(tmp_path):
    root = make_fake_source(tmp_path / "d" / "m", n=3, token_counts=[2, 5, 3])
    ds = QKVImageDataset(root)

    images, labels, mask, origins = collate([ds[0], ds[1], ds[2]])

    assert images.shape == (3, 5, 3, 3, L, C)   # padded to T_max = 5
    assert labels.shape == (3,)
    assert mask.shape == (3, 5)
    assert mask[0].tolist() == [True, True, False, False, False]
    assert mask[1].tolist() == [True] * 5
    assert mask[2].tolist() == [True, True, True, False, False]
    assert len(origins) == 3


def test_collate_zero_fills_padding(tmp_path):
    root = make_fake_source(tmp_path / "d" / "m", n=2, token_counts=[1, 4])
    ds = QKVImageDataset(root)
    images, _, mask, _ = collate([ds[0], ds[1]])
    # Everything outside the mask must be exactly zero.
    assert torch.all(images[0, 1:] == 0)
    assert torch.all(images[~mask] == 0)


def test_collate_preserves_real_content(tmp_path):
    root = make_fake_source(tmp_path / "d" / "m", n=2, token_counts=[2, 4])
    ds = QKVImageDataset(root)
    a, _, _ = ds[0]
    images, _, _, _ = collate([ds[0], ds[1]])
    torch.testing.assert_close(images[0, :2], a)


# --------------------------------------------------------------------------
# concat (leave-one-dataset-out)
# --------------------------------------------------------------------------

def test_concat_indexes_across_sources(tmp_path):
    a = QKVImageDataset(make_fake_source(tmp_path / "ds1" / "m", n=4, seed=1))
    b = QKVImageDataset(make_fake_source(tmp_path / "ds2" / "m", n=6, seed=2))
    cat = ConcatQKVDataset([a, b])

    assert len(cat) == 10
    assert len(cat.labels) == 10

    # Boundary indices must land in the right source.
    assert cat[0][2] == a.origin
    assert cat[3][2] == a.origin
    assert cat[4][2] == b.origin      # first item of the second source
    assert cat[9][2] == b.origin


def test_concat_rejects_mismatched_image_sizes(tmp_path):
    """Cross-LLM training with different layer counts must fail loudly, not
    produce a CNN that silently cannot consume one of its inputs."""
    a = QKVImageDataset(make_fake_source(tmp_path / "ds1" / "llama", n=3, n_rows=32))
    b = QKVImageDataset(make_fake_source(tmp_path / "ds2" / "qwen", n=3, n_rows=28))
    with pytest.raises(ValueError, match="different image sizes"):
        ConcatQKVDataset([a, b])


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------

def test_split_is_stratified_and_disjoint():
    labels = [0] * 80 + [1] * 20        # 20% positive
    train, val, test = make_split(labels, val_fraction=0.2, seed=0)

    assert len(train) + len(val) == 100
    assert test == [], "no test slice was requested"
    assert not set(train) & set(val), "train and val overlap"

    val_rate = sum(labels[i] for i in val) / len(val)
    assert abs(val_rate - 0.2) < 0.06, "stratification did not preserve class balance"


def test_three_way_split_is_disjoint_and_stratified():
    """The test slice must never intersect train -- that was the original bug:
    same-dataset 'test' metrics were being computed over the training rows."""
    labels = [0] * 80 + [1] * 20
    train, val, test = make_split(labels, val_fraction=0.2, test_fraction=0.2, seed=42)

    assert len(train) + len(val) + len(test) == 100
    assert not set(train) & set(test), "TRAIN/TEST OVERLAP -- test metrics would be inflated"
    assert not set(val) & set(test), "val and test overlap"
    assert not set(train) & set(val), "train and val overlap"

    # val_fraction is w.r.t. the whole dataset, so carving out a test slice must
    # not silently shrink val as a side effect.
    assert len(val) == 20 and len(test) == 20

    test_rate = sum(labels[i] for i in test) / len(test)
    assert abs(test_rate - 0.2) < 0.06, "test slice lost the class balance"


def test_split_is_cached_and_reused(tmp_path):
    labels = [0, 1] * 25
    cache = tmp_path / "split.json"

    t1, v1, s1 = make_split(labels, seed=0, cache=cache)
    assert cache.exists()
    t2, v2, s2 = make_split(labels, seed=0, cache=cache)
    assert t1 == t2 and v1 == v2 and s1 == s2


def test_split_cache_is_invalidated_when_data_changes(tmp_path):
    cache = tmp_path / "split.json"
    make_split([0, 1] * 25, seed=0, cache=cache)
    # A different dataset size must NOT silently reuse the stale split.
    t, v, _ = make_split([0, 1] * 50, seed=0, cache=cache)
    assert len(t) + len(v) == 100


def test_split_cache_is_invalidated_when_test_fraction_changes(tmp_path):
    """A cached two-way split must not be reused once a test slice is asked for,
    or the run would train on rows it later reports as held-out."""
    labels = [0, 1] * 50
    cache = tmp_path / "split.json"

    make_split(labels, val_fraction=0.2, test_fraction=0.0, seed=42, cache=cache)
    train, val, test = make_split(
        labels, val_fraction=0.2, test_fraction=0.2, seed=42, cache=cache
    )
    assert len(test) == 20, "stale two-way split was reused"
    assert not set(train) & set(test)
