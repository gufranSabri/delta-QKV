"""Dataset + collation for the extracted per-token Q/K/V images."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Index of each view on the stored view axis, as written by run_extraction.
ALL_VIEWS = ("Q", "K", "V")


class QKVImageDataset(Dataset):
    """One example = one folder holding tokens.npy of shape (T, V, L, C, 3).

    Returns (images, label, origin) where images is (T, V, 3, L, C) -- note the
    channel axis is moved into PyTorch's conv position (N, C, H, W) here, so the
    model never has to permute.
    """

    def __init__(
        self,
        root: str | Path,
        views: list[str] | None = None,
        stats: dict | None = None,
        max_tokens: int | None = None,
        origin: str | None = None,
        channels: str = "default",
    ):
        self.channels = channels
        self.root = Path(root)
        manifest = self.root / "manifest.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(
                f"no manifest at {manifest}. Run `python main.py extract` first."
            )

        geometry_path = self.root / "geometry.json"
        self.geometry = (
            json.loads(geometry_path.read_text()) if geometry_path.exists() else {}
        )
        stored_views = self.geometry.get("views", list(ALL_VIEWS))

        requested = views or stored_views
        missing = [v for v in requested if v not in stored_views]
        if missing:
            raise ValueError(
                f"requested views {missing} were not extracted into {self.root}. "
                f"Available: {stored_views}. Re-run extraction to add them."
            )
        # Positions of the requested views on the stored view axis.
        self.view_idx = [stored_views.index(v) for v in requested]
        self.views = requested

        self.records = []
        with open(manifest) as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))

        unlabeled = [r for r in self.records if r.get("label", -1) not in (0, 1)]
        if unlabeled:
            raise ValueError(
                f"{len(unlabeled)} examples in {self.root} have no valid label. "
                "Run `python main.py label` to (re)label them."
            )

        self.stats = stats
        self.max_tokens = max_tokens
        self.origin = origin or f"{self.root.parent.name}_{self.root.name}"

    def __len__(self) -> int:
        return len(self.records)

    @property
    def labels(self) -> list[int]:
        return [int(r["label"]) for r in self.records]

    def __getitem__(self, i):
        rec = self.records[i]
        path = self.root / rec["dir"] / "tokens.npy"

        arr = np.load(path)                       # (T, V_stored, L, C, 3) fp16
        images = torch.from_numpy(np.ascontiguousarray(arr)).float()

        # Select only the requested views.
        images = images[:, self.view_idx]         # (T, V, L, C, 3)

        if self.max_tokens is not None and images.shape[0] > self.max_tokens:
            images = images[: self.max_tokens]

        if self.stats is not None:
            images = normalize(images, self.stats, self.views)

        # (T, V, L, C, 3) -> (T, V, 3, L, C): channels into conv position.
        images = images.permute(0, 1, 4, 2, 3)

        images = regroup_channels(images, self.channels).contiguous()

        return images, float(rec["label"]), self.origin


def n_images(mode: str, n_views: int) -> int:
    """How many images (i.e. CNN streams) `mode` produces from `n_views` views.

    This is what the model must be built for -- NOT len(extract.views), which is
    only the image count in `default` mode. `n_views` is 1 for the hidden-state
    source and up to 3 (Q/K/V) for the qkv source.
    """
    if mode == "default":
        return n_views        # one image per view
    if mode == "first_only":
        return 1              # one image; the views became its channels
    if mode == "same":
        return 3              # one image per channel-type (raw, ch1, ch2)
    raise ValueError(f"unknown model.channels mode {mode!r}")


def n_channels(mode: str, n_views: int) -> int:
    """How many channels each image produced by `mode` carries."""
    if mode == "default":
        return 3              # the three extraction channels (raw + two others)
    if mode == "first_only":
        return n_views        # first channel of each view, stacked
    if mode == "same":
        return n_views        # views stacked onto the channel axis, per channel-type
    raise ValueError(f"unknown model.channels mode {mode!r}")


def regroup_channels(images: torch.Tensor, mode: str) -> torch.Tensor:
    """Regroup the (view, channel) axes into the images the CNNs consume.

    images: (T, V, 3, L, C) -- V views, each a 3-channel image. The three
    channels are (raw, delta-prev, delta-next) under extraction_type=delta, or
    (raw, DWT, FFT) under extraction_type=transforms; the regrouping is identical
    either way, since it only cares about channel POSITION, not meaning.
    Returns (T, V', C', L, C): V' images of C' channels each.

    This is pure re-slicing of what extraction already wrote -- no mode needs a
    re-extract. Normalisation has ALREADY run, per view and per channel, on the
    original layout; that ordering is load-bearing, because different views have
    very different magnitudes and stacking them onto one channel axis
    unnormalised would let the largest-scale view dominate the shared conv filters.

      default     (T, V, 3, L, C) unchanged -- one image per view.
      first_only  (T, 1, V, L, C) -- ONE image whose channels are the FIRST (raw)
                  channel of each view. The other two channels are dropped.
      same        (T, 3, V, L, C) -- transposed: image k holds channel k of every
                  view, i.e. (raw of all views), (ch1 of all), (ch2 of all).
    """
    if mode == "default":
        return images

    if mode == "first_only":
        # Channel 0 (raw) of every view -> one image, one channel per view.
        raw = images[:, :, 0]                      # (T, V, L, C)
        return raw.unsqueeze(1)                    # (T, 1, V, L, C)

    if mode == "same":
        # Swap the view and channel axes: image k = channel k across all views.
        return images.transpose(1, 2)              # (T, 3, V, L, C)

    raise ValueError(f"unknown model.channels mode {mode!r}")


def normalize(images: torch.Tensor, stats: dict, views: list[str]) -> torch.Tensor:
    """Standardise PER VIEW, PER CHANNEL.

    A single global statistic would be wrong: Q, K and V have very different
    magnitudes (and under GQA are pooled from different-width vectors), so
    whichever view happens to have the largest scale would dominate the fused
    representation before the fusion module ever got a say. The raw channel also
    dwarfs the two delta channels, which would make the deltas numerically
    invisible.

    images: (T, V, L, C, 3)
    """
    mean = torch.tensor(
        [[stats[v]["mean"][c] for c in range(3)] for v in views],
        dtype=images.dtype,
    ).view(1, len(views), 1, 1, 3)
    std = torch.tensor(
        [[stats[v]["std"][c] for c in range(3)] for v in views],
        dtype=images.dtype,
    ).view(1, len(views), 1, 1, 3)
    return (images - mean) / std.clamp(min=1e-6)


def _get_stats(dataset):
    """Read normalisation stats from a source or a concat-of-sources."""
    if hasattr(dataset, "datasets"):        # ConcatQKVDataset
        return dataset.datasets[0].stats
    return dataset.stats


def _set_stats(dataset, stats) -> None:
    """Attach stats to a source, or to every source inside a concat.

    A ConcatQKVDataset holds no images itself -- its children do -- so the stats
    have to be pushed down to each of them.
    """
    if hasattr(dataset, "datasets"):        # ConcatQKVDataset
        for d in dataset.datasets:
            d.stats = stats
    else:
        dataset.stats = stats


def compute_stats(
    dataset,
    indices: list[int],
    max_examples: int = 500,
) -> dict:
    """Per-view, per-channel mean/std over a sample of the TRAINING split only.

    Computed on train indices exclusively -- using val/test examples here would
    leak their distribution into the model's input scaling.
    """
    views = dataset.views
    n_chan = 3
    # Welford would be tidier, but a two-pass over a bounded sample is simpler
    # and plenty accurate for a normalisation constant.
    sums = torch.zeros(len(views), n_chan, dtype=torch.float64)
    sqs = torch.zeros(len(views), n_chan, dtype=torch.float64)
    count = torch.zeros(len(views), n_chan, dtype=torch.float64)

    sample = indices[:max_examples]
    logger.info("computing normalisation stats over %d training examples", len(sample))

    # Read RAW values, not already-normalised ones -- otherwise recomputing stats
    # on a dataset that already has them would standardise twice. `dataset` may
    # be a ConcatQKVDataset, whose stats live on its child sources, so toggle
    # through the helpers rather than touching `.stats` directly.
    saved = _get_stats(dataset)
    _set_stats(dataset, None)
    try:
        for i in sample:
            images, _, _ = dataset[i]             # (T, V, 3, L, C) -- permuted!
            x = images.double()
            # channel axis is dim 2 after the permute in __getitem__
            sums += x.sum(dim=(0, 3, 4))
            sqs += (x**2).sum(dim=(0, 3, 4))
            n = x.shape[0] * x.shape[3] * x.shape[4]
            count += n
    finally:
        _set_stats(dataset, saved)

    mean = sums / count.clamp(min=1)
    var = (sqs / count.clamp(min=1)) - mean**2
    std = var.clamp(min=0).sqrt()

    return {
        v: {
            "mean": mean[k].tolist(),
            "std": [max(s, 1e-6) for s in std[k].tolist()],
        }
        for k, v in enumerate(views)
    }


def collate(batch):
    """Pad a batch of variable-length responses and build the mask.

    Returns:
        images: (B, T_max, V, 3, L, C)
        labels: (B,)
        mask:   (B, T_max) bool -- True at real tokens
        origins: list[str]
    """
    images_list, labels, origins = zip(*batch)

    t_max = max(img.shape[0] for img in images_list)
    b = len(images_list)
    _, v, c, h, w = images_list[0].shape

    images = torch.zeros(b, t_max, v, c, h, w, dtype=images_list[0].dtype)
    mask = torch.zeros(b, t_max, dtype=torch.bool)

    for i, img in enumerate(images_list):
        t = img.shape[0]
        images[i, :t] = img
        mask[i, :t] = True

    return (
        images,
        torch.tensor(labels, dtype=torch.float32),
        mask,
        list(origins),
    )


class ConcatQKVDataset(Dataset):
    """Concatenate several (dataset, LLM) sources for multi-dataset training.

    Used by the leave-one-dataset-out setting: train on the union of N-1 sources,
    test on the held-out one. Each item keeps its `origin` tag so metrics can be
    reported per source, the way ACT-ViT does.
    """

    def __init__(self, datasets: list[QKVImageDataset]):
        if not datasets:
            raise ValueError("ConcatQKVDataset needs at least one dataset")

        # All sources must agree on image geometry, or the CNN cannot consume
        # them. Cross-LLM training therefore requires extract.l_eff to be set to
        # a common value (Llama has 32 layers, Qwen 28).
        shapes = {tuple(d.geometry.get("views", [])) for d in datasets}
        if len(shapes) > 1:
            raise ValueError(f"sources disagree on views: {shapes}")

        rows = {(d.geometry.get("n_rows"), d.geometry.get("n_cols")) for d in datasets}
        if len(rows) > 1:
            raise ValueError(
                f"sources have different image sizes {rows}. Cross-LLM training "
                "requires a common image size: set extract.l_eff (and n_cols) to "
                "the same value for every LLM and re-extract."
            )

        self.datasets = datasets
        self.offsets = [0]
        for d in datasets:
            self.offsets.append(self.offsets[-1] + len(d))
        self.views = datasets[0].views

    def __len__(self) -> int:
        return self.offsets[-1]

    @property
    def labels(self) -> list[int]:
        out: list[int] = []
        for d in self.datasets:
            out.extend(d.labels)
        return out

    def __getitem__(self, i):
        # Locate the source dataset this global index falls into.
        lo, hi = 0, len(self.datasets) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if i >= self.offsets[mid + 1]:
                lo = mid + 1
            else:
                hi = mid
        return self.datasets[lo][i - self.offsets[lo]]
