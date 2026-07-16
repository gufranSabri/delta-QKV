"""Config loading: YAML -> validated dataclasses, with deep-merge over defaults."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.extract.tensor_ops import BOUNDARY_MODES, POOL_MODES

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"

VALID_VIEWS = ("Q", "K", "V")
VALID_BACKBONES = ("scratch_cnn", "resnet18")
VALID_FUSIONS = ("gated", "concat_mlp", "bilinear", "cross_attn")

#: What activation the images are built from. This is an EXTRACTION-time choice --
#: it changes what is captured and therefore lives in a different data folder.
#:
#:   qkv  per-layer Q/K/V projection activations, captured via forward hooks.
#:        Produces one "view" per entry in extract.views (Q, K, V).
#:   hs   per-layer hidden states (the residual stream), captured for free via
#:        output_hidden_states. There is exactly ONE hidden-state stream, so this
#:        always yields a single image ("view").
VALID_SOURCES = ("qkv", "hs")

#: How each view's per-layer signal is turned into the 3 image channels. Also an
#: EXTRACTION-time choice (different channels are written to disk), so it too
#: partitions the data folder.
#:
#:   delta       (raw, delta-to-prev-layer, delta-to-next-layer). The original
#:               design: how the representation changes between adjacent layers.
#:   transforms  (raw, DWT, FFT) computed ALONG the layer axis. Multi-resolution
#:               and frequency views of how a dimension evolves with depth.
VALID_EXTRACTION_TYPES = ("delta", "transforms")

#: How the extracted (V views x 3 channels) tensor is regrouped into the images
#: the CNNs actually see. Purely a re-slicing of what extraction already wrote --
#: no mode here requires re-extracting. Generalises the old qkv-specific names.
#:
#:   default     V images, 3 channels each: view v -> its own (ch0, ch1, ch2).
#:               One CNN per view, meeting at fusion.
#:   first_only  ONE image, V channels: the FIRST channel (raw) of each view
#:               stacked (raw_v0, raw_v1, ...). The other two channels are
#:               dropped. Tests whether the extra channels were earning their keep.
#:   same        THREE images, regrouped BY CHANNEL instead of by view: image k
#:               holds channel k of every view. Same information as `default`,
#:               transposed -- each CNN sees one channel-type across all views.
VALID_CHANNEL_MODES = ("default", "first_only", "same")
VALID_SCHEMES = ("exact_match", "bleurt")


@dataclass
class LLMConfig:
    name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    dtype: str = "bfloat16"
    # Short alias used in output paths (avoids '/' in directory names).
    alias: str = "llama3_8b"


@dataclass
class DatasetConfig:
    name: str = "triviaqa"
    n_samples: int = 10000
    max_new_tokens: int = 100
    prompt_template: str = "Answer the question concisely. Q: {question} A:"


@dataclass
class ExtractConfig:
    # What to capture and how to turn it into channels. Both are EXTRACTION-time
    # choices that change what lands on disk, so they partition the data folder:
    #   data/{source}/{extraction_type}/{dataset}/{llm_alias}/
    source: str = "qkv"                 # qkv | hs   (VALID_SOURCES)
    extraction_type: str = "delta"      # delta | transforms (VALID_EXTRACTION_TYPES)
    views: list[str] = field(default_factory=lambda: ["Q", "K", "V"])
    pool: str = "max"
    boundary_mode: str = "zero"
    dtype: str = "float16"
    max_tokens: int = 100
    # Number of image columns. `null` in YAML -> use the model's layer count,
    # which makes the image square (the whole point of the design).
    n_cols: int | None = None
    # Pool the LAYER axis to this many rows. `null` -> keep the model's L.
    # Only needed for cross-LLM training, where Llama (32) and Qwen (28) differ.
    l_eff: int | None = None


@dataclass
class LabelingConfig:
    scheme: str = "exact_match"
    bleurt_threshold: float = 0.5
    bleurt_checkpoint: str = "models/BLEURT-20-D12"


@dataclass
class Stream2Config:
    # Stream 2 transposes each (T, L, D) image to (D, L, T) before the same
    # channels/regroup pipeline stream 1 uses -- T (generated tokens) becomes
    # a spatial axis instead of the sequence axis, so the model can find
    # "this happens over these tokens" structure directly inside a CNN. D
    # (a fixed extract.n_cols per run) takes T's old role as the axis folded
    # into the batch. Off by default: existing configs/checkpoints are
    # single-stream and unaffected.
    enable: bool = False
    # Same semantics as model.include, but selects among stream 2's own
    # regrouped images (same `channels` mode, independent selection).
    include: list[int] | None = None


@dataclass
class ModelConfig:
    backbone: str = "scratch_cnn"
    # How the extracted views/channels are regrouped into images. See
    # VALID_CHANNEL_MODES. Never requires re-extraction.
    channels: str = "default"
    # Which of the regrouped images to actually feed the model, by index --
    # applied as the LAST step in QKVImageDataset.__getitem__, after `channels`
    # has decided how many images there are. `null` (the default) keeps all of
    # them. E.g. under channels=same (3 images: raw, ch1, ch2), include=[0, 2]
    # drops the middle image and leaves the model with 2 CNN streams.
    include: list[int] | None = None
    stream2: Stream2Config = field(default_factory=Stream2Config)
    share_backbone: bool = False
    embed_dim: int = 128       # E: per-view CNN output
    fusion: str = "gated"
    fused_dim: int = 128       # F: fusion output
    conv1d_layers: int = 2
    lstm_hidden: int = 128
    lstm_layers: int = 1
    dropout: float = 0.3
    pretrained_backbone: bool = True   # only meaningful for resnet18


@dataclass
class TrainConfig:
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    # A pretrained backbone needs a gentler LR than the randomly-initialised
    # fusion/temporal/head, or the early high-LR steps wreck its ImageNet
    # features before the head stabilises. Its LR is `lr * backbone_lr_scale`.
    # 1.0 = single LR for everything (correct for scratch / random-init, where
    # there is nothing pretrained to protect).
    backbone_lr_scale: float = 1.0
    # Linear LR decay: hold the LR flat for the first `lr_decay_start` epochs,
    # then ramp linearly down to `lr_final_scale` x the initial LR by `epochs`.
    # Applied as a multiplier on each group's own LR, so backbone_lr_scale's
    # backbone:head ratio survives the decay.
    lr_decay_start: int = 5
    lr_final_scale: float = 0.0
    epochs: int = 30
    patience: int = 8
    seed: int = 42            # matches ACT-ViT's RANDOM_STATE and HalluShift's
    num_workers: int = 4
    val_fraction: float = 0.2
    # Only used for datasets with no separate held-out corpus (TruthfulQA).
    # Everything else tests on a `<name>_test` corpus, so it needs no slice.
    test_fraction: float = 0.2
    # Weight the positive (hallucination) class to counter imbalance.
    balance_classes: bool = True


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data_root: str = "data"
    runs_root: str = "runs"

    # ---- derived paths -------------------------------------------------
    def example_dir(self, root: str | None = None) -> Path:
        """Where this combo's features live:

            data/{source}/{extraction_type}/{dataset}/{llm_alias}/

        `source` (qkv|hs) and `extraction_type` (delta|transforms) are baked into
        the path because they change what is stored on disk -- the same dataset x
        LLM produces genuinely different tensors under each, so they must not
        collide in one directory.
        """
        return (
            Path(root or self.data_root)
            / self.extract.source
            / self.extract.extraction_type
            / self.dataset.name
            / self.llm.alias
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        e, m, la = self.extract, self.model, self.labeling

        if e.source not in VALID_SOURCES:
            raise ValueError(
                f"extract.source must be one of {VALID_SOURCES}, got {e.source!r}"
            )
        if e.extraction_type not in VALID_EXTRACTION_TYPES:
            raise ValueError(
                f"extract.extraction_type must be one of {VALID_EXTRACTION_TYPES}, "
                f"got {e.extraction_type!r}"
            )
        # Hidden states are a single residual stream: there is no Q/K/V axis to
        # subset, so extract.views is meaningless (and misleading) under source=hs.
        if e.source == "hs" and e.views != ["H"]:
            raise ValueError(
                "extract.source=hs has a single hidden-state stream; "
                "set extract.views: [H] (or leave it to be defaulted per-source)"
            )
        if not e.views:
            raise ValueError("extract.views must not be empty")
        if e.source == "qkv":
            bad = [v for v in e.views if v not in VALID_VIEWS]
            if bad:
                raise ValueError(
                    f"extract.views: unknown {bad}, valid are {list(VALID_VIEWS)}"
                )
        if len(set(e.views)) != len(e.views):
            raise ValueError(f"extract.views has duplicates: {e.views}")
        if e.pool not in POOL_MODES:
            raise ValueError(f"extract.pool must be one of {POOL_MODES}, got {e.pool!r}")
        if e.boundary_mode not in BOUNDARY_MODES:
            raise ValueError(
                f"extract.boundary_mode must be one of {BOUNDARY_MODES}, got {e.boundary_mode!r}"
            )
        if e.max_tokens < 1:
            raise ValueError("extract.max_tokens must be >= 1")
        if e.n_cols is not None and e.n_cols < 1:
            raise ValueError("extract.n_cols must be >= 1 or null")
        if e.l_eff is not None and e.l_eff < 2:
            raise ValueError("extract.l_eff must be >= 2 (deltas need 2 layers) or null")

        if m.backbone not in VALID_BACKBONES:
            raise ValueError(f"model.backbone must be one of {VALID_BACKBONES}")
        if m.channels not in VALID_CHANNEL_MODES:
            raise ValueError(
                f"model.channels must be one of {VALID_CHANNEL_MODES}, got {m.channels!r}"
            )
        # `same` stacks every view onto ONE image's channel axis, so a view
        # subset silently changes that image's channel count. It only carries the
        # intended meaning ("one channel-type across all views") with the full
        # view set. For the single-stream hidden-state source there is just one
        # view, so this constraint does not bite. `first_only` is fine with any
        # number of views (it drops all but the first channel of each).
        if m.channels == "same" and e.source == "qkv" and len(e.views) != 3:
            raise ValueError(
                f"model.channels='same' stacks Q/K/V on the channel axis and "
                f"needs all three views, but extract.views={e.views}"
            )
        def _check_include(include: list[int] | None, field_name: str) -> None:
            if include is None:
                return
            from src.data.dataset import n_images

            n_avail = n_images(m.channels, len(e.views))
            if not include:
                raise ValueError(f"{field_name} must not be empty")
            if len(set(include)) != len(include):
                raise ValueError(f"{field_name} has duplicates: {include}")
            bad = [i for i in include if not 0 <= i < n_avail]
            if bad:
                raise ValueError(
                    f"{field_name} {bad} out of range for model.channels={m.channels!r} "
                    f"with extract.views={e.views} ({n_avail} images available)"
                )

        _check_include(m.include, "model.include")
        _check_include(m.stream2.include, "model.stream2.include")

        if m.stream2.enable and m.backbone == "resnet18":
            raise ValueError(
                "model.stream2.enable=true is not supported with backbone=resnet18: "
                "torchvision's resnet18 internals aren't decomposable the way "
                "scratch_cnn's are, which stream 2's masked pooling needs. "
                "Use backbone=scratch_cnn, or disable stream2."
            )
        if m.fusion not in VALID_FUSIONS:
            raise ValueError(f"model.fusion must be one of {VALID_FUSIONS}")
        if la.scheme not in VALID_SCHEMES:
            raise ValueError(f"labeling.scheme must be one of {VALID_SCHEMES}")

        if not 0.0 < self.train.val_fraction < 1.0:
            raise ValueError("train.val_fraction must be in (0, 1)")
        if not 0.0 <= self.train.test_fraction < 1.0:
            raise ValueError("train.test_fraction must be in [0, 1)")
        if self.train.val_fraction + self.train.test_fraction >= 1.0:
            raise ValueError("train.val_fraction + train.test_fraction must be < 1")

        t = self.train
        if t.lr_decay_start < 0:
            raise ValueError("train.lr_decay_start must be >= 0")
        # Decay would never begin: the run ends before the flat phase does.
        if t.lr_decay_start >= t.epochs:
            raise ValueError(
                f"train.lr_decay_start ({t.lr_decay_start}) must be < train.epochs "
                f"({t.epochs}), or the LR never starts decaying"
            )
        if not 0.0 <= t.lr_final_scale < 1.0:
            raise ValueError("train.lr_final_scale must be in [0, 1)")
        if t.backbone_lr_scale <= 0.0:
            raise ValueError("train.backbone_lr_scale must be > 0")


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base`, returning a new dict."""
    out = copy.deepcopy(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


_SECTIONS = {
    "llm": LLMConfig,
    "dataset": DatasetConfig,
    "extract": ExtractConfig,
    "labeling": LabelingConfig,
    "model": ModelConfig,
    "train": TrainConfig,
}


def _build(raw: dict) -> Config:
    kwargs: dict[str, Any] = {}
    for name, cls in _SECTIONS.items():
        section = dict(raw.get(name) or {})
        if not isinstance(section, dict):
            raise ValueError(f"config section '{name}' must be a mapping")
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(section) - known
        if unknown:
            raise ValueError(
                f"unknown key(s) in '{name}': {sorted(unknown)}. Valid: {sorted(known)}"
            )
        # model.stream2 is a nested dataclass; YAML gives it to us as a plain
        # dict, which must be converted explicitly or it would silently stick
        # around as a dict and break attribute access (cfg.model.stream2.enable).
        if cls is ModelConfig and isinstance(section.get("stream2"), dict):
            section["stream2"] = Stream2Config(**section["stream2"])
        kwargs[name] = cls(**section)

    for top in ("data_root", "runs_root"):
        if top in raw:
            kwargs[top] = raw[top]

    unknown_top = set(raw) - set(_SECTIONS) - {"data_root", "runs_root"}
    if unknown_top:
        raise ValueError(f"unknown top-level config key(s): {sorted(unknown_top)}")

    cfg = Config(**kwargs)

    # Hidden-state extraction has a single stream. The Q/K/V view set is
    # meaningless under source=hs and only ever arrives via the base default
    # (nobody deliberately asks for Q/K/V hidden states), so normalise it to the
    # lone hidden-state view. A DIFFERENT explicit views (e.g. [Q], a typo) is a
    # real mistake and is left to fail loudly in validate().
    if cfg.extract.source == "hs" and cfg.extract.views == list(VALID_VIEWS):
        cfg.extract.views = ["H"]

    cfg.validate()
    return cfg


def load_config(path: str | Path, overrides: dict | None = None) -> Config:
    """Load `path`, layered over configs/default.yaml, then apply `overrides`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")

    raw: dict = {}
    if DEFAULT_CONFIG.exists() and path.resolve() != DEFAULT_CONFIG.resolve():
        with open(DEFAULT_CONFIG) as f:
            raw = yaml.safe_load(f) or {}

    with open(path) as f:
        raw = _deep_merge(raw, yaml.safe_load(f) or {})

    if overrides:
        raw = _deep_merge(raw, overrides)

    return _build(raw)
