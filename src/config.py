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
class ModelConfig:
    backbone: str = "scratch_cnn"
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
    epochs: int = 30
    patience: int = 8
    seed: int = 0
    num_workers: int = 4
    val_fraction: float = 0.2
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
        """data/{dataset}/{llm_alias}/ -- where this combo's features live."""
        return Path(root or self.data_root) / self.dataset.name / self.llm.alias

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        e, m, la = self.extract, self.model, self.labeling

        if not e.views:
            raise ValueError("extract.views must not be empty")
        bad = [v for v in e.views if v not in VALID_VIEWS]
        if bad:
            raise ValueError(f"extract.views: unknown {bad}, valid are {list(VALID_VIEWS)}")
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
        if m.fusion not in VALID_FUSIONS:
            raise ValueError(f"model.fusion must be one of {VALID_FUSIONS}")
        if la.scheme not in VALID_SCHEMES:
            raise ValueError(f"labeling.scheme must be one of {VALID_SCHEMES}")

        if not 0.0 < self.train.val_fraction < 1.0:
            raise ValueError("train.val_fraction must be in (0, 1)")


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
        section = raw.get(name) or {}
        if not isinstance(section, dict):
            raise ValueError(f"config section '{name}' must be a mapping")
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(section) - known
        if unknown:
            raise ValueError(
                f"unknown key(s) in '{name}': {sorted(unknown)}. Valid: {sorted(known)}"
            )
        kwargs[name] = cls(**section)

    for top in ("data_root", "runs_root"):
        if top in raw:
            kwargs[top] = raw[top]

    unknown_top = set(raw) - set(_SECTIONS) - {"data_root", "runs_root"}
    if unknown_top:
        raise ValueError(f"unknown top-level config key(s): {sorted(unknown_top)}")

    cfg = Config(**kwargs)
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
