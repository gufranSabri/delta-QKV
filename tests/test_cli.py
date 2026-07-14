"""CLI argument-plumbing tests.

These exist because of a real bug: --config and --set were declared only on the
top-level parser, so `main.py --config c.yaml train --set k=v` -- the exact form
every generated ablation script uses -- failed outright. Then the naive fix
(argparse `parents`) introduced a subtler one, where the subparser's default
silently overwrote the top-level value.
"""

import pytest
import yaml

import main


@pytest.fixture
def cfg_path(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"dataset": {"name": "triviaqa"}}))
    return str(p)


def parse(argv):
    """Run main()'s parser only, returning the resolved (config, overrides)."""
    captured = {}

    def fake_train(cfg, train_datasets, test_dataset, run_name=None):
        captured["cfg"] = cfg
        captured["train_datasets"] = train_datasets
        captured["test_dataset"] = test_dataset

    import src.train

    original = src.train.train
    src.train.train = fake_train
    try:
        main.main(argv)
    finally:
        src.train.train = original
    return captured


def test_set_works_after_the_subcommand(cfg_path):
    """The form every generated ablation script uses."""
    out = parse(["--config", cfg_path, "train", "--set", "model.fusion=bilinear"])
    assert out["cfg"].model.fusion == "bilinear"


def test_set_works_before_the_subcommand(cfg_path):
    out = parse(["--config", cfg_path, "--set", "model.fusion=bilinear", "train"])
    assert out["cfg"].model.fusion == "bilinear"


def test_config_works_after_the_subcommand(cfg_path):
    out = parse(["train", "--config", cfg_path, "--set", "model.fusion=bilinear"])
    assert out["cfg"].model.fusion == "bilinear"


def test_all_orderings_agree(cfg_path):
    """Argument position must never change the resulting config."""
    a = parse(["--config", cfg_path, "train", "--set", "train.epochs=7"])
    b = parse(["--config", cfg_path, "--set", "train.epochs=7", "train"])
    c = parse(["train", "--config", cfg_path, "--set", "train.epochs=7"])
    assert a["cfg"].train.epochs == b["cfg"].train.epochs == c["cfg"].train.epochs == 7


def test_set_values_are_typed_not_strings(cfg_path):
    out = parse([
        "--config", cfg_path, "train",
        "--set", "train.epochs=5",
        "--set", "train.lr=0.002",
        "--set", "model.share_backbone=true",
        "--set", "extract.views=[V]",
    ])
    cfg = out["cfg"]
    assert cfg.train.epochs == 5 and isinstance(cfg.train.epochs, int)
    assert cfg.train.lr == 0.002 and isinstance(cfg.train.lr, float)
    assert cfg.model.share_backbone is True
    assert cfg.extract.views == ["V"]


def test_train_datasets_flag_is_split_on_commas(cfg_path):
    out = parse([
        "--config", cfg_path, "train",
        "--train-datasets", "triviaqa,hotpotqa,imdb",
        "--test-dataset", "movies",
    ])
    assert out["train_datasets"] == ["triviaqa", "hotpotqa", "imdb"]
    assert out["test_dataset"] == "movies"


def test_train_datasets_defaults_to_the_config_dataset(cfg_path):
    out = parse(["--config", cfg_path, "train"])
    assert out["train_datasets"] == ["triviaqa"]


def test_missing_config_errors_cleanly():
    with pytest.raises(SystemExit):
        main.main(["train"])


def test_malformed_set_is_rejected(cfg_path):
    with pytest.raises(SystemExit, match="key=value"):
        main.main(["--config", cfg_path, "train", "--set", "nonsense"])


def test_invalid_override_is_caught_by_validation(cfg_path):
    with pytest.raises(ValueError, match="model.fusion"):
        main.main(["--config", cfg_path, "train", "--set", "model.fusion=telepathy"])
