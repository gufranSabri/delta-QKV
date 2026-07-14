"""Config validation and labeling-scheme tests."""

import pytest
import yaml

from src.config import load_config
from src.label.exact_match import (
    correctness_imdb,
    correctness_substring,
    score_exact_match,
)
from src.utils.metrics import compute_metrics, tpr_at_fpr


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

def write_cfg(tmp_path, data) -> str:
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(data))
    return str(p)


def test_config_layers_over_defaults(tmp_path):
    cfg = load_config(write_cfg(tmp_path, {"dataset": {"name": "imdb"}}))
    assert cfg.dataset.name == "imdb"
    # Untouched keys still carry the defaults.
    assert cfg.model.fusion == "gated"
    assert cfg.extract.views == ["Q", "K", "V"]


def test_config_overrides_apply_last(tmp_path):
    path = write_cfg(tmp_path, {"model": {"fusion": "gated"}})
    cfg = load_config(path, overrides={"model": {"fusion": "bilinear"}})
    assert cfg.model.fusion == "bilinear"


def test_unknown_key_is_rejected(tmp_path):
    path = write_cfg(tmp_path, {"model": {"fusionn": "gated"}})
    with pytest.raises(ValueError, match="unknown key"):
        load_config(path)


def test_unknown_section_is_rejected(tmp_path):
    path = write_cfg(tmp_path, {"nonsense": {"a": 1}})
    with pytest.raises(ValueError, match="unknown top-level"):
        load_config(path)


@pytest.mark.parametrize("bad,msg", [
    ({"extract": {"views": []}}, "must not be empty"),
    ({"extract": {"views": ["Q", "X"]}}, "unknown"),
    ({"extract": {"views": ["Q", "Q"]}}, "duplicates"),
    ({"extract": {"pool": "median"}}, "extract.pool"),
    ({"extract": {"boundary_mode": "bounce"}}, "boundary_mode"),
    ({"model": {"fusion": "magic"}}, "model.fusion"),
    ({"model": {"backbone": "vgg"}}, "model.backbone"),
    ({"labeling": {"scheme": "vibes"}}, "labeling.scheme"),
    ({"train": {"val_fraction": 1.5}}, "val_fraction"),
    ({"extract": {"l_eff": 1}}, "l_eff"),
])
def test_invalid_configs_are_caught_at_load(tmp_path, bad, msg):
    with pytest.raises(ValueError, match=msg):
        load_config(write_cfg(tmp_path, bad))


def test_shipped_configs_all_load():
    """Every config in configs/ must be valid -- a typo there wastes a GPU run."""
    from src.config import REPO_ROOT

    paths = sorted((REPO_ROOT / "configs").rglob("*.yaml"))
    assert len(paths) > 1
    for p in paths:
        cfg = load_config(p)
        cfg.validate()


def test_qwen_configs_set_a_valid_n_cols():
    """Qwen has L=28 but D_kv=512, and 28 does not divide 512. The configs must
    therefore override n_cols -- this test guards the fix."""
    from src.config import REPO_ROOT

    for p in sorted((REPO_ROOT / "configs").glob("*/qwen2.5_7b.yaml")):
        cfg = load_config(p)
        assert cfg.extract.n_cols is not None, f"{p} must set extract.n_cols"
        assert 3584 % cfg.extract.n_cols == 0, "must divide D_q"
        assert 512 % cfg.extract.n_cols == 0, "must divide D_kv"


# --------------------------------------------------------------------------
# labeling
# --------------------------------------------------------------------------

def test_substring_match_accepts_any_alias():
    assert correctness_substring("The answer is John F. Kennedy.",
                                 ["JFK", "John F. Kennedy"]) == 1
    assert correctness_substring("It was Lincoln.",
                                 ["JFK", "John F. Kennedy"]) == 0


def test_substring_match_is_case_insensitive():
    assert correctness_substring("paris is the capital", "Paris") == 1


def test_substring_match_handles_stringified_alias_lists():
    """TriviaQA aliases sometimes arrive as the string "['a', 'b']"."""
    assert correctness_substring("the answer is b", "['a', 'b']") == 1


def test_empty_answer_is_never_correct():
    assert correctness_substring("", ["anything"]) == 0


def test_imdb_takes_the_first_sentiment_mentioned():
    # "negative" appears first, so that is the model's answer.
    assert correctness_imdb("negative, though some call it positive", 0) == 1
    assert correctness_imdb("negative, though some call it positive", 1) == 0


def test_imdb_no_sentiment_named_is_wrong():
    assert correctness_imdb("I am not sure about this film.", 1) == 0


def test_imdb_accepts_word_or_int_gold():
    assert correctness_imdb("positive", 1) == 1
    assert correctness_imdb("positive", "positive") == 1


def test_label_is_the_complement_of_correctness():
    score, label = score_exact_match("triviaqa", "the answer is Paris", ["Paris"])
    assert score == 1.0 and label == 0, "correct answer -> NOT a hallucination"

    score, label = score_exact_match("triviaqa", "the answer is Berlin", ["Paris"])
    assert score == 0.0 and label == 1, "wrong answer -> hallucination"


def test_test_split_suffix_reuses_the_parent_scorer():
    assert score_exact_match("triviaqa_test", "Paris", ["Paris"]) == (1.0, 0)


def test_unknown_dataset_raises():
    with pytest.raises(KeyError, match="no exact-match scorer"):
        score_exact_match("nonexistent", "x", "y")


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def test_metrics_on_a_perfect_classifier():
    m = compute_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m["auroc"] == 1.0
    assert m["f1"] == 1.0


def test_metrics_on_a_random_classifier():
    m = compute_metrics([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])
    assert m["auroc"] == 0.5


def test_metrics_survive_a_single_class_slice():
    """Per-origin test slices can legitimately be all-one-class; that must not
    crash the run, it must report NaN."""
    m = compute_metrics([1, 1, 1], [0.2, 0.7, 0.9])
    assert m["auroc"] != m["auroc"]        # NaN
    assert m["n"] == 3


def test_tpr_at_fpr_respects_the_budget():
    # Perfectly separable -> we catch everything at zero false alarms.
    assert tpr_at_fpr([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.05) == 1.0
