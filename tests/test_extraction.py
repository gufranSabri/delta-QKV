"""Extraction pipeline tests, driven by a real (tiny) LLM.

run_extraction is the one module that touches a model, a tokenizer, the
filesystem and the labeler all at once. These tests run the genuine code path
against a tiny randomly-initialised Llama, so extraction is not left as the
untested link in the chain.
"""

import json

import numpy as np
import pytest
import torch

from src.config import Config
from src.extract.run_extraction import (
    build_images,
    is_complete,
    parse_meta,
    resolve_stop_tokens,
    write_manifest,
)

transformers = pytest.importorskip("transformers")
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

TINY = dict(
    vocab_size=64, hidden_size=32, intermediate_size=64,
    num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
    head_dim=8, max_position_embeddings=64,
)


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    m = LlamaForCausalLM(LlamaConfig(**TINY))
    m.eval()
    return m


# --------------------------------------------------------------------------
# build_images: raw activations -> the stored tensor
# --------------------------------------------------------------------------

def test_build_images_stacks_views_on_their_own_axis():
    """The view axis must be a REAL axis, never folded into channels."""
    cfg = Config()
    cfg.extract.views = ["Q", "K", "V"]
    L, T = 4, 3

    qkv = {
        "Q": torch.randn(T, L, 32),   # D_q
        "K": torch.randn(T, L, 16),   # D_kv -- narrower (GQA)
        "V": torch.randn(T, L, 16),
    }
    images = build_images(qkv, cfg, n_cols=4)

    # (T, V, L, C, 3): views are axis 1, channels are axis -1.
    assert images.shape == (T, 3, L, 4, 3)


def test_build_images_respects_the_view_subset():
    cfg = Config()
    cfg.extract.views = ["V"]
    qkv = {"V": torch.randn(2, 4, 16)}
    assert build_images(qkv, cfg, n_cols=4).shape == (2, 1, 4, 4, 3)


def test_build_images_applies_l_eff_layer_pooling():
    """Cross-LLM training pools the layer axis to a common size."""
    cfg = Config()
    cfg.extract.views = ["Q"]
    cfg.extract.l_eff = 2
    qkv = {"Q": torch.randn(3, 8, 32)}       # 8 layers
    images = build_images(qkv, cfg, n_cols=4)
    assert images.shape == (3, 1, 2, 4, 3)   # pooled down to 2


def test_build_images_view_order_matches_the_config():
    """Stored view i must correspond to cfg.extract.views[i] -- the dataset
    slices by this index, so a mismatch would silently swap Q and V."""
    cfg = Config()
    cfg.extract.views = ["V", "Q"]            # deliberately not alphabetical
    qkv = {
        "Q": torch.ones(1, 4, 32) * 7.0,
        "V": torch.ones(1, 4, 16) * 3.0,
    }
    images = build_images(qkv, cfg, n_cols=4)
    # Channel 0 is the raw pooled value: view 0 should be V (3.0), view 1 Q (7.0).
    assert images[0, 0, 0, 0, 0].item() == 3.0
    assert images[0, 1, 0, 0, 0].item() == 7.0


# --------------------------------------------------------------------------
# stop tokens -- the bug that would make every response run to max_new_tokens
# --------------------------------------------------------------------------

def test_resolve_stop_tokens_collects_more_than_eos_token_id(tiny_model):
    class FakeTok:
        eos_token_id = 2
        unk_token_id = 0

        def convert_tokens_to_ids(self, tok):
            return {"<|eot_id|>": 42}.get(tok, self.unk_token_id)

        def encode(self, text, add_special_tokens=False):
            return [9]

    tiny_model.config.eos_token_id = 2
    ids = resolve_stop_tokens(FakeTok(), tiny_model, "meta-llama/Meta-Llama-3-8B-Instruct")

    assert 2 in ids, "the plain eos id must be included"
    assert 42 in ids, (
        "<|eot_id|> must be included -- instruct models end turns with it, and "
        "missing it means generation never stops early"
    )
    # An instruct model must NOT stop at a bare newline.
    assert 9 not in ids


def test_base_models_also_stop_at_a_newline(tiny_model):
    class FakeTok:
        eos_token_id = 2
        unk_token_id = 0

        def convert_tokens_to_ids(self, tok):
            return self.unk_token_id

        def encode(self, text, add_special_tokens=False):
            return [9]

    tiny_model.config.eos_token_id = 2
    ids = resolve_stop_tokens(FakeTok(), tiny_model, "meta-llama/Llama-2-7b-hf")
    assert 9 in ids, "base models ramble; both baselines cut them off at a newline"


def test_resolve_stop_tokens_handles_a_list_eos(tiny_model):
    class FakeTok:
        eos_token_id = [2, 3]
        unk_token_id = 0

        def convert_tokens_to_ids(self, tok):
            return self.unk_token_id

        def encode(self, text, add_special_tokens=False):
            return [9]

    tiny_model.config.eos_token_id = [2, 3]
    ids = resolve_stop_tokens(FakeTok(), tiny_model, "x-instruct")
    assert 2 in ids and 3 in ids


# --------------------------------------------------------------------------
# manifest + resume
# --------------------------------------------------------------------------

def test_manifest_merge_preserves_earlier_chunks(tmp_path):
    """Chunked extraction must not have chunk 2 erase chunk 1."""
    write_manifest(tmp_path, [
        {"idx": 0, "dir": "00000", "n_tokens": 5, "score": 1.0, "label": 0},
        {"idx": 1, "dir": "00001", "n_tokens": 3, "score": 0.0, "label": 1},
    ], chunk=1)

    write_manifest(tmp_path, [
        {"idx": 2, "dir": "00002", "n_tokens": 4, "score": 1.0, "label": 0},
    ], chunk=2)

    lines = (tmp_path / "manifest.jsonl").read_text().strip().splitlines()
    idxs = [json.loads(line)["idx"] for line in lines]
    assert idxs == [0, 1, 2], "the second chunk erased the first"


def test_manifest_rewrite_updates_an_existing_record(tmp_path):
    write_manifest(tmp_path, [
        {"idx": 0, "dir": "00000", "n_tokens": 5, "score": 1.0, "label": 0},
    ], chunk=None)
    write_manifest(tmp_path, [
        {"idx": 0, "dir": "00000", "n_tokens": 5, "score": 0.0, "label": 1},
    ], chunk=None)

    lines = (tmp_path / "manifest.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["label"] == 1, "re-extraction should overwrite"


def test_is_complete_rejects_a_tensor_without_a_label(tmp_path):
    """The crash-between-write-and-label case. Skipping such an example on
    resume would strand it unlabeled and break dataset loading."""
    d = tmp_path / "00000"
    d.mkdir()
    np.save(d / "tokens.npy", np.zeros((2, 1, 4, 4, 3), dtype=np.float16))

    # tensor exists but no meta at all
    assert not is_complete(d)

    # tensor + meta, but the label is the "not yet labeled" sentinel
    (d / "meta.txt").write_text("prompt: p\nresponse: r\ngold: g\nscore: nan\nlabel: -1\n")
    assert not is_complete(d), "an unlabeled example must not be treated as done"

    # fully labeled
    (d / "meta.txt").write_text("prompt: p\nresponse: r\ngold: g\nscore: 1.0\nlabel: 0\n")
    assert is_complete(d)


def test_is_complete_rejects_a_missing_tensor(tmp_path):
    d = tmp_path / "00000"
    d.mkdir()
    (d / "meta.txt").write_text("prompt: p\nresponse: r\ngold: g\nscore: 1.0\nlabel: 0\n")
    assert not is_complete(d)


# --------------------------------------------------------------------------
# meta.txt round-trip
# --------------------------------------------------------------------------

def test_parse_meta_round_trips_a_multiline_response(tmp_path):
    """Responses contain newlines. A naive line-based parser would truncate them,
    and relabeling reads the response back out of this file."""
    p = tmp_path / "meta.txt"
    p.write_text(
        "prompt: What is 2+2?\n"
        "response: The answer is 4.\nIt is a simple sum.\n"
        "gold: ['4', 'four']\n"
        "score: 1.0\n"
        "label: 0\n"
    )
    meta = parse_meta(p)
    assert meta["prompt"] == "What is 2+2?"
    assert meta["response"] == "The answer is 4.\nIt is a simple sum."
    assert meta["gold"] == "['4', 'four']"
    assert meta["label"] == "0"


def test_stringified_gold_list_still_labels_correctly(tmp_path):
    """meta.txt stores gold as text, so a TriviaQA alias list comes back as a
    string. The labeler must still parse it -- otherwise relabeling silently
    scores against the literal characters "['a', 'b']"."""
    from src.label.exact_match import score_exact_match

    meta = parse_meta_from_text(
        "prompt: q\nresponse: the answer is four\ngold: ['4', 'four']\nscore: 0\nlabel: -1\n",
        tmp_path,
    )
    score, label = score_exact_match("triviaqa", meta["response"], meta["gold"])
    assert score == 1.0 and label == 0


def parse_meta_from_text(text, tmp_path):
    p = tmp_path / "m.txt"
    p.write_text(text)
    return parse_meta(p)


# --------------------------------------------------------------------------
# THE FULL CHAIN: run_extraction -> label -> dataset -> model
# --------------------------------------------------------------------------

def test_extract_to_model_end_to_end(tmp_path, tiny_model):
    """Runs the REAL run_extraction against a real tokenizer and model, then
    loads the result through the real dataset and pushes it through the real
    model. This is the only test that exercises extraction as an integrated
    whole; everything else stubs one side or the other.
    """
    from unittest.mock import patch

    from transformers import AutoTokenizer

    from src.data.dataset import QKVImageDataset, collate
    from src.extract.datasets import Example
    from src.extract.run_extraction import run_extraction
    from src.models.classifier import build_model

    try:
        tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    except Exception:
        pytest.skip("tokenizer unavailable (offline)")

    # A model whose vocab matches the real tokenizer.
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=32000, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=4,
        head_dim=8, max_position_embeddings=128,
    )).eval()

    cfg = Config()
    cfg.data_root = str(tmp_path)
    cfg.llm.alias = "tiny"
    cfg.llm.name = "tiny-llama-base"
    cfg.dataset.name = "triviaqa"
    cfg.dataset.max_new_tokens = 6
    cfg.extract.max_tokens = 6
    cfg.extract.n_cols = 4          # D_q=64 and D_kv=32 are both divisible by 4
    cfg.model.embed_dim = 16
    cfg.model.fused_dim = 16
    cfg.model.lstm_hidden = 8

    examples = [
        Example(prompt="Q: capital of France? A:", gold=["Paris"], idx=0),
        Example(prompt="Q: capital of Japan? A:", gold=["Tokyo"], idx=1),
        Example(prompt="Q: 2+2? A:", gold=["4", "four"], idx=2),
    ]

    with patch("src.extract.run_extraction.load_llm", return_value=(model, tok)), \
         patch("src.extract.run_extraction.load_examples", return_value=examples):
        run_extraction(cfg)

    root = cfg.example_dir()

    # -- extraction wrote what it should
    geom = json.loads((root / "geometry.json").read_text())
    assert geom["geometry"]["n_layers"] == 4
    assert geom["n_cols"] == 4

    arr = np.load(root / "00000" / "tokens.npy")
    assert arr.ndim == 5                      # (T, V, L, C, 3)
    assert arr.shape[1:] == (3, 4, 4, 3)
    assert arr.dtype == np.float16

    meta = parse_meta(root / "00000" / "meta.txt")
    assert meta["label"] in ("0", "1"), "extraction must leave a resolved label"

    # -- the dataset can load it
    ds = QKVImageDataset(root, views=["Q", "K", "V"])
    assert len(ds) == 3
    images, label, _ = ds[0]
    assert images.shape[1:] == (3, 3, 4, 4)   # (V, chans, L, C)

    # -- and the model can consume it
    batch_images, labels, mask, _ = collate([ds[i] for i in range(3)])
    model_out = build_model(cfg, n_views=3).eval()
    with torch.no_grad():
        logits = model_out(batch_images, mask)
    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()


def test_extraction_is_resumable(tmp_path, tiny_model):
    """A second run must skip completed examples rather than redo them."""
    from unittest.mock import patch

    from transformers import AutoTokenizer

    from src.extract.datasets import Example
    from src.extract.run_extraction import run_extraction

    try:
        tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    except Exception:
        pytest.skip("tokenizer unavailable (offline)")

    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=32000, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=4,
        head_dim=8, max_position_embeddings=128,
    )).eval()

    cfg = Config()
    cfg.data_root = str(tmp_path)
    cfg.llm.alias = "tiny"
    cfg.dataset.name = "triviaqa"
    cfg.dataset.max_new_tokens = 4
    cfg.extract.max_tokens = 4
    cfg.extract.n_cols = 4

    examples = [Example(prompt=f"Q{i}", gold=["x"], idx=i) for i in range(3)]

    with patch("src.extract.run_extraction.load_llm", return_value=(model, tok)), \
         patch("src.extract.run_extraction.load_examples", return_value=examples):
        run_extraction(cfg)
        first = np.load(cfg.example_dir() / "00000" / "tokens.npy").copy()

        # Re-run: everything is complete, so nothing should be re-extracted.
        run_extraction(cfg)
        second = np.load(cfg.example_dir() / "00000" / "tokens.npy")

    np.testing.assert_array_equal(first, second)

    # The manifest must still list all three, not be truncated to the (empty)
    # set of newly-extracted records.
    lines = (cfg.example_dir() / "manifest.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3, "resume wiped the manifest"


def test_hidden_state_transforms_extraction_end_to_end(tmp_path):
    """source=hs + extraction_type=transforms: one view (H), (raw,DWT,FFT) channels,
    and its own data folder. Exercises the full extract -> load -> model path."""
    from unittest.mock import patch

    from transformers import AutoTokenizer

    from src.data.dataset import QKVImageDataset, collate, n_images
    from src.extract.datasets import Example
    from src.extract.run_extraction import run_extraction
    from src.models.classifier import build_model

    try:
        tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    except Exception:
        pytest.skip("tokenizer unavailable (offline)")

    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=32000, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=4,
        head_dim=8, max_position_embeddings=128,
    )).eval()

    cfg = Config()
    cfg.data_root = str(tmp_path)
    cfg.llm.alias = "tiny"
    cfg.dataset.name = "triviaqa"
    cfg.dataset.max_new_tokens = 6
    cfg.extract.max_tokens = 6
    cfg.extract.source = "hs"
    cfg.extract.views = ["H"]              # single hidden-state stream
    cfg.extract.extraction_type = "transforms"
    cfg.extract.n_cols = 4                 # hidden_size=64 divisible by 4
    cfg.model.channels = "default"         # 1 view -> 1 image, 3 channels
    cfg.model.embed_dim = 16
    cfg.model.fused_dim = 16
    cfg.model.lstm_hidden = 8

    examples = [Example(prompt=f"Q{i}", gold=["x"], idx=i) for i in range(3)]

    with patch("src.extract.run_extraction.load_llm", return_value=(model, tok)), \
         patch("src.extract.run_extraction.load_examples", return_value=examples):
        run_extraction(cfg)

    root = cfg.example_dir()
    # The path must carry source + extraction_type so it never collides with qkv.
    assert root.as_posix().endswith("hs/transforms/triviaqa/tiny")

    geom = json.loads((root / "geometry.json").read_text())
    assert geom["source"] == "hs"
    assert geom["extraction_type"] == "transforms"
    assert geom["views"] == ["H"]

    arr = np.load(root / "00000" / "tokens.npy")
    assert arr.shape[1:] == (1, 4, 4, 3)   # (V=1, L=4, C=4, 3 channels)
    # Transform channels (1=DWT, 2=FFT) are magnitudes -> non-negative.
    assert (arr[..., 1] >= 0).all()
    assert (arr[..., 2] >= 0).all()

    # Load + run the model on the single-view (H) images.
    ds = QKVImageDataset(root, views=["H"], channels=cfg.model.channels)
    images, _, _ = ds[0]
    assert images.shape[1:] == (1, 3, 4, 4)   # (V=1, chans=3, L, C)

    n_streams = n_images(cfg.model.channels, 1)
    batch_images, labels, mask, _ = collate([ds[i] for i in range(3)])
    model_out = build_model(cfg, n_views=n_streams).eval()
    with torch.no_grad():
        logits = model_out(batch_images, mask)
    assert logits.shape == (3,) and torch.isfinite(logits).all()


def test_reuses_already_generated_examples_without_loading_the_model(tmp_path):
    """A run whose generation finished but crashed before labeling must resume at
    the post-generation step: label + write the manifest from the on-disk tensors
    and meta.txt, WITHOUT ever loading the LLM (the expensive part is done)."""
    from unittest.mock import patch

    from src.extract.datasets import Example
    from src.extract.run_extraction import run_extraction

    cfg = Config()
    cfg.data_root = str(tmp_path)
    cfg.llm.alias = "tiny"
    cfg.dataset.name = "triviaqa"
    cfg.labeling.scheme = "exact_match"
    root = cfg.example_dir()
    root.mkdir(parents=True)

    # Pre-generate: tensor + meta (label -1), NO manifest -- exactly the state a
    # crash-after-generation leaves behind.
    golds = ["['Paris']", "['Tokyo']", "['4', 'four']"]
    responses = ["Paris.", "Kyoto.", "4"]           # #1 is wrong -> hallucinated
    for i, (g, r) in enumerate(zip(golds, responses)):
        d = root / f"{i:05d}"
        d.mkdir()
        np.save(d / "tokens.npy", np.zeros((2, 3, 4, 4, 3), np.float16))
        (d / "meta.txt").write_text(
            f"prompt: p{i}\nresponse: {r}\ngold: {g}\nscore: nan\nlabel: -1\n"
        )
    (root / "geometry.json").write_text(json.dumps({
        "views": ["Q", "K", "V"], "n_rows": 4, "n_cols": 4,
        "source": "qkv", "extraction_type": "delta", "geometry": {"n_layers": 4},
    }))

    examples = [Example(prompt=f"p{i}", gold=eval(g), idx=i)
                for i, g in enumerate(golds)]

    def fail_if_loaded(_cfg):
        raise AssertionError("load_llm must NOT be called when all examples exist")

    with patch("src.extract.run_extraction.load_llm", side_effect=fail_if_loaded), \
         patch("src.extract.run_extraction.load_examples", return_value=examples):
        run_extraction(cfg)   # must not raise -> model never loaded

    # Manifest now exists with all three, and every example is labeled.
    lines = (root / "manifest.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3
    recs = sorted((json.loads(l) for l in lines), key=lambda r: r["idx"])
    labels = [r["label"] for r in recs]
    assert labels == [0, 1, 0], labels    # "Kyoto." for Tokyo is the hallucination
    for r in recs:
        assert r["label"] in (0, 1)
        assert r["n_tokens"] == 2


# --------------------------------------------------------------------------
# build_prompt_ids -- transformers v5 changed apply_chat_template's return type
# --------------------------------------------------------------------------

def test_build_prompt_ids_always_returns_a_2d_tensor():
    """apply_chat_template returns a BatchEncoding in transformers v5 but a bare
    tensor in v4. Either way capture_qkv needs a (1, prompt_len) LongTensor --
    handing it a dict fails later and confusingly, at `.ndim`.
    """
    from transformers import BatchEncoding

    from src.extract.run_extraction import build_prompt_ids

    class TokReturnsTensor:
        chat_template = "x"

        def apply_chat_template(self, msgs, return_tensors=None, add_generation_prompt=None):
            return torch.tensor([[1, 2, 3]])

    class TokReturnsBatchEncoding:
        chat_template = "x"

        def apply_chat_template(self, msgs, return_tensors=None, add_generation_prompt=None):
            return BatchEncoding({"input_ids": torch.tensor([[1, 2, 3]])})

    class TokReturns1D:
        chat_template = "x"

        def apply_chat_template(self, msgs, return_tensors=None, add_generation_prompt=None):
            return torch.tensor([1, 2, 3])          # no batch axis

    for tok in (TokReturnsTensor(), TokReturnsBatchEncoding(), TokReturns1D()):
        ids = build_prompt_ids("hi", tok, "meta-llama/Meta-Llama-3-8B-Instruct", "cpu")
        assert isinstance(ids, torch.Tensor), type(tok).__name__
        assert ids.ndim == 2 and ids.shape[0] == 1, type(tok).__name__


def test_build_prompt_ids_uses_raw_text_for_base_models():
    from src.extract.run_extraction import build_prompt_ids

    called = {}

    class Tok:
        chat_template = "x"

        def __call__(self, prompt, return_tensors=None):
            called["raw"] = True
            return {"input_ids": torch.tensor([[5, 6]])}

        def apply_chat_template(self, *a, **kw):
            raise AssertionError("a base model must not use the chat template")

    ids = build_prompt_ids("hi", Tok(), "meta-llama/Llama-2-7b-hf", "cpu")
    assert called.get("raw")
    assert ids.shape == (1, 2)


def test_instruct_model_without_a_chat_template_falls_back_to_raw_text():
    """Do not crash just because the name says 'instruct' but no template exists."""
    from src.extract.run_extraction import build_prompt_ids

    class Tok:
        chat_template = None

        def __call__(self, prompt, return_tensors=None):
            return {"input_ids": torch.tensor([[7]])}

    ids = build_prompt_ids("hi", Tok(), "some-instruct-model", "cpu")
    assert ids.shape == (1, 1)
