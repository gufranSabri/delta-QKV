"""Gate tests for Q/K/V capture.

The single most important test in this repo is
`test_captured_q_equals_manual_matmul`: if the hook does not actually return
q_proj(hidden_state), then every image, every model and every number downstream
is meaningless. Everything else is built on this.

Uses a tiny randomly-initialised Llama so the tests run on CPU in seconds and
need no network access or HF auth.
"""

import pytest
import torch

from src.extract.qkv_hooks import (
    capture_qkv,
    get_decoder_layers,
    qkv_hooks,
    read_geometry,
)

transformers = pytest.importorskip("transformers")
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402


# A deliberately GQA-shaped tiny model: 4 query heads but only 2 kv heads, so
# D_q (4*8=32) != D_kv (2*8=16). Any code that assumes Q/K/V share a feature
# dim will fail here -- which is the point, since all three real LLMs use GQA.
TINY = dict(
    vocab_size=64,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    max_position_embeddings=64,
)


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(**TINY))
    model.eval()
    return model


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def test_read_geometry_reports_gqa_dims(tiny_model):
    g = read_geometry(tiny_model)
    assert g.n_layers == 3
    assert g.n_heads == 4
    assert g.n_kv_heads == 2
    assert g.head_dim == 8
    assert g.d_q == 32          # 4 heads x 8
    assert g.d_kv == 16         # 2 kv-heads x 8  -- NARROWER, this is GQA
    assert g.feature_dim("Q") == 32
    assert g.feature_dim("K") == g.feature_dim("V") == 16


def test_geometry_matches_the_actual_projection_weights(tiny_model):
    """Guards against reading the config wrong: compare against real Linears."""
    g = read_geometry(tiny_model)
    attn = get_decoder_layers(tiny_model)[0].self_attn
    assert attn.q_proj.out_features == g.d_q
    assert attn.k_proj.out_features == g.d_kv
    assert attn.v_proj.out_features == g.d_kv


def test_get_decoder_layers_finds_all_layers(tiny_model):
    assert len(get_decoder_layers(tiny_model)) == 3


# --------------------------------------------------------------------------
# THE GATE: is what we capture actually Q, K and V?
# --------------------------------------------------------------------------

def test_captured_qkv_equals_manual_matmul(tiny_model):
    """Hook output must equal hidden_state @ W.T + b, computed independently.

    This is the load-bearing test of the project. We take the hidden states that
    HF gives us for free, multiply them by the projection weights BY HAND, and
    assert the hooks captured exactly that.
    """
    torch.manual_seed(1)
    ids = torch.randint(0, TINY["vocab_size"], (1, 7))

    # Run a plain forward, capturing hooks AND hidden states together.
    with qkv_hooks(tiny_model) as cap:
        cap._recording = True
        out = tiny_model(input_ids=ids, output_hidden_states=True, use_cache=False)
        cap._recording = False

    captured = cap.stack()  # (T=7, L=3, D) per view

    layers = get_decoder_layers(tiny_model)
    for layer_idx, layer in enumerate(layers):
        attn = layer.self_attn
        # hidden_states[i] is the INPUT to layer i (hidden_states[0] = embeddings).
        # The attention block sees it only after the input layernorm, so that is
        # what actually reaches q_proj -- not the raw residual stream.
        h_in = out.hidden_states[layer_idx]              # (1, 7, hidden)
        h_normed = layer.input_layernorm(h_in)           # (1, 7, hidden)

        for view, proj in (("Q", attn.q_proj), ("K", attn.k_proj), ("V", attn.v_proj)):
            expected = torch.nn.functional.linear(h_normed, proj.weight, proj.bias)
            expected = expected[0]                       # (7, D)
            got = captured[view][:, layer_idx, :]        # (7, D)
            assert got.shape == expected.shape, f"{view} L{layer_idx} shape"
            torch.testing.assert_close(
                got, expected, rtol=1e-4, atol=1e-5,
                msg=f"captured {view} at layer {layer_idx} != manual q_proj(hidden)",
            )


def test_capture_shapes_respect_gqa(tiny_model):
    """Q must be wider than K/V -- proving we capture pre-reshape projections."""
    torch.manual_seed(2)
    ids = torch.randint(0, TINY["vocab_size"], (1, 5))
    with qkv_hooks(tiny_model) as cap:
        cap._recording = True
        tiny_model(input_ids=ids, use_cache=False)
        cap._recording = False
    got = cap.stack()
    assert got["Q"].shape == (5, 3, 32)   # (T, L, D_q)
    assert got["K"].shape == (5, 3, 16)   # (T, L, D_kv)  -- half as wide
    assert got["V"].shape == (5, 3, 16)


def test_hooks_are_removed_on_exit(tiny_model):
    attn = get_decoder_layers(tiny_model)[0].self_attn
    before = len(attn.q_proj._forward_hooks)
    with qkv_hooks(tiny_model):
        during = len(attn.q_proj._forward_hooks)
    after = len(attn.q_proj._forward_hooks)
    assert during == before + 1
    assert after == before, "hooks leaked -- they must be removed on context exit"


def test_hooks_are_removed_even_on_exception(tiny_model):
    attn = get_decoder_layers(tiny_model)[0].self_attn
    before = len(attn.q_proj._forward_hooks)
    with pytest.raises(RuntimeError, match="boom"):
        with qkv_hooks(tiny_model):
            raise RuntimeError("boom")
    assert len(attn.q_proj._forward_hooks) == before


def test_subset_of_views_hooks_only_those(tiny_model):
    torch.manual_seed(3)
    ids = torch.randint(0, TINY["vocab_size"], (1, 4))
    with qkv_hooks(tiny_model, views=("V",)) as cap:
        cap._recording = True
        tiny_model(input_ids=ids, use_cache=False)
        cap._recording = False
    got = cap.stack()
    assert set(got) == {"V"}


def test_unknown_view_rejected(tiny_model):
    with pytest.raises(ValueError, match="unknown views"):
        with qkv_hooks(tiny_model, views=("Q", "Z")):
            pass


def test_recording_off_captures_nothing(tiny_model):
    """The prefill-exclusion mechanism must actually work."""
    torch.manual_seed(4)
    ids = torch.randint(0, TINY["vocab_size"], (1, 4))
    with qkv_hooks(tiny_model) as cap:
        cap._recording = False
        tiny_model(input_ids=ids, use_cache=False)
        with pytest.raises(RuntimeError, match="no activations captured"):
            cap.stack()


# --------------------------------------------------------------------------
# capture_qkv: the generation loop, and the prefill/decode boundary
# --------------------------------------------------------------------------

def test_capture_qkv_returns_one_activation_per_generated_token(tiny_model):
    torch.manual_seed(5)
    ids = torch.randint(0, TINY["vocab_size"], (1, 6))
    qkv, gen = capture_qkv(tiny_model, ids, max_new_tokens=9, eos_token_id=[])

    assert gen.shape == (9,), "should have generated exactly max_new_tokens"
    for view in ("Q", "K", "V"):
        # T must equal the number of GENERATED tokens -- NOT prompt+generated.
        assert qkv[view].shape[0] == 9, f"{view}: prompt activations leaked in"
        assert qkv[view].shape[1] == 3  # layers
    assert qkv["Q"].shape[2] == 32
    assert qkv["K"].shape[2] == 16


def test_capture_qkv_excludes_the_prompt(tiny_model):
    """A longer prompt must not change T. This is the prefill-leak canary."""
    torch.manual_seed(6)
    short = torch.randint(0, TINY["vocab_size"], (1, 3))
    long = torch.randint(0, TINY["vocab_size"], (1, 20))

    qkv_s, _ = capture_qkv(tiny_model, short, max_new_tokens=5, eos_token_id=[])
    qkv_l, _ = capture_qkv(tiny_model, long, max_new_tokens=5, eos_token_id=[])

    assert qkv_s["Q"].shape[0] == 5
    assert qkv_l["Q"].shape[0] == 5, (
        "T grew with prompt length -- prefill activations are leaking into the "
        "capture, which would make every image partly the PROMPT's activations"
    )


def test_capture_qkv_decode_matches_a_full_teacher_forced_forward(tiny_model):
    """The strongest end-to-end check.

    Generate with the incremental KV-cache decode loop, then re-run the FULL
    (prompt + generated) sequence in one teacher-forced forward pass with no
    cache, and assert the Q/K/V we captured incrementally match the positions
    they should correspond to.

    This simultaneously validates: the cache is being used correctly, the
    prefill/decode split is right, and the token<->activation alignment is
    off-by-none.
    """
    torch.manual_seed(7)
    prompt = torch.randint(0, TINY["vocab_size"], (1, 5))
    qkv, gen = capture_qkv(tiny_model, prompt, max_new_tokens=6, eos_token_id=[])

    # Teacher-force the whole thing in one pass, no cache, no hooks games.
    full = torch.cat([prompt, gen.view(1, -1)], dim=1)  # (1, 5+6)
    with qkv_hooks(tiny_model) as cap:
        cap._recording = True
        tiny_model(input_ids=full, use_cache=False)
        cap._recording = False
    full_qkv = cap.stack()  # (11, L, D)

    # The activations for generated token i were computed when that token was
    # the model's INPUT. In `full`, generated token i sits at position 5+i.
    prompt_len = prompt.shape[1]
    for view in ("Q", "K", "V"):
        expected = full_qkv[view][prompt_len : prompt_len + 6]
        torch.testing.assert_close(
            qkv[view], expected, rtol=1e-3, atol=1e-4,
            msg=f"{view}: incremental decode capture disagrees with a full "
                f"teacher-forced forward -- token/activation alignment is off",
        )


def test_capture_qkv_stops_at_eos(tiny_model):
    """EOS must truncate, and images must still align with the kept tokens."""
    torch.manual_seed(8)
    ids = torch.randint(0, TINY["vocab_size"], (1, 4))
    # First, see what it generates unconstrained.
    _, gen = capture_qkv(tiny_model, ids, max_new_tokens=8, eos_token_id=[])
    # Now declare its 3rd token to be EOS; generation must stop before emitting it.
    eos = int(gen[3])
    qkv, gen2 = capture_qkv(tiny_model, ids, max_new_tokens=8, eos_token_id=[eos])

    assert len(gen2) == 3, "should have stopped just before the EOS token"
    assert qkv["Q"].shape[0] == 3, "images must match the truncated token count"
    assert torch.equal(gen2, gen[:3])


def test_capture_qkv_rejects_batched_input(tiny_model):
    ids = torch.randint(0, TINY["vocab_size"], (2, 4))
    with pytest.raises(ValueError, match=r"expected input_ids \(1, prompt_len\)"):
        capture_qkv(tiny_model, ids, max_new_tokens=2)


def test_capture_qkv_handles_immediate_eos(tiny_model):
    """If the model emits EOS as its very first token there are no images."""
    torch.manual_seed(9)
    ids = torch.randint(0, TINY["vocab_size"], (1, 4))
    _, gen = capture_qkv(tiny_model, ids, max_new_tokens=4, eos_token_id=[])
    first = int(gen[0])
    qkv, gen2 = capture_qkv(tiny_model, ids, max_new_tokens=4, eos_token_id=[first])
    assert len(gen2) == 0
    assert qkv["Q"].numel() == 0
