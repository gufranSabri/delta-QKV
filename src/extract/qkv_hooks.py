"""Capture per-layer Q, K and V activations from a HuggingFace causal LM.

HuggingFace exposes hidden states (`output_hidden_states=True`) and attention
probabilities (`output_attentions=True`) for free, but gives you NOTHING for the
Q/K/V projections -- they are internal to the attention module's forward(). This
module fills that gap with forward hooks on the projection Linears.

WHAT THE HOOK SEES
------------------
In every RoPE-family decoder (Llama, Mistral, Qwen, ...) the attention forward
begins:

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states   = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, ...)

so a forward hook on `q_proj` / `k_proj` / `v_proj` yields the projection output
BEFORE the head reshape and BEFORE RoPE, with shape (B, seq, n_heads*head_dim).

This is deliberate. We capture PRE-RoPE Q/K:
  - RoPE is a position-dependent rotation. Post-RoPE Q/K would entangle the
    token's *content* with its *absolute position*, so identical content at
    different positions would produce different images. We want content.
  - V never receives RoPE at all, so for V the distinction does not exist and
    pre/post capture are identical.

GROUPED-QUERY ATTENTION (GQA)
-----------------------------
K and V are NARROWER than Q in every model we use:

    Llama-3-8B    D_q = 4096 (32 heads x 128)   D_kv = 1024 (8 kv-heads x 128)
    Mistral-7B    D_q = 4096 (32 heads x 128)   D_kv = 1024 (8 kv-heads x 128)
    Qwen2.5-7B    D_q = 3584 (28 heads x 128)   D_kv =  512 (4 kv-heads x 128)

The image builder pools each view's D down to the same number of columns, so the
resulting images are the same size across views -- only the pooling chunk width
differs. Never hardcode any of these numbers; read them from `model.config`.

GENERATION SEMANTICS
--------------------
With a KV cache active, `model.generate` calls each attention module:
  - once during PREFILL, with seq = prompt_len   (the prompt's Q/K/V)
  - once per DECODE step, with seq = 1           (the new token's Q/K/V)

We tag each hook firing with a step counter and keep only the decode steps, so
we naturally end up with exactly one Q/K/V vector per layer per GENERATED token
-- which is precisely the (T, L, D) tensor we want. The prefill activations are
the prompt's, not the response's, and are discarded.

CAVEAT: this assumes `use_cache=True` (the default). With `use_cache=False` every
decode step re-runs the full sequence and the hook fires with seq = prompt+t, so
we would have to slice out the last position. `capture_qkv` asserts on this
rather than silently producing garbage.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.nn as nn

VIEWS = ("Q", "K", "V")
_PROJ_FOR_VIEW = {"Q": "q_proj", "K": "k_proj", "V": "v_proj"}


@dataclass
class ModelGeometry:
    """The per-model numbers the image builder needs. Read, never assumed."""

    n_layers: int
    hidden_size: int
    n_heads: int
    n_kv_heads: int
    head_dim: int

    @property
    def d_q(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def d_kv(self) -> int:
        return self.n_kv_heads * self.head_dim

    def feature_dim(self, view: str) -> int:
        return self.d_q if view == "Q" else self.d_kv

    def __str__(self) -> str:
        return (
            f"L={self.n_layers} hidden={self.hidden_size} "
            f"heads={self.n_heads} kv_heads={self.n_kv_heads} "
            f"head_dim={self.head_dim} D_q={self.d_q} D_kv={self.d_kv}"
        )

    def valid_n_cols(self, views=VIEWS) -> list[int]:
        """Column counts that evenly divide EVERY requested view's feature dim.

        Q and K/V have different widths under GQA, so a legal n_cols must divide
        their GCD. Notably Qwen2.5-7B (L=28, D_kv=512) has NO valid n_cols equal
        to its layer count -- 28 does not divide 512 -- so it cannot use the
        default square image and must set extract.n_cols explicitly.
        """
        from math import gcd

        g = 0
        for view in views:
            g = gcd(g, self.feature_dim(view))
        return [d for d in range(1, g + 1) if g % d == 0]

    def check_n_cols(self, n_cols: int, views=VIEWS) -> None:
        """Raise with an actionable message if n_cols cannot pool every view."""
        bad = [
            (v, self.feature_dim(v))
            for v in views
            if self.feature_dim(v) % n_cols != 0
        ]
        if not bad:
            return

        valid = self.valid_n_cols(views)
        nearby = sorted(valid, key=lambda d: abs(d - n_cols))[:6]
        detail = ", ".join(f"{v} has D={d}" for v, d in bad)
        raise ValueError(
            f"n_cols={n_cols} does not evenly divide the feature dim of: {detail}. "
            f"This model has L={self.n_layers} layers, D_q={self.d_q}, D_kv={self.d_kv}. "
            f"Set extract.n_cols to one of {sorted(nearby)} in your config "
            f"(closest valid choices to {n_cols})."
        )


def read_geometry(model) -> ModelGeometry:
    """Extract layer/head geometry from a model config, with GQA fallbacks."""
    cfg = model.config
    # Some configs nest the text config (multimodal wrappers).
    cfg = getattr(cfg, "text_config", cfg)

    n_heads = cfg.num_attention_heads
    hidden = cfg.hidden_size
    # head_dim is explicit on newer configs; otherwise it's the classic ratio.
    head_dim = getattr(cfg, "head_dim", None) or hidden // n_heads
    # Models without GQA omit num_key_value_heads; then kv heads == q heads.
    n_kv = getattr(cfg, "num_key_value_heads", None) or n_heads

    return ModelGeometry(
        n_layers=cfg.num_hidden_layers,
        hidden_size=hidden,
        n_heads=n_heads,
        n_kv_heads=n_kv,
        head_dim=head_dim,
    )


def get_decoder_layers(model) -> nn.ModuleList:
    """Locate the ModuleList of decoder layers across model wrappers."""
    for path in ("model.layers", "model.model.layers", "transformer.h"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if isinstance(obj, (nn.ModuleList, list)):
            return obj
    raise AttributeError(
        "could not locate decoder layers on this model; expected one of "
        "model.layers / model.model.layers / transformer.h"
    )


@dataclass
class QKVCapture:
    """Accumulates hook firings, keyed by (view, layer), split by decode step."""

    n_layers: int
    views: tuple[str, ...]
    # steps[view][layer] -> list of (D,) tensors, one per generation step captured
    steps: dict = field(default_factory=dict)
    _recording: bool = False

    def __post_init__(self):
        self.steps = {v: [[] for _ in range(self.n_layers)] for v in self.views}

    def record(self, view: str, layer: int, out: torch.Tensor) -> None:
        """out: (B, seq, D) -- the raw projection output for this firing."""
        if not self._recording:
            return
        if out.shape[0] != 1:
            raise NotImplementedError(
                f"batch size must be 1 during extraction, got {out.shape[0]}. "
                "Per-example token counts differ, so batching would require "
                "unpadding logic we deliberately avoid here."
            )
        # Detach immediately and move off the accelerator: keeping these on GPU
        # across a 100-token generation is what blows up memory.
        self.steps[view][layer].append(out[0].detach().to("cpu", torch.float32))

    def stack(self) -> dict[str, torch.Tensor]:
        """Concatenate captured steps into (T, L, D) per view.

        Every firing is concatenated along the sequence axis, so this works
        whether the generation produced one (seq=1) firing per step or a single
        multi-token firing.
        """
        out: dict[str, torch.Tensor] = {}
        for view in self.views:
            per_layer = []
            for layer in range(self.n_layers):
                chunks = self.steps[view][layer]
                if not chunks:
                    raise RuntimeError(
                        f"no activations captured for view {view} layer {layer}; "
                        "did the hooks fire? (is the model actually running?)"
                    )
                per_layer.append(torch.cat(chunks, dim=0))  # (T, D)

            n_tok = {t.shape[0] for t in per_layer}
            if len(n_tok) != 1:
                raise RuntimeError(
                    f"view {view}: layers disagree on token count: {sorted(n_tok)}. "
                    "This means some layers fired more often than others."
                )
            out[view] = torch.stack(per_layer, dim=1)  # (T, L, D)
        return out


@contextmanager
def qkv_hooks(model, views=VIEWS):
    """Attach Q/K/V projection hooks for the duration of the context.

    Yields the QKVCapture. Recording is OFF on entry -- call `capture.start()`
    (via the `_recording` flag, which `capture_qkv` manages) to begin, so that a
    prefill pass can be run without being recorded.

    Hooks are always removed on exit, including on exception.
    """
    bad = set(views) - set(VIEWS)
    if bad:
        raise ValueError(f"unknown views {sorted(bad)}; valid views are {VIEWS}")

    geom = read_geometry(model)
    layers = get_decoder_layers(model)
    if len(layers) != geom.n_layers:
        raise RuntimeError(
            f"config says {geom.n_layers} layers but found {len(layers)} modules"
        )

    capture = QKVCapture(n_layers=geom.n_layers, views=tuple(views))
    handles = []

    def make_hook(view: str, layer_idx: int):
        def hook(_module, _inputs, output):
            capture.record(view, layer_idx, output)
        return hook

    try:
        for layer_idx, layer in enumerate(layers):
            attn = layer.self_attn
            for view in views:
                proj = getattr(attn, _PROJ_FOR_VIEW[view])
                handles.append(proj.register_forward_hook(make_hook(view, layer_idx)))
        yield capture
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def capture_qkv(
    model,
    input_ids: torch.Tensor,
    views=VIEWS,
    max_new_tokens: int = 100,
    eos_token_id=None,
    pad_token_id=None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Greedy-generate and capture Q/K/V for the GENERATED tokens only.

    Args:
        input_ids: (1, prompt_len) prompt token ids, already on the model device.

    Returns:
        (qkv, generated_ids) where
          qkv["Q"] is (T, L, D_q), qkv["K"] and qkv["V"] are (T, L, D_kv),
          generated_ids is (T,) -- the response tokens, prompt excluded.
        T is the number of generated tokens (<= max_new_tokens).

    Only decode-step activations are recorded. The prefill pass -- which carries
    the PROMPT's Q/K/V, not the response's -- is explicitly excluded by leaving
    recording off until after the prompt has been consumed.

    We drive the decode loop by hand rather than calling `model.generate`,
    because generate() gives us no reliable hook to distinguish prefill from
    decode; doing it manually makes the prefill/decode boundary explicit and
    unambiguous, which is exactly the thing that would otherwise silently
    corrupt every tensor we produce.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"expected input_ids (1, prompt_len), got {tuple(input_ids.shape)}")

    device = input_ids.device
    if eos_token_id is None:
        eos_token_id = model.config.eos_token_id
    # eos may be a list in some configs (e.g. Llama-3 has <|eot_id|>).
    eos_ids = set()
    if eos_token_id is not None:
        eos_ids = set(eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id])

    with qkv_hooks(model, views=views) as capture:
        # ---- PREFILL: consume the prompt. NOT recorded. ----
        capture._recording = False
        out = model(input_ids=input_ids, use_cache=True)
        past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1, 1)

        # ---- DECODE: one token at a time. Recorded. ----
        # The token fed at step t is the token GENERATED at step t-1 (or the
        # argmax of the prompt's last logit, for t=0). So the Q/K/V captured at
        # step t are exactly the activations the model computed *for* that
        # generated token -- which is the correspondence we want between an
        # image and a response token.
        capture._recording = True
        generated = []

        for _ in range(max_new_tokens):
            token_id = int(next_token.item())
            if token_id in eos_ids:
                break
            generated.append(token_id)

            out = model(input_ids=next_token, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        capture._recording = False

    if not generated:
        # Model emitted EOS immediately: no response tokens, so no images.
        return {v: torch.empty(0) for v in views}, torch.empty(0, dtype=torch.long)

    qkv = capture.stack()

    n_gen = len(generated)
    for view, tensor in qkv.items():
        if tensor.shape[0] != n_gen:
            raise RuntimeError(
                f"view {view}: captured {tensor.shape[0]} token activations but "
                f"generated {n_gen} tokens. The prefill/decode split is wrong."
            )

    return qkv, torch.tensor(generated, dtype=torch.long, device=device)
