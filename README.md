# delta-QKV

**Hallucination detection by rendering an LLM's per-token Q/K/V activations as images.**

For each token an LLM generates, we build three small images — one from the **Q**
projections, one from **K**, one from **V** — where rows are transformer layers, columns
are pooled feature groups, and the three colour channels are `(raw activation, delta to the
previous layer, delta to the next layer)`. Each view gets its own CNN; a learned fusion
module combines them; a BiLSTM reads the resulting sequence of token embeddings and predicts
whether the response is a hallucination.

```
                per-token, three separate 3-channel images
       Q (32×32×3)        K (32×32×3)        V (32×32×3)
            │                  │                  │
       [ CNN_Q ]          [ CNN_K ]          [ CNN_V ]     ← untied weights, shared across tokens
            │                  │                  │
          e_Q (E)            e_K (E)            e_V (E)
            └──────────────┬──┴──────────────────┘
                    [ Fusion ]                             ← gated | concat_mlp | bilinear | cross_attn
                           │
                 stack over tokens → (T, F)
                           │
              [ Conv1d ×2 ] → [ BiLSTM ] → [ masked attention-pool ]
                           │
                    p(hallucination)
```

---

## Why this design

Two published baselines live in this repo for comparison:

| | What it uses | What it does with it |
|---|---|---|
| **HalluShift** (IJCNN'25) | hidden states + attention probs | collapses a whole response into ~40 scalars (Wasserstein/cosine between layers) → MLP |
| **ACT-ViT** (NeurIPS'25) | hidden states only | keeps the full `(L, N, D)` activation tensor as **one** image → ViT |
| **delta-QKV** (this repo) | **Q, K and V** | one image **per token per view**, layer-deltas as channels, per-view CNNs + learned fusion → BiLSTM |

Three things are different here:

1. **Q/K/V instead of hidden states.** Neither baseline looks at the attention projections
   themselves — only at what attention produced. Q, K and V are three distinct views of what
   the attention mechanism is actually *doing*.
2. **Layer-deltas as spatial channels.** HalluShift's insight is that hallucination shows up
   as a *distribution shift between layers*, but it averages that shift into a scalar. We keep
   it as a channel of the image, so the CNN can see *where* in the layer stack the shift happens.
3. **A sequence of per-token images, not one flattened image.** ACT-ViT collapses the token
   axis into an image dimension. We keep it as time, so the model can learn *where in the
   response* the hallucination emerges.

### Q, K and V are never channel-stacked

The obvious shortcut — concatenate Q, K and V into one 9-channel image — is deliberately
rejected. A 9-channel conv lets the very first kernel blend the three views, so by layer one
there is no longer any such thing as "the Q representation". That destroys the premise of the
method and makes it impossible to ask which view carries the signal.

Keeping them separate means the fusion weights are **explicit and inspectable**. With
`fusion: gated`, training reports a softmax distribution over views:

```
view gates (mean softmax weight): Q=0.375, K=0.108, V=0.518
```

That line is a result, not a diagnostic. "The model relies mostly on V" is a claim you can
make and defend; a 9-channel conv gives you no such statement.

### The images are square by construction

A token's activation vector has `D = 4096` dimensions, but the image needs a manageable width.
We **max-pool D down to L** (the layer count) over contiguous chunks, so a 32-layer model
yields a 32×32 image with **no reshaping and no interpolation artifact**. This drops storage
from ~2.3 TB to ~18 GB per (dataset × LLM) pair.

A pleasant accident for Llama-3-8B: `D=4096` with 32 heads of 128 dims means the contiguous
chunks of 128 *coincide with attention heads*, so column *j* is literally "the peak activation
of head *j*". This does **not** hold for Qwen — do not over-claim it.

---

## Install

```bash
pip install -r requirements.txt
```

BLEURT labeling is optional and needs a separate TensorFlow install (see
`src/label/bleurt.py`). The default `exact_match` scheme requires nothing extra.

---

## Quickstart

```bash
# 1. Generate responses, capture Q/K/V, build and save the token images.
python main.py --config configs/triviaqa/llama3_8b.yaml extract

# 2. Look at what you produced BEFORE training on it.
python main.py --config configs/triviaqa/llama3_8b.yaml inspect --idx 0

# 3. Train.
python main.py --config configs/triviaqa/llama3_8b.yaml train

# 4. Evaluate a checkpoint.
python main.py --config configs/triviaqa/llama3_8b.yaml test \
  --checkpoint runs/<run>/best.pt
```

Extraction is the expensive step (one `generate()` per example). It is **restartable** — an
example that is already complete is skipped — and **chunkable** for splitting across machines:

```bash
python main.py --config configs/triviaqa/llama3_8b.yaml extract --chunk 3   # examples 2000–2999
```

### `inspect` first, always

`inspect` prints a cross-view correlation matrix and renders the token images to PNG:

```
  cross-view correlation of the raw channel (flattened):
           Q       K       V
    Q   1.000   0.412   0.087
    K   0.412   1.000   0.103
    V   0.087   0.103   1.000
```

If those off-diagonal numbers came back near 1.0, Q, K and V would be redundant and the whole
separate-CNN + fusion design would be buying nothing. **Check this on day one**, not after a
week of training runs.

---

## Configuration

`configs/default.yaml` holds every setting; each `configs/{dataset}/{llm}.yaml` overrides only
what differs. Any key can be overridden from the command line, on either side of the subcommand:

```bash
python main.py --config configs/triviaqa/llama3_8b.yaml train \
  --set model.fusion=bilinear \
  --set extract.views='[V]' \
  --set train.epochs=50
```

The knobs that matter:

| Key | Options | Notes |
|---|---|---|
| `extract.views` | any subset of `[Q, K, V]` | A single view bypasses fusion entirely. This is the ablation that asks *which view carries the signal*. |
| `extract.pool` | `max` (default), `mean`, `l2` | How `D` is reduced to columns. `max` follows ACT-ViT. |
| `extract.boundary_mode` | `zero` (default), `replicate`, `wrap` | What to do at layer 0 (no previous) and layer L-1 (no next). See below. |
| `model.fusion` | `gated` (default), `concat_mlp`, `bilinear`, `cross_attn` | `concat_mlp` is the honest control. |
| `model.share_backbone` | `false` (default) | `true` ties one CNN across all views. |
| `model.backbone` | `scratch_cnn` (default), `resnet18` | |
| `labeling.scheme` | `exact_match` (default), `bleurt` | ACT-ViT's protocol vs. HalluShift's. |

### Boundary handling

Layer 0 has no previous layer and layer L-1 has no next one, so two delta values are undefined.
The default is `zero` (the delta is 0 — "no delta exists here"). `wrap` — where layer 0's
"previous" is the *last* layer — is available but **not** the default: the embedding output and
the final layer are very far apart, so the wrapped rows carry a much larger magnitude than any
genuine adjacent-layer delta, and a CNN will happily latch onto those two outlier rows.

### Relabeling is free

`meta.txt` stores each response and its gold answer alongside the tensor, so switching labeling
schemes never requires re-extracting features:

```bash
python main.py --config configs/triviaqa/llama3_8b.yaml label --set labeling.scheme=bleurt
```

---

## Experiments

```bash
python scripts/gen_experiment_scripts.py --mode all
bash scripts/generated/lodo/run_all.sh
```

Three settings are generated:

- **`same`** — train and test on one dataset.
- **`lodo`** — leave-one-dataset-out: train on the other N−1 datasets, evaluate **zero-shot**
  on the held-out one.
- **`ablate`** — the tables that decide whether the architecture's claims hold: view subsets,
  fusion variants, boundary modes, backbone choices.

Or drive it directly:

```bash
python main.py --config configs/triviaqa/llama3_8b.yaml train \
  --train-datasets hotpotqa,imdb,movies \
  --test-dataset triviaqa            # never seen during training
```

### Cross-LLM training

Llama has 32 layers, Qwen has 28 — so their images are 32×32 and 28×32, and one CNN cannot
consume both. Set `extract.l_eff` to a common value for every LLM and re-extract. Same-LLM
experiments are unaffected. `ConcatQKVDataset` refuses to combine mismatched sources rather
than failing later and mysteriously.

---

## Repo layout

```
main.py                     single entry point: extract | label | train | test | inspect
configs/                    default.yaml + one file per (dataset × LLM)
src/
  extract/
    tensor_ops.py           pooling, delta channels, boundary modes  (PURE, no model needed)
    qkv_hooks.py            forward hooks on q_proj/k_proj/v_proj    ← the load-bearing part
    run_extraction.py       generate → capture → build images → save
    datasets.py             prompt templates + gold-answer extraction
  label/
    exact_match.py          ACT-ViT's protocol (substring / sentiment match)
    bleurt.py               HalluShift's protocol (BLEURT-20-D12 @ 0.5)
  models/
    backbones.py            ScratchCNN, ResNet18Adapted → a per-view token embedding
    fusion.py               gated | concat_mlp | bilinear | cross_attn
    temporal.py             Conv1d + BiLSTM + masked attention pooling
    classifier.py           the full detector
  data/dataset.py           dataset, per-view normalisation, padding + masking
  train.py / test.py
tests/                      162 tests
```

On disk, extraction produces:

```
data/{dataset}/{llm}/
  00000/
    tokens.npy    (T, V, L, C, 3) float16   ← (tokens, views, layers, cols, channels)
    meta.txt      prompt / response / gold / score / label
  manifest.jsonl  one line per example — the training index
  geometry.json   the model geometry the images were built with
```

The **view axis is a real axis**, never folded into channels. That is what makes dropping a
view (`extract.views: [Q]`) a pure slicing operation with no re-extraction.

---

## Implementation notes

Things that are easy to get wrong, and how they are handled here.

**Q/K/V capture is the load-bearing part.** HuggingFace gives you `hidden_states` and
`attentions` for free but *nothing* for Q/K/V — they are internal to the attention module.
We hook the `q_proj`/`k_proj`/`v_proj` Linears, which yields the projections **pre-RoPE**
(RoPE is a position-dependent rotation; post-RoPE activations would entangle token *content*
with token *position*, and V never receives RoPE at all). `tests/test_qkv_hooks.py` proves the
captured tensors equal `q_proj(input_layernorm(hidden))` computed by hand, and that an
incremental decode matches a full teacher-forced forward.

**Grouped-query attention makes K and V narrower than Q.** Llama-3 has 32 query heads but only
8 KV heads, so `D_q = 4096` while `D_kv = 1024`. Every dimension is read from `model.config` at
runtime; nothing is hardcoded. Both views still pool to the same number of columns — only the
chunk width differs.

**Qwen2.5-7B cannot use the default square image.** It has `L=28` layers but `D_kv=512`, and 28
does not divide 512 — so `n_cols = L` is impossible and the Qwen configs set `n_cols: 32`
explicitly (giving 28×32 images). The config loader raises with the valid alternatives rather
than failing three hours into an extraction run.

**Padding must not leak.** Responses have different lengths. Getting the mask wrong does not
crash — it silently contaminates every metric. The conv zeroes padded positions before and
after (a conv has a receptive field), the BiLSTM uses `pack_padded_sequence`, and the attention
pool sets padded logits to `-inf`. `tests/test_models.py` asserts that writing garbage into the
padded slots changes the output by *exactly zero*.

**Normalisation is per-view, per-channel.** Q, K and V have very different magnitudes, and the
raw channel dwarfs the two delta channels. A single global statistic would let the
largest-scale view swamp the others before the fusion module ever saw them, and would make the
deltas numerically invisible. Statistics are computed on the **training split only**.

---

## Tests

```bash
pytest tests/ -q          # 162 tests
```

The suite is structured around the ways this pipeline can fail *silently*:

- `test_tensor_ops.py` — hand-verified pooling and boundary deltas on a synthetic ramp.
- `test_qkv_hooks.py` — **the gate**: captured Q/K/V vs. a manual matmul; prefill exclusion;
  a longer prompt must not change `T`.
- `test_models.py` — padding invariance; that view *i* actually reaches backbone *i* (a
  shape-only test would pass even with a transposed axis); that all three CNNs receive gradient.
- `test_data.py` — per-view normalisation, collation, splits, train-only statistics.
- `test_extraction.py` — the full chain (`run_extraction` → label → dataset → model) against a
  real tiny Llama.
- `test_train_e2e.py` — plants a signal in **one** view and asserts the pipeline finds it, which
  proves every view's path is live, and that the gated fusion's weights actually shift toward
  the informative view.

---

## Status and open questions

The pipeline is implemented and tested end to end; **no results on real LLMs yet**. The honest
open questions, in the order they should be answered:

1. **Does Q/K/V beat hidden states?** Q, K and V are linear projections of the hidden state, so
   information-theoretically a probe on the hidden state has access to strictly more. The bet is
   that the *inductive bias* — three views, layer-deltas as channels — makes the signal easier
   to learn. Budget for the ablation that tests it. If Q/K/V does not win, the delta-channel
   idea may still carry the work on its own.
2. **Does `gated` beat `concat_mlp`?** If not, "learned fusion" is a fancy concatenation and we
   should say so. A negative result here is still a clean result; the separate-CNN design is
   defensible regardless, since it is what makes the per-view ablations and the gate figure
   possible.
3. **Which view carries the signal?** `--set extract.views='[V]'` is one config line. If V alone
   nearly matches Q+K+V, that says the hallucination signal lives in *what gets retrieved*
   rather than *what gets attended to* — the most quotable claim available here.

Start with TriviaQA or HotpotQA (10k examples, and both baselines use them). TruthfulQA is only
817 examples — too small to train a CNN+BiLSTM from scratch.

## References

- Dasgupta et al., *HalluShift: Measuring Distribution Shifts towards Hallucination Detection in
  LLMs*, IJCNN 2025. ([`hallushift/`](hallushift/))
- Bar-Shalom et al., *Beyond Token Probes: Hallucination Detection via Activation Tensors with
  ACT-ViT*, NeurIPS 2025. ([`ACT-ViT/`](ACT-ViT/))
