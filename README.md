# delta-QKV

**Hallucination detection by rendering an LLM's per-token Q/K/V activations as images.**

For each token an LLM generates, we build small images from its **Q**, **K** and **V**
projections — rows are transformer layers, columns are pooled feature groups, and the colour
channels carry either layer-deltas or wavelet transforms of the pooled activation. Each image
stream gets its own CNN; a fusion module (or a pure single-stream pass-through) combines them; a
Conv1d + BiLSTM temporal encoder reads the resulting sequence of token embeddings and predicts
whether the response is a hallucination.

```
           per-token image stream(s) -- shape and count set by
           extract.views x model.channels x model.include
                              │
                        [ CNN per stream ]        ← untied weights (or shared), per src.models.backbones
                              │
                          e_1 ... e_V
                              │
                        [ Fusion ]                ← identity (single stream) | gated | concat_mlp | bilinear | cross_attn
                              │
                    stack over tokens → (T, F)
                              │
              [ Conv1d x N ] → [ BiLSTM ] → [ masked attention-pool ]
                              │
                       p(hallucination)
```

---

## Why this design

Two published baselines are the main comparators:

| | What it uses | What it does with it |
|---|---|---|
| **HalluShift** (IJCNN'25) | hidden states + attention probs | collapses a whole response into ~40 scalars (Wasserstein/cosine between layers) → MLP |
| **ACT-ViT** (NeurIPS'25) | hidden states only | keeps the full `(L, N, D)` activation tensor as **one** image → ViT |
| **delta-QKV** (this repo) | **Q, K and V** (or hidden states, for comparison) | one image **per token**, layer-deltas or wavelet transforms as channels, per-stream CNN(s) + optional fusion → BiLSTM |

Neither baseline looks at the attention projections themselves — only at what attention
produced. Q, K and V are three distinct views of what the attention mechanism is actually
*doing*, and this repo captures them directly via forward hooks (see
[Implementation notes](#implementation-notes)).

### Q, K and V are never channel-stacked by default

The obvious shortcut — concatenate Q, K and V into one 9-channel image — is deliberately
avoidable but not forced: `model.channels` controls whether views stay as separate image streams
(`default`) or get regrouped by channel type across views (`same`), and `model.include` further
selects which of the resulting streams actually reach a CNN. See
[Configuration](#configuration) for the full mechanics. Stacking views onto the channel axis at
the very first conv layer would let that kernel blend Q, K and V immediately, destroying the
ability to ask *which view carries the signal* — which is why `channels=default` keeps every
view on its own CNN, meeting only at fusion.

With `model.fusion: gated`, training reports a softmax distribution over the fed-in streams —
"the model relies mostly on stream 2" is a measurable, inspectable claim
(`QKVHalluDetector.view_gates`). Note: whenever `model.include` narrows the input down to a
**single** stream (the current best-performing setting — see
[docs/ablation_insights.md](docs/ablation_insights.md)), `build_fusion` returns `IdentityFusion`
and the whole fusion-module axis is a no-op — worth remembering before quoting a gate weight.

### The images are square by construction (when possible)

A token's activation vector has `D` dimensions (e.g. 4096 for most 7-8B models), but the image
needs a manageable width. We pool `D` down to `n_cols` columns (default: the model's layer
count `L`, giving a square image) over contiguous chunks — see
[src/extract/tensor_ops.py](src/extract/tensor_ops.py) for the four pooling modes (`max`,
`mean`, `l2`, `sdk`). Not every model can use the default square size: Qwen2.5-7B has `L=28`
layers but `D_kv=512` under GQA, and 28 does not divide 512, so its config sets `extract.n_cols`
explicitly. `ModelGeometry.check_n_cols` in
[src/extract/qkv_hooks.py](src/extract/qkv_hooks.py) raises with the valid alternatives rather
than failing hours into an extraction run.

### The evaluation protocol is a design choice, not a detail

Both baselines were read from their **source**, not their papers, and they do different things:

- **ACT-ViT** keeps a genuinely separate test corpus from the benchmark's dev/test split.
- **HalluShift** does not: a single `train_test_split(test_size=0.25, random_state=42)` produces
  75/25 with **no validation set**, and the same 25% drives both early stopping and the reported
  score.

This repo's datasets (TriviaQA, CoQA, TruthfulQA — see [Datasets](#datasets)) each carve their
own held-out slice out of a single upstream split via `train.val_fraction` /
`train.test_fraction`, matching HalluShift's `test_size` convention rather than assuming a
separate benchmark test split exists for every dataset. Seed is **42** everywhere, matching both
baselines. Normalisation statistics are fit on the **training split only** and baked into the
checkpoint (`test` re-uses them rather than recomputing on the eval set, unless
`--recompute-stats` is passed — see [CLI](#cli-reference)).

---

## Install

There is **no `requirements.txt`**. `scripts/install.sh` is the single source of truth for
dependencies, and every SLURM/interactive script calls it.

```bash
bash scripts/install.sh              # core deps
bash scripts/install.sh --bleurt     # + BLEURT (required for the HalluShift/BLEURT labeling scheme)
```

`--bleurt` installs TensorFlow-CPU, clones and `pip install ./bleurt`, and downloads the
BLEURT-20-D12 checkpoint (~1.5 GB) into `models/`. It self-tests and exits non-zero if BLEURT
cannot score, so a broken install fails *now* rather than hours into a generation run.

**TensorFlow-CPU is deliberate.** The GPU build reserves VRAM at import and collides with the
torch CUDA context during extraction. BLEURT scoring is cheap; keep TF off the GPU.

The checkpoint is cached in `models/`, so it survives across allocations. The core deps are
installed with `--no-index` on Compute Canada (the local wheelhouse) and from PyPI elsewhere;
BLEURT is a git install and always needs the network.

---

## Quickstart

```bash
# 0. Dependencies (there is no requirements.txt).
bash scripts/install.sh --bleurt

# 1. Generate responses, capture Q/K/V + hidden states, build and save every
#    (source x extraction_type) combination of token images in one pass.
python main.py --config configs/triviaqa/llama2_7b.yaml extract --set extract.batch_size=8

# 2. Look at what you produced BEFORE training on it.
python main.py --config configs/triviaqa/llama2_7b.yaml inspect --idx 0

# 3. Train.
python main.py --config configs/triviaqa/llama2_7b.yaml train --run-name runs/same_llama2_7b_triviaqa

# 4. Evaluate. `test` writes/updates docs/results.csv.
python main.py --config configs/triviaqa/llama2_7b.yaml test \
  --checkpoint runs/same_llama2_7b_triviaqa/best.pt \
  --dataset triviaqa
```

`all-datasets_extract.sh` and `all-datasets_run.sh` loop the same two calls over every
`(dataset, model)` combination for the currently-active `DATASETS`/`MODELS` lists (edit those
arrays at the top of each script). `all-datasets_run.sh` skips a train/test call whose output
already exists unless `RERUN=1` is set. `single-dataset_ablation.sh` sweeps
`model.channels x model.include` for one `(dataset, model)` pair, training, testing, and
rendering a Grad-CAM figure for every combination.

Extraction is the expensive step (one `generate()` per example). It is **restartable** — an
example already complete in every `(source, extraction_type)` combo's directory is skipped
unless `--overwrite` is passed — and **chunkable**:

```bash
python main.py --config configs/triviaqa/llama2_7b.yaml extract --chunk 3   # examples 2000-2999
```

### `inspect` first, always

`inspect` renders the token images to PNG and prints a cross-view correlation matrix:

```
  cross-view correlation of the raw channel (flattened):
           Q       K       V
    Q   1.000   0.412   0.087
    K   0.412   1.000   0.103
    V   0.087   0.103   1.000
```

If those off-diagonal numbers came back near 1.0, Q, K and V would be redundant and the whole
separate-CNN design would be buying nothing. `src/viz.py`'s `cross_view_correlation_figure`
produces the same check as a heatmap PNG, averaged over many examples rather than one.

---

## Configuration

`configs/default.yaml` holds the base settings; each `configs/{dataset}/{llm}.yaml` overrides
only what differs. Any key can be overridden from the command line, on either side of the
subcommand:

```bash
python main.py --config configs/triviaqa/llama2_7b.yaml train \
  --set model.fusion=bilinear \
  --set extract.views='[V]' \
  --set train.epochs=50
```

The knobs that matter:

| Key | Options | Notes |
|---|---|---|
| `extract.source` | `qkv` (default) or `hs` | `extract` always captures and writes **both** in one run; this field only selects which folder `train`/`test`/`cam`/`inspect` read. `qkv` = 3 views (Q/K/V); `hs` = 1 view (hidden state, residual stream). |
| `extract.extraction_type` | `delta` (default) or `transforms` | Also always both-written by `extract`, selects which folder is read. `delta` = channels (raw, Δprev, Δnext); `transforms` = channels (raw, DWT-Haar, DWT-Sym3), computed along the layer axis. |
| `extract.views` | any subset of `[Q, K, V]` (or `[H]` under `source=hs`) | Dropping a view is a pure slicing operation, no re-extraction needed. The ablation that asks *which view carries the signal*. |
| `extract.pool` | `max` (default), `mean`, `l2`, `sdk` | How the feature axis `D` is reduced to `n_cols` columns. |
| `extract.boundary_mode` | `zero` (default), `replicate`, `wrap` | What to do at layer 0 (no previous) and layer L-1 (no next). `delta` only — see [tensor_ops.py](src/extract/tensor_ops.py) for why `zero` is the safe default. |
| `extract.n_cols` | `null` (default: model's layer count) | Must evenly divide every requested view's feature dim; the config loader raises with valid alternatives if not (e.g. Qwen2.5-7B needs this set explicitly). |
| `extract.l_eff` | `null` (default: no pooling) | Pools the layer axis to a fixed size — only needed for cross-LLM training where layer counts differ. |
| `extract.batch_size` | `1` (default) | Examples generated together per batch; a pure throughput knob, produces identical tensors to `batch_size=1`. |
| `model.channels` | `default` (default), `first_only`, `same` | How the (view × channel) axes regroup into CNN input images. Pure train-time re-slice, no re-extraction. `default` = one image per view; `first_only` = one image, raw channel of every view stacked; `same` = one image per channel-type, views stacked. |
| `model.include` | `null` (default: keep all) | Which of the regrouped images to actually feed the model, by index. E.g. `channels=same, include=[0]` trains on a single stream: the raw channel only. |
| `model.fusion` | `gated` (default), `concat_mlp`, `bilinear`, `cross_attn` | Ignored (bypassed by `IdentityFusion`) whenever `include` narrows the input to one stream. `gated` is the inspectable variant; `concat_mlp` is the plain control. |
| `model.share_backbone` | `false` (default) | `true` ties one CNN across all streams. |
| `model.backbone` | `scratch_cnn` (default), `resnet18` | |
| `model.pretrained_backbone` | `true` (default) | Only read by `resnet18`. When `true`, images are **upscaled to 224×224** and the ImageNet network is used unmodified — see below. |
| `model.embed_dim` / `model.fused_dim` | `128` / `128` | Per-stream CNN output size / fusion output size. |
| `train.batch_size` | `32` | See [docs/ablation_insights.md](docs/ablation_insights.md) for why this interacts strongly with BatchNorm in both the backbone and temporal encoder. |
| `train.lr` / `train.backbone_lr_scale` | `1e-3` / `1.0` | `backbone_lr_scale` only matters when the backbone is pretrained. |
| `labeling.scheme` | `exact_match` (default), `bleurt` | ACT-ViT's protocol vs. HalluShift's. `bleurt` needs `install.sh --bleurt`. |
| `train.seed` | `42` | Matches ACT-ViT's `RANDOM_STATE` and HalluShift's `random_state`. |
| `train.val_fraction` / `train.test_fraction` | `0.25` / `0.25` | Every supported dataset carves its held-out slice out of a single upstream split this way — see [Datasets](#datasets). |
| `data_root` | `/scratch/ahmedubc/delta-QKV-data` | Where extracted tensors live; not inside the repo. |

### Pretrained ResNet-18 resizes the input rather than rewiring the network

A stock ResNet-18 opens with a 7×7 stride-2 conv and a stride-2 maxpool. Fed a 32×32 activation
image, those crush it to 8×8 before the first residual block — so the usual fix is to swap in a
3×3 stride-1 conv and delete the maxpool. But that **throws away the pretrained stem**, which is
the whole reason for loading the checkpoint.

So `pretrained_backbone: true` instead **upscales the image to 224×224** and uses the ImageNet
network completely unmodified — every filter sees inputs at the scale it was trained on. That is
also what makes "pretrained" an honest row in the ablation table: with the stem rebuilt, the
comparison against `scratch_cnn` was never really testing *pretraining*.

`pretrained_backbone: false` keeps the native resolution and adapts the stem CIFAR-style, since
with random weights there is nothing to preserve and upscaling would cost ~49x the compute for
no information gain. `scratch_cnn` remains the default for exactly that reason.

### Relabeling is free

`meta.txt` stores each response and its gold answer alongside the tensor, so switching labeling
schemes never requires re-extracting features:

```bash
python main.py --config configs/triviaqa/llama2_7b.yaml label --set labeling.scheme=bleurt
```

---

## Datasets

Configs currently exist for three datasets, each crossed with the models below:

| dataset | notes |
|---|---|
| TriviaQA | question + answer, no context |
| CoQA | conversational QA with a `{story}` context in the prompt |
| TruthfulQA | 817 examples — small; the held-out slice comes from a single split, same as the others |

Each dataset's train/val/test split is carved from a single upstream split via
`train.val_fraction` / `train.test_fraction` (see [src/extract/datasets.py](src/extract/datasets.py)
`SPLIT_SOURCES`), matching HalluShift's protocol rather than assuming a separate benchmark test
split exists for every one of them.

## Models

Configs currently exist for:

- `llama2_7b` — `meta-llama/Llama-2-7b-hf`
- `llama3.1_8b` — `meta-llama/Llama-3.1-8B`
- `qwen2.5_7b` — `Qwen/Qwen2.5-7B` (GQA: `D_kv=512` forces `extract.n_cols=32` explicitly)
- `opt_6.7b` — `facebook/opt-6.7b`

All are **base** models, not instruction-tuned, matching HalluShift's protocol. Every
`(dataset, model)` config sets `bleurt` labeling with the BLEURT-20-D12 checkpoint and 64
greedy-decoded new tokens.

---

## CLI reference

```
main.py --config <path> [--set key=val ...] <subcommand> [subcommand args]
```

| Subcommand | Purpose |
|---|---|
| `extract` | Generate responses, capture Q/K/V + hidden states, build and save every (source × extraction_type) combination of token images. `--chunk N` restricts to a 1000-example block; `--overwrite` re-extracts examples that already have `tokens.npy`. |
| `label` | Recompute labels from stored responses — no re-extraction needed. |
| `train` | Train the detector. `--dataset` overrides the config's dataset; `--run-name` names the output directory under `runs/`. |
| `test` | Evaluate a checkpoint. `--checkpoint` required; `--dataset` defaults to the config's dataset; `--recompute-stats` normalizes with statistics recomputed from the eval corpus instead of the checkpoint's training stats; `--out-name` avoids clobbering an in-distribution result file. |
| `inspect` | Render token images to PNG (`--idx`, `--tokens` = how many to render, `--out`). |
| `cam` | Grad-CAM/Eigen-CAM for one example: where the detector's CNN looks, per image stream, per generated token. `--checkpoint` required; `--method gradcam\|eigencam`; `--max-tokens` caps rendered token columns. |

`--set key=val` can appear before or after the subcommand and works with nested keys
(`--set extract.views='[V]'`) and lists (YAML-parsed).

### `src/viz.py` — not wired into the CLI

Four figure-generating functions exist for deeper analysis but are called by importing the
module directly, not through `main.py`:

- `aggregate_gradcam` — Grad-CAM/Eigen-CAM averaged over N hallucinated vs. N truthful examples.
- `cross_view_correlation_figure` — the `inspect` correlation check, as a heatmap averaged over many examples.
- `temporal_attention_profile` — the `MaskedAttentionPool`'s real per-token attention weight for one example.
- `qk_alignment_proxy` — an explicitly-labeled **proxy** (not real attention) for token-to-token influence: cosine similarity between pooled, pre-RoPE Q and K images. See [docs/ablation_insights.md](docs/ablation_insights.md) §3 for what it can and cannot tell you.

Every figure writes a `README.txt` next to the PNG explaining what it shows, how to read it, its
provenance, and its caveats.

```python
from src.config import load_config
from src.viz import qk_alignment_proxy

cfg = load_config("configs/triviaqa/opt_6.7b.yaml")
qk_alignment_proxy(cfg, dataset_name="triviaqa", idx=0)
```

---

## Results

`docs/results.csv` is written/updated automatically by every `main.py ... test` run — an
existing row is overwritten, a new one appended.

---

## Repo layout

```
main.py                     single entry point: extract | label | train | test | inspect | cam
main.slurm                  SLURM wrapper
all-datasets_extract.sh     extract every (dataset x model) in the current DATASETS/MODELS lists
all-datasets_run.sh         train + test every (dataset x model); RERUN=1 to force re-runs
single-dataset_ablation.sh  channels x include sweep for one (dataset, model), + Grad-CAM per cell

configs/                    default.yaml + one file per (dataset x model)
scripts/
  install.sh                the only dependency list. --bleurt adds BLEURT.
  troubleshooting.sh         the pipeline command by command, for an interactive session
docs/
  ablation_insights.md      batch-size/BatchNorm, heatmap interpretation, and ablation writeups
  results.csv                per-(model, llm, metric) results table, updated by `test`
src/
  extract/
    tensor_ops.py           pooling, delta channels, DWT transforms, boundary modes
    qkv_hooks.py             forward hooks on q_proj/k_proj/v_proj; model geometry; decoder-layer lookup
    run_extraction.py        generate -> capture -> build images -> save
    datasets.py               prompt templates + gold-answer extraction + split policy
  label/
    exact_match.py            exact-match labeling
    bleurt.py                 BLEURT-20-D12 labeling
    registry.py                scheme dispatch
  models/
    backbones/                scratch_cnn.py, resnet18.py
    fusion.py                  identity | gated | concat_mlp | bilinear | cross_attn
    temporal.py                 Conv1d + BiLSTM + masked attention pooling
    classifier.py               the full detector
  data/dataset.py             dataset, per-view normalisation, padding + masking
  train.py / test.py          training loop and evaluation
  cam.py                       Grad-CAM/Eigen-CAM for a single example
  viz.py                       aggregate figures (see CLI reference above)
  inspect_images.py           renders token images + cross-view correlation to PNG
```

On disk, extraction produces:

```
{data_root}/{source}/{extraction_type}/{dataset}/{model_alias}/
  00000/
    tokens.npy    (T, V, L, C, 3) float16   <- (tokens, views, layers, cols, channels)
    meta.txt      prompt / response / gold / score / label
  manifest.jsonl  one line per example -- the training index
  geometry.json   the model geometry the images were built with
  progress.log    "i/total" appended every 100 generated examples
```

`source` and `extraction_type` head the path because they change what is captured — a single
`extract` call populates all four `(source, extraction_type)` combinations from one generation
pass, so nothing is generated twice. The **view axis is a real axis**, never folded into
channels, which is what makes dropping a view (`extract.views: [Q]`) a pure slicing operation.

---

## Implementation notes

**Q/K/V capture is the load-bearing part.** HuggingFace gives you `hidden_states` and
`attentions` for free but *nothing* for Q/K/V — they are internal to the attention module. We
hook the `q_proj`/`k_proj`/`v_proj` Linears, which yields the projections **pre-RoPE** (RoPE is a
position-dependent rotation; post-RoPE activations would entangle token *content* with token
*position*, and V never receives RoPE at all).

**Decoder layers are located across model families.** `get_decoder_layers` in
[src/extract/qkv_hooks.py](src/extract/qkv_hooks.py) tries `model.layers` (Llama/Mistral/Qwen),
`model.model.layers`, `transformer.h` (GPT-2 family), and `model.decoder.layers` (OPT), raising
a clear error if none match.

**Grouped-query attention makes K and V narrower than Q.** Every dimension is read from
`model.config` at runtime via `read_geometry`; nothing is hardcoded. Both views still pool to
the same number of columns — only the chunk width differs.

**Padding must not leak.** Responses have different lengths. Getting the mask wrong does not
crash — it silently contaminates every metric. The temporal Conv1d zeroes padded positions
before and after convolving, the BiLSTM uses `pack_padded_sequence`, and the attention pool sets
padded logits to `-inf`.

**Normalisation is per-view, per-channel.** Q, K and V have very different magnitudes, and the
raw channel dwarfs the delta/transform channels. A single global statistic would let the
largest-scale stream swamp the others before fusion ever saw them. Statistics are computed on
the **training split only** and baked into the checkpoint.

**BatchNorm makes this pipeline batch-size sensitive.** Both CNN backbones and the temporal
Conv1d stack use `BatchNorm2d`/`BatchNorm1d`. See
[docs/ablation_insights.md](docs/ablation_insights.md) §1 for why this specific architecture is
unusually sensitive to `train.batch_size`, and what to check before trusting a reported score.

---

## Status and open questions

See [docs/ablation_insights.md](docs/ablation_insights.md) for a full write-up. In short, at the
time of writing the best-performing configuration is `model.channels=same, model.include=[0]`
(a single image stream: the raw channel, all views stacked) — which means the fusion-module axis
and the layer-delta/transform channels are currently **not** being exercised by the winning
config. The two open questions this raises, in priority order:

1. **Is the batch-size sensitivity a normalization artifact or a real property of the signal?**
   (BatchNorm vs. GroupNorm sweep.)
2. **Does Q/K/V actually beat hidden states, and does any view beat V alone?** Q, K and V are
   linear projections of the hidden state, so information-theoretically a probe on the hidden
   state has access to at least as much information — the bet is that the inductive bias (deltas,
   pooling, per-view CNNs) makes the signal easier to learn. Currently unresolved.

## References

- Dasgupta et al., *HalluShift: Measuring Distribution Shifts towards Hallucination Detection in
  LLMs*, IJCNN 2025. Protocol mirrored in [src/label/bleurt.py](src/label/bleurt.py) and
  [src/extract/datasets.py](src/extract/datasets.py).
- Bar-Shalom et al., *Beyond Token Probes: Hallucination Detection via Activation Tensors with
  ACT-ViT*, NeurIPS 2025. Split policy and comparison wiring live in
  [src/extract/datasets.py](src/extract/datasets.py) and [src/test.py](src/test.py).
