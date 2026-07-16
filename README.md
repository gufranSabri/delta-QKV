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

Two published baselines are the main comparators:

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

### The evaluation protocol is a design choice, not a detail
Both baselines were read from their **source**, not their papers, and they do different things:

- **ACT-ViT** keeps a genuinely separate test corpus: `LIST_OF_DATASETS` pairs every dataset with
  a `*_test` twin generated from the benchmark's dev/test split, and `create_datasets_split.py:59`
  refuses to split anything named `*_test`. Train/val is 80/20 of the *train* pool, seed 42.
- **HalluShift** does not. A single `train_test_split(test_size=0.25, random_state=42)` produces
  75/25 with **no validation set**, and the same 25% is used both for early stopping
  (`classifier.py:190`) and as the reported score (`classifier.py:224`). Their headline AUROC is
  therefore selected on the split it reports.

We follow ACT-ViT's structure for both comparisons, because HalluShift's is a flaw and copying it
would import the flaw. That means our HalluShift numbers are slightly **pessimistic** relative to
theirs — the safe direction, and one to state plainly rather than paper over.

Two consequences worth internalising:

- **Never carve a test slice out of a dataset that has a real held-out split** — you would be
  discarding training data to build a test set you already have. `train.test_fraction` is applied
  *only* to TruthfulQA, which has no alternative.
- **Normalisation statistics are fit on the train indices only** and baked into the checkpoint, so
  `test` re-uses them rather than recomputing on the test set. Recomputing would leak the test
  distribution into the input scaling.

---

## Install

There is **no `requirements.txt`**. `scripts/install.sh` is the single source of truth for
dependencies, and every SLURM script calls it.

```bash
bash scripts/install.sh              # core deps
bash scripts/install.sh --bleurt     # + BLEURT (required for the HalluShift comparison)
```

`--bleurt` installs TensorFlow-CPU, clones and `pip install ./bleurt`, and downloads the
BLEURT-20-D12 checkpoint (~1.5 GB) into `models/`. It self-tests and exits non-zero if BLEURT
cannot score, so a broken install fails *now* rather than six hours into a generation run.

**TensorFlow-CPU is deliberate.** The GPU build reserves VRAM at import and collides with the
torch CUDA context during extraction. BLEURT scoring is cheap; keep TF off the GPU.

The checkpoint is cached in `models/`, so it survives across allocations. The core deps are
installed with `--no-index` on Compute Canada (the local wheelhouse) and from PyPI elsewhere;
BLEURT is a git install and always needs the network.

---

## Quickstart

```bash
# 0. Dependencies (there is no requirements.txt).
bash scripts/install.sh

# 1. Generate responses, capture Q/K/V, build and save the token images.
#    This is the TRAIN corpus, built from the benchmark's train split.
python main.py --config configs/triviaqa/llama3_8b.yaml extract

# 2. The HELD-OUT test corpus, from the benchmark's dev split. Without this
#    there is no honest same-dataset test set. (Skip for truthfulqa — it has
#    only one split, so it gets a stratified slice instead.)
python main.py --config configs/triviaqa/llama3_8b.yaml extract \
  --set dataset.name=triviaqa_test

# 3. Look at what you produced BEFORE training on it.
python main.py --config configs/triviaqa/llama3_8b.yaml inspect --idx 0

# 4. Train.
python main.py --config configs/triviaqa/llama3_8b.yaml train --run-name same_llama3_8b_triviaqa

# 5. Evaluate. Pass the plain dataset name — `test` picks the held-out corpus
#    itself and appends the result to docs/results.csv.
python main.py --config configs/triviaqa/llama3_8b.yaml test \
  --checkpoint runs/same_llama3_8b_triviaqa/best.pt \
  --dataset triviaqa
```

Or just edit `current_run.sh` and run `sbatch main.slurm` — the SLURM wrapper boots the env and
then calls that script, so the same five steps stay resumable.

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

`configs/default.yaml` holds the base settings; each `configs/{dataset}/{llm}.yaml` overrides only
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
| `extract.source` | `qkv` (base default), `hs` | `qkv` captures Q/K/V projections (3 views); `hs` captures hidden states (1 view). Changes the data folder: `data/{source}/…`. |
| `extract.extraction_type` | `transforms` (base default), `delta` | `delta` = channels (raw, Δprev, Δnext); `transforms` = channels (raw, DWT, FFT) along the layer axis. Changes the data folder: `data/{source}/{extraction_type}/…`. |
| `extract.views` | any subset of `[Q, K, V]` (or `[H]` for `hs`) | A single view bypasses fusion entirely. This is the ablation that asks *which view carries the signal*. |
| `extract.pool` | `mean` (base default), `max`, `l2`, `sdk` | How `D` is reduced to columns. `max` follows ACT-ViT. |
| `extract.boundary_mode` | `wrap` (base default), `zero`, `replicate` | What to do at layer 0 (no previous) and layer L-1 (no next). `delta` only. See below. |
| `model.channels` | `default` (base default), `first_only`, `same` | How (view × channel) is regrouped into CNN images. Pure train-time re-slice; no re-extraction. |
| `model.fusion` | `concat_mlp` (base default), `gated`, `bilinear`, `cross_attn` | `gated` is the inspectable variant; `concat_mlp` is the plain control. |
| `model.share_backbone` | `false` (base default) | `true` ties one CNN across all views. |
| `model.backbone` | `scratch_cnn` (base default), `resnet18` | |
| `model.pretrained_backbone` | `true` (base default) | Only read by `resnet18`. When `true` the images are **upscaled to 224×224** and the ImageNet network is used unmodified — see below. |
| `model.embed_dim` | `128` | Per-view CNN output size. |
| `model.fused_dim` | `128` | Fusion output size. |
| `train.batch_size` | `32` | |
| `train.lr` | `1e-3` | |
| `train.backbone_lr_scale` | `1.0` | Only matters when the backbone is pretrained; keeps the head/backbone LR ratio stable. |
| `labeling.scheme` | `exact_match` (default), `bleurt` | ACT-ViT's protocol vs. HalluShift's. `bleurt` needs `install.sh --bleurt`. |
| `train.seed` | `42` | Matches ACT-ViT's `RANDOM_STATE` and HalluShift's `random_state`. |
| `train.test_fraction` | `0.2` | **Only** used by datasets with no held-out corpus (TruthfulQA). Everything else tests on `<ds>_test`, so nothing is carved out. |

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
with random weights there is nothing to preserve and upscaling would cost ~49× the compute for
no information gain.

The trade is real: 224×224 is ~49× the pixels of 32×32, so the pretrained path is substantially
slower per batch. `scratch_cnn` remains the default for exactly that reason.

### Boundary handling

Layer 0 has no previous layer and layer L-1 has no next one, so two delta values are undefined.
The shipped base config uses `wrap` — where layer 0's "previous" is the *last* layer. That keeps
the channel layout simple, but the embedding output and the final layer are very far apart, so the
wrapped rows carry a much larger magnitude than any genuine adjacent-layer delta, and a CNN will
happily latch onto those two outlier rows. Switch to `zero` if you want the conservative choice
instead.

### Relabeling is free

`meta.txt` stores each response and its gold answer alongside the tensor, so switching labeling
schemes never requires re-extracting features:

```bash
python main.py --config configs/triviaqa/llama3_8b.yaml label --set labeling.scheme=bleurt
```

---

## How the test set is built — read this before running anything

This is the part that is easiest to get wrong, and getting it wrong silently inflates every
number you report. The rule:

> **If a dataset has a real held-out split, use it. If it doesn't, carve a stratified slice and
> keep it out of both training and model selection.**

Concretely, extraction produces **two corpora** per (dataset, LLM):

```
data/triviaqa/llama3_8b/          <- from the benchmark's TRAIN split; train + val come from here
data/triviaqa_test/llama3_8b/     <- from the benchmark's dev/test split; TEST, never trained on
```

Validation is always carved out of the **train** corpus, so no test row ever influences early
stopping or checkpoint selection. Per dataset:

| dataset | train + val (80/20 of…) | test |
|---|---|---|
| TriviaQA | `train` | `validation` → `triviaqa_test` |
| HotpotQA (± context) | `train` | `validation` → `hotpotqa_test` |
| IMDB | `train` | `test` → `imdb_test` |
| Movies | `movie_qa_train.csv` | `movie_qa_test.csv` → `movies_test` |
| CoQA | train JSON | `coqa-dev-v1.0.json` → `coqa_test` |
| TyDiQA-GP | `train` (English) | `validation` → `tydiqa_test` |
| **TruthfulQA** | *817 rows, one split* | stratified **60/20/20** — no twin corpus exists |

TriviaQA's and HotpotQA's upstream `test` splits are **unlabelled leaderboard blind sets**, so
`validation` is the real held-out set and plays the test role. Both baselines make the same call.

`main.py test --dataset triviaqa` resolves the target for you — it evaluates `triviaqa_test` if
that corpus exists, the held-out slice if not, and the **full** corpus when the checkpoint never
trained on that dataset (the zero-shot / LODO case). You always pass the plain dataset name.

Seed is **42** everywhere, matching both baselines.

---

## Experiments

Two comparisons, two suites. They differ in datasets, LLMs, and labelling — they are not
interchangeable, and mixing them produces numbers that compare to nothing.

|  | ACT-ViT | HalluShift |
|---|---|---|
| datasets | TriviaQA, HotpotQA, HotpotQA+ctx, IMDB, Movies | TruthfulQA, TriviaQA, CoQA, TyDiQA-GP |
| LLMs | Mistral-7B-Instruct, Llama-3-8B-Instruct, Qwen2.5-7B-Instruct | Llama-2-7b, Llama-3.1-8B, OPT-6.7B (**base**, not instruct) |
| labels | `exact_match` (substring / sentiment) | `bleurt` (BLEURT-20-D12, hallucination iff score ≤ 0.5) |
| generation | 100 new tokens | 64 new tokens, greedy |
| grid | 15 cells | 12 cells |
| BLEURT | not needed | **required** |

The two overlap on **TriviaQA only**. ACT-ViT never evaluated on TruthfulQA/CoQA/TyDiQA, and
HalluShift never on HotpotQA/IMDB/Movies — so most cells of `docs/results.csv` are legitimately
`-`, not missing work.

### Run a suite

The literal command lists live under `scripts/main/{actvit,hallushift}/` and
`scripts/ablation/{actvit,hallushift}/`. Each directory has `extract.sh`, `train.sh`, and
`test.sh`; the files are meant to be read top to bottom and copied into a shell or SLURM job as
needed.

For the main experiments:

```bash
bash scripts/main/actvit/extract.sh
bash scripts/main/actvit/train.sh
bash scripts/main/actvit/test.sh
```

Use the matching `hallushift/` scripts for the BLEURT-based comparison, and the corresponding
`scripts/ablation/...` files for the ablation grids.

Re-running a command that is already done will redo it — there is no skip logic. If a job dies
partway, comment out the lines that finished (or just run the remaining ones) before resubmitting.

### One config end to end

```bash
sbatch main.slurm        # edit current_run.sh first
```

Runs extract → inspect → train → test for a single config, skipping any step already done.

### The three settings

- **`same`** — train on a dataset, test on its held-out corpus. The headline table.
- **`lodo`** — leave-one-dataset-out: train on the other N−1, evaluate **zero-shot** on the
  held-out one. ACT-ViT reports this; HalluShift does not, so those cells have no baseline.
- **`ablate`** — the tables that decide whether the architecture's claims hold: view subsets,
  fusion variants, boundary modes, backbone choices. Run on one (dataset, LLM) — the full grid
  multiplies cost without adding a claim.

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

## Results

`docs/results.csv` is keyed on **(model, llm, metric)** with one column per dataset, and is
updated automatically by every `main.py … test` run — an existing row is overwritten, a new one
appended. The baseline rows are pre-filled from the two papers and are never touched.

The LLM column is load-bearing: ACT-ViT reports one number per *(LLM, dataset)* cell, so without
it our rows cannot line up against theirs.

### Comparing honestly

Three caveats that decide whether a comparison is meaningful:

1. **AUROC compares; precision/recall/F1 do not.** ACT-ViT's positive class is *correct*; ours is
   *hallucinated*. AUROC is invariant under a consistent label flip, so it compares directly.
   P/R/F1 are defined w.r.t. opposite positive classes and **do not**.
2. **TruthfulQA needs BLEURT to be comparable to HalluShift.** They label with BLEURT ≤ 0.5. With
   `exact_match` you are measuring a different task, and the numbers do not belong side by side.
3. **HalluShift's own numbers are optimistically biased.** They use a single 75/25 split with *no
   validation set* — the same 25% drives early stopping *and* is reported (`classifier.py:190`
   and `:224`). We deliberately do not copy this. Our numbers are therefore slightly pessimistic
   relative to theirs, which is the safe direction.

There is one more, inherited rather than introduced: ACT-ViT's shipped `movie_qa_*.csv` files
share ~102 questions between train and test (and the train file has internal duplicates: 10,000
rows, 9,856 unique). We reproduce their corpora faithfully rather than deduplicating, so the
head-to-head stays apples-to-apples. It affects ~1.3% of the Movies test set.

---

## Repo layout

```
main.py                     single entry point: extract | label | train | test | inspect
main.slurm                  SLURM wrapper: bootstrap env, then run current_run.sh
current_run.sh              tiny local driver: set DATASET and MODEL, then run the steps

configs/                    default.yaml + one file per (dataset × LLM)
scripts/
  install.sh                the only dependency list. --bleurt adds BLEURT.
  main/                     ready-made run scripts for the main experiments
  ablation/                 ablation run scripts and helpers
  troubleshooting.sh        the pipeline command by command, for an interactive session
src/
  extract/
    tensor_ops.py           pooling, delta channels, boundary modes
    qkv_hooks.py            forward hooks on q_proj/k_proj/v_proj
    run_extraction.py       generate -> capture -> build images -> save
    datasets.py             prompt templates + gold-answer extraction + split policy
  label/
    exact_match.py          exact-match / sentiment labeling
    bleurt.py               BLEURT-20-D12 labeling
  models/
    backbones.py            ScratchCNN, ResNet18Adapted
    fusion.py               gated | concat_mlp | bilinear | cross_attn
    temporal.py             Conv1d + BiLSTM + masked attention pooling
    classifier.py           the full detector
  data/dataset.py           dataset, per-view normalisation, padding + masking
  train.py / test.py        training loop and evaluation
tests/                      the suite (see below)
```

On disk, extraction produces:

```
data/{source}/{extraction_type}/{dataset}/{llm}/   ← benchmark's TRAIN split (train + val)
  00000/
    tokens.npy    (T, V, L, C, 3) float16   ← (tokens, views, layers, cols, channels)
    meta.txt      prompt / response / gold / score / label
  manifest.jsonl  one line per example — the training index
  geometry.json   the model geometry the images were built with

data/{source}/{extraction_type}/{dataset}_test/{llm}/   ← dev/test split. TEST ONLY.
  …same structure. Disjoint from the corpus above; never trained on.
```

`source` and `extraction_type` head the path because they change what is captured, so their
outputs must not collide:

- **`extract.source`** — `qkv` (per-layer Q/K/V projections) or `hs` (per-layer hidden states,
  a single stream → `V == 1`).
- **`extract.extraction_type`** — `delta` (channels = raw, delta-to-prev-layer, delta-to-next-layer)
  or `transforms` (channels = raw, DWT, FFT computed *along the layer axis*).

The **view axis is a real axis**, never folded into channels. That is what makes dropping a
view (`extract.views: [Q]`) a pure slicing operation with no re-extraction. How the (view ×
channel) axes are regrouped into the images the CNNs consume is a *training-time* choice,
`model.channels`: `default` (one image per view), `first_only` (one image, raw channel of each
view stacked), or `same` (one image per channel-type, views stacked).

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
pytest tests/ -q
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
  LLMs*, IJCNN 2025. Protocol mirrored in [src/label/bleurt.py](src/label/bleurt.py) and [src/extract/datasets.py](src/extract/datasets.py).
- Bar-Shalom et al., *Beyond Token Probes: Hallucination Detection via Activation Tensors with
  ACT-ViT*, NeurIPS 2025. Split policy and comparison wiring live in [src/extract/datasets.py](src/extract/datasets.py) and [src/test.py](src/test.py).
