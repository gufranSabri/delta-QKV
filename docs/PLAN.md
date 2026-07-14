# delta-QKV — Build Plan

Hallucination detection by rendering an LLM's per-token Q/K/V activations as images and
learning over them with a CNN → 1D-conv → BiLSTM → classifier.

Positioned against two baselines that live in this repo:
- **HalluShift** (IJCNN'25): hand-crafted scalar features (Wasserstein + cosine between
  every-other layer's hidden states and attentions, plus token-probability stats) → small MLP.
  Collapses the whole response to ~40 numbers. Labels via BLEURT.
- **ACT-ViT** (NeurIPS'25): the full hidden-state *activation tensor* (L, N, D) treated as one
  image with D as channels → ViT. Keeps the tensor but only uses **hidden states**, and
  max-pools it down to L_eff=8 × N_eff=100 before training.

**Our contribution sits between them:** we use **Q, K, V** rather than hidden states (three
views of what attention is actually doing, not just what it produced), we add **explicit
layer-to-layer delta channels** (HalluShift's distribution-shift intuition, but kept as a
spatial channel instead of averaged into a scalar), and we model the response as a
**sequence of per-token images** rather than one flattened image — so the temporal
structure of where hallucination emerges is preserved.

---

## 0. Decisions already locked (from discussion)

| Question | Decision |
|---|---|
| Feature axis | Max-pool D → L over contiguous chunks, giving a **square L×L image** per token. No reshape step needed. |
| Heads | Concatenated (D = d_model), then pooled. Not per-head (GQA would make K/V non-square). |
| Delta channels | **Signed** delta, computed **after** pooling: `ch1 = p[l] - p[l-1]`, `ch2 = p[l] - p[l+1]`. Not true Wasserstein. |
| Boundaries | **Zero-pad**: `img[0,:,1] = 0`, `img[L-1,:,2] = 0`. Not circular wrap (wrap would inject a huge embedding-vs-final-layer outlier row). |
| Sequence model | CNN per token → 1D-conv → BiLSTM → classifier. |
| Q/K/V handling | **Three separate CNN passes, combined by a learned fusion module.** Never channel-stacked into one 9-channel image — that would let conv1 blend the views and destroy the three-view premise. |
| Fusion | Config-selectable: `gated` (default), `concat_mlp` (control), `bilinear`, `cross_attn`. |
| CNN backbone | Config-selectable: `scratch_cnn` (default) or `resnet18` (CIFAR-adapted). Untied across views by default. |
| Labeling | Config-selectable: `exact_match` (ACT-ViT style, default) or `bleurt` (HalluShift style). |

---

## 1. The core object: what one token's image is

For one example (prompt + generated response of T tokens), an LLM with L layers and hidden
dim D, we extract Q, K, V at every layer for every **generated** token.

For each of Q, K, V independently:

```
raw[t, l, :]          # (D,)  — the Q (or K, or V) vector at token t, layer l
pooled[t, l, j] = max( raw[t, l, jC : (j+1)C] )      where C = D // L,  j = 0..L-1
```

`C = 4096 // 32 = 128` for Llama-3-8B. So `pooled[t]` is an **L×L = 32×32** matrix.

Then the three channels:

```
img[t, l, j, 0] = pooled[t, l, j]                          # raw
img[t, l, j, 1] = pooled[t, l, j] - pooled[t, l-1, j]      # delta to previous layer  (0 at l=0)
img[t, l, j, 2] = pooled[t, l, j] - pooled[t, l+1, j]      # delta to next layer      (0 at l=L-1)
```

**One token → three 32×32×3 images**, one each for Q, K and V. Rows = layers, columns = pooled
feature groups, channels = {raw, backward delta, forward delta}.

The three images stay **separate all the way through the CNN** and are combined only by a
learned fusion module (§5a). They are *stored* as one array of shape `(T, 3, 32, 32, 3)`
= (tokens, views, layers, cols, channels) for I/O convenience, but the view axis is never
folded into channels.

Size: 32·32·3·2 bytes (fp16) = **6 KB per token per view**, 18 KB for Q+K+V. At T=100 tokens
that's **1.8 MB per example**, ~18 GB for a 10k-example dataset. Tractable. (Compare: full-D
would have been 2.3 TB.)

### Why this shape is defensible in a paper
- The image is square **because D was pooled to L**, not because we resized — no interpolation
  artifact, no arbitrary aspect ratio.
- Max-pool over contiguous D-chunks is exactly ACT-ViT's `max_pool` down-sampling, so we
  inherit their empirical justification for it.
- With Llama's D=4096 and 32 heads of 128 dims, the contiguous chunks of size 128 **coincide
  with attention heads** — so column *j* is literally "the peak activation of head *j*". That's
  a free interpretability story. Worth stating; note it does *not* hold for Qwen (3584 dim,
  28 layers, 4 kv-heads) so don't over-claim it.

---

## 2. Repo layout

```
delta-QKV/
├── main.py                       # single entry point, subcommands
├── configs/
│   ├── default.yaml              # base config, everything else overrides it
│   ├── triviaqa/
│   │   ├── llama3_8b.yaml
│   │   ├── mistral_7b.yaml
│   │   └── qwen2.5_7b.yaml
│   ├── hotpotqa/
│   │   ├── llama3_8b.yaml
│   │   └── ...
│   ├── imdb/ ...
│   ├── movies/ ...
│   └── truthfulqa/ ...
├── src/
│   ├── config.py                 # yaml load + merge + validate (dataclass-backed)
│   ├── extract/
│   │   ├── qkv_hooks.py          # forward hooks on q_proj/k_proj/v_proj  ← the tricky part
│   │   ├── tensor_ops.py         # pool_D_to_L, add_delta_channels, boundary handling (PURE, unit-testable)
│   │   ├── generate.py           # prompt templating + model.generate per dataset
│   │   └── run_extraction.py     # orchestrates: generate → hook capture → build images → save
│   ├── label/
│   │   ├── exact_match.py        # ACT-ViT's correctness fns (triviaqa/hotpotqa/imdb/movies/…)
│   │   ├── bleurt.py             # HalluShift's BLEURT-20-D12 @ 0.5
│   │   └── registry.py           # LABELERS = {"exact_match": ..., "bleurt": ...}
│   ├── data/
│   │   ├── dataset.py            # QKVImageDataset (one example = folder of token images)
│   │   ├── combined.py           # ConcatDataset across (dataset × LLM), tags each item's origin
│   │   └── collate.py            # pad variable-T batches + build mask
│   ├── models/
│   │   ├── backbones.py          # ScratchCNN, ResNet18Adapted → per-view token embedding
│   │   ├── fusion.py             # GatedFusion, ConcatMLP, BilinearFusion, CrossAttnFusion
│   │   ├── temporal.py           # Conv1d stack + BiLSTM + attention-pool
│   │   ├── classifier.py         # QKVHalluDetector = 3×backbone + fusion + temporal + head
│   │   └── registry.py
│   ├── train.py
│   ├── test.py
│   └── utils/
│       ├── seed.py, logger.py, metrics.py (AUROC, TPR@5%FPR, PR-AUC, F1)
│       └── splits.py             # stratified train/val, saved to disk like ACT-ViT
├── scripts/
│   ├── gen_experiment_scripts.py # ← generates the LODO / same-dataset run scripts
│   └── generated/                # output of the above (gitignored)
├── tests/
│   ├── test_tensor_ops.py        # wrap/zero-pad boundary tests on synthetic L=4,T=2,D=12
│   └── test_qkv_hooks.py         # tiny model, assert captured Q/K/V match manual matmul
├── data/                         # extracted features (gitignored)
│   └── {dataset}/{llm}/
│       ├── 00000/
│       │   ├── tokens.npy        # (T, 3, 32, 32, 3) fp16 = (tokens, views, L, cols, chans)
│       │   └── meta.txt          # prompt, response, gold, score, label
│       ├── 00001/
│       ├── stats.json            # per-view per-channel mean/std over the train split
│       └── manifest.jsonl        # one line per example: {idx, n_tokens, label, score}
├── hallushift/                   # baseline (already present)
├── ACT-ViT/                      # baseline (already present)
└── docs/
    ├── main_idea.txt
    └── PLAN.md                   # this file
```

### Note on the on-disk format
Your spec says "one image per token" as separate files. That's **10k examples × 100 tokens =
1M files per dataset-LLM pair**, which will make any filesystem (and any dataloader) miserable.

**Recommendation:** keep the per-example folder, but store all tokens as **one** `tokens.npy`
of shape `(T, 3, 32, 32, 3)` inside it, plus `meta.txt`. You get the same conceptual layout, one
`np.load(mmap_mode='r')` per example, and a 100× reduction in inode count. A
`--save_per_token_files` flag can emit the individual PNGs/`.npy`s for a handful of examples
when you want to *look* at them for a figure. I'll build it this way unless you object.

Note the **view axis is a real axis in the stored array**, not folded into channels — the
dataloader hands the model `(B, T, V, 3, 32, 32)` and the model folds `V` into the batch for
the CNN and unfolds it before fusion. Storing it this way keeps the "drop a view" ablation
(`views: [Q]`) a pure slicing operation with no re-extraction.

---

## 3. The part that needs real care: capturing Q/K/V

HuggingFace gives you `output_hidden_states=True` and `output_attentions=True` for free. It
gives you **nothing** for Q/K/V — they're internal to `LlamaAttention.forward`. Both baselines
sidestep this (ACT-ViT uses hidden states; HalluShift uses hidden states + attention probs).
**This is the actual novel engineering in the project, and the main risk.**

Approach: register `forward_hook`s on `model.model.layers[i].self_attn.q_proj` / `.k_proj` /
`.v_proj`. The hook output is the projection result, pre-RoPE, pre-reshape:
- `q_proj` output: `(B, seq, num_heads * head_dim)` = (B, seq, 4096)
- `k_proj` / `v_proj` output: `(B, seq, num_kv_heads * head_dim)` = (B, seq, **1024**) for
  Llama-3 (8 KV heads × 128) — **not 4096**. GQA means K and V are narrower than Q.

### Consequences you must handle
1. **K and V have a different D than Q.** For Llama-3-8B: D_q=4096, D_kv=1024. Pooling
   1024 → 32 gives chunks of 32 instead of 128; the image is still 32×32, so **the shapes
   still work out** — good. But the pooling chunk size differs per view. Handle by computing
   `C = D_view // L` per view. Mistral/Qwen have their own numbers; read them from
   `model.config` (`num_attention_heads`, `num_key_value_heads`, `head_dim`), never hardcode.
2. **Decisions:** do we take Q/K/V **pre-RoPE** (the raw projections, what the hook gives) or
   **post-RoPE**? Pre-RoPE is simpler and position-independent. Post-RoPE is what attention
   actually consumes. I'll implement pre-RoPE (hook on the projection) and note post-RoPE as a
   config flag `rope: pre | post` that we can add later by hooking the attention module itself.
3. **Generation loop:** with KV-cache on, at decode step *t* the `q_proj` hook fires with
   `seq=1` (just the new token) — which is exactly what we want. But `k_proj`/`v_proj` also
   fire with `seq=1`, giving the K/V *of the new token*, also what we want. The **prefill** step
   fires once with `seq=len(prompt)` — we discard that (or optionally keep it, mirroring
   ACT-ViT's `save_activations_input`). So: **hook fires once per layer per generated token,
   naturally.** Clean.
4. **Memory during generation:** we accumulate `T × L × D` floats on GPU. Move to CPU + fp16
   every step, or every K steps. Do the D-pooling **on GPU immediately** so we only ever hold
   `T × L × 32` — this makes the whole thing nearly free.

**Milestone 1 must be:** write `qkv_hooks.py`, run it on a tiny model, and *assert against a
manual matmul* that the captured Q equals `hidden_state @ W_q`. Do not build anything else
until this test passes. Everything downstream is worthless if the capture is wrong.

---

## 4. Labeling (re-annotation)

Both baselines re-annotate, because — as you correctly noted — once the model generates its
*own* response, the dataset's original hallucination label is meaningless.

**HalluShift's scheme:** greedy-decode 64 new tokens → score generated answer against all
reference answers with **BLEURT-20-D12** → `hallucination = 1 if max_score <= 0.5 else 0`.
Semantic, handles paraphrase, needs a TF/BLEURT install.

**ACT-ViT's scheme:** greedy-decode → per-dataset **exact/substring match**:
- `triviaqa`: any alias appears in the answer (case-insensitive) → correct
- `hotpotqa`, `movies`: gold answer is a substring of the answer → correct
- `imdb`: first token of the answer parses to the gold sentiment → correct
- `natural_questions`: falls back to an **LLM judge** (Mistral-7B prompted for 0/1)

`label = 1 - correctness` (1 = hallucinated).

**Plan: implement both behind `labeling.scheme`.** Default `exact_match` (free, deterministic,
lets you iterate fast); `bleurt` available so we can report numbers directly comparable to
HalluShift's table. Both write `score` and `label` into `meta.txt` and the manifest, so
**relabeling never requires re-extracting features** — that separation matters, because
extraction is the expensive step.

---

## 5. The model

**Q, K and V each get their own CNN pass, and are combined by a learned fusion module — never
by channel-stacking.** Stacking them into one 9-channel image would let the very first conv
kernel mix Q, K and V together, destroying the three-view structure that is the whole premise
of the method. Keeping them separate through the CNN means the fusion weights are explicit,
inspectable, and reportable ("the model relies mostly on V" is a result; a 9-channel conv gives
you no such statement).

```
                per-token, three separate 3-channel images
       Q (32,32,3)        K (32,32,3)        V (32,32,3)
            │                  │                  │
       [ CNN_Q ]          [ CNN_K ]          [ CNN_V ]     ← weights shared across tokens
            │                  │                  │           (see "tied or untied" below)
          e_Q (E)            e_K (E)            e_V (E)
            └──────────────┬──┴──────────────────┘
                    [ FusionModule ]                        ← parameterized, config-selectable
                           │
                    token embedding  (F,)
                           │
                 stack over tokens → (T, F)
                           │
              [ Conv1d(F→F, k=3) × 2, + GELU ]              # local n-gram structure over tokens
                           │
              [ BiLSTM(F → H, 1–2 layers) ] → (T, 2H)
                           │
              [ masked attention-pool ]                     # must respect the padding mask
                           │
              [ Linear(2H → 1) ] → sigmoid                  # p(hallucination)
```

### 5a. The fusion module (`model.fusion`)

All variants take three `E`-dim embeddings and return one `F`-dim token embedding. Implemented
behind a registry so they're one config line apart:

| `fusion` | What it does | Params | Why you'd pick it |
|---|---|---|---|
| `gated` **(default)** | Learn per-view scalar-or-vector gates: `g = softmax(W·[e_Q;e_K;e_V])`, then `out = Σ gᵥ ⊙ eᵥ`, projected to F. | small | Gates are directly readable as "how much does the model use Q vs K vs V" — a figure in the paper. |
| `concat_mlp` | `out = MLP([e_Q ; e_K ; e_V])`, a Linear(3E→F) + GELU + Linear(F→F). | 3E·F | Simplest expressive baseline; the honest control the gated version must beat. |
| `bilinear` | Low-rank bilinear/tensor-fusion capturing pairwise interactions (Q·K is *meaningful* — it's what attention computes). | medium | The one with a principled story: attention scores are Q·Kᵀ, so an explicit multiplicative Q–K term is theory-motivated, not decorative. |
| `cross_attn` | Treat {e_Q, e_K, e_V} as a length-3 sequence, run 1 transformer block, pool. | medium | Most general; likely overkill for 3 tokens but cheap to have. |

**Recommendation: `gated` as the default, `concat_mlp` as the control, `bilinear` as the
interesting ablation.** Report all three — the fusion comparison is a natural results table and
it's the part of the architecture that is genuinely *yours*.

### 5b. Tied vs untied CNN weights (`model.share_backbone`)

- `share_backbone: false` **(default)** — three independent CNNs. Q, K and V have different
  statistics (V isn't even in the same space as Q/K; and under GQA the K/V images are pooled
  from a 1024-dim vector while Q comes from 4096), so forcing shared filters is a real
  assumption, not a free saving.
- `share_backbone: true` — one CNN applied three times. 3× fewer backbone params, and can be
  regularizing on small datasets. Worth an ablation line; don't make it the default.

Cost of untied: 3 × ~250k = ~750k params for ScratchCNN. Trivial. The CNN forward runs on a
`(B·T·3, 3, 32, 32)` batch either way (fold the view axis into the batch), so **untied costs
essentially nothing in wall-clock** as long as you don't loop over views in Python.

### 5c. Backbones

**ScratchCNN** (default — activation maps are not natural images, ImageNet priors buy little):
```
Conv(3→32, 3×3) + BN + GELU        # 3 in-channels now: raw, Δprev, Δnext
ResBlock(32)              → 32×32
ResBlock(64,  stride 2)   → 16×16
ResBlock(128, stride 2)   → 8×8
AdaptiveAvgPool → E = 128
```
~250k params each. Fast enough to train the whole thing on one GPU.

**ResNet18Adapted** (ablation): `torchvision.resnet18(weights=IMAGENET1K_V1)`, replace `conv1`
with `Conv2d(3, 64, 3, stride=1, padding=1)`, `maxpool = Identity()`, `fc = Identity()` → E=512.
Standard CIFAR adaptation. **Note the separate-view design makes the pretrained weights more
usable than the stacked one would have:** with 3 input channels we *could* keep pretrained
`conv1` weights (they expect 3 channels), where a 9-channel stack would have forced us to throw
`conv1` away. Worth trying both `conv1` re-init vs. keeping the pretrained RGB filters.

### 5d. What `qkv_mode` now means

Since views are always processed separately, `qkv_mode` selects **which** views are used, not
how they're packed:
```yaml
extract:
  views: [Q, K, V]     # or [Q], [Q,K], [V], … — fusion module adapts to len(views)
```
`len(views) == 1` bypasses fusion entirely (identity). This gives the Q-only / K-only / V-only
ablations for free, which you want anyway: **"which of Q, K, V actually carries the hallucination
signal" is one of the more interesting questions this project can answer.**

**Baselines to implement in the same repo:** ACT-ViT-style single-image ViT over our QKV tensor;
a "raw-channel-only" ablation (drop channels 1/2) to isolate whether the delta channels earn
their keep; and a hidden-states-instead-of-QKV ablation (see §9.1).

### Critical implementation details
- **Batching variable T:** pad to max-T in batch, carry a `(B, T)` bool mask, and make the
  BiLSTM use `pack_padded_sequence` and the pool use the mask. Getting this wrong silently
  poisons the metric — this is the #1 source of bugs in this architecture.
- **Folding the view axis:** reshape `(B, T, V, 3, 32, 32)` → `(B·T·V, 3, 32, 32)` for the CNN,
  then unfold back to `(B, T, V, E)` before fusion. Never loop over views or tokens in Python.
- **Normalization is per-view, not global.** Q, K and V have wildly different magnitudes (and
  under GQA are pooled from different-width vectors). Compute mean/std **per view per channel**
  over the training split, cache to `stats.json`, apply at load. A single global statistic would
  let whichever view has the largest scale swamp the others before the fusion module ever sees
  them.
- **Metric:** AUROC primary (both baselines report it), plus TPR@5%FPR (ACT-ViT), plus PR-AUC /
  F1 (HalluShift). Select the best checkpoint on **val AUROC**.

---

## 6. Config system

`configs/default.yaml` holds everything; a dataset×LLM config overrides only what differs.

```yaml
# configs/triviaqa/llama3_8b.yaml
llm:
  name: meta-llama/Meta-Llama-3-8B-Instruct
  dtype: bfloat16
  # L, D_q, D_kv, n_heads read from model.config at runtime — never hardcoded

dataset:
  name: triviaqa
  n_samples: 10000
  max_new_tokens: 100
  prompt_template: "Answer the question concisely. Q: {question} A:"

extract:
  views: [Q, K, V]         # which views to use; subset for ablations (fusion adapts to len)
  pool: max                # max | mean
  boundary_mode: zero      # zero | wrap | replicate
  dtype: float16
  max_tokens: 100          # truncate T (matches ACT-ViT's N_MAX)

labeling:
  scheme: exact_match      # exact_match | bleurt
  bleurt_threshold: 0.5

model:
  backbone: scratch_cnn    # scratch_cnn | resnet18
  share_backbone: false    # false = one CNN per view (default); true = tied weights
  embed_dim: 128           # E, per-view embedding out of each CNN
  fusion: gated            # gated | concat_mlp | bilinear | cross_attn
  fused_dim: 128           # F, output of the fusion module
  conv1d_layers: 2
  lstm_hidden: 128
  lstm_layers: 1
  dropout: 0.3

train:
  batch_size: 32
  lr: 1e-3
  weight_decay: 1e-4
  epochs: 30
  patience: 8
  seed: 0
```

`main.py` subcommands:
```bash
python main.py extract --config configs/triviaqa/llama3_8b.yaml [--chunk 3]
python main.py label   --config configs/triviaqa/llama3_8b.yaml --scheme bleurt   # relabel w/o re-extract
python main.py train   --config configs/triviaqa/llama3_8b.yaml \
                       [--train-datasets triviaqa,hotpotqa,imdb] [--test-dataset movies]
python main.py test    --config ... --checkpoint runs/xyz/best.pt
```

---

## 7. Cross-dataset experiment scripts (your "train on all-but-1" requirement)

`scripts/gen_experiment_scripts.py` reads the config tree and emits shell scripts:

- **`--mode same`** → for each (dataset, LLM): train and test on that dataset. N scripts.
- **`--mode lodo`** → leave-one-dataset-out: for each held-out dataset D, train on all
  datasets except D (for a given LLM, or pooled across LLMs), test zero-shot on D.
- **`--mode lolo`** → leave-one-LLM-out (ACT-ViT's 10/15 setting), if you want it later.

Emits into `scripts/generated/{mode}/` with one `.sh` per run plus a `run_all.sh`. Mirrors how
ACT-ViT organizes `sweeps/{15_15, 1_15, leaving_one_dataset_out_14_15, ...}` but as plain shell
rather than W&B sweeps (no W&B dependency; add it later if you want).

**Cross-LLM caveat:** different LLMs have different L (32 vs 28) and different D. Our images are
L×L, so **a Llama image is 32×32 and a Qwen image is 28×28** — the CNN cannot consume both without
handling it. Options, to decide before running cross-LLM experiments:
- (a) fix a global `L_eff = 28` (or 32) and pool/pad the layer axis too, so every LLM yields the
  same image size — this is what ACT-ViT effectively does with `L_eff=8`;
- (b) per-LLM linear adapter, like ACT-ViT's `ModuleListPerLLMLinear`.
**Recommendation: (a)**, set `extract.L_eff` in the config and pool the layer axis to it. Simpler,
and it makes the image size a fixed hyperparameter rather than an LLM property. Same-LLM
experiments are unaffected.

---

## 8. Build order (each step is independently verifiable)

| # | Deliverable | Verification |
|---|---|---|
| 1 | `src/extract/tensor_ops.py` — pool, deltas, boundary modes | `tests/test_tensor_ops.py` on a hand-computed L=4, T=2, D=12 tensor. Assert `img[0,:,1]==0`, `img[3,:,2]==0`, and that ch1 at l=2 equals `p[2]-p[1]`. **Pure math, no model needed.** |
| 2 | `src/extract/qkv_hooks.py` | Load `hf-internal-testing/tiny-random-LlamaForCausalLM`, capture Q, and assert it equals `hidden @ q_proj.weight.T` to within fp tolerance. **Gate: do not proceed until green.** |
| 3 | `src/extract/run_extraction.py` + one config | Run on 20 TriviaQA examples with Llama-3-8B. Inspect `data/triviaqa/.../00000/`: correct T, correct `(T,3,32,32,3)` shape, `meta.txt` populated. Render Q, K and V token images side by side to PNG and *look at them* — is there visible structure across layers, and do the three views look meaningfully different from each other? (If Q, K and V images look identical, the separate-CNN premise is in trouble and you want to know that on day one, not after training.) |
| 4 | `src/label/` both schemes | Relabel the 20 examples with `exact_match` and with `bleurt`; check they mostly agree and eyeball the disagreements. |
| 5 | Full extraction, 1 dataset × 1 LLM | ~10k examples, chunked like ACT-ViT (`--chunk 1..10`) so it's restartable. Check the label balance isn't degenerate (if 95% correct, the task is uninteresting — pick a harder dataset). |
| 6 | `src/models/` — backbones + **fusion registry** + temporal | Shape test: feed `(B=2, T=5, V=3, 3, 32, 32)` through each fusion variant and assert output `(2,)`. Assert `views: [Q]` bypasses fusion cleanly. |
| 7 | `src/data/` + `train.py` | Overfit 50 examples to ~100% train AUROC. If it can't overfit, the model or the mask is broken. |
| 8 | Real train/val on 1 dataset | Compare val AUROC against HalluShift's reported number for the same dataset+LLM. **This is the go/no-go.** |
| 9 | Ablations: fusion variants, view subsets, tied/untied backbone | The fusion + view-subset table is a core paper result, not an afterthought. |
| 10 | Remaining datasets × LLMs, then `gen_experiment_scripts.py`, LODO runs | Full results table. |

Steps 1–2 are the whole risk. If Q/K/V capture doesn't work cleanly the project doesn't exist,
so it goes first and gets a real test.

---

## 9. Open questions / things I'd flag now

1. **Does Q/K/V beat hidden states?** Nobody has shown it does. Q/K/V are linear projections of
   the hidden state, so information-theoretically a probe on the hidden state has access to
   *everything* Q/K/V has. Our bet is that the *inductive bias* of splitting into three views
   and exposing layer-deltas as spatial channels makes the signal easier to learn. That's a
   legitimate bet, but **budget for the ablation that tests it**: same architecture, hidden
   states instead of Q/K/V. If Q/K/V doesn't win, the delta-channel idea may still carry the
   paper on its own.
2. **Max-pooling 4096 → 32 throws away a lot.** It's the same move ACT-ViT makes and it works
   for them, but consider a `mean` and an `L2-norm` pool variant in the config for a cheap ablation.
3. **Dataset choice.** TruthfulQA (HalluShift's headline) is only 817 examples — too small to
   train a CNN+BiLSTM from scratch. Start with **TriviaQA** or **HotpotQA** (10k, and both
   baselines use them, so we get a shared comparison point).
4. **Compute.** Extraction is one forward-generate per example — 10k examples × 5 datasets ×
   3 LLMs is the dominant cost. Do 1 dataset × 1 LLM end-to-end before scaling out.
5. **Does the fusion module actually earn its keep?** `gated` must beat `concat_mlp` for the
   "learned fusion" framing to mean anything — otherwise you've built a fancy concatenation.
   Run that comparison early (step 9) and be willing to report that concat wins if it does; a
   negative result on fusion is still a clean result, and the separate-CNN design is defensible
   on its own (it's what makes the per-view ablations and the gate-weight figure possible).
6. **The most likely interesting finding** is the per-view ablation: if V alone nearly matches
   Q+K+V, that says the hallucination signal lives in *what gets retrieved*, not *what gets
   attended to* — a genuinely quotable claim. Make sure the code makes that experiment one
   config line.

