#!/bin/bash
# ============================================================================
# delta-QKV: interactive pipeline
# ============================================================================
# Run this file command-by-command in an interactive SLURM terminal. It is not
# meant to be executed as a whole -- copy-paste one section at a time and look
# at the output before moving on.
# ============================================================================


# ── STEP 0: ALLOCATE ───────────────────────────────────────────────────────
# Run ONE of these.

salloc --gpus-per-node=l40s:1 --cpus-per-task=24 --mem=60G --time=3:00:00 --account=aip-lsigal
# salloc --gpus-per-node=l40s:1 --cpus-per-task=8 --mem=16G --time=1:00:00 --account=aip-lsigal


# Longer / bigger:
# salloc --gpus-per-node=h100:1 --cpus-per-task=24 --mem=60G --time=3:00:00 --account=aip-lsigal


# ── STEP 1: ENVIRONMENT ────────────────────────────────────────────────────
# There is no requirements.txt -- scripts/install.sh is the single source of
# truth for dependencies, and every SLURM script calls it too.

module load StdEnv/2023 gcc/12.3 cuda/13.2 arrow/23.0.1 python/3.11.5
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_DISABLE_XET=1
export TF_CPP_MIN_LOG_LEVEL=3
export HF_TOKEN=token
export HF_HOME=/home/ahmedubc/scratch/hf_cache

# Core deps only -- enough for the ACT-ViT comparison (exact_match labels):
bash scripts/install.sh --bleurt

# ...OR with BLEURT, required for the HalluShift comparison. Adds TensorFlow-CPU
# and downloads the ~1.5GB BLEURT-20-D12 checkpoint into models/ (cached, so
# later allocations reuse it). It self-tests and fails loudly if broken.
# bash scripts/install.sh --bleurt

python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name())"


bash all-datasets_extract.sh
bash single-dataset_ablation.sh

# ══════════════════════════════════════════════════════════════════════════
# SINGLE CONFIG, STEP BY STEP
# ══════════════════════════════════════════════════════════════════════════

CONFIG="configs/triviaqa/llama3_8b.yaml"
DATASET=$(python -c "from src.config import load_config; print(load_config('$CONFIG').dataset.name)")
LLM=$(python -c "from src.config import load_config; print(load_config('$CONFIG').llm.alias)")
RUN="same_${LLM}_${DATASET}"


# ── STEP 2: EXTRACT ────────────────────────────────────────────────────────
# Generates responses, hooks q_proj/k_proj/v_proj, builds the token images.
# The expensive step -- one manual decode loop per batch of examples.
# Restartable: finished examples are skipped.

python main.py --config "$CONFIG" extract --set extract.batch_size=16

# Split across jobs if needed (1-indexed blocks of 1000):
# python main.py --config "$CONFIG" extract --chunk 1 --set extract.batch_size=16   # examples 0-999


# ── STEP 3: INSPECT (do this before training, on day one) ──────────────────
# Prints the cross-view correlation matrix. If Q/K/V come back correlated near
# 1.0 they are redundant, and the per-view CNN + fusion design buys nothing.

python main.py --config "$CONFIG" inspect --idx 0


# ── STEP 4: TRAIN ──────────────────────────────────────────────────────────

python main.py --config "$CONFIG" train --run-name "$RUN"


# ── STEP 5: TEST ───────────────────────────────────────────────────────────
# Pass the plain dataset name. `test` evaluates the held-out slice carved out
# at train time. Writes docs/results.csv.

python main.py --config "$CONFIG" test --checkpoint "runs/$RUN/best.pt" --dataset "$DATASET"


# ── RELABEL (optional, free) ───────────────────────────────────────────────
# meta.txt keeps each response + gold answer next to the tensor, so switching
# labeling schemes never needs a re-extract.

# python main.py --config "$CONFIG" label --set labeling.scheme=bleurt


# ══════════════════════════════════════════════════════════════════════════
# WHOLE SUITES
# ══════════════════════════════════════════════════════════════════════════
# The .bash files ARE the experiment lists -- every command spelled out, no
# loops, no functions. Open one and read it; run it whole, or copy-paste any
# single line out of it.

# bash experiments_actvit.bash        # 5 datasets x 3 LLMs, exact_match labels
# bash experiments_hallushift.bash    # 4 datasets x 3 LLMs, BLEURT (needs --bleurt above)

# As batch jobs (each just sets up the env, then runs the .bash file):
# sbatch main.slurm                   # one config (edit CONFIG at the top)
# sbatch experiments_actvit.slurm
# sbatch experiments_hallushift.slurm


# ── UTILITIES ──────────────────────────────────────────────────────────────

# nvidia-smi
# ls -lh runs/
# cat docs/results.csv | column -t -s,
# watch -n 2 'find data -name "tokens.npy" | wc -l'      # extraction progress
