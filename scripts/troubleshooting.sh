#!/bin/bash
# ============================================================================
# delta-QKV: Interactive Pipeline
# ============================================================================
# Run this file command-by-command in an interactive SLURM terminal.
# Start with one of the salloc commands below, then execute each step.
#
# Usage:
#   1. Run one of the salloc commands to allocate resources
#   2. Copy-paste each section of this file into the terminal
#   3. Check output before moving to the next step
# ============================================================================

# ── STEP 0: ALLOCATE INTERACTIVE SLURM RESOURCES ──────────────────────────
# Choose based on your needs. Run ONE of these:

# Single GPU (l40s), 12 hours - good for prototyping
salloc --gpus-per-node=l40s:1 --cpus-per-task=24 --mem=200G --time=12:00:00 --account=aip-lsigal

# Or: Two GPUs (l40s), 3 hours - for faster training
# salloc --gpus-per-node=l40s:1 --cpus-per-task=12 --mem=100G --time=3:00:00 --account=aip-lsigal

# Or: Single H100 GPU, 12 hours - highest perf
# salloc --gpus-per-node=h100:1 --cpus-per-task=24 --mem=200G --time=12:00:00 --account=aip-lsigal


# ── STEP 1: SETUP ENVIRONMENT ──────────────────────────────────────────────
# Load modules and create virtual environment
module load StdEnv/2023 gcc/12.3 cuda/13.2 arrow/23.0.1 python/3.11.5
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

# Install dependencies
pip install -q -r requirements.txt

# Set environment variables
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_DISABLE_XET=1

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name()}')"


# ── STEP 2: EXTRACT Q/K/V TOKEN IMAGES ─────────────────────────────────────
# Generates responses, captures Q/K/V activations, and builds token images.
# This is the most time-consuming step. It is restartable — already-processed
# examples are skipped. You can also split across multiple jobs with --chunk:
#   --chunk 1  → examples 0–999
#   --chunk 2  → examples 1000–1999
#   --chunk 3  → examples 2000–2999

echo "Step 2/4: Extracting Q/K/V token images..."
CONFIG="configs/triviaqa/llama3_8b.yaml"  # Edit this to use a different config
python main.py --config "$CONFIG" extract

# Alternative: extract only a specific chunk (for parallel jobs)
# python main.py --config "$CONFIG" extract --chunk 1

# Alternative: re-extract even if tokens.npy already exists
# python main.py --config "$CONFIG" extract --overwrite


# ── STEP 3: INSPECT EXTRACTED DATA ─────────────────────────────────────────
# Verify image quality and print cross-view correlation matrix.
# If the off-diagonal correlations are near 1.0, Q/K/V are redundant
# and the whole fusion architecture is wasted effort. Check this on day one!

echo "Step 3/4: Inspecting extracted token images..."
python main.py --config "$CONFIG" inspect --idx 0

# Inspect additional examples:
# python main.py --config "$CONFIG" inspect --idx 10 --tokens 8
# python main.py --config "$CONFIG" inspect --idx 20 --out /tmp/inspect_idx20.png


# ── STEP 4: RELABEL (OPTIONAL) ─────────────────────────────────────────────
# Recompute labels from stored responses without re-extracting.
# Useful if you want to try a different labeling scheme (e.g., bleurt).

# echo "Relabeling with BLEURT..."
# python main.py --config "$CONFIG" label --set labeling.scheme=bleurt


# ── STEP 5: TRAIN HALLUCINATION DETECTOR ───────────────────────────────────
# Train the model. This saves checkpoints to runs/{run_name}/
# Training is on a single GPU by default. For multi-GPU, use torchrun:

echo "Step 5/4: Training hallucination detector..."
python main.py --config "$CONFIG" train

# Alternative: custom run name and train datasets
# python main.py --config "$CONFIG" train \
#   --run-name my_experiment_v1 \
#   --train-datasets triviaqa,hotpotqa \
#   --test-dataset truthfulqa

# Alternative: leave-one-dataset-out validation
# python main.py --config "$CONFIG" train \
#   --train-datasets triviaqa,hotpotqa \
#   --test-dataset imdb


# ── STEP 6: FIND BEST CHECKPOINT ───────────────────────────────────────────
# The training step saved checkpoints. Find the most recent best.pt:

CHECKPOINT=$(find runs -name "best.pt" -type f -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
echo "Best checkpoint: $CHECKPOINT"


# ── STEP 7: EVALUATE ON TEST SET ───────────────────────────────────────────
# Evaluate the trained detector on the test set.

echo "Step 7/4: Evaluating detector..."
python main.py --config "$CONFIG" test --checkpoint "$CHECKPOINT"

# Alternative: evaluate on a different dataset
# python main.py --config "$CONFIG" test --checkpoint "$CHECKPOINT" --dataset hotpotqa


# ── UTILITIES ──────────────────────────────────────────────────────────────

# List all completed runs:
# ls -lh runs/

# Remove a run to free up space:
# rm -r runs/run_20250714_120000_my_experiment

# Check current GPU usage:
# nvidia-smi

# Monitor extraction progress (in another terminal):
# watch -n 2 'find data -name "tokens.npy" | wc -l'