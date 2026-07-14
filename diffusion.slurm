#!/bin/bash
#SBATCH --job-name=delta-qkv-pipeline
#SBATCH --output=slurm/%x.%j.out
#SBATCH --error=slurm/%x.%j.err
#SBATCH --open-mode=append
#SBATCH --account=aip-lsigal
#SBATCH --mail-user=ahmed.ghuf@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00

module load StdEnv/2023 gcc/12.3 cuda/13.2 arrow/23.0.1 python/3.11.5
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

# Install dependencies
pip install -q -r requirements.txt

# Configuration
CONFIG="${1:-configs/triviaqa/llama3_8b.yaml}"
SKIP_EXTRACTION="${2:-false}"
SKIP_TRAINING="${3:-false}"

if [ -z "$CONFIG" ]; then
    echo "Usage: sbatch delta-qkv-pipeline.slurm [CONFIG] [SKIP_EXTRACTION] [SKIP_TRAINING]"
    echo "  CONFIG: path to config file (default: configs/triviaqa/llama3_8b.yaml)"
    echo "  SKIP_EXTRACTION: true to skip extraction (default: false)"
    echo "  SKIP_TRAINING: true to skip training (default: false)"
    exit 1
fi

export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_DISABLE_XET=1

echo "=========================================="
echo "delta-QKV Pipeline"
echo "Config: $CONFIG"
echo "Skip Extraction: $SKIP_EXTRACTION"
echo "Skip Training: $SKIP_TRAINING"
echo "=========================================="

# STEP 1: EXTRACT Q/K/V IMAGES (if not skipped)
if [ "$SKIP_EXTRACTION" != "true" ]; then
    echo ""
    echo "STEP 1/4: Extracting Q/K/V token images..."
    echo "  Generating responses, capturing activations, and building token images."
    python main.py --config "$CONFIG" extract
    if [ $? -ne 0 ]; then
        echo "ERROR: Extraction failed"
        exit 1
    fi
fi

# STEP 2: INSPECT EXTRACTED DATA
echo ""
echo "STEP 2/4: Inspecting extracted token images..."
echo "  Verifying image quality and cross-view correlation."
python main.py --config "$CONFIG" inspect --idx 0
if [ $? -ne 0 ]; then
    echo "ERROR: Inspection failed"
    exit 1
fi

# STEP 3: TRAIN DETECTOR (if not skipped)
if [ "$SKIP_TRAINING" != "true" ]; then
    echo ""
    echo "STEP 3/4: Training hallucination detector..."
    python main.py --config "$CONFIG" train
    if [ $? -ne 0 ]; then
        echo "ERROR: Training failed"
        exit 1
    fi

    # Extract the run directory from the most recent checkpoint
    RUN_DIR=$(ls -td runs/*/ | head -1 | sed 's:/$::')
    CHECKPOINT="$RUN_DIR/best.pt"

    if [ ! -f "$CHECKPOINT" ]; then
        echo "ERROR: Best checkpoint not found at $CHECKPOINT"
        exit 1
    fi
else
    # If training was skipped, find the most recent checkpoint
    CHECKPOINT=$(find runs -name "best.pt" -type f -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
    if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
        echo "ERROR: No checkpoint found. Either run training (SKIP_TRAINING=false) or ensure runs/*/best.pt exists"
        exit 1
    fi
fi

# STEP 4: EVALUATE ON TEST SET
echo ""
echo "STEP 4/4: Evaluating detector on test set..."
echo "  Checkpoint: $CHECKPOINT"
python main.py --config "$CONFIG" test --checkpoint "$CHECKPOINT"
if [ $? -ne 0 ]; then
    echo "ERROR: Testing failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Pipeline completed successfully!"
echo "Checkpoint: $CHECKPOINT"
echo "=========================================="