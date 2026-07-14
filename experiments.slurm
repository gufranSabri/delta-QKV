#!/bin/bash
#SBATCH --job-name=delta-qkv-experiments
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
#SBATCH --time=30-00:00:00

module load StdEnv/2023 gcc/12.3 cuda/13.2 arrow/23.0.1 python/3.11.5
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install -q -r requirements.txt

export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_DISABLE_XET=1

echo "=========================================="
echo "delta-QKV Full Experiment Suite"
echo "=========================================="
echo "This script runs all experiments:"
echo "  1. SAME      - train/test on individual datasets"
echo "  2. LODO      - leave-one-dataset-out cross-validation"
echo "  3. ABLATE    - architecture ablations (views, fusion, backbone)"
echo ""
echo "Total experiments: ~90 (varies by config)"
echo "Estimated runtime: 7-14 days on single GPU"
echo "=========================================="

# Generate all experiment scripts
python scripts/gen_experiment_scripts.py --mode all

# Create a log file for this experiment run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="slurm/experiments_${TIMESTAMP}"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master.log"

echo "[$(date)] Starting experiment suite" | tee "$MASTER_LOG"

# ── STEP 1: EXTRACT DATA FOR ALL DATASETS AND LLMS ──────────────────────────
# This is a one-time cost. Once complete, all configs can reuse the extracted data.
echo ""
echo "========== STEP 1: DATA EXTRACTION ==========" | tee -a "$MASTER_LOG"
echo "[$(date)] Extracting Q/K/V token images for all datasets/LLMs..." | tee -a "$MASTER_LOG"

# Extract for all dataset-LLM pairs
CONFIGS=(
  "configs/triviaqa/llama3_8b.yaml"
  "configs/triviaqa/mistral_7b.yaml"
  "configs/triviaqa/qwen2.5_7b.yaml"
  "configs/hotpotqa/llama3_8b.yaml"
  "configs/hotpotqa/mistral_7b.yaml"
  "configs/hotpotqa/qwen2.5_7b.yaml"
  "configs/hotpotqa_with_context/llama3_8b.yaml"
  "configs/hotpotqa_with_context/mistral_7b.yaml"
  "configs/hotpotqa_with_context/qwen2.5_7b.yaml"
  "configs/imdb/llama3_8b.yaml"
  "configs/imdb/mistral_7b.yaml"
  "configs/imdb/qwen2.5_7b.yaml"
  "configs/movies/llama3_8b.yaml"
  "configs/movies/mistral_7b.yaml"
  "configs/movies/qwen2.5_7b.yaml"
  "configs/truthfulqa/llama3_8b.yaml"
  "configs/truthfulqa/mistral_7b.yaml"
  "configs/truthfulqa/qwen2.5_7b.yaml"
)

for config in "${CONFIGS[@]}"; do
  dataset=$(echo "$config" | cut -d'/' -f2)
  llm=$(basename "$config" .yaml)
  data_marker="data/${dataset}/${llm}/manifest.jsonl"

  if [ -f "$data_marker" ]; then
    echo "[$(date)] SKIPPED (already extracted): $config" | tee -a "$MASTER_LOG"
  else
    echo "[$(date)] Extracting: $config" | tee -a "$MASTER_LOG"
    python main.py --config "$config" extract 2>&1 | tee -a "$LOG_DIR/extract_${llm}_${dataset}.log"
    if [ $? -ne 0 ]; then
      echo "[$(date)] ERROR: Extraction failed for $config" | tee -a "$MASTER_LOG"
    fi
  fi
done

echo "[$(date)] Data extraction complete" | tee -a "$MASTER_LOG"


# Helper function to run experiments and skip completed ones
run_experiment_suite() {
  local mode=$1
  local suite_script="scripts/generated/${mode}/run_all.sh"
  local temp_script="/tmp/run_${mode}_filtered.sh"

  echo "[$(date)] Processing ${mode} experiments..." | tee -a "$MASTER_LOG"

  # Read the original script and filter out completed runs
  {
    head -3 "$suite_script"  # Keep header lines
    echo ""
    grep -E '^(echo|bash)' "$suite_script" | while read line; do
      if [[ "$line" =~ "bash" ]]; then
        # Extract run name from the bash command
        script_name=$(echo "$line" | sed -E 's/.*\/(.*\.sh).*/\1/')
        run_name=$(basename "$script_name" .sh)

        # Check if this run already exists
        if [ -d "runs/*${run_name}" ] 2>/dev/null || ls runs/*${run_name}* &>/dev/null 2>&1; then
          echo "# [SKIPPED - already exists] $run_name" >&2
          echo "# [SKIPPED] $line"
        else
          echo "$line"
        fi
      else
        echo "$line"
      fi
    done
  } > "$temp_script"

  bash "$temp_script" 2>&1 | tee -a "$LOG_DIR/${mode}_experiments.log"
  local exit_code=$?
  rm -f "$temp_script"
  return $exit_code
}


# ── STEP 2: RUN SAME EXPERIMENTS ──────────────────────────────────────────
# Train and test on individual datasets
echo ""
echo "========== STEP 2: SAME EXPERIMENTS ==========" | tee -a "$MASTER_LOG"
echo "[$(date)] Running same-dataset experiments (skipping completed)..." | tee -a "$MASTER_LOG"

run_experiment_suite "same"
SAME_EXIT=$?
if [ $SAME_EXIT -eq 0 ]; then
  echo "[$(date)] ✓ SAME experiments completed successfully" | tee -a "$MASTER_LOG"
else
  echo "[$(date)] ⚠ SAME experiments finished (check logs for details)" | tee -a "$MASTER_LOG"
fi


# ── STEP 3: RUN LODO EXPERIMENTS ──────────────────────────────────────────
# Leave-one-dataset-out cross-validation
echo ""
echo "========== STEP 3: LODO EXPERIMENTS ==========" | tee -a "$MASTER_LOG"
echo "[$(date)] Running leave-one-dataset-out experiments (skipping completed)..." | tee -a "$MASTER_LOG"

run_experiment_suite "lodo"
LODO_EXIT=$?
if [ $LODO_EXIT -eq 0 ]; then
  echo "[$(date)] ✓ LODO experiments completed successfully" | tee -a "$MASTER_LOG"
else
  echo "[$(date)] ⚠ LODO experiments finished (check logs for details)" | tee -a "$MASTER_LOG"
fi


# ── STEP 4: RUN ABLATION EXPERIMENTS ──────────────────────────────────────
# Architecture ablations: views, fusion, boundary modes, backbones
echo ""
echo "========== STEP 4: ABLATION EXPERIMENTS ==========" | tee -a "$MASTER_LOG"
echo "[$(date)] Running ablation experiments (skipping completed)..." | tee -a "$MASTER_LOG"
echo "  - View subsets: [Q], [K], [V], [Q,K], [Q,K,V]" | tee -a "$MASTER_LOG"
echo "  - Fusion types: gated, concat_mlp, bilinear, cross_attn" | tee -a "$MASTER_LOG"
echo "  - Boundary modes: zero, replicate, wrap" | tee -a "$MASTER_LOG"
echo "  - Backbones: scratch_cnn, resnet18, shared_backbone" | tee -a "$MASTER_LOG"

run_experiment_suite "ablate"
ABLATE_EXIT=$?
if [ $ABLATE_EXIT -eq 0 ]; then
  echo "[$(date)] ✓ Ablation experiments completed successfully" | tee -a "$MASTER_LOG"
else
  echo "[$(date)] ⚠ Ablation experiments finished (check logs for details)" | tee -a "$MASTER_LOG"
fi


# ── SUMMARY ───────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Experiment Suite Complete" | tee -a "$MASTER_LOG"
echo "=========================================="

# Count completed runs
NUM_RUNS=$(find runs -maxdepth 1 -type d | wc -l)
echo "Total runs completed: $((NUM_RUNS - 1))" | tee -a "$MASTER_LOG"

# Report results
echo "" | tee -a "$MASTER_LOG"
echo "Results Summary:" | tee -a "$MASTER_LOG"
echo "  SAME Experiments:     $([ $SAME_EXIT -eq 0 ] && echo "✓ PASS" || echo "✗ FAIL")" | tee -a "$MASTER_LOG"
echo "  LODO Experiments:     $([ $LODO_EXIT -eq 0 ] && echo "✓ PASS" || echo "✗ FAIL")" | tee -a "$MASTER_LOG"
echo "  Ablation Experiments: $([ $ABLATE_EXIT -eq 0 ] && echo "✓ PASS" || echo "✗ FAIL")" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "Log files:" | tee -a "$MASTER_LOG"
echo "  Master log: $MASTER_LOG" | tee -a "$MASTER_LOG"
echo "  Per-run logs: $LOG_DIR/" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "Run directory: runs/" | tee -a "$MASTER_LOG"

# Overall exit code
if [ $SAME_EXIT -eq 0 ] && [ $LODO_EXIT -eq 0 ] && [ $ABLATE_EXIT -eq 0 ]; then
  echo "[$(date)] All experiments completed successfully!" | tee -a "$MASTER_LOG"
  exit 0
else
  echo "[$(date)] Some experiments failed. Check logs for details." | tee -a "$MASTER_LOG"
  exit 1
fi
