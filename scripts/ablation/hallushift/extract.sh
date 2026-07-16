#!/bin/bash
# ============================================================================
# HalluShift ablations -- EXTRACTION. Anchor: triviaqa x llama2_7b.
#
# As with the ACT-ViT ablations, extraction stores the full Q/K/V stack plus
# delta channels, so every view/fusion/boundary/backbone ablation reuses this
# one corpus -- no per-ablation extraction. BLEURT is required (the slurm
# wrapper installs it).
#
# If you already ran scripts/main/hallushift/extract.sh this corpus is on disk
# and both lines are no-ops.
#
#   bash scripts/ablation/hallushift/extract.sh
# ============================================================================
set -x

HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"
purge() {  # purge <models--...> [datasets--...]
  for entry in "$@"; do
    [ -n "$entry" ] && rm -rf "$HUB/$entry"
  done
}

python main.py --config configs/triviaqa/llama2_7b.yaml extract
python main.py --config configs/triviaqa/llama2_7b.yaml extract --set dataset.name=triviaqa_test
purge "models--meta-llama--Llama-2-7b-hf" "datasets--trivia_qa"
