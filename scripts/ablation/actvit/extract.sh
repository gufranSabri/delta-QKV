#!/bin/bash
# ============================================================================
# ACT-ViT ablations -- EXTRACTION.
#
# Every ablation runs on ONE anchor: triviaqa x llama3_8b. Extraction always
# stores the full Q/K/V stack plus the delta channels (features live at
# data/{source}/{extraction_type}/{dataset}/{llm_alias}/ -- here qkv/delta/...),
# so "drop a view" and "change boundary_mode" are TRAIN-time slices -- they need
# no separate corpus. One extraction of the anchor covers all of them.
#
# If you already ran scripts/main/actvit/extract.sh this corpus is on disk and
# both lines are no-ops (an already-extracted corpus is skipped).
#
#   bash scripts/ablation/actvit/extract.sh
# ============================================================================
set -x

HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"
purge() {  # purge <models--...> [datasets--...]
  for entry in "$@"; do
    [ -n "$entry" ] && rm -rf "$HUB/$entry"
  done
}

python main.py --config configs/triviaqa/llama3_8b.yaml extract
python main.py --config configs/triviaqa/llama3_8b.yaml extract --set dataset.name=triviaqa_test
purge "models--meta-llama--Meta-Llama-3-8B-Instruct" "datasets--trivia_qa"
