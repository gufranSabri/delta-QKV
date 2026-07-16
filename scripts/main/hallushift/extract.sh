#!/bin/bash
# ============================================================================
# HalluShift -- EXTRACTION for every (dataset x LLM), train + held-out test.
#
#   datasets  truthfulqa, triviaqa, coqa, tydiqa
#   LLMs      llama2_7b, llama3.1_8b, opt_6.7b     (BASE models, not instruct)
#   labels    BLEURT-20-D12, hallucination iff score <= 0.5
#   gen       greedy, 64 new tokens
#
# BLEURT is REQUIRED -- the slurm wrapper installs it (scripts/install.sh --bleurt).
#
# One (dataset x LLM) at a time: extract train corpus, extract held-out test
# corpus (truthfulqa has none -- it gets a stratified slice at train time),
# then WIPE that LLM + dataset from the HF hub cache before moving on.
#
#   bash scripts/main/hallushift/extract.sh
# ============================================================================
set -x

HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"
purge() {  # purge <models--...> [datasets--...]
  for entry in "$@"; do
    [ -n "$entry" ] && rm -rf "$HUB/$entry"
  done
}
# CoQA is downloaded to data/raw/coqa/ (not the HF hub), so it has no hub entry
# to purge -- the coqa lines below pass an empty dataset arg.

# ── truthfulqa (no held-out twin) ───────────────────────────────────────────
python main.py --config configs/truthfulqa/llama2_7b.yaml extract
purge "models--meta-llama--Llama-2-7b-hf"
python main.py --config configs/truthfulqa/llama3.1_8b.yaml extract
purge "models--meta-llama--Llama-3.1-8B"
python main.py --config configs/truthfulqa/opt_6.7b.yaml extract
purge "models--facebook--opt-6.7b" "datasets--truthful_qa"

# ── triviaqa ────────────────────────────────────────────────────────────────
python main.py --config configs/triviaqa/llama2_7b.yaml extract
python main.py --config configs/triviaqa/llama2_7b.yaml extract --set dataset.name=triviaqa_test
purge "models--meta-llama--Llama-2-7b-hf"
python main.py --config configs/triviaqa/llama3.1_8b.yaml extract
python main.py --config configs/triviaqa/llama3.1_8b.yaml extract --set dataset.name=triviaqa_test
purge "models--meta-llama--Llama-3.1-8B"
python main.py --config configs/triviaqa/opt_6.7b.yaml extract
python main.py --config configs/triviaqa/opt_6.7b.yaml extract --set dataset.name=triviaqa_test
purge "models--facebook--opt-6.7b" "datasets--trivia_qa"

# ── coqa (downloaded to data/raw/coqa/, no hub dataset entry) ────────────────
python main.py --config configs/coqa/llama2_7b.yaml extract
python main.py --config configs/coqa/llama2_7b.yaml extract --set dataset.name=coqa_test
purge "models--meta-llama--Llama-2-7b-hf" ""
python main.py --config configs/coqa/llama3.1_8b.yaml extract
python main.py --config configs/coqa/llama3.1_8b.yaml extract --set dataset.name=coqa_test
purge "models--meta-llama--Llama-3.1-8B" ""
python main.py --config configs/coqa/opt_6.7b.yaml extract
python main.py --config configs/coqa/opt_6.7b.yaml extract --set dataset.name=coqa_test
purge "models--facebook--opt-6.7b" ""

# ── tydiqa ──────────────────────────────────────────────────────────────────
python main.py --config configs/tydiqa/llama2_7b.yaml extract
python main.py --config configs/tydiqa/llama2_7b.yaml extract --set dataset.name=tydiqa_test
purge "models--meta-llama--Llama-2-7b-hf" "datasets--tydiqa"
python main.py --config configs/tydiqa/llama3.1_8b.yaml extract
python main.py --config configs/tydiqa/llama3.1_8b.yaml extract --set dataset.name=tydiqa_test
purge "models--meta-llama--Llama-3.1-8B" "datasets--tydiqa"
python main.py --config configs/tydiqa/opt_6.7b.yaml extract
python main.py --config configs/tydiqa/opt_6.7b.yaml extract --set dataset.name=tydiqa_test
purge "models--facebook--opt-6.7b" "datasets--tydiqa"
