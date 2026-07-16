#!/bin/bash
# ============================================================================
# ACT-ViT -- EXTRACTION for every (dataset x LLM), train + held-out test.
#
#   datasets  triviaqa, hotpotqa, hotpotqa_with_context, imdb, movies
#   LLMs      mistral_7b, llama3_8b, qwen2.5_7b
#   labels    exact_match  -- no BLEURT needed
#
# One (dataset x LLM) at a time: extract the train corpus, extract the held-out
# `<name>_test` corpus, then WIPE that LLM + dataset from the HuggingFace hub
# cache before moving on -- weights are re-downloaded on next use, but storage
# never piles up.  Run this once, in one go, up front.
#
#   bash scripts/main/actvit/extract.sh
# ============================================================================
set -x

# Delete one model + one dataset from the HF hub cache. Model dirs are named
# models--<org>--<model>, dataset dirs datasets--<name-with-slashes-as-dashes>.
HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"
purge() {  # purge <models--...> [datasets--...]
  for entry in "$@"; do
    [ -n "$entry" ] && rm -rf "$HUB/$entry"
  done
}

# ── triviaqa ────────────────────────────────────────────────────────────────
python main.py --config configs/triviaqa/mistral_7b.yaml extract
python main.py --config configs/triviaqa/mistral_7b.yaml extract --set dataset.name=triviaqa_test
purge "models--mistralai--Mistral-7B-Instruct-v0.2" "datasets--trivia_qa"
python main.py --config configs/triviaqa/llama3_8b.yaml extract
python main.py --config configs/triviaqa/llama3_8b.yaml extract --set dataset.name=triviaqa_test
purge "models--meta-llama--Meta-Llama-3-8B-Instruct" "datasets--trivia_qa"
python main.py --config configs/triviaqa/qwen2.5_7b.yaml extract
python main.py --config configs/triviaqa/qwen2.5_7b.yaml extract --set dataset.name=triviaqa_test
purge "models--Qwen--Qwen2.5-7B-Instruct" "datasets--trivia_qa"

# ── hotpotqa ──────────────────────────────────────────────────────────────
python main.py --config configs/hotpotqa/mistral_7b.yaml extract
python main.py --config configs/hotpotqa/mistral_7b.yaml extract --set dataset.name=hotpotqa_test
purge "models--mistralai--Mistral-7B-Instruct-v0.2" "datasets--hotpot_qa"
python main.py --config configs/hotpotqa/llama3_8b.yaml extract
python main.py --config configs/hotpotqa/llama3_8b.yaml extract --set dataset.name=hotpotqa_test
purge "models--meta-llama--Meta-Llama-3-8B-Instruct" "datasets--hotpot_qa"
python main.py --config configs/hotpotqa/qwen2.5_7b.yaml extract
python main.py --config configs/hotpotqa/qwen2.5_7b.yaml extract --set dataset.name=hotpotqa_test
purge "models--Qwen--Qwen2.5-7B-Instruct" "datasets--hotpot_qa"

# ── hotpotqa_with_context ─────────────────────────────────────────────────
python main.py --config configs/hotpotqa_with_context/mistral_7b.yaml extract
python main.py --config configs/hotpotqa_with_context/mistral_7b.yaml extract --set dataset.name=hotpotqa_with_context_test
purge "models--mistralai--Mistral-7B-Instruct-v0.2" "datasets--hotpot_qa"
python main.py --config configs/hotpotqa_with_context/llama3_8b.yaml extract
python main.py --config configs/hotpotqa_with_context/llama3_8b.yaml extract --set dataset.name=hotpotqa_with_context_test
purge "models--meta-llama--Meta-Llama-3-8B-Instruct" "datasets--hotpot_qa"
python main.py --config configs/hotpotqa_with_context/qwen2.5_7b.yaml extract
python main.py --config configs/hotpotqa_with_context/qwen2.5_7b.yaml extract --set dataset.name=hotpotqa_with_context_test
purge "models--Qwen--Qwen2.5-7B-Instruct" "datasets--hotpot_qa"

# ── imdb ──────────────────────────────────────────────────────────────────
python main.py --config configs/imdb/mistral_7b.yaml extract
python main.py --config configs/imdb/mistral_7b.yaml extract --set dataset.name=imdb_test
purge "models--mistralai--Mistral-7B-Instruct-v0.2" "datasets--imdb"
python main.py --config configs/imdb/llama3_8b.yaml extract
python main.py --config configs/imdb/llama3_8b.yaml extract --set dataset.name=imdb_test
purge "models--meta-llama--Meta-Llama-3-8B-Instruct" "datasets--imdb"
python main.py --config configs/imdb/qwen2.5_7b.yaml extract
python main.py --config configs/imdb/qwen2.5_7b.yaml extract --set dataset.name=imdb_test
purge "models--Qwen--Qwen2.5-7B-Instruct" "datasets--imdb"

# ── movies ────────────────────────────────────────────────────────────────
# movies ships as a local CSV (ACT-ViT/data/), not an HF dataset -- nothing to
# purge on the dataset side, so the second purge arg is empty.
python main.py --config configs/movies/mistral_7b.yaml extract
python main.py --config configs/movies/mistral_7b.yaml extract --set dataset.name=movies_test
purge "models--mistralai--Mistral-7B-Instruct-v0.2" ""
python main.py --config configs/movies/llama3_8b.yaml extract
python main.py --config configs/movies/llama3_8b.yaml extract --set dataset.name=movies_test
purge "models--meta-llama--Meta-Llama-3-8B-Instruct" ""
python main.py --config configs/movies/qwen2.5_7b.yaml extract
python main.py --config configs/movies/qwen2.5_7b.yaml extract --set dataset.name=movies_test
purge "models--Qwen--Qwen2.5-7B-Instruct" ""
