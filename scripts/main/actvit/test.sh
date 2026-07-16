#!/bin/bash
# ============================================================================
# ACT-ViT -- TEST commands (main experiments: SAME + LODO).
#
# Each line scores a trained run on the held-out corpus and appends to
# docs/results.csv. Run the matching train.sh line first.
# ============================================================================
set -x

# ── SAME ────────────────────────────────────────────────────────────────────
python main.py --config configs/triviaqa/mistral_7b.yaml test --checkpoint runs/same_mistral_7b_triviaqa/best.pt --dataset triviaqa
python main.py --config configs/hotpotqa/mistral_7b.yaml test --checkpoint runs/same_mistral_7b_hotpotqa/best.pt --dataset hotpotqa
python main.py --config configs/hotpotqa_with_context/mistral_7b.yaml test --checkpoint runs/same_mistral_7b_hotpotqa_with_context/best.pt --dataset hotpotqa_with_context
python main.py --config configs/imdb/mistral_7b.yaml test --checkpoint runs/same_mistral_7b_imdb/best.pt --dataset imdb
python main.py --config configs/movies/mistral_7b.yaml test --checkpoint runs/same_mistral_7b_movies/best.pt --dataset movies
python main.py --config configs/triviaqa/llama3_8b.yaml test --checkpoint runs/same_llama3_8b_triviaqa/best.pt --dataset triviaqa
python main.py --config configs/hotpotqa/llama3_8b.yaml test --checkpoint runs/same_llama3_8b_hotpotqa/best.pt --dataset hotpotqa
python main.py --config configs/hotpotqa_with_context/llama3_8b.yaml test --checkpoint runs/same_llama3_8b_hotpotqa_with_context/best.pt --dataset hotpotqa_with_context
python main.py --config configs/imdb/llama3_8b.yaml test --checkpoint runs/same_llama3_8b_imdb/best.pt --dataset imdb
python main.py --config configs/movies/llama3_8b.yaml test --checkpoint runs/same_llama3_8b_movies/best.pt --dataset movies
python main.py --config configs/triviaqa/qwen2.5_7b.yaml test --checkpoint runs/same_qwen2.5_7b_triviaqa/best.pt --dataset triviaqa
python main.py --config configs/hotpotqa/qwen2.5_7b.yaml test --checkpoint runs/same_qwen2.5_7b_hotpotqa/best.pt --dataset hotpotqa
python main.py --config configs/hotpotqa_with_context/qwen2.5_7b.yaml test --checkpoint runs/same_qwen2.5_7b_hotpotqa_with_context/best.pt --dataset hotpotqa_with_context
python main.py --config configs/imdb/qwen2.5_7b.yaml test --checkpoint runs/same_qwen2.5_7b_imdb/best.pt --dataset imdb
python main.py --config configs/movies/qwen2.5_7b.yaml test --checkpoint runs/same_qwen2.5_7b_movies/best.pt --dataset movies

# ── LODO ────────────────────────────────────────────────────────────────────
python main.py --config configs/triviaqa/mistral_7b.yaml test --checkpoint runs/lodo_mistral_7b_holdout_triviaqa/best.pt --dataset triviaqa
python main.py --config configs/hotpotqa/mistral_7b.yaml test --checkpoint runs/lodo_mistral_7b_holdout_hotpotqa/best.pt --dataset hotpotqa
python main.py --config configs/hotpotqa_with_context/mistral_7b.yaml test --checkpoint runs/lodo_mistral_7b_holdout_hotpotqa_with_context/best.pt --dataset hotpotqa_with_context
python main.py --config configs/imdb/mistral_7b.yaml test --checkpoint runs/lodo_mistral_7b_holdout_imdb/best.pt --dataset imdb
python main.py --config configs/movies/mistral_7b.yaml test --checkpoint runs/lodo_mistral_7b_holdout_movies/best.pt --dataset movies
python main.py --config configs/triviaqa/llama3_8b.yaml test --checkpoint runs/lodo_llama3_8b_holdout_triviaqa/best.pt --dataset triviaqa
python main.py --config configs/hotpotqa/llama3_8b.yaml test --checkpoint runs/lodo_llama3_8b_holdout_hotpotqa/best.pt --dataset hotpotqa
python main.py --config configs/hotpotqa_with_context/llama3_8b.yaml test --checkpoint runs/lodo_llama3_8b_holdout_hotpotqa_with_context/best.pt --dataset hotpotqa_with_context
python main.py --config configs/imdb/llama3_8b.yaml test --checkpoint runs/lodo_llama3_8b_holdout_imdb/best.pt --dataset imdb
python main.py --config configs/movies/llama3_8b.yaml test --checkpoint runs/lodo_llama3_8b_holdout_movies/best.pt --dataset movies
python main.py --config configs/triviaqa/qwen2.5_7b.yaml test --checkpoint runs/lodo_qwen2.5_7b_holdout_triviaqa/best.pt --dataset triviaqa
python main.py --config configs/hotpotqa/qwen2.5_7b.yaml test --checkpoint runs/lodo_qwen2.5_7b_holdout_hotpotqa/best.pt --dataset hotpotqa
python main.py --config configs/hotpotqa_with_context/qwen2.5_7b.yaml test --checkpoint runs/lodo_qwen2.5_7b_holdout_hotpotqa_with_context/best.pt --dataset hotpotqa_with_context
python main.py --config configs/imdb/qwen2.5_7b.yaml test --checkpoint runs/lodo_qwen2.5_7b_holdout_imdb/best.pt --dataset imdb
python main.py --config configs/movies/qwen2.5_7b.yaml test --checkpoint runs/lodo_qwen2.5_7b_holdout_movies/best.pt --dataset movies
