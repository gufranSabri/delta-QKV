#!/bin/bash
# ============================================================================
# ACT-ViT -- TRAIN commands (main experiments: SAME + LODO).
#
# Run extract.sh first. These assume every corpus is already on disk.
# Copy lines one at a time into main.slurm, or run the whole file.
#
#   SAME  train on a dataset, test on its own held-out twin
#   LODO  train on the other N-1 datasets, zero-shot the held-out one
# ============================================================================
set -x

# ── SAME ────────────────────────────────────────────────────────────────────
python main.py --config configs/triviaqa/mistral_7b.yaml train --run-name same_mistral_7b_triviaqa
python main.py --config configs/hotpotqa/mistral_7b.yaml train --run-name same_mistral_7b_hotpotqa
python main.py --config configs/hotpotqa_with_context/mistral_7b.yaml train --run-name same_mistral_7b_hotpotqa_with_context
python main.py --config configs/triviaqa/llama3_8b.yaml train --run-name same_llama3_8b_triviaqa
python main.py --config configs/hotpotqa/llama3_8b.yaml train --run-name same_llama3_8b_hotpotqa
python main.py --config configs/hotpotqa_with_context/llama3_8b.yaml train --run-name same_llama3_8b_hotpotqa_with_context
python main.py --config configs/triviaqa/qwen2.5_7b.yaml train --run-name same_qwen2.5_7b_triviaqa
python main.py --config configs/hotpotqa/qwen2.5_7b.yaml train --run-name same_qwen2.5_7b_hotpotqa
python main.py --config configs/hotpotqa_with_context/qwen2.5_7b.yaml train --run-name same_qwen2.5_7b_hotpotqa_with_context

# ── LODO ────────────────────────────────────────────────────────────────────
python main.py --config configs/triviaqa/mistral_7b.yaml train --train-datasets hotpotqa,hotpotqa_with_context --test-dataset triviaqa --run-name lodo_mistral_7b_holdout_triviaqa
python main.py --config configs/hotpotqa/mistral_7b.yaml train --train-datasets triviaqa,hotpotqa_with_context --test-dataset hotpotqa --run-name lodo_mistral_7b_holdout_hotpotqa
python main.py --config configs/hotpotqa_with_context/mistral_7b.yaml train --train-datasets triviaqa,hotpotqa --test-dataset hotpotqa_with_context --run-name lodo_mistral_7b_holdout_hotpotqa_with_context
python main.py --config configs/triviaqa/llama3_8b.yaml train --train-datasets hotpotqa,hotpotqa_with_context --test-dataset triviaqa --run-name lodo_llama3_8b_holdout_triviaqa
python main.py --config configs/hotpotqa/llama3_8b.yaml train --train-datasets triviaqa,hotpotqa_with_context --test-dataset hotpotqa --run-name lodo_llama3_8b_holdout_hotpotqa
python main.py --config configs/hotpotqa_with_context/llama3_8b.yaml train --train-datasets triviaqa,hotpotqa --test-dataset hotpotqa_with_context --run-name lodo_llama3_8b_holdout_hotpotqa_with_context
python main.py --config configs/triviaqa/qwen2.5_7b.yaml train --train-datasets hotpotqa,hotpotqa_with_context --test-dataset triviaqa --run-name lodo_qwen2.5_7b_holdout_triviaqa
python main.py --config configs/hotpotqa/qwen2.5_7b.yaml train --train-datasets triviaqa,hotpotqa_with_context --test-dataset hotpotqa --run-name lodo_qwen2.5_7b_holdout_hotpotqa
python main.py --config configs/hotpotqa_with_context/qwen2.5_7b.yaml train --train-datasets triviaqa,hotpotqa --test-dataset hotpotqa_with_context --run-name lodo_qwen2.5_7b_holdout_hotpotqa_with_context
