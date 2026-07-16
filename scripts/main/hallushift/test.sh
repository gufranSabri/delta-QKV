#!/bin/bash
# ============================================================================
# HalluShift -- TEST commands (main experiments: SAME).
#
# Scores each trained run on its held-out corpus, appending to docs/results.csv.
# ============================================================================
set -x

python main.py --config configs/truthfulqa/llama2_7b.yaml test --checkpoint runs/same_llama2_7b_truthfulqa/best.pt --dataset truthfulqa
python main.py --config configs/triviaqa/llama2_7b.yaml test --checkpoint runs/same_llama2_7b_triviaqa/best.pt --dataset triviaqa
python main.py --config configs/coqa/llama2_7b.yaml test --checkpoint runs/same_llama2_7b_coqa/best.pt --dataset coqa
python main.py --config configs/tydiqa/llama2_7b.yaml test --checkpoint runs/same_llama2_7b_tydiqa/best.pt --dataset tydiqa
python main.py --config configs/truthfulqa/llama3.1_8b.yaml test --checkpoint runs/same_llama3.1_8b_truthfulqa/best.pt --dataset truthfulqa
python main.py --config configs/triviaqa/llama3.1_8b.yaml test --checkpoint runs/same_llama3.1_8b_triviaqa/best.pt --dataset triviaqa
python main.py --config configs/coqa/llama3.1_8b.yaml test --checkpoint runs/same_llama3.1_8b_coqa/best.pt --dataset coqa
python main.py --config configs/tydiqa/llama3.1_8b.yaml test --checkpoint runs/same_llama3.1_8b_tydiqa/best.pt --dataset tydiqa
python main.py --config configs/truthfulqa/opt_6.7b.yaml test --checkpoint runs/same_opt_6.7b_truthfulqa/best.pt --dataset truthfulqa
python main.py --config configs/triviaqa/opt_6.7b.yaml test --checkpoint runs/same_opt_6.7b_triviaqa/best.pt --dataset triviaqa
python main.py --config configs/coqa/opt_6.7b.yaml test --checkpoint runs/same_opt_6.7b_coqa/best.pt --dataset coqa
python main.py --config configs/tydiqa/opt_6.7b.yaml test --checkpoint runs/same_opt_6.7b_tydiqa/best.pt --dataset tydiqa
