#!/bin/bash
# ============================================================================
# HalluShift -- TRAIN commands (main experiments: SAME).
#
# Run extract.sh first. Copy lines one at a time into main.slurm, or run whole.
# LODO is omitted here on purpose -- add it back later if wanted.
# ============================================================================
set -x

python main.py --config configs/truthfulqa/llama2_7b.yaml train --run-name same_llama2_7b_truthfulqa
python main.py --config configs/triviaqa/llama2_7b.yaml train --run-name same_llama2_7b_triviaqa
python main.py --config configs/coqa/llama2_7b.yaml train --run-name same_llama2_7b_coqa
python main.py --config configs/tydiqa/llama2_7b.yaml train --run-name same_llama2_7b_tydiqa
python main.py --config configs/truthfulqa/llama3.1_8b.yaml train --run-name same_llama3.1_8b_truthfulqa
python main.py --config configs/triviaqa/llama3.1_8b.yaml train --run-name same_llama3.1_8b_triviaqa
python main.py --config configs/coqa/llama3.1_8b.yaml train --run-name same_llama3.1_8b_coqa
python main.py --config configs/tydiqa/llama3.1_8b.yaml train --run-name same_llama3.1_8b_tydiqa
python main.py --config configs/truthfulqa/opt_6.7b.yaml train --run-name same_opt_6.7b_truthfulqa
python main.py --config configs/triviaqa/opt_6.7b.yaml train --run-name same_opt_6.7b_triviaqa
python main.py --config configs/coqa/opt_6.7b.yaml train --run-name same_opt_6.7b_coqa
python main.py --config configs/tydiqa/opt_6.7b.yaml train --run-name same_opt_6.7b_tydiqa
