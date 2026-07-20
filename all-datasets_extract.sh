# DATASET=truthfulqa
# MODEL=qwen2.5_1.5b

# python main.py --config configs/$DATASET/$MODEL.yaml extract
# python main.py --config configs/$DATASET/$MODEL.yaml extract --set dataset.name=${DATASET}_test


# for loop for running inspect from 0 to 9
# for i in {0..9}; do
#     python main.py --config configs/$DATASET/$MODEL.yaml inspect --idx $i
#     python main.py --config configs/$DATASET/$MODEL.yaml cam \
#         --checkpoint runs/same_$MODEL_$DATASET_x2/best.pt \
#         --dataset $DATASET --idx 0 --method gradcam
# done

# python main.py --config configs/$DATASET/$MODEL.yaml train --run-name same_${MODEL}_${DATASET}_x2
# python main.py --config configs/$DATASET/$MODEL.yaml test --checkpoint "runs/same_${MODEL}_${DATASET}_x2/best.pt" --dataset ${DATASET}_test


#!/bin/bash

DATASETS=(
    truthfulqa
    tydiqa
    coqa
    hotpotqa
    hotpotqa_with_context
    triviaqa
)

MODELS=(
    qwen2.5_7b
)

# TruthfulQA has no separate upstream test/validation split (src/extract/datasets.py
# SPLIT_SOURCES) -- it gets a stratified held-out slice carved out of the single
# corpus at train time instead of a `<name>_test` extraction.
declare -A HAS_TEST_CORPUS=(
    [truthfulqa]=0
    [tydiqa]=1
    [coqa]=1
    [hotpotqa]=1
    [hotpotqa_with_context]=1
    [triviaqa]=1
)

for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do

        echo "========================================"
        echo "Dataset: $DATASET"
        echo "Model:   $MODEL"
        echo "========================================"

        # Extract training set
        python main.py --config configs/$DATASET/$MODEL.yaml extract

        # Extract test set (only for datasets with a real held-out corpus)
        if [ "${HAS_TEST_CORPUS[$DATASET]}" = "1" ]; then
            python main.py \
                --config configs/$DATASET/$MODEL.yaml \
                extract \
                --set dataset.name=${DATASET}_test
        fi

        # Inspect first 10 examples
        for i in {0..9}; do
            python main.py \
                --config configs/$DATASET/$MODEL.yaml \
                inspect \
                --idx $i

        done
    done
done
