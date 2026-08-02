# DATASET=truthfulqa
# MODEL=qwen2.5_1.5b

# python main.py --config configs/$DATASET/$MODEL.yaml extract


# for loop for running inspect from 0 to 9
# for i in {0..9}; do
#     python main.py --config configs/$DATASET/$MODEL.yaml inspect --idx $i
#     python main.py --config configs/$DATASET/$MODEL.yaml cam \
#         --checkpoint runs/same_$MODEL_$DATASET_x2/best.pt \
#         --dataset $DATASET --idx 0 --method gradcam
# done

# python main.py --config configs/$DATASET/$MODEL.yaml train --run-name same_${MODEL}_${DATASET}_x2
# python main.py --config configs/$DATASET/$MODEL.yaml test --checkpoint "runs/same_${MODEL}_${DATASET}_x2/best.pt" --dataset ${DATASET}


#!/bin/bash

DATASETS=(
    triviaqa
    truthfulqa
    coqa
)

MODELS=(
    qwen2.5_7b
    # llama2_7b
    # llama3.1_8b
    # opt_6.7b
)

# truthfulqa/triviaqa/coqa all mirror HalluShift's single-split
# protocol (src/extract/datasets.py SPLIT_SOURCES): each loads ONE upstream
# split and gets a stratified held-out slice carved out of it at train time,
# so there is no separately extracted `<name>_test` corpus to pull here.

for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do

        echo "========================================"
        echo "Dataset: $DATASET"
        echo "Model:   $MODEL"
        echo "========================================"

        # Extract training set. One call populates every (source, extraction_type)
        # combination (qkv/hs x delta/transforms) from a single generation pass --
        # extract.source/extraction_type are not consulted by `extract` (see
        # src/extract/run_extraction.py), only by `train`/`test`/`cam`/`inspect`.
        python main.py --config configs/$DATASET/$MODEL.yaml extract \
            --set extract.batch_size=8

        # Inspect first 10 examples
        # for i in {0..9}; do
        #     python main.py \
        #         --config configs/$DATASET/$MODEL.yaml \
        #         inspect \
        #         --idx $i
        # done
    done
done
