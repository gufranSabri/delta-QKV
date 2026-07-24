#!/bin/bash
# Train + test every (dataset x LLM) with default settings -- no ablation
# overrides. Run all-datasets_extract.sh first so the corpora are on disk.

DATASETS=(
    triviaqa
    truthfulqa
    coqa
)

MODELS=(
    llama2_7b
)

for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do

        echo "========================================"
        echo "Dataset: $DATASET"
        echo "Model:   $MODEL"
        echo "========================================"

        RUN_NAME=same_${MODEL}_${DATASET}
        RUN_NAME_DELTA=same_${MODEL}_${DATASET}_delta

        python main.py --config configs/$DATASET/$MODEL.yaml train --run-name runs/$RUN_NAME
        python main.py --config configs/$DATASET/$MODEL.yaml test --checkpoint "runs/${RUN_NAME}/best.pt" --dataset $DATASET

        python main.py --config configs/$DATASET/$MODEL.yaml train --run-name runs/$RUN_NAME_DELTA --set extract.extraction_type=delta
        python main.py --config configs/$DATASET/$MODEL.yaml test --checkpoint "runs/${RUN_NAME_DELTA}/best.pt" --dataset $DATASET --set extract.extraction_type=delta

    done
done
