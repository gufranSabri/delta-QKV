DATASET=triviaqa
MODEL=llama2_7b

# python main.py --config configs/$DATASET/$MODEL.yaml extract
python main.py --config configs/$DATASET/$MODEL.yaml extract --set dataset.name=${DATASET}_test

python main.py --config configs/$DATASET/$MODEL.yaml inspect --idx 0

python main.py --config configs/$DATASET/$MODEL.yaml train --run-name same_${MODEL}_${DATASET}
python main.py --config configs/$DATASET/$MODEL.yaml test --checkpoint "runs/same_${MODEL}_${DATASET}/best.pt" --dataset ${DATASET}_test