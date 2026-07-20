DATASET=tydiqa
MODEL=qwen2.5_1.5b

# ============================================================================
# Extraction: qkv (Q/K/V projections) AND hs (hidden states) for this dataset.
# The qkv ablation loop below reads the qkv extraction; the hs block at the
# bottom reads the hs one. Both must run before their respective train/test
# calls -- extraction only needs to happen once per (source, dataset, model).
# ============================================================================
python main.py --config configs/$DATASET/$MODEL.yaml extract
python main.py --config configs/$DATASET/$MODEL.yaml extract --set extract.source=hs --set "extract.views=[H]"

# ============================================================================
# Ablation (qkv source): every (channels, include) combination.
#
#   channels=default -> one image PER VIEW      (0=Q, 1=K, 2=V)
#   channels=same     -> one image PER CHANNEL   (0=raw, 1=ch1, 2=ch2)
#
# include indices mean different things depending on channels (see above),
# but the sweep itself is the same 6 include values under each mode:
#   null, [0], [0,1], [1], [2], [1,2]
# ============================================================================

for CHANNELS in same default; do
  for INCLUDE_LABEL in null 0 01 1 2 12; do
    case $INCLUDE_LABEL in
      null) INCLUDE=null ;;
      0)    INCLUDE='[0]' ;;
      01)   INCLUDE='[0,1]' ;;
      1)    INCLUDE='[1]' ;;
      2)    INCLUDE='[2]' ;;
      12)   INCLUDE='[1,2]' ;;
    esac

    RUN_NAME=same_${MODEL}_${DATASET}_channels-${CHANNELS}_include-${INCLUDE_LABEL}
    python main.py --config configs/$DATASET/$MODEL.yaml train --run-name $RUN_NAME --set model.channels=$CHANNELS --set "model.include=$INCLUDE"
    python main.py --config configs/$DATASET/$MODEL.yaml test --checkpoint "runs/${RUN_NAME}/best.pt" --dataset $DATASET --set model.channels=$CHANNELS --set "model.include=$INCLUDE"

    python main.py \
      --config configs/$DATASET/$MODEL.yaml \
      cam \
      --checkpoint runs/${RUN_NAME}/best.pt \
      --dataset $DATASET \
      --idx 0 \
      --method gradcam
  done
done

# ============================================================================
# hs (hidden states) setting: source=hs, views=[H] -- a single view, so the
# channels/include sweep above doesn't apply the same way. One train+test run,
# default channels mode.
# ============================================================================
RUN_NAME=same_${MODEL}_${DATASET}_source-hs
python main.py --config configs/$DATASET/$MODEL.yaml train --run-name $RUN_NAME --set extract.source=hs --set "extract.views=[H]"
python main.py --config configs/$DATASET/$MODEL.yaml test --checkpoint "runs/${RUN_NAME}/best.pt" --dataset $DATASET --set extract.source=hs --set "extract.views=[H]"

python main.py \
  --config configs/$DATASET/$MODEL.yaml \
  cam \
  --checkpoint runs/${RUN_NAME}/best.pt \
  --dataset $DATASET \
  --idx 0 \
  --method gradcam
