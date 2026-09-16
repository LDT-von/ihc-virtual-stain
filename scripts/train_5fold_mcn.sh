#!/usr/bin/env bash
# 5-fold MCN training: fold 0..4, 50 epochs each
set -e
cd /data1/AIC
PY=/home/ubuntu/.conda/envs/trisurv/bin/python
DEVICE=cuda:1
EPOCHS=50
WIDTH=32
BS=8
LR=5e-4

for FOLD in 0 1 2 3 4; do
  echo "================================================================"
  echo "Starting MCN 5-fold fold $FOLD ($EPOCHS epochs, width=$WIDTH)"
  echo "================================================================"
  $PY -m src.train_marker_context train \
    --data-root /data1/AIC/data \
    --manifest /data1/AIC/configs/roi_split_5fold/fold_${FOLD}.json \
    --output checkpoints/mcn_5fold_fold${FOLD} \
    --epochs $EPOCHS --width $WIDTH --batch-size $BS --lr $LR \
    --device $DEVICE 2>&1 | tee logs/mcn_5fold_fold${FOLD}.log
done

echo "All 5 folds complete!"
