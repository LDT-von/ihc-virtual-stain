#!/usr/bin/env bash
# Sequential MCN runs: 100-epoch resume + 20-epoch warm-start + ResNet50 v2.
# Each run uses cuda:1; DCT training on cuda:0 is untouched.
set -euo pipefail
PY=/home/ubuntu/.conda/envs/trisurv/bin/python
TR=/data1/AIC
LOG=/data1/AIC/logs
mkdir -p "$LOG" "$TR/checkpoints"

DEVICE="cuda:1"
WIDTH=32
BS=8
LR=5e-4

V1_BEST="$TR/checkpoints/mcn_v1/best.pt"

echo "================================================================"
echo "[1/3] MCN 100-epoch RESUME from $V1_BEST (adaptive weights)"
echo "================================================================"
"$PY" -m src.train_marker_context train \
  --data-root /data1/AIC/data \
  --manifest /data1/AIC/configs/roi_split_5fold/fold_0.json \
  --output "$TR/checkpoints/mcn_v1_100ep" \
  --resume "$V1_BEST" \
  --extend-epochs \
  --epochs 100 \
  --width "$WIDTH" \
  --batch-size "$BS" \
  --lr "$LR" \
  --device "$DEVICE" \
  --seed 42 \
  --adaptive-marker-weights \
  --adaptive-alpha 0.95 \
  --adaptive-eps 1e-3 \
  > "$LOG/mcn_v1_100ep.log" 2>&1
tail -5 "$LOG/mcn_v1_100ep.log"
echo "[1/3] done at $(date -u +%FT%TZ)"

echo "================================================================"
echo "[2/3] MCN 20-epoch warm-start (cosine schedule 0..20, adaptive)"
echo "================================================================"
"$PY" -m src.train_marker_context train \
  --data-root /data1/AIC/data \
  --manifest /data1/AIC/configs/roi_split_5fold/fold_0.json \
  --output "$TR/checkpoints/mcn_v1_20ep" \
  --init-checkpoint "$V1_BEST" \
  --epochs 20 \
  --width "$WIDTH" \
  --batch-size "$BS" \
  --lr "$LR" \
  --device "$DEVICE" \
  --seed 42 \
  --adaptive-marker-weights \
  --adaptive-alpha 0.95 \
  --adaptive-eps 1e-3 \
  > "$LOG/mcn_v1_20ep.log" 2>&1
tail -5 "$LOG/mcn_v1_20ep.log"
echo "[2/3] done at $(date -u +%FT%TZ)"

echo "================================================================"
echo "[3/3] ResNet50-UNet v2 (bs=8, lr=5e-4, 300ep, 8x sample loader)"
echo "================================================================"
cd /data1/AIC
# ResNet training script location:
ls "$TR"/scripts/train_resnet_unet*.sh 2>/dev/null || true
echo "TODO: launch ResNet50 v2 training script"
