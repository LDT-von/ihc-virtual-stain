@echo off
REM 用 67.4586 那组老 ckpt 跑 4xTTA test 推理
REM 目标：恢复 67.45+ 分数

set ROOT=E:\aic\ihc-virtual-stain
cd /d %ROOT%

set OUTROOT=%ROOT%\results_best_4xtta

echo === Best v3: Old 67.4586 ckpts with 4xTTA ===
echo Output root: %OUTROOT%
echo.

REM HLA-DR
echo [1/4] HLA-DR with old ckpt epoch89
python inference_8x_tta.py ^
  --marker HLA-DR ^
  --ckpt "%ROOT%\checkpoints\pix2pix_v2_HLA-DR_1788962683\epoch89.pt" ^
  --split test ^
  --tta-mode 4x ^
  --batch-size 4 ^
  --output-root "%OUTROOT%"

REM CD68
echo [2/4] CD68 with old ckpt best
python inference_8x_tta.py ^
  --marker CD68 ^
  --ckpt "%ROOT%\checkpoints\pix2pix_v2_CD68_1788969904\best.pt" ^
  --split test ^
  --tta-mode 4x ^
  --batch-size 4 ^
  --output-root "%OUTROOT%"

REM CD45RO
echo [3/4] CD45RO with old ckpt best
python inference_8x_tta.py ^
  --marker CD45RO ^
  --ckpt "%ROOT%\checkpoints\pix2pix_v2_CD45RO_1788978395\best.pt" ^
  --split test ^
  --tta-mode 4x ^
  --batch-size 4 ^
  --output-root "%OUTROOT%"

REM Vimentin
echo [4/4] Vimentin with old ckpt epoch69
python inference_8x_tta.py ^
  --marker Vimentin ^
  --ckpt "%ROOT%\checkpoints\pix2pix_v2_Vimentin_1788988426\epoch69.pt" ^
  --split test ^
  --tta-mode 4x ^
  --batch-size 4 ^
  --output-root "%OUTROOT%"

echo.
echo === Done ===
