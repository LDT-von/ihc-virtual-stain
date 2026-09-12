@echo off
REM v6 一键推理 + 打包脚本 (使用 v6 训练出的 best.pt)
REM 由 Cursor Agent 自动运行

setlocal enabledelayedexpansion
set DATA_ROOT=E:/aic/ihc-virtual-stain/初赛数据集（包含训练集和测试集输入）/初赛数据集（包含训练集和测试集输入）

echo ============================================================
echo v6 PyramidPix2Pix 一键推理
echo ============================================================

REM 1. 找到最新的 v6 checkpoint 目录
for /f "delims=" %%d in ('powershell -NoProfile -Command "Get-ChildItem -Directory E:\aic\ihc-virtual-stain\checkpoints | Where-Object { $_.Name -like 'pix2pix_v6_*' } | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName"') do set V6_DIR=%%d
echo [INFO] 最新 v6 checkpoint 目录: !V6_DIR!

if "!V6_DIR!"=="" (
    echo [ERROR] 找不到 v6 checkpoint 目录！
    exit /b 1
)

REM 2. 按 marker 推理
for %%M in (HLA-DR CD68 CD45RO Vimentin) do (
    echo.
    echo [INFER] 推理 marker=%%M ...
    cd /d E:\aic\ihc-virtual-stain
    python -m src.inference_pix2pix --ckpt "!V6_DIR!\best.pt" --marker %%M --data-root "%DATA_ROOT%" --split test --output-dir results --tta --jpeg-quality 95
    if !ERRORLEVEL! neq 0 (
        echo [WARN] %%M TTA 推理失败，尝试 no-TTA
        python -m src.inference_pix2pix --ckpt "!V6_DIR!\best.pt" --marker %%M --data-root "%DATA_ROOT%" --split test --output-dir results --no-tta --jpeg-quality 95
    )
)

REM 3. 打包提交
echo.
echo [SUBMIT] 打包所有 marker
cd /d E:\aic\ihc-virtual-stain
python -m src.submit --results-dir results --marker all --out-zip submission_v6.zip

echo.
echo ============================================================
echo v6 完成！提交包: submission_v6.zip
echo ============================================================

endlocal
