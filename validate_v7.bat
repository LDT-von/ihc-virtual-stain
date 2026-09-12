@echo off
REM 在真实验证集上评估 v7 训练结果

cd /d E:\aic\ihc-virtual-stain

echo ============================================================
echo v7 Validation: Check SSIM on 20%% val split
echo ============================================================

REM 查找最新的 v7 checkpoint 目录
for /f "delims=" %%d in ('dir /b /ad /o-n "checkpoints\pix2pix_v7_*" 2^>nul') do (
    set latest_cd45ro=%%d
    goto :found_cd45ro
)
:found_cd45ro

echo Found v7 checkpoints directory: %latest_cd45ro%
echo.

REM CD45RO
echo [1/2] Validating CD45RO...
for /f "delims=" %%f in ('dir /b "%latest_cd45ro%\CD45RO\best.pt" 2^>nul') do (
    python validate.py --marker CD45RO --ckpt "%latest_cd45ro%\CD45RO\best.pt" --batch_size 8 --num_workers 0 >> validate_v7.log 2>&1
    echo CD45RO done, check validate_v7.log
)

REM Vimentin
for /f "delims=" %%f in ('dir /b "%latest_cd45ro%\Vimentin\best.pt" 2^>nul') do (
    python validate.py --marker Vimentin --ckpt "%latest_cd45ro%\Vimentin\best.pt" --batch_size 8 --num_workers 0 >> validate_v7.log 2>&1
    echo Vimentin done, check validate_v7.log
)

echo.
echo ============================================================
echo Validation complete. Check validate_v7.log for results.
echo ============================================================
