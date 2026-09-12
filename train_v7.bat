@echo off
REM 训练改进脚本 v7 - 同时训练 CD45RO 和 Vimentin
REM 关键改进：1) 排除验证集重叠 2) 15 epoch 3) Cosine LR 衰减

cd /d E:\aic\ihc-virtual-stain

echo ============================================================
echo v7 Training: CD45RO + Vimentin (15 epochs each)
echo ============================================================

REM 启动 CD45RO 训练
start "CD45RO_v7" python train_v7.py --marker CD45RO --epochs 15 --batch_size 16 --lambda_pyr 150 --lambda_ssim 50 --d_every_n 3 --log_every 50 > train_v7_CD45RO.log 2>&1

REM 等待一小会让第一个任务启动
timeout /t 5 /nobreak >nul

REM 启动 Vimentin 训练
start "Vimentin_v7" python train_v7.py --marker Vimentin --epochs 15 --batch_size 16 --lambda_pyr 150 --lambda_ssim 50 --d_every_n 3 --log_every 50 > train_v7_Vimentin.log 2>&1

echo Training started!
echo CD45RO log: train_v7_CD45RO.log
echo Vimentin log: train_v7_Vimentin.log
echo.
echo Check progress with:
echo   type train_v7_CD45RO.log
echo   type train_v7_Vimentin.log
echo.
echo Wait for both to complete, then run validate_all.bat
