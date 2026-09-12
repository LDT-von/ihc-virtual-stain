@echo off
REM 等待 HLA-DR 完成，然后启动 chain
:loop
timeout /t 60 /nobreak > nul
powershell -NoProfile -Command "Get-ChildItem E:\aic\ihc-virtual-stain\checkpoints -Directory | Where-Object { $_.Name -like 'pix2pix_v6_HLA-DR*' -and (Test-Path (Join-Path $_.FullName 'best.pt')) } | Select-Object -First 1"
if exist "E:\aic\ihc-virtual-stain\checkpoints\pix2pix_v6_HLA-DR*\best.pt" (
    echo [MONITOR] HLA-DR best.pt 出现！启动 chain
    python E:\aic\ihc-virtual-stain\train_v6_chain.py
    exit /b 0
)
goto loop
