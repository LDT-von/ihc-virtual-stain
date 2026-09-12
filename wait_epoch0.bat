@echo off
REM 等待 v6 第一个 epoch 完成 (日志里出现 "Epoch 0")
:loop
timeout /t 30 /nobreak > nul
powershell -NoProfile -Command "$log = Get-Content 'E:\aic\ihc-virtual-stain\train_v6_HLA-DR.log' -Tail 100; if ($log -match 'Epoch 0|Saved|Done') { exit 0 } else { exit 1 }"
if %ERRORLEVEL% equ 0 (
    echo [MONITOR] Epoch 0 已完成！
    exit /b 0
)
goto loop
