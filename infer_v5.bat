@echo off
set ROOT=E:\aic\ihc-virtual-stain
set OUTROOT=%ROOT%\results_best_all_bestpt
set LOG=%ROOT%\infer_best_all_bestpt.log

if not exist "%OUTROOT%" mkdir "%OUTROOT%"

echo === v5: ALL best.pt + 4xTTA started %date% %time% >> "%LOG%"

for %%M in (HLA-DR CD68 CD45RO Vimentin) do (
    if %%M==HLA-DR set CKP=pix2pix_v2_HLA-DR_1788962683\best.pt
    if %%M==CD68 set CKP=pix2pix_v2_CD68_1788969904\best.pt
    if %%M==CD45RO set CKP=pix2pix_v2_CD45RO_1788978395\best.pt
    if %%M==Vimentin set CKP=pix2pix_v2_Vimentin_1788988426\best.pt

    set CKPT=%ROOT%\checkpoints\%CKP%

    echo [%%M] using %%CKPT%% >> "%LOG%"

    if exist "%OUTROOT%\test\%%M" (
        for /f %%c in ('dir /b "%OUTROOT%\test\%%M" ^| find /c /v ""') do (
            if %%c GEQ 1346 (
                echo [%%M] skip (done, %%c files) >> "%LOG%"
            ) else (
                echo [%%M] incomplete (%%c files), re-running >> "%LOG%"
                rmdir /s /q "%OUTROOT%\test\%%M" 2>nul
                echo [%%M] launching >> "%LOG%"
                python "%ROOT%\inference_8x_tta.py" --marker %%M --ckpt "%%CKPT%%" --split test --tta-mode 4x --batch-size 4 --output-root "%OUTROOT%" >> "%LOG%.%%M.out" 2>> "%LOG%.%%M.err"
                for /f %%n in ('dir /b "%OUTROOT%\test\%%M" ^| find /c /v ""') do (
                    echo [%%M] done, files=%%n, exit=!ERRORLEVEL! >> "%LOG%"
                )
            )
        )
    ) else (
        echo [%%M] launching >> "%LOG%"
        python "%ROOT%\inference_8x_tta.py" --marker %%M --ckpt "%%CKPT%%" --split test --tta-mode 4x --batch-size 4 --output-root "%OUTROOT%" >> "%LOG%.%%M.out" 2>> "%LOG%.%%M.err"
        for /f %%n in ('dir /b "%OUTROOT%\test\%%M" ^| find /c /v ""') do (
            echo [%%M] done, files=%%n, exit=!ERRORLEVEL! >> "%LOG%"
        )
    )
)

echo === v5 ALL DONE %date% %time% >> "%LOG%"
