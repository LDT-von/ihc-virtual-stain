$ROOT = 'E:\aic\ihc-virtual-stain'
$OUTROOT = "$ROOT\results_best_all_bestpt"
$LOG = "$ROOT\infer_best_all_bestpt.log"
$STATUS = "$ROOT\status_v5.txt"

$markers = @{
    'HLA-DR'   = 'pix2pix_v2_HLA-DR_1788962683\best.pt'
    'CD68'     = 'pix2pix_v2_CD68_1788969904\best.pt'
    'CD45RO'   = 'pix2pix_v2_CD45RO_1788978395\best.pt'
    'Vimentin' = 'pix2pix_v2_Vimentin_1788988426\best.pt'
}

if (-not (Test-Path $OUTROOT)) {
    New-Item -ItemType Directory -Force -Path $OUTROOT | Out-Null
}

"=== v5: ALL best.pt + 4xTTA test, started $(Get-Date) ===" | Out-File $LOG
"" | Out-File $LOG -Append

foreach ($m in @('HLA-DR','CD68','CD45RO','Vimentin')) {
    $ckpt = "$ROOT\checkpoints\$($markers[$m])"
    $outdir = "$OUTROOT\test\$m"
    $n = 0
    if (Test-Path $outdir) {
        $n = (Get-ChildItem $outdir).Count
    }
    "[$m] ckpt=$ckpt existing=$n" | Out-File $LOG -Append

    if ($n -ge 1346) {
        "[$m] skip (done)" | Out-File $LOG -Append
        continue
    }

    if (Test-Path $outdir) {
        Remove-Item $outdir -Recurse -Force
    }

    "[$m] launching..." | Out-File $LOG -Append
    $arglist = "python inference_8x_tta.py --marker $m --ckpt `"$ckpt`" --split test --tta-mode 4x --batch-size 4 --output-root `"$OUTROOT`" > `"$LOG.$m.out`" 2> `"$LOG.$m.err`""

    cmd.exe /c $arglist
    "[$m] finished exit=$LASTEXITCODE, files=$((Get-ChildItem $outdir -ErrorAction SilentlyContinue).Count)" | Out-File $LOG -Append
}

"=== v5 ALL DONE ===" | Out-File $LOG -Append
