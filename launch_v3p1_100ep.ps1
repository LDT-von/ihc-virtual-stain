"""Launch a 100-epoch CD68=3.0 refit starting from v3 final.pt.
Independent run (does NOT touch fullplus_cd68_v3_continue).
"""
$ErrorActionPreference = 'Stop'
Set-Location E:\aic\final-ihc

$log = 'logs\fullplus_cd68_v3p1_100ep.log'
$err = 'logs\fullplus_cd68_v3p1_100ep.err'

$ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Add-Content -Path $log -Value "[$ts] launching fullplus_cd68_v3p1 (100 epochs from v3 final, lr=1e-5, cd68=3.0)"

# Start job so it survives this shell closing
$job = Start-Job -ScriptBlock {
    param($log, $err)
    & python train_fullplus_cd68.py `
        --data-root 'E:\aic\复赛数据集(包括训练集和测试集输入)' `
        --manifest 'E:\aic\final-ihc\configs\roi_split_semifinal_2026_expanded.json' `
        --init-checkpoint 'E:\aic\final-ihc\checkpoints\fullplus_cd68_v3\final.pt' `
        --output 'E:\aic\final-ihc\checkpoints\fullplus_cd68_v3p1' `
        --epochs 100 `
        --batch-size 8 `
        --lr 1e-5 `
        --cd68-weight 3.0 `
        --seed 2026 `
        1> $log 2> $err
} -ArgumentList $log, $err -Name 'fullplus_cd68_v3p1'

Write-Host "[$ts] Job started. Use Get-Job -Name fullplus_cd68_v3p1 | Receive-Job to peek output."
Write-Host "  log: $log"
Write-Host "  err: $err"
Write-Host "  history: checkpoints\fullplus_cd68_v3p1\history.jsonl"
