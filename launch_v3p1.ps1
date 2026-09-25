$ErrorActionPreference = 'Stop'
Set-Location E:\aic\final-ihc

New-Item -ItemType Directory -Force -Path 'logs' | Out-Null
New-Item -ItemType Directory -Force -Path 'checkpoints\fullplus_cd68_v3p1' | Out-Null

$ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
$entry = "[$ts] launching fullplus_cd68_v3p1"
Add-Content -Path 'logs/fullplus_cd68_v3p1.log' -Value $entry

# Use Start-Process so the python survives the launcher exiting
$proc = Start-Process `
    -FilePath 'python' `
    -ArgumentList @(
        'train_fullplus_cd68.py',
        '--data-root','E:\aic\复赛数据集(包括训练集和测试集输入)',
        '--manifest','E:\aic\final-ihc\configs\roi_split_semifinal_2026_expanded.json',
        '--init-checkpoint','E:\aic\final-ihc\checkpoints\fullplus_cd68_v3\final.pt',
        '--output','E:\aic\final-ihc\checkpoints\fullplus_cd68_v3p1',
        '--epochs','100',
        '--batch-size','8',
        '--lr','3e-5',
        '--cd68-weight','3.0',
        '--seed','2026'
    ) `
    -RedirectStandardOutput 'logs/fullplus_cd68_v3p1.out' `
    -RedirectStandardError 'logs/fullplus_cd68_v3p1.err' `
    -WindowStyle Hidden `
    -PassThru

Write-Host "started pid=$($proc.Id)"
$entry2 = "[$ts] pid=$($proc.Id)"
Add-Content -Path 'logs/fullplus_cd68_v3p1.log' -Value $entry2
