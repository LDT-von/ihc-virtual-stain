$ROOT = 'E:\aic\ihc-virtual-stain\results_best_4xtta\test'
Write-Host '=== Files per marker ==='
foreach ($m in 'HLA-DR','CD68','CD45RO','Vimentin') {
    $p = Join-Path $ROOT $m
    if (Test-Path $p) {
        Write-Host "$m : $((Get-ChildItem $p).Count)"
    } else {
        Write-Host "$m : no dir"
    }
}
Write-Host ''
Write-Host '=== Python processes ==='
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id, StartTime, @{N='Min';E={[math]::Round(([DateTime]::Now - $_.StartTime).TotalMinutes,1)}} | Format-Table -AutoSize | Out-String
