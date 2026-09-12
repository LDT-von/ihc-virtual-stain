$SRC = 'E:\aic\ihc-virtual-stain\results_best_4xtta\test'
$DST = 'E:\aic\ihc-virtual-stain\results_pack'

if (Test-Path $DST) {
    Remove-Item $DST -Recurse -Force
}

# 创建 results/test 目录结构
New-Item -ItemType Directory -Force -Path "$DST\results\test" | Out-Null

foreach ($m in 'HLA-DR','CD68','CD45RO','Vimentin') {
    Copy-Item -Path "$SRC\$m" -Destination "$DST\results\test\$m" -Recurse -Force
    $n = (Get-ChildItem "$DST\results\test\$m").Count
    Write-Host "$m : $n files"
}

Write-Host ''
Write-Host '=== Verify structure ==='
Get-ChildItem $DST -Recurse | Select-Object -First 5 FullName | Format-Table -AutoSize
