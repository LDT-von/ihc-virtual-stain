$ROOT = 'E:\aic\ihc-virtual-stain'
$markers = 'HLA-DR','CD68','CD45RO','Vimentin'

foreach ($dirName in @('results_best_4xtta','results_best_all_bestpt','results_v2_8xtta')) {
    Write-Host "=== $dirName ==="
    foreach ($m in $markers) {
        $p = "$ROOT\$dirName\test\$m"
        if (Test-Path $p) {
            $sizes = Get-ChildItem $p | Where-Object { $_.Name -like '*_fake.jpg' } | Measure-Object Length -Sum -Average
            Write-Host ("  {0,-10} count={1}  avg={2:N1} KB  total={3:N1} MB" -f $m, $sizes.Count, ($sizes.Average / 1024), ($sizes.Sum / 1MB))
        }
    }
    Write-Host ''
}

Write-Host "=== Reference: Real test images ==="
$REAL = "$ROOT\初赛数据集（包含训练集和测试集输入）\初赛数据集（包含训练集和测试集输入）\test"
foreach ($m in $markers) {
    $p = "$REAL\$m"
    if (Test-Path $p) {
        $sizes = Get-ChildItem $p | Where-Object { $_.Name -like '*_real_A.jpg' } | Measure-Object Length -Sum -Average
        Write-Host ("  {0,-10} count={1}  avg={2:N1} KB  total={3:N1} MB" -f $m, $sizes.Count, ($sizes.Average / 1024), ($sizes.Sum / 1MB))
    }
}
