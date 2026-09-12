$root = 'E:\aic\初赛数据集（包含训练集和测试集输入）\train'
foreach ($m in 'HLA-DR','CD68','CD45RO','Vimentin') {
    $p = "$root\$m"
    if (Test-Path $p) {
        $files = Get-ChildItem $p -ErrorAction SilentlyContinue
        Write-Host "--- train/$m ---"
        foreach ($f in $files | Select-Object -First 3) {
            Write-Host ("  {0,-30} {1:N1} KB" -f $f.Name, ($f.Length/1024))
        }
        $realB = $files | Where-Object { $_.Name -like '*_real_B.jpg' }
        if ($realB.Count -gt 0) {
            $avg = ($realB | Measure-Object Length -Average).Average
            Write-Host ("  real_B avg: {0:N1} KB  (n={1})" -f ($avg/1024), $realB.Count)
        }
        $realA = $files | Where-Object { $_.Name -like '*_real_A.jpg' }
        if ($realA.Count -gt 0) {
            $avg = ($realA | Measure-Object Length -Average).Average
            Write-Host ("  real_A avg: {0:N1} KB  (n={1})" -f ($avg/1024), $realA.Count)
        }
    }
}
