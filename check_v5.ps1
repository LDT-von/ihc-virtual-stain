$ROOT = 'E:\aic\ihc-virtual-stain'
Write-Host '=== v5 log ==='
Get-Content "$ROOT\infer_best_all_bestpt.log"
Write-Host '---'
Write-Host '=== file counts ==='
foreach ($m in 'HLA-DR','CD68','CD45RO','Vimentin') {
    $p = "$ROOT\results_best_all_bestpt\test\$m"
    if (Test-Path $p) {
        $c = (Get-ChildItem $p | Where-Object { $_.Name -like '*_fake.jpg' }).Count
        Write-Host "$m : $c files"
    } else {
        Write-Host "$m : no dir yet"
    }
}
