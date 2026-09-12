$ROOT = 'E:\aic\ihc-virtual-stain\checkpoints'
Write-Host '=== All checkpoints ==='
foreach ($d in (Get-ChildItem $ROOT | Where-Object { $_.PSIsContainer })) {
    Write-Host ("--- " + $d.Name + " ---")
    Get-ChildItem $d.FullName -Filter '*.pt' | ForEach-Object {
        Write-Host ("  " + $_.Name + "  " + ("{0:N1} MB" -f ($_.Length/1MB)))
    }
}
