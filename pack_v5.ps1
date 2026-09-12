$SRC = 'E:\aic\ihc-virtual-stain\results_best_all_bestpt'
$TMP = 'E:\aic\ihc-virtual-stain\results_v5_pack'
$DST = 'E:\aic\ihc-virtual-stain\submission_v5.zip'

if (Test-Path $TMP) {
    Remove-Item $TMP -Recurse -Force
}
New-Item -ItemType Directory -Force -Path "$TMP\results\test" | Out-Null

foreach ($m in 'HLA-DR','CD68','CD45RO','Vimentin') {
    Copy-Item -Path "$SRC\test\$m" -Destination "$TMP\results\test\$m" -Recurse -Force
}

if (Test-Path $DST) {
    Remove-Item $DST -Force
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory($TMP, $DST, [System.IO.Compression.CompressionLevel]::Optimal, $false)

$size = (Get-Item $DST).Length / 1MB
Write-Host ("Created: $DST, Size: {0:N2} MB" -f $size)

Remove-Item $TMP -Recurse -Force
