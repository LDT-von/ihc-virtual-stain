$SRC = 'E:\aic\ihc-virtual-stain\results_pack'
$DST = 'E:\aic\ihc-virtual-stain\submission_v4.zip'

if (Test-Path $DST) {
    Remove-Item $DST -Force
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory($SRC, $DST, [System.IO.Compression.CompressionLevel]::Optimal, $false)

$size = (Get-Item $DST).Length / 1MB
Write-Host "Created: $DST"
Write-Host ("Size: {0:N2} MB" -f $size)
