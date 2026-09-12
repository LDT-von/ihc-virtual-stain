$ZIP = 'E:\aic\ihc-virtual-stain\submission_v4.zip'
Add-Type -AssemblyName System.IO.Compression.FileSystem

$z = [System.IO.Compression.ZipFile]::OpenRead($ZIP)
Write-Host "Total entries: $($z.Entries.Count)"

# 统计每个 marker
$entries = $z.Entries | ForEach-Object { $_.FullName }
$groups = $entries | Group-Object { ($_ -split '/')[0..1] -join '/' }
Write-Host ''
Write-Host '=== Top-level dirs ==='
$groups | ForEach-Object { Write-Host "$($_.Name) : $($_.Count)" }

Write-Host ''
Write-Host '=== Sample entries (top 10) ==='
$entries | Select-Object -First 10 | ForEach-Object { Write-Host $_ }
