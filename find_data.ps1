$ROOT = 'E:\aic\ihc-virtual-stain'
$REAL = 'E:\aic\初赛数据集（包含训练集和测试集输入）\初赛数据集（包含训练集和测试集输入）\test'
if (-not (Test-Path $REAL)) {
    Write-Host "Try alternative path"
    Get-ChildItem E:\aic | Where-Object { $_.PSIsContainer } | Select-Object Name
}
