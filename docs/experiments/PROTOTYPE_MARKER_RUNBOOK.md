# 多 marker 原型架构：运行与对照

`prototype_marker` 接收单通道 DAPI，预测四个 IHC marker。它采用共享表征、共享原型和各 marker 专用原型的设计；[ProtoMTG 作者项目](https://jj-zhou-code.github.io/ProtoMTG-website/)只提供概念参考。其 GitHub 仓库入口不完整，且未列出可复用许可证，因此这里是独立实现，不复制其代码或权重。结构相似不代表已复现论文结果，也不保证平台分提升。

该候选通过 `train_semifinal_v7.py fit` 训练；旧 `src.train_marker_context train` 入口不接受该架构。

## 数据与比较原则

- 这台机器目前只有初赛配对图；与现有平台分记录对应的目标轮次原图和权重不在本工作区，75.2485 的阶段尚未证实；下列命令**尚未运行**。用户在有数据的机器上填写 `$DataRoot` 与 `$Manifest`。数据根目录须直接包含 `train\DAPI`、`train\<marker>` 和推理时的 `test\DAPI`；manifest 的文件名必须与该目录一致。
- 对 `context`、`marker_specific`、`prototype_marker` 使用同一 manifest、ROI 五折、种子、增强、损失、宽度、batch、epoch、学习率、验证间隔与 TTA。示例宽度 32、batch 4，避免默认宽度 96 对四路解码器造成显存压力。相同宽度仍不等于相同参数量；核对各折 `run.json` 的 `parameters`。
- `summarize` 按五折样本加权的 OOF SSIM 选轮，同时报告 PSNR 和逐 marker、逐 ROI 结果。这是开发集证据，不是平台分。重点检查提升是否跨 ROI 稳定、CD68 等 marker 是否退步，并结合最差样本图判断。模型对照只改变架构；原型数量和正则强度属于 `prototype_marker` 自身配方，不应把调参收益称为纯架构收益。
- 每折验证 ROI 不进入该折训练；`final` 会使用全部标注 ROI，之后不能再把其中任何 ROI 当作独立测试。当前没有患者 ID，ROI 分组也不能证明患者独立。

## 五折开发与选轮

在数据机器上的仓库目录运行；每次实验使用全新输出目录。`$DataRoot` 和 `$Manifest` 仅为待填写占位值。

```powershell
Set-Location E:\aic
$DataRoot = '填写目标轮次配对数据的根目录'
$Manifest = '填写与该数据一致的完整 ROI manifest 路径'
$Out = 'E:\aic\results\prototype_marker_comparison'

$Common = @(
    '--data-root', $DataRoot, '--manifest', $Manifest,
    '--width', '32', '--batch-size', '4', '--epochs', '300',
    '--eval-every', '50', '--lr', '0.0003', '--tta', '4',
    '--augmentation', 'geometry', '--loss', 'normalized'
)
$Proto = @(
    '--num-shared-prototypes', '8', '--num-task-prototypes', '4',
    '--prototype-temperature', '0.25',
    '--prototype-diversity-weight', '0.001'
)

foreach ($Arch in @('context', 'marker_specific', 'prototype_marker')) {
    $Extra = @()
    if ($Arch -eq 'prototype_marker') { $Extra = $Proto }
    foreach ($Fold in 0..4) {
        $FoldOut = Join-Path $Out ("{0}_f{1}" -f $Arch, $Fold)
        python .\train_semifinal_v7.py fit @Common --architecture $Arch @Extra --fold $Fold --output $FoldOut
        if ($LASTEXITCODE -ne 0) { throw "训练失败：$Arch fold $Fold" }
    }
    $FoldDirs = @(0..4 | ForEach-Object { Join-Path $Out ("{0}_f{1}" -f $Arch, $_) })
    $Selection = Join-Path $Out ("{0}_selection.json" -f $Arch)
    python .\train_semifinal_v7.py summarize --runs $FoldDirs --output $Selection
    if ($LASTEXITCODE -ne 0) { throw "选轮失败：$Arch" }
}

python .\compare_semifinal_v7.py --require-matched-training --baseline (Join-Path $Out 'context_selection.json') --candidate (Join-Path $Out 'prototype_marker_selection.json')
python .\compare_semifinal_v7.py --require-matched-training --baseline (Join-Path $Out 'marker_specific_selection.json') --candidate (Join-Path $Out 'prototype_marker_selection.json')
```

默认原型参数为共享原型 8 个、每任务原型 4 个、温度 0.25、多样性权重 0.001。若修改任一参数，五折应全部重跑到新目录。`summarize` 拒绝混合不同配方或不完整的折。中断时只可在原输出目录中用相同参数和该目录的 `last.pt` 追加 `--resume`。

## 锁定胜出配方、推理与打包

下面以 `prototype_marker` 的五折结果胜出为**条件示例**。实际应先审阅三份 `*_selection.json` 和逐 marker、逐 ROI 结果；若其他架构胜出，`$Winner`、`$Extra` 和 selection 路径须一起替换。最终重训从零开始，只训练到五折选中的轮次。

```powershell
$Winner = 'prototype_marker'
$Extra = $Proto
$Selection = Join-Path $Out ("{0}_selection.json" -f $Winner)
$Final = Join-Path $Out ("{0}_final" -f $Winner)
python .\train_semifinal_v7.py fit @Common --stage final --architecture $Winner @Extra --selection $Selection --output $Final
if ($LASTEXITCODE -ne 0) { throw '最终重训失败' }

$TestInput = Join-Path $DataRoot 'test\DAPI'
$PredictionDir = Join-Path $Out ("{0}_test_predictions" -f $Winner)
python .\train_semifinal_v7.py infer --checkpoint (Join-Path $Final 'final.pt') --input $TestInput --output $PredictionDir
if ($LASTEXITCODE -ne 0) { throw '推理失败' }

$Zip = Join-Path $Out ("{0}_submission.zip" -f $Winner)
python .\scripts\package_final.py --input $TestInput --prediction-dir $PredictionDir --output $Zip
if ($LASTEXITCODE -ne 0) { throw '打包失败' }
```

打包器核对四个 marker 的文件数、名称、尺寸和格式；打包完成不等于已提交。盲测平台成绩才能检验真实泛化。不要把旧训练涉及的 ROI078 或最终重训使用过的训练 ROI 写成独立测试结果。
