# 半决赛 V7：诊断与运行

这是一套待运行的新实验，目前没有 V7 权重、验证成绩或平台分。它与旧初赛 `train_v7.py` 无关。

## 已确认的问题

1. 旧 V3/V6 把 manifest 的 train、val、holdout 合并训练，再用 ROI078 对比。ROI078 对它们不是独立 holdout。本地 V3→V6 SSIM 升高，平台分 75.1128→75.0463，不能用该本地分推断泛化。
2. 旧 `PairedMarkers._gaussian_noise` 在输入数组仍为 `uint8` 时把负高斯噪声转成 `uint8`，会回绕；噪声和镜面亮斑还作用于四个目标 marker。代码错误已确认，分数影响尚未实测。
3. 当前 75.2485 的 ALL-IN 文档写五模型，现有同名脚本却使用六个不同权重且只处理 ROI078。原提交 ZIP、权重和实际打包脚本不在此工作区，不能逐项归因。V6 的逐 marker 数字复制了另一提交，目标 78.3845 的逐 marker 数字不是官方分项。
4. 当前训练集只有 17 个 ROI、每 ROI 140 patch。应按 ROI 分组评估；因缺少患者 ID，ROI 独立仍不等于患者独立。
5. [官方赛题说明](https://www.aicomp.cn/wp-content/uploads/2026/05/6%E3%80%81%E5%9F%BA%E4%BA%8E%E8%99%9A%E6%8B%9F%E6%9F%93%E8%89%B2%E7%9A%84%E5%85%8D%E7%96%AB%E7%BB%84%E5%8C%96%E5%9B%BE%E5%83%8F%E7%94%9F%E6%88%90.pdf)在初赛阶段公布 `0.7 × SSIM + 0.3 × Normalize(PSNR)`；复赛沿用 SSIM、PSNR 和综合分，半决赛 test3 的归一化细节没有公布。半决赛综合分写为 `0.1 × test1 + 0.2 × test2 + 0.7 × test3`。需确认平台显示的 75.2485 是 test3 单项还是半决赛综合分。若是综合分且前两项不变，到 78.0000 需要 test3 约提高 `(78-75.2485)/0.7 = 3.93` 分；这是条件计算。
6. [ProtoMTG 作者项目](https://jj-zhou-code.github.io/ProtoMTG-website/)处理相同四个 marker，使用共享与任务特定原型。这是有依据的架构候选，不代表移植后必达 78。

## 新代码的实验设计

- `src/data/paired_clean.py` 同步几何增强；可选噪声只加在 DAPI；目标与缓存保持原始像素。
- `train_semifinal_v7.py` 做固定种子的五折 ROI 分组。每折验证 ROI 从未进入该折训练。增强臂：`legacy-v6`、`geometry`、`dapi-noise`；损失臂：`legacy-v6`、`normalized`。当 CD68 权重为 3 时，归一化仅把整个损失除以 6，理论最优解不变；它主要影响梯度裁剪与优化器数值尺度，不应期待单靠此项从 75 跳到 78。
- 每次验证保存按 marker、ROI、单图的 SSIM/PSNR 与该轮 EMA 权重。`summarize` 只接受同配方、同源码、完整五折，按 OOF 平均 SSIM 选轮，完整保留 PSNR。官方分还含未知归一化的 PSNR，所以这是选轮代理指标，不是平台分。
- `fit --stage final` 按选中轮次用全部标注 ROI 重训；V7 的 `infer` 入口从权重读取验证时的 TTA。`inspect_semifinal_v7.py` 用选中轮次权重生成四个 marker 的最差样本拼图（DAPI、真值、预测、绝对误差），并重算 SSIM 检查图像是否漂移。
- `marker_specific` 同时改变编码器/解码器深度与容量，是整体架构候选；若它提高，不能只归因于专用 decoder。

OOF 指标用于配方与轮次选择，属于开发证据；官方盲测才检验泛化。

## 用户运行命令（PowerShell；这里没有代跑）

在实际保存半决赛训练集的机器上修改 `$DataRoot`。它必须包含 `train\DAPI` 和 `train\<marker>`，并与 manifest 全部文件名一致。此工作区目前找不到文档所写的数据路径，也没有 75.2485 的当前图片/权重。

```powershell
Set-Location E:\aic
$DataRoot = 'E:\aic\复赛数据集(包括训练集和测试集输入)'
$Manifest = 'E:\aic\configs\roi_split_semifinal_2026_expanded.json'
$Out = 'E:\aic\results\semifinal_v7'

# 同一折的成对筛查；除消融开关外保持参数相同。
python .\train_semifinal_v7.py fit --data-root $DataRoot --manifest $Manifest --output "$Out\legacy_f0" --fold 0 --augmentation legacy-v6 --loss legacy-v6
python .\train_semifinal_v7.py fit --data-root $DataRoot --manifest $Manifest --output "$Out\geometry_f0" --fold 0 --augmentation geometry --loss legacy-v6
python .\train_semifinal_v7.py fit --data-root $DataRoot --manifest $Manifest --output "$Out\normalized_f0" --fold 0 --augmentation geometry --loss normalized
```

先比较三个目录的 `best_validation.json` 与 `history.jsonl`，重点看逐 marker、逐 ROI 的 SSIM 和 PSNR。第一折仅用于筛查。`legacy-v6` 臂是本训练器中的对照，随机数时序与历史 V6 脚本不完全相同，并非旧提交的逐字节复现。

若要判断哪处改动真正有效，旧增强、干净几何增强、归一化损失三个配方都要跑**相同的五折**；fold 0 可复用，已有目录不能重复运行。计算资源有限时可只选两臂做成对五折，但不能对未跑的那一项做因果归因。

```powershell
foreach ($recipe in @(
    @{ Name='legacy'; Aug='legacy-v6'; Loss='legacy-v6' },
    @{ Name='geometry'; Aug='geometry'; Loss='legacy-v6' },
    @{ Name='normalized'; Aug='geometry'; Loss='normalized' }
)) {
    foreach ($fold in 1..4) {
        $FoldOut = Join-Path $Out ("{0}_f{1}" -f $recipe.Name, $fold)
        python .\train_semifinal_v7.py fit --data-root $DataRoot --manifest $Manifest --output $FoldOut --fold $fold --augmentation $recipe.Aug --loss $recipe.Loss
    }
    $FoldDirs = @(0..4 | ForEach-Object { Join-Path $Out ("{0}_f{1}" -f $recipe.Name, $_) })
    $SummaryPath = Join-Path $Out ("{0}_selection.json" -f $recipe.Name)
    python .\train_semifinal_v7.py summarize --runs $FoldDirs --output $SummaryPath
}
$Selection = Join-Path $Out 'normalized_selection.json'
python .\inspect_semifinal_v7.py --selection $Selection --data-root $DataRoot --output (Join-Path $Out 'normalized_worst') --top-k 6
```

比较三个 selection 的 OOF 总体、逐 marker、逐 ROI 的同折差异，并看最差样本图，再决定是否最终重训。若五折几何增强没有稳定改善，检查 ROI/器官分布差异与正样本漏检，再尝试 `dapi-noise`；若监督和输入问题排除后仍持续欠拟合，才投入多任务原型或专用解码架构。下面最终重训命令仅以 `normalized` 获胜为例；获胜配方不同，必须同步更改 `--augmentation`、`--loss` 和 selection 路径。

```powershell
$Final = Join-Path $Out 'normalized_final'
python .\train_semifinal_v7.py fit --stage final --data-root $DataRoot --manifest $Manifest --selection $Selection --output $Final --augmentation geometry --loss normalized
$TestInput = Join-Path $DataRoot 'test\DAPI'
$PredictionDir = Join-Path $Out 'normalized_test_predictions'
python .\train_semifinal_v7.py infer --checkpoint (Join-Path $Final 'final.pt') --input $TestInput --output $PredictionDir
python .\scripts\package_final.py --input $TestInput --prediction-dir $PredictionDir --output (Join-Path $Out 'semifinal_v7_normalized.zip')
```

训练内存不足时，所有对照臂都用相同的 `--batch-size 4` 重做，不能把 batch 8 和 4 的结果说成纯增强/损失差异。每折默认 300 epoch、每 50 epoch 存一份 EMA 验证权重；预留时间和磁盘。中断后仅可从该输出目录的 `last.pt` 以相同参数追加 `--resume <last.pt>`。最终 `final.pt` 用全体标注 ROI 训练，不可再用其中某 ROI 当独立测试。

要对 78 做可信判断，还需找回 75.2485 的实际 ZIP、对应 checkpoint、四 marker 平台分项，以及 test1/test2/test3 成绩口径。当前代码与记录只能定位**可证的实验错误**并建立新对照，不能声称 V7 已提升平台分。
