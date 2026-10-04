# SMP ResNet34-U-Net：DAPI → IHC 适配与运行

## 来源与方法

直接使用 [segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)
的 `smp.Unet`，不是自行重写的 ResNet 或 U-Net。
固定上游版本 **v0.5.0**，提交 `420ce84b0c2df0286fa9bb2bd1499eea625c9b33`（MIT 许可）。
上游源代码已拉取用于核对和接口检查；项目通过固定版本依赖使用它，不提交整个第三方仓库或缓存。

- 适配器：`src/models/marker_smp_unet.py`；架构名 `smp_resnet34_unet`。
- 单通道 DAPI 输入 `[0,1]`，随机初始化，不使用外部预训练权重。
- ResNet34 编码器、五级 U-Net 解码器，不添加额外注意力或分类辅助头。
- 四通道分别预测 HLA-DR、CD68、CD45RO、Vimentin 的连续强度。
  输出使用独立 sigmoid，不使用 softmax、Dice 或分割交叉熵。
- `--width 16` 对应标准解码通道 `(256,128,64,32,16)`；只缩放解码器，编码器容量固定。
- 非方形输入补到 32 的倍数，必要时至少补到 64，再裁回原尺寸；小尺寸单样本也满足 BatchNorm 条件。

复用 V7 的 ROI 分组、同步几何增强、原始标签强度、SSIM/L1/MSE 和多尺度 L1、EMA、TTA，
以及实际 RGB JPEG 保存像素上的 SSIM/PSNR。EMA 同时同步 BatchNorm 的运行均值、方差和计数器。
checkpoint 记录模型版本、项目与上游实现的源码哈希，加载 SMP 权重时检查一致性。

## 安装

先在训练服务器安装互相匹配的 PyTorch/torchvision CUDA 版本，然后在项目根目录执行：

```powershell
python -m pip install -r requirements-smp-unet.txt
```

接口检查环境：Python 3.10.11、torch 2.6.0+cu124、torchvision 0.21.0+cu124、SMP 0.5.0。
本机直接导入拉取的 v0.5.0 源码，没有修改全局 Python 安装。
模型构造使用 `encoder_weights=None`，不会下载预训练权重。
不安装 SMP 也能继续使用 MCN/NAFNet；选择 SMP 模型时才检查版本。

## 四标记训练、选模与最终重训

以下命令交给使用者运行。本次没有启动真实训练、测试集推理或提交。
把 `$DataRoot` 改为实际目标轮次的数据目录，不要直接沿用其他器官的清单。
根目录须包含 `train/DAPI`、四种 `train/<marker>` 和 `test/DAPI`。
当前单标记模式仍通过四标记读取器选取目标，完整四标记训练标签须可读取。

```powershell
$DataRoot = 'E:\aic\LIVER_DATA_ROOT'
$Manifest = 'configs/roi_split_liver_2026.json'
$RunRoot = 'checkpoints/liver_smp_resnet34_v1'
python -m src.train_marker_context split --data-root $DataRoot --output $Manifest --val-rois 2 --holdout-rois 2
if ($LASTEXITCODE -ne 0) { throw '清单生成失败' }

$Common = @('--data-root', $DataRoot, '--manifest', $Manifest, '--architecture', 'smp_resnet34_unet', '--width', '16', '--folds', '5', '--epochs', '120', '--eval-every', '10', '--batch-size', '4', '--lr', '0.0003', '--warmup-epochs', '5', '--weaken-start-epoch', '80', '--augmentation', 'geometry', '--loss', 'normalized', '--cd68-weight', '1', '--tta', '4', '--no-cache')
foreach ($Fold in 0..4) {
    python train_semifinal_v7.py fit --stage dev --fold $Fold --output "$RunRoot/fold_$Fold" @Common
    if ($LASTEXITCODE -ne 0) { throw "第 $Fold 折失败，停止后续步骤" }
}
$FoldRuns = 0..4 | ForEach-Object { "$RunRoot/fold_$_" }
python train_semifinal_v7.py summarize --runs $FoldRuns --output "$RunRoot/selection.json"
if ($LASTEXITCODE -ne 0) { throw '选模失败' }
python train_semifinal_v7.py fit --stage final --selection "$RunRoot/selection.json" --output "$RunRoot/final" @Common
if ($LASTEXITCODE -ne 0) { throw '最终重训失败' }
python train_semifinal_v7.py infer --checkpoint "$RunRoot/final/final.pt" --input "$DataRoot/test/DAPI" --output 'predictions/liver_smp_resnet34_v1' --batch-size 4
if ($LASTEXITCODE -ne 0) { throw '推理失败' }
```

输出目录须全新；恢复使用相同参数加 `--resume <原目录>/last.pt`。
五折完成后按 OOF 平均 SSIM 选轮，同时保留 PSNR；最终重训使用全部官方训练标签，
不再用训练内 ROI 调参。推理 TTA、种子和标记从 checkpoint 读取。
提交时只打包预测目录下的 `results/`，保留 `provenance.json` 作为来源记录。

单标记：给 `$Common` 增加 `--target-marker CD68`（或当期指定标记），并换用全新运行和预测目录。
全部折、选模、重训须使用同一目标标记。

Linux 服务器按同样参数调用，例如第 0 折：

```bash
python train_semifinal_v7.py fit --architecture smp_resnet34_unet --width 16 \
  --data-root /path/to/LIVER_DATA_ROOT --manifest configs/roi_split_liver_2026.json \
  --output checkpoints/liver_smp_resnet34_v1/fold_0 --stage dev --fold 0 \
  --folds 5 --epochs 120 --eval-every 10 --batch-size 4 --lr 0.0003 \
  --warmup-epochs 5 --weaken-start-epoch 80 --augmentation geometry \
  --loss normalized --cd68-weight 1 --tta 4 --no-cache
```

## 与现有方法公平比较

安装 SMP 后，用同一当前代码版本重跑 MCN/NAFNet 与 SMP 的全部折。
固定清单、ROI 折、种子、增强、损失、标记权重、batch size、学习率、日程、评估频率、TTA 与 JPEG。
本次新增源码哈希，旧源码下的折结果和 `selection.json` 不能直接沿用。

```powershell
python compare_semifinal_v7.py --baseline checkpoints/liver_mcn_v1/selection.json --candidate checkpoints/liver_smp_resnet34_v1/selection.json --require-matched-training
```

MCN 的 width 缩放编码器和解码器，SMP 的 width 只缩放解码器。
跨这两类架构比较时允许 width 不同，但检查其余训练控制，并展示架构和 width。
这是完整模型比较，不是等参数量比较，不能把全部差异归因于某一个模块。
同一架构之间仍要求 width 一致。显存不足时，两边全部折统一降低 batch size 并建新实验。

## 接口验证与证据边界

```powershell
python -m pytest tests/test_marker_smp_unet.py -q
```

7 项 CPU 接口检查通过：上游模型四通道梯度、非方形尺寸和八方向 TTA、独立输出强度、
单标记 checkpoint 恢复与依赖哈希拒绝、BatchNorm EMA、输入/依赖约束、CLI 与公平比较约束。
检查未下载预训练权重。这些证明接口可用，不能替代真实目标数据完整训练、Liver OOF 或平台成绩。
