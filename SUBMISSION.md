# AIC 虚拟染色赛题提交指南

本指南依据项目中的《基于虚拟染色的免疫组化图像生成》赛题 PDF（第 7–8 页）整理。赛事方后续发布的新通知优先。

## 已确认的提交规则

- 输入为测试集 `DAPI` 图像；输出为官方要求的一个目标标记图像，例如 `HLA-DR`。
- 输入与输出必须空间尺寸一致。官方 patch 为 `256×256`，图像格式为 JPG。
- 输出文件名必须与输入图像一一对应，并使用 `_fake.jpg` 后缀。例如 `ROI025_00_00.jpg` 对应 `ROI025_00_00_fake.jpg`。
- 初赛结果目录为：`results/test/<marker>/`。PDF 示例为 `results/test/CD68/ROI025_00_00_fake.jpg`。
- 测试集只能自动推理；不得人工逐张修改或使用测试集标签、衍生信息进行训练或后处理。

PDF 没有规定压缩包文件名或要求初赛必须包含代码、模型。若提交平台要求上传 ZIP，可将上述 `results/` 目录原样压缩；本项目的 `src.submit` 会创建该结构。

## 当前官方数据位置

本地数据根目录应直接包含 `train/` 与 `test/`：

```text
E:\aic\ihc-virtual-stain\初赛数据集（包含训练集和测试集输入）\初赛数据集（包含训练集和测试集输入）
├── train/
│   ├── DAPI/
│   ├── HLA-DR/
│   ├── CD45RO/
│   ├── Vimentin/
│   └── CD68/
└── test/
    └── DAPI/
```

当前初赛测试集共有 1,346 张 DAPI JPG 图像。

## 生成初赛结果

以当前最新的 HLA-DR checkpoint 为例，在项目根目录执行：

```powershell
D:\Anaconda3\python.exe -m src.inference `
  --ckpt "checkpoints\HLA-DR_1788855131\epoch109.pt" `
  --marker HLA-DR `
  --data-root "E:\aic\ihc-virtual-stain\初赛数据集（包含训练集和测试集输入）\初赛数据集（包含训练集和测试集输入）" `
  --split test `
  --output-dir results
```

输出会写入：

```text
results/
└── test/
    └── HLA-DR/
        ├── ROI025_00_00_fake.jpg
        └── ...
```

命令默认使用 checkpoint 中保存的 50 个采样步数。只用于检查流程时，可额外传入 `--max-samples 2 --num-steps 2`；该低步数输出不能作为正式成绩提交。

## 打包初赛结果（如平台要求 ZIP）

```powershell
D:\Anaconda3\python.exe -m src.submit `
  --results-dir results `
  --marker HLA-DR `
  --stage preliminary `
  --out-zip submission_hla_dr.zip
```

压缩包内仅包含：

```text
results/test/HLA-DR/ROI025_00_00_fake.jpg
```

## 复赛与半决赛的额外材料

除同样格式的结果图像外，赛题 PDF 要求提交：

1. 完整 Python 代码：数据预处理、训练和预测推理；
2. 训练好的模型文件，以及模型加载与运行说明、所需环境和依赖；
3. PDF 技术报告，不少于 2,000 字，涵盖算法设计、模型架构、训练设置、数据增强、指标分析、创新点和不足。

对应打包命令：

```powershell
D:\Anaconda3\python.exe -m src.submit `
  --results-dir results `
  --marker HLA-DR `
  --stage rematch `
  --ckpt "checkpoints\HLA-DR_1788855131\epoch109.pt" `
  --report "技术报告.pdf" `
  --out-zip rematch_hla_dr.zip
```

半决赛使用 `--stage semifinal`。多个目标标记同时输出时，赛题说明会对多个输出成绩取平均；最终采用最后一次有效提交的成绩。
