# 75.0876 - w96_expanded + TTA=4 Submission

## 概述

这是当前最佳提交：**分数 75.0876**，使用 `ultimate_w96_expanded/baseline/best.pt` checkpoint + TTA=4 推理。

## 模型配置

- **Checkpoint**: `checkpoints/best.pt`
- **架构**: MarkerContextNet (width=96, 4 尺度)
- **训练数据**: expanded dataset (2800 ROIs)
- **TTA**: 4x (翻转/旋转)
- **训练 epochs**: 见 `checkpoints/history.jsonl`

## 平台分数拆解

| Marker | SSIM | PSNR |
|--------|------|------|
| CD68 | 0.773 | 28.100 |
| CD45RO | 0.894 | 26.533 |
| HLA-DR | 0.850 | 25.484 |
| Vimentin | 0.865 | 25.958 |
| **Total** | **75.0876** | — |

## 关键洞察

1. **CD45RO 已达标** (0.894 > 0.893 目标)
2. **CD68 是最大短板** (SSIM 差 0.120)
3. **目标分数**: 78.3845

## 文件说明

- `checkpoints/` - 模型权重和训练历史
- `configs/` - ROI 分割配置
- `submission/` - 提交文件

## 复现命令

```bash
# 推理
cd E:\aic\final-ihc
D:\Anaconda3\python.exe src\inference.py --checkpoint checkpoints/ultimate_w96_expanded/baseline/best.pt --tta 4

# 打包提交
D:\Anaconda3\python.exe src\package_submission.py --output submission.zip
```
