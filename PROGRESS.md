# 比赛训练进展报告

## 当前最佳：ultimate_w96_expanded/baseline (TTA=8, NoJPEG)

| 指标 | Val (2100 train) | Holdout (含污染) | ROI078 only (真实holdout) |
|------|------------------|-------------------|---------------------------|
| SSIM | 0.8423 | 0.8319 ⚠️ | 0.7876 |
| PSNR | 38.05 | 26.05 | 22.97 |

⚠️ 重要：expanded 模型在训练时见过 ROI045/052，所以 holdout 分数有"数据污染"。真实泛化指标看 ROI078（未污染）。

## 关键观察

### ROI078 是真正的瓶颈
| 模型 | ROI078 SSIM |
|------|-------------|
| w48 refiner | 0.7600 |
| w96 baseline (1540 train) | 0.7870 |
| w96 expanded (2100 train) | **0.7876** (+0.0006) |

扩大训练量对 ROI078 几乎没帮助！CD68 marker 也是瓶颈：
- CD68 SSIM = 0.775（vs CD45RO = 0.871）
- CD68 是巨噬细胞标记，形态/分布多变

## 历史最佳
| 模型 | Holdout SSIM | ROI078 SSIM |
|------|--------------|-------------|
| w48 refiner | 0.8157 | 0.7600 |
| w96 baseline | 0.8193 | 0.7870 |
| **w96 expanded** | **0.8319** ⚠️ | **0.7876** |

## 下一步思路

1. **不能再用 expanded split 给 holdout 评分** — 污染了
2. 在原始 split 上继续训练，优化 ROI078
3. 可能要做 **per-marker refiner**，专项解决 CD68

## 时间线
- 2026-09-22 训练开始
- expanded split 创建（从1540->2100样本）
- 完成 50 epochs，最佳 SSIM=0.8423 (val)
- 真实 holdout 泛化：+0.0006（几乎没提升）

