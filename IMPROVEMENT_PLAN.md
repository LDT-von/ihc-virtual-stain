# 提升计划：以SSIM=0.893, PSNR=26.499为目标

## 当前状况
- 最佳Holdout SSIM: 0.815 (w48 refiner, TTA=8)
- 目标: 0.893 (差距 -0.078)
- 最佳Val SSIM: 0.829 (TTA=8)
- Holdout vs Val差距: -0.014 (说明泛化有问题)

## 根本问题分析
1. **Refiner几乎无效** - 最佳epoch=1，说明训练没收敛
   - lr太小 (0.0001) + epochs太少 (20)
   - residual_limit=0.2 限制了表达能力
   
2. **CD68泛化最差** - holdout上0.744 vs val 0.793
   - 可能CD68的特征模式在holdout的ROI上不同
   - 需要marker-specific的关注
   
3. **模型可能欠拟合** - SSIM=0.82-0.83对于IHC虚拟染色来说还有空间

## 行动方案（从快到慢，从有效到彻底）

### 方案1: 快速改进 - 重新训练Refiner（立即可跑，约1小时）
**问题**: 当前refiner训练失败（最佳epoch=1）
**修复**:
- epochs: 20 → 60
- lr: 0.0001 → 0.001 (提高10倍)
- residual_limit: 0.2 → 0.5

```bash
cd E:\aic\final-ihc
python -m src.train_marker_context train \
  --output "E:\aic\final-ihc\checkpoints\ultimate_w48_2380\refiner_v2" \
  --device cuda --batch-size 4 --seed 2026 \
  --manifest "E:\aic\final-ihc\configs\roi_split_semifinal_2026_2380.json" \
  --architecture anchored \
  --baseline-checkpoint "E:\aic\final-ihc\checkpoints\ultimate_w48_2380\baseline\best.pt" \
  --width 24 --epochs 60 --lr 0.001 --no-cache \
  --data-root "E:\aic\复赛数据集(包括训练集和测试集输入)" \
  --residual-limit 0.5
```

### 方案2: 中期改进 - 训练更大的Baseline（约2-3小时）
**问题**: 当前w48 baseline可能不够强
**修复**: 训练width=96的baseline

### 方案3: 长期改进 - 全数据训练 + 更优架构

## 预期效果
- 方案1: SSIM提升约+0.01-0.02 (从0.815到0.825-0.835)
- 方案2: SSIM提升约+0.02-0.04
- 方案3: 可能达到0.85+
- 要达到0.893，可能需要方案2+方案3的组合

## 立即执行
先跑方案1（最快验证）
