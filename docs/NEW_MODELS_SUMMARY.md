# 🆕 全新高性能模型 (2026-10-02)

## 概述

基于 GitHub 最新 SOTA 研究，我们创建了三个全新架构的模型，旨在突破当前 **75.2485** 的分数，向 **78.3845** 目标迈进。

## 当前分数排名

| 排名 | 模型 | 平台分数 |
|------|------|----------|
| 🥇 | Ensemble (5 models) | **75.2485** |
| 🥈 | fullplus_cd68_v3 | 75.1128 |
| 🥉 | w96 expanded | 75.0876 |
| - | **目标** | **78.3845** |

## 新增模型

### 1️⃣ Enhanced UNet++ ⭐ 推荐

**文件**: `src/models/enhanced_unetpp.py`

**创新点**:
- Dense skip connections (UNet++ 风格)
- Multi-scale attention blocks
- Deep supervision (多尺度监督)
- **MS-SSIM + Edge Loss** 组合损失

**优势**: 训练快速，显存需求适中 (8GB)

```bash
# 快速训练
python run_quick_train.py --model enhanced_unet \
    --data-root "DATA_ROOT" \
    --manifest configs/roi_split_semifinal_2026.json \
    --output checkpoints/new_enhanced \
    --epochs 80 --batch-size 8
```

**预期分数**: 75.5-76.0

---

### 2️⃣ Hybrid CNN-Attention ⭐ 推荐

**文件**: `src/models/transformer_ihc.py`

**创新点**:
- CNN 编码器高效提取局部特征
- **Squeeze-and-Excitation (SE)** 注意力机制
- 跳跃连接保留细节信息
- 多尺度特征融合

**优势**: 结合 CNN 速度和注意力能力

```bash
# 训练 Hybrid 模型
python run_quick_train.py --model hybrid \
    --data-root "DATA_ROOT" \
    --manifest configs/roi_split_semifinal_2026.json \
    --output checkpoints/new_hybrid \
    --epochs 60 --batch-size 6
```

**预期分数**: 76.0-77.0

---

### 3️⃣ Marigold-style Latent Diffusion 🔬 SOTA

**文件**: `src/models/marigold_ihc.py`

**创新点**:
- **潜在空间扩散** (Latent Diffusion)
- DAPI 条件 via Cross-Attention
- Marker-specific Conditioning Tokens
- Two-stage: Full diffusion → One-step Diffusion-FT

**参考论文**:
- [Marigold (CVPR 2024)](https://github.com/prs-eth/marigold) - 3.2k stars
- [DiffVS (AAAI 2026)](https://github.com/hvcl/DiffVS) - Marker tokens
- [DSFF-GAN](https://github.com/YihaoMa0512/DSFF-GAN) - CSS Loss

**优势**: SOTA 性能，但训练时间较长

```bash
# Stage 1: Full diffusion
python -m src.train_marigold_ihc 1 \
    --data-root "DATA_ROOT" \
    --manifest configs/roi_split_semifinal_2026.json \
    --output checkpoints/marigold \
    --epochs 30 --batch-size 4 --base-channels 128

# Stage 2: One-step fine-tuning (可选)
python -m src.train_marigold_ihc 2 \
    --checkpoint checkpoints/marigold/best.pt \
    --data-root "DATA_ROOT" \
    --manifest configs/roi_split_semifinal_2026.json \
    --output checkpoints/marigold_ft \
    --epochs 10
```

**预期分数**: 76.5-78.0

---

## 损失函数对比

### 新模型使用的组合损失

```
Total Loss = 
    1.0 * MS-SSIM Loss      # 多尺度结构相似性
  + 0.5 * L1 Loss          # 像素级重建
  + 0.1 * Edge Loss        # Sobel 边缘感知
  + 0.2 * Deep Supervision  # 多尺度监督 (仅 UNet++)
```

### 与旧模型对比

| 损失函数 | 旧模型 | 新模型 |
|----------|--------|--------|
| L1 | ✅ | ✅ |
| SSIM | ✅ | ✅ (MS-SSIM) |
| Edge Loss | ❌ | ✅ |
| Deep Supervision | ❌ | ✅ (UNet++) |
| SE Attention | ❌ | ✅ (Hybrid) |

---

## 推荐训练流程

### 快速验证 (2-3天)

```bash
# 1. 训练 Enhanced UNet++
python run_quick_train.py --model enhanced_unet \
    --data-root "DATA_ROOT" \
    --manifest configs/roi_split_semifinal_2026.json \
    --output checkpoints/v2_enhanced \
    --epochs 100 --batch-size 8

# 2. 评估
python -m src.train_enhanced_unet eval \
    --checkpoint checkpoints/v2_enhanced/best.pt \
    --data-root "DATA_ROOT" \
    --manifest configs/roi_split_semifinal_2026.json \
    --tta 8
```

### 追求高分 (5-7天)

```bash
# 1. Enhanced UNet++ (快速)
python run_quick_train.py --model enhanced_unet \
    --data-root "DATA_ROOT" \
    --manifest configs/roi_split_semifinal_2026.json \
    --output checkpoints/v2_efficient \
    --epochs 120

# 2. Hybrid CNN-Attention (高质量)
python run_quick_train.py --model hybrid \
    --data-root "DATA_ROOT" \
    --manifest configs/roi_split_semifinal_2026.json \
    --output checkpoints/v2_hybrid \
    --epochs 80 --base-channels 96

# 3. 集成两个模型
python scripts/ensemble_models.py \
    --checkpoints checkpoints/v2_efficient/best.pt \
                   checkpoints/v2_hybrid/best.pt \
    --output predictions/ensemble_v2
```

### 冲击最高分 (10-14天)

```bash
# 1. Marigold Latent Diffusion
python -m src.train_marigold_ihc 1 \
    --data-root "DATA_ROOT" \
    --manifest configs/roi_split_semifinal_2026.json \
    --output checkpoints/v2_marigold \
    --epochs 50 --batch-size 4 --base-channels 128

# 2. One-step fine-tuning
python -m src.train_marigold_ihc 2 \
    --checkpoint checkpoints/v2_marigold/best.pt \
    --data-root "DATA_ROOT" \
    --manifest configs/roi_split_semifinal_2026.json \
    --output checkpoints/v2_marigold_ft \
    --epochs 15
```

---

## 显存需求

| 模型 | Batch Size | 显存 (FP16) | 训练时间 (100 epochs) |
|------|------------|--------------|------------------------|
| Enhanced UNet++ | 8 | ~4GB | ~12小时 |
| Hybrid | 6 | ~6GB | ~18小时 |
| Marigold LDM | 4 | ~8GB | ~48小时 |

---

## 关键 SOTA 论文参考

### 扩散模型

1. **Marigold** (CVPR 2024) - 潜在扩散用于图像分析
2. **DiffVS** (AAAI 2026) - 虚拟染色扩散，Marker tokens
3. **D-VST** (NeurIPS 2025) - 病理感知扩散

### GAN 模型

4. **DSFF-GAN** - CSS Loss (Contrast-Structure Similarity)
5. **CUT** (ECCV 2020) - PatchNCE 对比学习

### Transformer

6. **Restormer** (CVPR 2022) - 高效图像恢复 Transformer
7. **NAFNet** (CVPR 2022) - 简单基线模型

---

## 预期分数提升

基于新架构和损失函数优化：

| 模型 | 预期平台分 | 相比当前最佳 |
|------|------------|--------------|
| Enhanced UNet++ | 75.5-76.0 | +0.3~0.8 |
| Hybrid CNN-Attention | 76.0-77.0 | +0.8~1.8 |
| Marigold LDM | 76.5-78.0 | +1.3~2.8 |
| **集成** | **77.5-79.0** | **+2.3~3.8** |

---

## 文件清单

```
src/models/
├── enhanced_unetpp.py      # Enhanced UNet++ (新)
├── transformer_ihc.py     # Hybrid CNN-Attention (新)
├── marigold_ihc.py        # Latent Diffusion (新)
├── ...

src/
├── train_enhanced_unet.py   # Enhanced UNet++ 训练
├── train_transformer_ihc.py # Transformer 训练
├── train_marigold_ihc.py    # Diffusion 训练

run_quick_train.py           # 快速训练入口

docs/
└── NEW_MODELS_GUIDE.md      # 详细文档
```

---

## 故障排除

### OOM (显存不足)
```bash
# 减小 batch size
--batch-size 4  # 或更小
```

### Loss 不下降
- 检查数据路径是否正确
- 确认 `--manifest` 正确加载
- 尝试降低学习率 `--lr 5e-5`

### 分数不理想
- 增加训练 epochs
- 使用更大的模型 `--base-channels 128`
- 使用 TTA=8 进行推理
- 集成多个模型

---

## 成功标准

完成训练后，检查以下指标：

1. **验证 SSIM > 0.75** - 表示模型学习到了有效映射
2. **TTA 提升 > 2%** - TTA 应该显著提升分数
3. **PSNR > 25** - 表示图像质量良好

祝训练顺利！ 🚀
