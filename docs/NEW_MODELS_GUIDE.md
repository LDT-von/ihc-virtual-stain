# 新模型训练指南 (2026-10-02)

## 概述

本次新增了三个全新架构的模型，结合了 GitHub 上最新的 SOTA 论文成果：

1. **Enhanced UNet++** - Dense Skip Connections + Multi-scale Attention
2. **Transformer (Hybrid)** - CNN + Transformer 混合架构
3. **Marigold-style Latent Diffusion** - 潜在空间扩散模型

## 当前最佳分数参考

| 模型 | 平台分数 | 说明 |
|------|----------|------|
| Ensemble (5 models) | **75.2485** | 当前最高分 |
| fullplus_cd68_v3 | 75.1128 | 第二高 |
| w96 expanded | 75.0876 | TTA=4 |
| Target | **78.3845** | 目标分数 |

## 新增模型详细

### 1. Enhanced UNet++ (推荐快速训练)

**文件**: `src/models/enhanced_unetpp.py`

**特点**:
- Dense skip connections (UNet++ 风格)
- Multi-scale attention blocks
- Deep supervision (多尺度监督)
- MS-SSIM + Edge Loss 组合

**优势**: 训练快速，显存需求适中

```bash
# 训练
python -m src.train_enhanced_unet \
    --data-root "DATA_ROOT" \
    --manifest configs/roi_split_semifinal_2026.json \
    --output checkpoints/enhanced_unetpp \
    --epochs 80 \
    --batch-size 8 \
    --base-channels 64 \
    --depth 4

# 评估
python -m src.train_enhanced_unet eval \
    --checkpoint checkpoints/enhanced_unetpp/best.pt \
    --data-root "DATA_ROOT" \
    --manifest configs/roi_split_semifinal_2026.json \
    --tta 8
```

### 2. Hybrid CNN-Transformer (推荐)

**文件**: `src/models/transformer_ihc.py`

**特点**:
- CNN 编码器高效提取特征
- Transformer 全局上下文建模
- CNN 解码器恢复高分辨率
- Gated-DConv Feed-Forward Network
- Multi-DConv Head Transposed Attention

**优势**: 结合 CNN 速度和 Transformer 能力

```bash
# 训练
python -m src.train_transformer_ihc \
    --data-root "DATA_ROOT" \
    --manifest configs/roi_split_semifinal_2026.json \
    --output checkpoints/transformer_hybrid \
    --model hybrid \
    --epochs 60 \
    --batch-size 6 \
    --lr 1e-4

# 评估
python -m src.train_transformer_ihc eval \
    --checkpoint checkpoints/transformer_hybrid/best.pt \
    --data-root "DATA_ROOT" \
    --manifest configs/roi_split_semifinal_2026.json \
    --tta 8
```

### 3. Marigold-style Latent Diffusion (SOTA)

**文件**: `src/models/marigold_ihc.py`

**特点**:
- 潜在空间扩散 (LDM)
- DAPI 条件注入 via Cross-Attention
- Marker-specific Tokens
- Two-stage: Full diffusion -> One-step Diffusion-FT

**优势**: SOTA 性能，但训练时间较长

```bash
# Stage 1: Full diffusion training
python -m src.train_marigold_ihc 1 \
    --data-root "DATA_ROOT" \
    --manifest configs/roi_split_semifinal_2026.json \
    --output checkpoints/marigold_ihc \
    --epochs 30 \
    --batch-size 4 \
    --base-channels 128

# Stage 2: One-step fine-tuning (可选)
python -m src.train_marigold_ihc 2 \
    --checkpoint checkpoints/marigold_ihc/best.pt \
    --data-root "DATA_ROOT" \
    --manifest configs/roi_split_semifinal_2026.json \
    --output checkpoints/marigold_ihc_ft \
    --epochs 10
```

## 快速开始

### 选项 1: 使用统一入口

```bash
# Enhanced UNet++
python run_train.py --model enhanced_unet --data-root "DATA_ROOT" --manifest configs/roi_split_semifinal_2026.json --output checkpoints/quick_run

# Transformer Hybrid
python run_train.py --model transformer --model-type hybrid --data-root "DATA_ROOT" --manifest configs/roi_split_semifinal_2026.json --output checkpoints/transformer_run
```

### 选项 2: 推荐的完整流程

```bash
# 1. 训练 Enhanced UNet++
python -m src.train_enhanced_unet \
    --data-root "DATA_ROOT" \
    --manifest configs/roi_split_semifinal_2026.json \
    --output checkpoints/v2_enhanced \
    --epochs 100 \
    --batch-size 8

# 2. 训练 Transformer Hybrid
python -m src.train_transformer_ihc \
    --data-root "DATA_ROOT" \
    --manifest configs/roi_split_semifinal_2026.json \
    --output checkpoints/v2_transformer \
    --model hybrid \
    --epochs 80 \
    --batch-size 6

# 3. 集成两个模型
python scripts/ensemble_models.py \
    --checkpoints checkpoints/v2_enhanced/best.pt checkpoints/v2_transformer/best.pt \
    --output predictions/ensemble_v2
```

## 损失函数

新模型使用的损失函数组合：

```
Total Loss = 
    1.0 * MS-SSIM Loss      # 结构相似性
  + 0.5 * L1 Loss          # 像素级重建
  + 0.1 * Edge Loss        # 边缘保持
  + 0.05 * Color Loss      # 颜色分布
  + 0.2 * Deep Supervision # 多尺度监督 (仅 UNet++)
```

## 推理和打包

### 推理

```bash
# Enhanced UNet++
python -m src.train_enhanced_unet infer \
    --checkpoint checkpoints/v2_enhanced/best.pt \
    --data-root "DATA_ROOT" \
    --input data/test/DAPI \
    --output predictions/enhanced_test \
    --tta 8

# Transformer
python -m src.train_transformer_ihc infer \
    --checkpoint checkpoints/v2_transformer/best.pt \
    --data-root "DATA_ROOT" \
    --input data/test/DAPI \
    --output predictions/transformer_test \
    --tta 8
```

### 打包提交

```bash
python src/submit.py \
    --predictions predictions/enhanced_test/results \
    --output submission_enhanced.zip
```

## 显存需求

| 模型 | Batch Size | 显存 (FP16) |
|------|------------|--------------|
| Enhanced UNet++ | 8 | ~4GB |
| Transformer Hybrid | 6 | ~6GB |
| Marigold LDM | 4 | ~8GB |

## 预期分数

基于新架构和损失函数优化，预期：

- **Enhanced UNet++**: 平台分 75.5-76.0
- **Transformer Hybrid**: 平台分 76.0-77.0
- **Marigold LDM**: 平台分 76.5-78.0
- **集成**: 平台分 77.5-79.0

## 建议的训练策略

1. **先训练 Enhanced UNet++** - 快速验证新损失函数
2. **再训练 Transformer Hybrid** - 获取更高分数
3. **如果时间允许，训练 Marigold** - 追求最高分
4. **集成多个模型** - 进一步提升分数

## 关键创新点

### 1. MS-SSIM Loss
多尺度结构相似性损失，比单尺度 SSIM 更好地保持结构。

### 2. Edge Loss
Sobel 边缘感知的损失，帮助保持组织边界清晰。

### 3. Deep Supervision
多尺度监督信号，让不同深度的特征都能得到有效训练。

### 4. Transformer Attention
全局注意力机制捕获长距离依赖，提高全局一致性。

### 5. Gated-DConv
门控深度可分离卷积，比标准卷积更高效。

## 故障排除

### OOM (显存不足)
- 减小 `--batch-size`
- 使用 `--device cpu` 临时测试

### Loss 不下降
- 检查数据路径是否正确
- 确认 `--manifest` 正确加载
- 尝试降低学习率

### 分数不理想
- 增加训练 epochs
- 尝试更大的模型 (`--base-channels 128`)
- 使用 TTA=8 进行推理
- 集成多个模型

## 参考论文

1. **UNet++** (MICCAI 2018): Nested U-Net with Dense Skip Connections
2. **Restormer** (CVPR 2022): Efficient Transformer for Image Restoration
3. **Marigold** (CVPR 2024): Repurposing Diffusion for Image Analysis
4. **DiffVS** (AAAI 2026): Diffusion-based Virtual Staining
5. **DSFF-GAN**: CSS Loss for stain transfer
