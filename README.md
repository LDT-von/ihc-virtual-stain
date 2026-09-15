# IHC Virtual Stain (DAPI → IHC)

AIC 2026 初赛 — DAPI 荧光图到 IHC 染色图的虚拟染色任务。

## 新候选：MarkerContextNet v1（2026-09-15）

新增共享多尺度形态特征、上下文注意力、四个独立 marker 输出头的配对重建模型。
包括 ROI 隔离划分、局部 SSIM、EMA、可核对的断点恢复和全量标签微调/推理流程；
同时修复旧 8x TTA 漏还原翻转的问题。已通过集成检查，**完整训练与是否提分尚待验证**。
方法、历史结果审计与服务器命令见 [运行交接说明](docs/MARKER_CONTEXT_V1.md)。

## 任务

- 输入：测试集 DAPI patch
- 输出：对应 marker 的 IHC 染色预测图（`*_fake.jpg`）
- 四个 marker：`HLA-DR`、`CD68`、`CD45RO`、`Vimentin`
- 评价指标：PSNR / SSIM（4 个 marker 平均分，平台评分）

## 方案

Pix2Pix GAN (U-Net 生成器 + PatchGAN 判别器)，DAPI 同时作为输入与条件。

- **生成器**：6 层 U-Net，`base_filters=64`，含残差块，skip connections
- **判别器**：PatchGAN（70×70 receptive field）
- **损失**：`L_gan (LSGAN) + 100 * L1 + 50 * L_SSIM`
- **增强**：随机水平/垂直翻转、90° 旋转、ColorJitter、GaussianBlur
- **优化器**：Adam (lr=2e-4, betas=(0.5, 0.999))，梯度裁剪 max_norm=1.0
- **patch size**：256×256
- **2025 SOTA 改进（CSSP2P GAN + UNIStainNet）**：
    - **CSS Loss**（Contrast-Structure Similarity）：DAPI ↔ fake ↔ IHC 三方联合约束，保留 DAPI 形态结构同时学习 IHC 染色分布（论文：arXiv 2511.18946, 2025）
    - **LPIPS Perceptual Loss**：替换 VGG 多层 L1，与人类感知更对齐（支持 `--use_lpips`，需 `pip install lpips`）
    - **Edge Loss**：fake ↔ IHC Sobel 边缘对齐

## 目录结构

```
src/
  data/dataset.py          # DAPI/IHC 配对数据集 + UnpairedDataset + 增强
  models/
    pix2pix_gan.py        # U-Net 生成器 + PatchGAN 判别器
    losses.py              # SSIM / L1 / CSS / LPIPS / Edge Loss
    cut_model.py           # CUT/FastCUT 模型（ResNet Generator + PatchNCE）
    patchnce.py            # PatchNCE 对比损失
    train_pix2pix_v2.py   # Pix2Pix 训练入口
    inference_pix2pix.py  # 推理入口
  train_cut.py             # CUT/FastCUT 训练入口
  metrics/ssim_psnr.py    # 评价指标
scripts/
  train_cut_unpaired.py    # CUT 自动化脚本（4 marker）
  inference_8x_tta.py     # 8x/4x TTA 推理入口
train_all_markers.py       # 顺序训练多个 marker
resume_train_sota.py       # 从 best.pt 续训 + CSS/LPIPS 改进
inference_8x_tta.py       # TTA 推理（8x 翻转+旋转 / 4x 仅翻转）
make_submission.py         # 打包 submission.zip
```

## 训练

### 标准训练（无 SOTA 改进）

```bash
python -m src.train_pix2pix_v2 --marker CD68 --epochs 30 --batch_size 24
```

### SOTA 改进续训（推荐）

从各 marker 最新的 `best.pt` 出发，加 CSS Loss + LPIPS：

```bash
python resume_train_sota.py --epochs 12 --lr 1e-5 --css 20.0 --perceptual 5.0
```

参数说明：
- `--css 20.0`：CSS Loss 权重（论文推荐 10-25）
- `--perceptual 5.0`：LPIPS 感知损失权重
- `--lr 1e-5`：极小学习率避免 catastrophic forgetting
- `--use_lpips`：使用官方 lpips 库替代 VGG 实现（需先 `pip install lpips`）

所有 marker：
```bash
python train_all_markers.py
```

## 推理

```bash
python -m src.inference_pix2pix \
  --ckpt checkpoints/pix2pix_v2_HLA-DR_xxx/final.pt \
  --marker HLA-DR \
  --batch-size 16
```

输出到 `results/test/<marker>/<input_name>_fake.jpg`。

## 打包

```bash
python make_submission.py
```

## 当前最佳成绩

| 提交时间 | 分数 | 说明 |
|---------|------|------|
| 2026-09-10 | 64.0891 | 初始 4 marker 联合 |
| 2026-09-10 | 67.4586 | 全量数据微调后 TTA 推理 |

### 4x TTA 验证结果（train val 划分子集）

| Marker | 4x TTA SSIM | PSNR |
|--------|-------------|------|
| HLA-DR | 0.7783 | 22.45 |
| CD68 | 0.7336 | 24.60 |
| CD45RO | 0.7085 | 23.24 |
| Vimentin | 0.7239 | 21.90 |
| **平均** | **0.7361** | **23.05** |

> 注：平台评分基于 test 集，val 划分子集的 SSIM 仅供横向对比参考。4x TTA 相比 8x TTA 在 CD68/CD45RO/Vimentin 上更稳定（SSIM 更高），已用于生成本次 submission。8x TTA 在 HLA-DR 上表现略优，可作为后续混合策略探索。

## CUT / FastCUT（新增：Unpaired 对比学习）

### 为什么新增 CUT

CUT（Contrastive Unpaired Translation, ECCV 2020）和 FastCUT 是 **unpaired 图像翻译**的 SOTA 方法，与你的 paired pix2pix 是两种互补的路线：

| | Paired Pix2Pix | CUT / FastCUT |
|--|--|--|
| 数据需求 | 需要 pixel-level 配对 | **不需要配对**，DAPI/IHC 各自独立采样 |
| 约束方式 | L1 + SSIM pixel-level 监督 | **PatchNCE 对比学习**（特征空间） |
| 泛化能力 | 强依赖配对质量 | **更强泛化**（不记忆像素映射） |
| 训练稳定性 | 依赖 L1 收敛 | PatchNCE 更稳定 |

**你的数据是 paired，但用 CUT 训练也有价值**：打破 pixel-level 配对约束，强迫生成器学习更本质的特征映射。

### 训练

```bash
# FastCUT unpaired（推荐，比标准 CUT 快 2-3 倍）
python scripts/train_cut_unpaired.py --CUT_mode FastCUT

# 标准 CUT unpaired
python scripts/train_cut_unpaired.py --CUT_mode CUT

# paired 模式（CUT Loss + paired 采样）
python scripts/train_cut_unpaired.py --mode paired
```

### 关键参数

- `--CUT_mode FastCUT`：轻量单向翻译（推荐）
- `--mode unpaired`：DAPI 和 IHC 各自独立采样
- `--lambda_NCE 1.0`：PatchNCE 损失权重（FastCUT 推荐 10.0）
- `--nce_T 0.07`：NCE temperature
- `--nce_layers 0,4,8,12,16`：在 ResNet 的 5 个层级计算 PatchNCE

### 单 marker 训练

```bash
python -m src.train_cut \
    --marker HLA-DR \
    --mode unpaired \
    --CUT_mode FastCUT \
    --epochs 80 --n_epochs_decay 40 \
    --batch_size 8 --lr 2e-4 \
    --lambda_NCE 1.0 --lambda_GAN 1.0 \
    --nce_T 0.07 --num_patches 256
```

## SOTA 论文来源

| 论文 | 来源 | 核心贡献 |
|------|------|---------|
| CSSP2P GAN (arXiv 2511.18946, 2025) | Pix2Pix 改进 | CSS Loss：DAPI↔fake↔IHC 三方约束 |
| UNIStainNet (arXiv 2603.12716, 2026) | Pix2Pix 改进 | 多尺度 LPIPS + 边缘约束（misalignment-tolerant） |
| HistDiST (arXiv 2505.06793, 2025) | LDM | Latent Diffusion + DDIM inversion（架构级改动） |

## 待优化

- [x] CSS Loss（DAPI↔fake↔IHC 三方约束）
- [x] LPIPS Perceptual Loss（替代 VGG L1）
- [ ] HistDiST LDM 架构（Latent Diffusion + DDIM inversion）
- [ ] PAINT/VAR Transformer 路线
- [ ] 多模型集成（不同 seed / 不同 loss 组合取平均）
- [ ] 增大 base_filters 64→128
- [ ] 8x TTA（翻转 + 90°旋转）
