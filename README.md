# IHC Virtual Stain (DAPI → IHC)

AIC 2026 初赛 — DAPI 荧光图到 IHC 染色图的虚拟染色任务。

## 任务

- 输入：测试集 DAPI patch
- 输出：对应 marker 的 IHC 染色预测图（`*_fake.jpg`）
- 四个 marker：`HLA-DR`、`CD68`、`CD45RO`、`Vimentin`

## 方案

Pix2Pix GAN (U-Net 生成器 + PatchGAN 判别器)，DAPI 同时作为输入与条件。

- **生成器**：6 层 U-Net，`base_filters=64`，含残差块，skip connections
- **判别器**：PatchGAN（70×70 receptive field）
- **损失**：`L_gan (LSGAN) + 100 * L1 + 50 * L_SSIM`
- **增强**：随机水平/垂直翻转、90° 旋转、ColorJitter、GaussianBlur
- **优化器**：Adam (lr=2e-4, betas=(0.5, 0.999))，梯度裁剪 max_norm=1.0
- **patch size**：256×256

## 目录结构

```
src/
  data/dataset.py        # DAPI/IHC 配对数据集 + 增强
  models/
    pix2pix_gan.py       # 生成器 + 判别器
    losses.py            # GANLoss / CombinedLoss (L1+SSIM)
  train_pix2pix_v2.py    # 训练入口
  inference_pix2pix.py   # 推理入口
  metrics/ssim_psnr.py   # 评价指标
train_all_markers.py     # 顺序训练多个 marker
make_submission.py       # 打包 submission.zip
```

## 训练

单 marker：

```bash
python -m src.train_pix2pix_v2 --marker CD68 --epochs 30 --batch_size 24
```

所有 marker：

```bash
python train_all_markers.py
```

可调参数（`train_all_markers.py`）：`epochs`、`batch_size`、`lr`、`lambda_l1`、`lambda_ssim`、`num_workers`。

checkpoint 输出到 `checkpoints/pix2pix_v2_<marker>_<timestamp>/epoch{}.pt`，末尾保存 `final.pt`。

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

生成 `submission.zip`，目录结构：

```
test/<marker>/<input_name>_fake.jpg
```

## 复现

1. 准备数据集（`data_root/train/{DAPI,IHC_<marker>}`、`data_root/test/DAPI`）
2. `python train_all_markers.py` （约 ~5 小时，RTX 3090）
3. 按 marker 逐个跑 `inference_pix2pix.py`
4. `python make_submission.py`

## 当前最佳成绩

- 初赛平台提交：64.0891（4 个 marker 联合）

## 待优化

- [ ] val 评估 + 保存 best.pt
- [ ] 增大训练 epoch（30→80+）
- [ ] 加 patch overlap + 滑窗融合
- [ ] 尝试 Pix2Pix + Perceptual Loss (VGG)
- [ ] 尝试 diffusion 模型（Flow Matching 已留接口）
