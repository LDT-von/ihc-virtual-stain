# 📦 提交指南

> ⚠️ **重要**：最终提交格式请以官方最新通知为准（QQ 群 `1084060012`）

## 一、数据准备

按以下结构放置数据到 `data/` 目录：

```
data/
├── train/
│   ├── DAPI/                 # 输入：所有 DAPI patch
│   │   ├── patch_001.png
│   │   ├── patch_002.png
│   │   └── ...
│   ├── IHC_HLA-DR/           # 真实标签：HLA-DR 染色
│   │   ├── patch_001.png     # 同名配对
│   │   └── ...
│   ├── IHC_CD45RO/           # 可选：训练多标记时
│   ├── IHC_Vimentin/
│   └── IHC_CD68/
├── val/                      # 可选：本地验证（与 train 同结构）
└── test/                     # 测试集（仅 DAPI/，用于推理提交）
    └── DAPI/
        └── ...
```

> 💡 官方数据下载方式详见 [aicomp.cn 赛题页](https://www.aicomp.cn/tracks/tracks-1/3759.html)

## 二、训练

```bash
# 单标记训练（HLA-DR）
python -m src.train --config configs/default.yaml --marker HLA-DR

# 多标记训练（CD45RO）
python -m src.train --marker CD45RO --epochs 200

# 从断点恢复
python -m src.train --resume checkpoints/HLA-DR_xxx/epoch50.pt
```

训练输出：

- `checkpoints/<marker>_<timestamp>/epochN.pt`：每 5 个 epoch 保存
- `checkpoints/<marker>_<timestamp>/final.pt`：最终模型
- `logs/train_<marker>.log`：训练日志

## 三、推理与本地评测

```bash
# 在验证集上评测（计算 SSIM/PSNR）
python -m src.inference \
  --ckpt checkpoints/HLA-DR_xxx/final.pt \
  --marker HLA-DR \
  --split val \
  --save_images

# 在测试集上生成（生成结果用于提交）
python -m src.inference \
  --ckpt checkpoints/HLA-DR_xxx/final.pt \
  --marker HLA-DR \
  --split test \
  --output_dir submissions/run_test/ \
  --save_images
```

推理输出：

- `submissions/<run>_<timestamp>/images/<name>.png`：生成的 IHC 图
- `submissions/<run>_<timestamp>/metrics.txt`：SSIM / PSNR

## 四、打包提交

```bash
python -m src.submit \
  --marker HLA-DR \
  --ckpt checkpoints/HLA-DR_xxx/final.pt \
  --output_dir submissions/run_test/ \
  --out_zip submission.zip
```

`submission.zip` 会包含：

- `submissions/run_test/images/` — 生成的 IHC 结果
- `src/` — 源代码（用于官方复现）
- `configs/` — 配置
- `checkpoints/.../final.pt` — 模型权重

## 五、关键时间节点

| 日期 | 事项 |
|------|------|
| 2026-04-28 起 | 官方开放报名 |
| 2026-10-15 20:00 | 报名截止 |
| 赛前/赛中 | 提交作品进入官方评测 |

## 六、注意事项

1. **GPU 资源**：Flow Matching 在 256×256 上训练大约需要 8GB+ 显存；本地若无 GPU，推荐 **AutoDL / 恒源云**（按小时计费，4090/3090 性价比高）。
2. **数据隐私**：训练好的模型权重仅限本人/团队使用，遵守官方数据使用协议。
3. **可复现**：提交时务必保留 `configs/` 与 `src/`，方便官方核对。
4. **多标记输出**：若做"一对多"挑战，需要为每个标记训练或共享一个多任务模型。

## 七、改进方向（可选）

- 用 **DDPM scheduler** 做对比基线
- 引入 **histogram matching** / **stain normalization** 后处理
- 用 **GAN discriminator** 做对抗损失（提高锐度）
- 集成 **patch-level diffusion**（大尺寸 WSI 分块）
- 试试 [UniPath](https://github.com/Hanminghao/UniPath)、[CytoSyn](https://arxiv.org/pdf/2603.18089) 等前沿方案