# AIC 比赛项目 — 基于虚拟染色的免疫组化图像生成

> 第八届全球校园人工智能算法精英大赛（AIC）· 算法挑战赛道
> 官方链接：https://www.aicomp.cn/tracks/tracks-1/3759.html

## 赛题概览

- **任务**：给定 **DAPI 染色图像 patches**，生成对应的 **IHC 标记图像**
- **目标标记（4 选 1 / 可选多输出）**：
  - `HLA-DR` （MHC II 类，抗原递呈）
  - `CD45RO` （记忆 T 细胞）
  - `Vimentin` （间质细胞/上皮-间质转化标志）
  - `CD68` （巨噬细胞）
- **评测指标**：SSIM（结构相似度）+ PSNR（峰值信噪比）
- **报名截止**：2026-10-15 20:00
- **官方交流**：QQ 群 `1084060012`（验证：学校名+姓名）

## 方法：Flow Matching（条件生成）

选用 **Flow Matching** 作为基础生成模型，相比传统 DDPM：

| 维度 | DDPM | Flow Matching |
|------|------|--------------|
| 采样步数 | 50-1000 | 20-50 |
| 训练稳定性 | 一般 | 高 |
| 病理图像保真 | 良好 | 更优 |
| 推理速度 | 慢 | 快 ~10x |

**核心思路**：学习一个从噪声到真实 IHC 图像的连续流场 `v_θ(x_t, t, x_dapi)`，通过 ODE 求解器（Euler / Heun）逐步去噪生成。

## 目录结构

```
aic/
├── data/                   # 训练/测试数据（已 gitignore）
│   ├── train/
│   │   ├── DAPI/           # 输入 DAPI patches
│   │   └── IHC_<marker>/   # 对应标记 IHC patches
│   └── test/
├── notebooks/              # EDA / 试跑 notebook
├── src/
│   ├── data/
│   │   ├── dataset.py      # PyTorch Dataset
│   │   └── transforms.py   # 数据增强
│   ├── models/
│   │   ├── unet.py         # 条件 UNet（含 DAPI 条件输入）
│   │   └── flow_matching.py# FM 训练/采样
│   ├── losses/             # 损失函数
│   ├── metrics/            # SSIM / PSNR
│   ├── train.py            # 训练入口
│   ├── inference.py        # 推理入口
│   └── submit.py           # 打包提交
├── configs/                # YAML 配置
├── checkpoints/            # 模型权重（已 gitignore）
├── logs/                   # TensorBoard / 日志（已 gitignore）
├── submissions/            # 提交结果
├── requirements.txt
└── README.md
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 把数据放到 data/ 目录
#    训练：DAPI/ 与 IHC_<marker>/ 下分别放同名 patch
#    测试：只放 DAPI/

# 3. 训练
python -m src.train --config configs/train_<marker>.yaml

# 4. 推理并提交
python -m src.inference --ckpt checkpoints/best.pt --marker <marker>
python -m src.submit --out submissions/run_<timestamp>/
```

## 提交记录

| 时间 | 模型 | Marker | SSIM | PSNR | 备注 |
|------|------|--------|------|------|------|
|      |      |        |      |      |      |