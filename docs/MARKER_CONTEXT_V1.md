# MarkerContextNet v1：运行交接与结果审计

2026-09-15；基于已拉取的 `f8c2e45bbf1747661cac2c5b45b2ce3dd5ed41ac`。

## 当前结论

这是为本项目重新实现的多 marker 像素预测模型，不是 Pix2Pix 改名，也不是 DiffVS/NAFNet 论文的复现。它使用已有的门控卷积、U 形多尺度结构和注意力组件组合，**不宣称这些组件是原创，也不宣称已超过现有最佳模型**。

按用户要求，完整训练交由另一台机器执行。本地做了源码审计、17,946 张已保存预测图的重新评测、CPU 集成测试，以及真实数据/GPU 小规模训练。完整训练未完成，没有新模型官方分数。测试用的 smoke 权重只能检查流程，不能提交或当作正式实验结果。

## 先看赛题，而不是先选生成模型名字

- 输入：同位置的 DAPI，输出四个 marker：HLA-DR、CD68、CD45RO、Vimentin。
- 当前数据每种通道有 6,296 张训练图、DAPI 测试图 1,346 张，256×256 JPEG。当前读取的训练图为 RGB 三通道相同的灰度信号；新读取器会逐图检查，不会静默把彩色标签转灰度。
- 赛题公开口径是 `0.7 × SSIM + 0.3 × Normalize(PSNR)`；PSNR 归一化细节和完整 SSIM 实现尚未确认。这里报告标准局部 SSIM 与 PSNR，**不把 SSIM×100 当官方分，也不自行规定 PSNR/50**。
- 输出目录：`results/test/<marker>/<原输入名去扩展名>_fake.jpg`。推理只读取 DAPI。
- 数据来自比赛训练标签，不下载外部数据或预训练模型，不使用在线商业图像生成接口。
- 题目来源：[AIC 赛题页面](https://www.aicomp.cn/tracks/tracks-1/3759.html) 及项目内赛题说明。后续器官/彩色数据需要重新核对适用性。

## 为什么原方案可能丢分

### 已确认：8 次增强没有正确恢复坐标

旧 `inference_8x_tta.py::tta_8x_forward` 与 `tta_eval.py::tta_8x` 先翻转、再旋转，预测后只逆旋转，漏了逆翻转；而且实际调用 16 次，只有 8 种独立几何变换。平均的图像并未对齐。已修复为 8 种唯一变换，完整逆变换。

相同 314 个输入名称的已保存 JPEG 重新计算如下；这些是历史图片分数，不是修复后重新推理的结果：

| 历史结果 | HLA-DR SSIM | CD68 SSIM | CD45RO SSIM | Vimentin SSIM | 四项平均 |
|---|---:|---:|---:|---:|---:|
| results_4xtta | 0.7833 | 0.7455 | 0.7140 | 0.7137 | 0.7391 |
| results_8xtta | 0.7212 | 0.5731 | 0.6139 | 0.6620 | 0.6426 |

平均 PSNR 分别约 23.064 和 20.793 dB。不能从已经混合错位的 JPEG 无损还原正确预测；需要原权重重新推理。仓库没有这些权重，因此本轮没有声称恢复了多少平台分。

### 已确认：评测和训练目标不一致

- 共享 `MetricAggregator` 原来使用全图均值/方差近似，而不是局部窗口 SSIM；**不是一律偏高，而是不可与局部 SSIM 混比**。已改为局部窗口，旧算法保留为显式 `legacy_global_ssim_single` 以便追溯。
- v11 的验证函数已调用 skimage，但训练 `SSIMLoss` 仍直接使用 `[-1,1]`，常数按单位范围设置，且采用另一种窗口/边界处理。新方法统一在 `[0,1]`、7×7 局部窗口上计算。
- 原 `build_transforms` 把 IHC 也登记为 image，颜色扰动、噪声和模糊因此也作用到标签。新方法只对输入和四个目标做同步直角旋转/翻转，保持真实强度。
- 原 CSS 还要求 fake 与 DAPI 在对比/结构上相似。DAPI 是核信号，不等同每个 marker 的表达；这一约束存在任务不匹配风险。**这是设计假设，是否损害得分仍需消融**，新方法不使用它。

### 已确认：部分“验证”不能证明泛化

- `split_data.py` 把验证样本复制到 val，但未从原 train 移除；若继续使用原 train，就会直接重叠。
- v11 使用不重叠的随机 patch 子集，不能说它直接重叠，但相同 ROI 的相邻 patch 仍跨训练/验证。
- 部分 ensemble/eval 脚本从 train 随机抽样，再命名为验证；没有对应权重与划分归档，不能证明它们未被训练过。
- 本轮历史 JPG 分数没有配套 checkpoint/split 证据，**均不是认证的独立测试结果**。README 记载的官方最高分 67.4586 也只是仓库记录，本轮没有登录平台核验。

## 新方法

网络输入 `N×1×H×W`，输出 `N×4×H×W`，四个输出通道固定对应上述 marker 顺序。

1. 全分辨率卷积提取核形态；四个尺度共享编码，通道数为 `w, 2w, 4w, 8w`。
2. 门控空间块：逐像素通道归一化 → 1×1 扩展 → 深度可分离 5×5 空间卷积 → 门控与通道选择 → 残差更新。
3. 低分辨率处加入空洞卷积和池化至最多 8×8 token 的上下文注意力，补充细胞邻域信息，避免全分辨率注意力的开销。
4. 解码时融合各层跳接，包括完整 256×256 层；四个独立卷积头分别预测 marker 强度。
5. 输出 sigmoid 限定到 `[0,1]`。**不把 DAPI 直接加到输出，不假设输入与目标同色同亮度，也不对每张目标独立归一化**。

默认建议服务器使用 width=32（1,729,157 个参数）；本地 width=16 有 452,933 个参数。较宽配置不是已验证的最优选择，应由干净验证集决定。

损失（所有 marker 等权平均）：

```text
L = (1 - local_SSIM(pred, target))
    + 0.5 * L1(pred, target)
    + 2.0 * MSE(pred, target)
    + 0.1 * L1(avgpool2(pred), avgpool2(target))
    + 0.1 * L1(avgpool4(pred), avgpool4(target))
```

这些是待验证的训练权重，不是对官方评分公式的等价改写。这里没有对抗损失、VGG/LPIPS 或随机扩散采样；优先检验精确配对重建能否改善此任务。新模型或更好看的纹理不自动等于更高 SSIM/PSNR。

AdamW、余弦学习率、梯度裁剪、EMA；支持 BF16 的显卡使用 BF16，其余 CUDA 用带溢出保护的 FP16。验证、续训和推理明确使用对应 EMA 权重，不混淆 raw/EMA 的分数。

## 划分与证据

使用 `configs/roi_split_v1.json`，seed=42：

| 用途 | 图像数 | ROI |
|---|---:|---|
| 训练 | 4,813 | 除下列六个 ROI 以外的 19 个 ROI |
| 选模型/超参数 | 677 | ROI009、ROI012、ROI016 |
| 锁定方案后检查一次 | 806 | ROI006、ROI018、ROI019 |

ROI 独立不等于患者独立，因为没有患者映射。官方测试 ROI025–029 不参与这些划分和调参。split 校验码验证的是分组与文件名清单，不是全部图像字节内容。

每次 run 保存 split、实际样本名、marker 顺序、配置、Git HEAD、实际源码哈希及源码副本、优化器/精度状态、随机数状态、每轮指标、逐图/逐 marker 分数。输出目录非空时拒绝意外覆盖；续训必须保持源代码、模型、划分、批量、学习率计划一致。

## 在另一台机器运行

以下命令在项目根目录运行。把 `/data/ihc` 换成**直接包含 `train/DAPI` 与 `test/DAPI` 的目录**；中文/空格路径需加引号。先安装适合服务器 CUDA 的 PyTorch，再安装 `requirements-marker-context.txt`。旧代码回归测试还需要原项目依赖（如 albumentations）。

也可使用 `artifacts/marker_context_v1_20260915.zip` 源码交接包，解压到新目录后运行。包内不含数据、权重和正式预测，`BUNDLE_MANIFEST.json` 列出文件哈希；若合并到另一份已有项目，先检查同名文件差异，不要直接覆盖对方的未提交修改。

### 1. 先检查环境和短流程

```bash
python -m pip install -r requirements-marker-context.txt
python scripts/run_marker_context.py --data-root /data/ihc --mode smoke --run-dir checkpoints/mcn_smoke
```

仅 32 张训练、8 张验证、两轮小模型；成功只表示服务器可运行。不要用该权重提交。

### 2. 正式候选训练

```bash
python scripts/run_marker_context.py --data-root /data/ihc --mode train --run-dir checkpoints/mcn_w32_s42 --width 32 --epochs 60 --batch-size 8 --lr 0.0005
```

训练后自动对**同一套验证集**比较 JPEG 保存后的 1/4/8 次增强，生成 `tta_selection.json`，不自动提交、不自动读取独立检查集。这里选择标准仍是验证 SSIM，PSNR 同时报告；官方综合最优未被保证。若内存小，加入 `--no-cache`；若显存不足，降低批量，或者另开 width=16 的 run。

中断续训：保持全部原参数，用 `--resume checkpoints/mcn_w32_s42/last.pt`；`--epochs 60` 是原总轮数，不是额外再跑 60 轮。改源码/模型/训练计划，应新开 run 并使用底层入口的 `--init-checkpoint`，不能冒充完全恢复。

### 3. 对照和锁定

```bash
python scripts/run_marker_context.py --data-root /data/ihc --mode ablation --run-dir checkpoints/mcn_no_context_w32_s42 --width 32 --epochs 60 --batch-size 8 --lr 0.0005
```

这会去掉空洞卷积/池化注意力上下文分支，其余训练协议一致。它检验上下文分支是否值得，不等于已经与 Pix2Pix 同条件比较。原模型公平比较也必须重用这一 ROI 清单、目标处理、局部 SSIM/JPEG 评测。历史随机 patch 分数不能直接与本划分比较。

锁定架构、训练配置和 TTA 后，再执行一次独立检查；下面 `--tta 4` 是示例，应换成已在验证集锁定的值：

```bash
python -m src.train_marker_context eval --data-root /data/ihc --manifest configs/roi_split_v1.json --checkpoint checkpoints/mcn_w32_s42/best.pt --split holdout --tta 4 --output checkpoints/mcn_w32_s42/holdout_locked.json
```

### 4. 全量标签微调（可选，方案锁定之后）

```bash
python scripts/run_marker_context.py --data-root /data/ihc --mode refit --run-dir checkpoints/mcn_refit --checkpoint checkpoints/mcn_w32_s42/best.pt --width 32 --epochs 5 --batch-size 8 --lr 0.0001
```

会使用全部 6,296 张训练标签，**关闭验证与 best 选择，只保存预先固定轮数的 final.pt**。五轮是起始方案，不是实测最优。全量微调后禁止再以本划分的 val/holdout 宣称泛化；程序也会拒绝这种评测。

### 5. 生成提交图片

```bash
python scripts/run_marker_context.py --data-root /data/ihc --mode predict --run-dir results_mcn_final --checkpoint checkpoints/mcn_refit/final.pt --batch-size 8 --tta 4
```

生成 `results_mcn_final/results/test/<marker>/*_fake.jpg`，每个 marker 应有 1,346 张，256×256 灰度 JPEG，quality=95。目录外的 `provenance.json` 仅用于追踪，不放进初赛图片包。

按现有单 marker 打包入口，例如：

```bash
python -m src.submit --results-dir results_mcn_final/results --marker HLA-DR --stage preliminary --out-zip submission_mcn_HLA-DR.zip
```

其他 marker 分别替换参数；若平台允许合并四个 marker，压缩包顶层只能是 `results/`，不要多套 `results_mcn_final/`。打包前核对平台本阶段要求。本轮未生成正式测试预测、未上传或推送。

## 本地验证边界

- 10 项自动测试通过，覆盖局部 SSIM 对 skimage 的误差（≤3e-5）、形状/梯度、增强可逆性、旧 TTA 回归、划分泄漏、目标强度保持、彩色数据拒绝，以及训练→续训→评测→推理→全量微调的集成路径。
- 真实 GPU smoke：width=16、64 张训练/16 张验证、两轮完成，零溢出；另验证了服务器 launcher 的 width=8、32/8 图像、两轮和 8 次增强/JPEG 评测。这些分数不列为正式质量证据。
- 第一轮 FP16 长训练曾在首轮发生梯度溢出；已改 BF16/受保护 FP16。后续本地长训练按用户指示主动停止，未产生完整实验结果。
- 仅用训练 ROI 拟合的简单亮度/邻域线性基线，在 677 张验证图上 SSIM=0.61906、PSNR=21.4098 dB。它是 sanity baseline，不是深度学习基线，也不能证明新方法优势。
- 完整结果审计见 `docs/experiments/saved_prediction_audit_20260915.json`；逐图线性基线见 `docs/experiments/roi_ridge_baseline.json`。

真正是否提分，需要那边完成干净对照与平台评测后再下结论。
