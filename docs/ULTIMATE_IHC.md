# IHC Ultimate v1：基准保留与四标记残差精修

2026-09-20。目标：DAPI → HLA-DR、CD68、CD45RO、Vimentin 的配对灰度重建。

## 目前交付到哪里

方法、训练、续训、验证选模、锁定后独立检查、推理和提交打包均已实现。复赛版强制 seed=2026，并将固定验证选择写入唯一的 `final.pt`。**seed=42 的旧检查及旧权重均为历史证据，不符合复赛统一种子要求。复赛版已完成完整训练，并于 2026-09-22 以提交号 AIC-2026-51704013 取得平台分 74.6531，超过历史最高 67.9114。**

代码以远程分支 `cursor/fix-flow-matching-reverse-ode-and-add-stable-diffusion` 的 `6e60f0194d5dda9c19a96eb0d3acaa8f22425f6d` 为基础，独立工作分支为 `codex/final-ihc`。交付包附逐文件 SHA256；基础提交号不代表这些新增修改已推送。

## 已核验的平台成绩

| 提交号 | 提交时间 | 平台分 | 对应模型与产物 |
|---|---|---:|---|
| AIC-2026-51704013 | 2026-09-22 10:29 | **74.6531** | w48 冻结基准（复赛数据 1540/420/420、seed=2026、TTA=4、8-bit RGB JPEG）；`predictions/ultimate_w48_2380_rgb/submission.zip`，2240 张 |

对应本地 holdout（420 张）：SSIM 0.813827 / PSNR 25.3676；同口径 width32 为 0.810559 / 25.1978。容量提升（+0.0033 SSIM）在平台上体现为 67.9114 → 74.6531，说明本地同口径 holdout 差值可用于选型。

## 分数先对上实际模型

以下为用户提供的历史平台记录，尚未通过提交 ID、归档和权重哈希复核：

| 历史名称 | 用户记录分数 | 当前能确认的边界 |
|---|---:|---|
| MCN 20ep | 67.9114 | 当前验收参照；尚未取得对应权重 |
| MCN 60ep | 67.8401 | 与上项相差 0.0713；不能据此确定过拟合或显著差异 |
| per_marker_unet_v2 | 63.2290 | 不应只凭文件夹名认定为 ResNet50 |
| MCN v1 | 61.8401 | 需核对实际训练及推理配置 |
| ResNet50 UNet v2 | 53.5875 | 仅作为历史记录 |
| ResNet50 v1 | 50.9368 | 仅作为历史记录 |

本机原工作区 `scripts/train_mcn_resnet_chain.sh` 的 `mcn_v1_20ep` 调用的是 `src.train_marker_context`，由 `mcn_v1/best.pt` 初始化，再训练 20 轮。因而“MCN 20ep = pix2pix_v11”“20ep 是总训练轮数”都不能直接采用；最终以服务器 checkpoint 内参数键、保存配置、源代码与提交归档为准。

该脚本使用的本机 `configs/roi_split_5fold/fold_0.json` 为 train=4029、val=1007、holdout=0，仅含 5036 个样本。当前完整数据划分是 6296 个样本，二者不同。**不能把旧权重直接放进新划分，再把旧模型看过的样本称作独立验证。** 程序会拒绝不匹配的历史权重；没有完整原始划分和训练来源时，默认重新训练干净基准。

本机配置的 `rtx-5090`（经 `jump-server`）及直连均连接超时，因此没有远程启动训练或取回权重。用户提到的本地 `mcn_v1_20ep_test_submission.zip` 在本次环境中也未找到。待连接恢复，应在服务器先生成权重身份记录：

```bash
python scripts/inspect_checkpoint.py /data1/AIC/checkpoints/mcn_v1 /data1/AIC/checkpoints/mcn_v1_20ep --output checkpoint_identity.json
```

同时保留对应的 `run.json`、`split.json`、`sources/`、`history.jsonl`、推理配置及提交 ZIP。若实际高分权重是 Pix2Pix/ResNet，本版不会强行按 MCN 加载；需要对应原始模型实现后再接入基准适配。

## 模型与任务的对应关系

输入仅为 DAPI，保持原始强度 `/255`。首先得到冻结基准 `B(x)` 的四通道结果，再将 `[x, B(x)]` 共五通道送入精修网络。训练和推理都使用基准预测作为输入，不向精修器输入真实 IHC 标签。

精修网络使用四级共享形态编码器、门控空间卷积、低分辨率上下文注意力，以及四个独立的完整解码器。各 marker 可以学习不同尺度的表达修正，不必共用最后一张解码特征。输出为：

`candidate_m(x) = clamp(B_m(x) + 0.2 * tanh(R_m(x, B(x))), 0, 1)`。

四个修正输出头权重和偏置均从零开始，初始输出严格等于基准；基准没有梯度，EMA 也不更新其权重。单像素修正幅度上限为 0.2。默认基准 width=32、精修 width=24，总参数 3,468,634，其中可训练 1,739,477。这是任务化工程方案，不宣称组件原创或未经验证的领先性能。

相比重新随机训练一个单独的大模型，这个设计的动机是显式利用已有配对重建结果，并让新分支学习其误差。是否带来收益必须由配对实验回答；基准信息本身不足时，精修也可能没有收益。

监督目标为 `1-local_SSIM + 0.5*L1 + 2*MSE + 0.1*L1(pool2) + 0.1*L1(pool4)`，四 marker 等权。仅用同步翻转、直角旋转增强，不改变标签强度。SSIM 使用 7×7 有效窗口和样本协方差，已与 skimage 数值核对。

[官方赛题](https://www.aicomp.cn/tracks/tracks-1/3759.html) 给出 `0.7*SSIM + 0.3*Normalize(PSNR)`，公开文本没有完整 PSNR 归一化细节，所以本地分别报告 SSIM、PSNR，不编造本地“官方总分”。

## 验证和部署规则

复赛固定随机种子为 2026。`--manifest` 默认值为 `configs/roi_split_semifinal_2026.json`（初赛数据 4711/782/803，哈希 `02aff2b61f62da71c0fa65a5748f90761f23f7e1d245b5530531d306796f04d6`）；**取得 74.6531 的模型实际使用 `configs/roi_split_semifinal_2026_2380.json`：train=1540、val=420、holdout=420，哈希 `c050a1378ffa4027d628a24502e5431174a390050d4b6fb9f23f469ab8726444`。** 三个集合的 ROI 完全互斥；ROI 隔离不等于患者隔离，目前没有患者映射。

1. 基准和精修只使用 train 标签训练；每轮按固定 JPEG 序列化后的 val 平均 SSIM 选 best，PSNR 同时记录。序列化固定为 8-bit RGB JPEG（三通道相同）、quality=100、subsampling=0、optimize=false，不提供可调压缩参数。
2. 完成精修后，在完整 val 上用预先固定的 TTA=4 和同一固定序列化，分别评估冻结基准与精修结果。
3. 每个 marker 只有 SSIM 严格提高且 PSNR 不下降才启用精修；平局保留基准。该固定选择写入组合模型状态，并导出唯一的 `final.pt`；`deployment.json` 只作为验证审计记录，不参与测试推理。
4. 锁定后在 holdout 上一次性比较原基准与部署模型，输出逐图、逐 ROI、逐 marker 的结果及差值。**holdout 不再调整分支选择，也不承诺不下降。** 报告若下降，应如实作为独立验收失败记录，不围绕它继续搜索再声称独立。
5. 推理只允许读取 `final.pt`，从 checkpoint 取得 seed、TTA 和固定 marker mask；模型保持 eval/no-grad，不更新参数或统计量。正式打包入口要求已有匹配的独立检查报告，拒绝训练 checkpoint 和小规模流程测试模型。
6. TTA 只包含固定旋转/翻转，每个预测精确逆变换后等权平均。平均结果直接转为官方要求的 8-bit RGB uint8 JPEG（三通道相同）；没有阈值、形态学、去噪、平滑、锐化、Gamma/亮度/对比度、直方图匹配或 CLAHE。

这一规则只保证用于选择的本地 val 指标按该协议不退步。与历史 67.9114 的比较已由平台验证：提交号 AIC-2026-51704013 得 74.6531。完整训练最少归档基准、精修、部署三行本地指标，再归档平台提交 ID 与实际分数。

## 服务器运行

把代码包解压到一个新目录，在该目录执行 `python scripts/verify_bundle.py` 核对源码文件。`DATA_ROOT` 必须直接包含 `train/DAPI`、四个 marker 标签目录和 `test/DAPI`。先使用已支持服务器 GPU 的 PyTorch 环境，再安装 `requirements-marker-context.txt`。本方法不依赖 torchvision 或在线预训练权重。

以下是可跨机器运行的入口；`--device cuda:1` 仅沿用历史脚本的设备选择，正式运行前以服务器当前空闲 GPU 为准。显存不足时从 batch=4 降到 2，并从新 run 开始；续训必须保留原 batch 和总轮数。

```bash
# 先用独立目录检查两阶段流程。它不会产生可提交模型。
python scripts/run_final.py --data-root /data1/AIC/data --run-dir checkpoints/ultimate_smoke --mode smoke --device cuda:1 --batch-size 4

# 从零训练干净 MCN 基准 60 轮，再冻结它训练精修 20 轮。
python scripts/run_final.py --data-root /data1/AIC/data --run-dir checkpoints/ultimate_v1 --mode train --device cuda:1 --batch-size 4

# 中断后：保持所有原参数，追加一个布尔开关 --resume。
python scripts/run_final.py --data-root /data1/AIC/data --run-dir checkpoints/ultimate_v1 --mode train --device cuda:1 --batch-size 4 --resume

# 锁定后的一次独立检查。结果可能下降，需要如实保留。
python scripts/run_final.py --data-root /data1/AIC/data --run-dir checkpoints/ultimate_v1 --mode evaluate --device cuda:1 --batch-size 4

# 生成四 marker JPEG 并自动核对、打包到 output/submission.zip。
python scripts/run_final.py --data-root /data1/AIC/data --run-dir checkpoints/ultimate_v1 --mode predict --device cuda:1 --batch-size 4 --output predictions/ultimate_v1
```

默认基准 AdamW、lr=5e-4，精修 lr=1e-4；余弦衰减、EMA、梯度裁剪、BF16/受保护 FP16。60/20 是可复现起始配方，尚非实验选出的最佳轮数。每个阶段的 best.pt 来自其验证表现，而非最后一轮。

若有**当前完整划分一致、训练来源可核对**的历史 MCN 权重，可省去重训基准：

```bash
python scripts/run_final.py --data-root /data1/AIC/data --run-dir checkpoints/ultimate_from_verified_baseline --mode train --device cuda:1 --batch-size 4 --baseline-checkpoint /absolute/path/to/verified/best.pt
```

这不是当前 `mcn_v1_20ep` 可直接接入的承诺。程序会检查 marker 顺序、架构、划分和训练样本；原始暖启动链的真实性仍需用对应归档核实。路径名不是证据。

本主线不自动对全部 6296 个标签继续微调，因为换了权重就不能继续套用旧部署选择和独立报告。低层 trainer 的 full-data 功能保留供另立实验，但不能把其训练集指标作为独立性能，也不能绕过本主线权重绑定。

## 产物和已执行检查

每个 run 包含 `recipe.json`、两个阶段的配置/划分/源码快照/完整 last.pt/EMA best.pt/逐轮指标、验证审计用 `deployment.json`、唯一测试模型 `final.pt` 和一次性的 `holdout_locked.json`。测试入口只读取 `final.pt`。预测目录有 `provenance.json`，ZIP 只包含 `results/test/<marker>/<原文件名>_fake.jpg`。1346 个测试输入对应四 marker 共 5384 张 JPEG。

- 24 项回归检查通过，覆盖：强制 seed=2026、单 final checkpoint、冻结推理、TTA 精确逆变换与等权平均、无预测后处理、初始恒等映射、冻结基准、私有分支梯度、修正幅度、指标选择、拒绝泄漏、训练/恢复/独立检查/推理和完整打包。
- seed=2026 真实数据 smoke 已完成：32 个 Train、8 个 Val，基准和精修各 2 轮、各 32 次优化更新，零溢出；它只证明复赛流程可执行，不是性能证据，不能提交。
- 默认宽度 GPU 检查：RTX 3060 Laptop，真实图像 batch=4，三次 AdamW 更新，梯度有限，峰值已分配显存约 3894 MiB；不等于 5090 吞吐或完整训练时间。
- 目前缺少：服务器历史权重/归档复核、完整训练、锁定后的正式独立检查、平台成绩。不能把以上结构检查或小规模指标当作这些结果。

机器可读本次记录见 `docs/experiments/ultimate_ihc_audit_20260920.json`。此前 FM 修复见 `FINAL_IHC.md`，与本次 MCN 历史分数归因分开。
