# 此前阶段记录：FM 修复与独立解码器候选（2026-09-20）

当前统一入口及服务器命令以 [ULTIMATE_IHC.md](ULTIMATE_IHC.md) 为准。本文保留此前检查证据；FM 的错误不解释用户后来提供的 MCN/UNet 分数。

## 交付状态

本次从 `origin/cursor/fix-flow-matching-reverse-ode-and-add-stable-diffusion` 完整检出，基准提交为 `6e60f0194d5dda9c19a96eb0d3acaa8f22425f6d`。代码位于独立的 `codex/final-ihc` 工作分支；原 main 和已有结果保留。

交付的是可训练、验证、恢复、推理、打包的一套代码主线。**尚无本主线完整训练后的正式权重、独立检查集结果或平台分数，不能把它称为已经提分的最终获胜模型。** 小规模训练权重只验证流程，正式入口拒绝用它生成提交。

## 已确认的问题

| 问题 | 证据与处理 |
|---|---|
| Flow Matching 采样方向反了 | `44eaecc7` 将更新改成减号；而训练目标一直为 `x0-epsilon`。恢复为正向加该目标，Euler/Heun 均修正。现有 FM 权重兼容，可以先重新推理。 |
| 真实标签被扰动 | 原配对增强同时对 IHC 加颜色、模糊、噪声。改为同步翻转/直角旋转，保留真实强度。 |
| train_full 无验证选模 | 该脚本用全部训练图，仅保存固定轮次，无法用训练 loss 判断 SSIM/PSNR。它作为兼容入口保留，正式开发使用当前主线的 ROI 隔离流程。 |
| latest.pt 无法完整续训 | 原 latest 只存模型，缺少优化器；还把样本数硬编码为 6296。现在每轮原子保存优化器、实际步数及随机状态；旧权重文件不再伪装为完整恢复。 |
| FM 提交命名错误 | 原 infer_fm_test 保存原名，没有 `_fake`，目录也缺 `results/test/`。修正为官方层级并核对 marker。 |
| FM 参数未生效 | `--num-steps` 被 checkpoint 默认覆盖。现在明确传入采样器，同时记录 solver/seed。 |
| 旧 SD 不能直接等同 Stable Diffusion | 实际为自写像素扩散网络，无预训练 VAE；DAPI 经全局池化变成一个向量，缺少逐位置条件跳接。这是结构上的任务匹配风险，未做消融量化。 |

### 采样方向的可复核推导

训练：`x(t)=(1-t)*x0+t*epsilon`，网络学的是 `v=x0-epsilon`。
因此 `dx/dt=epsilon-x0=-v`。从 `t` 到 `t-h`，应更新 `x <- x+h*v`。

解析回归例：`x0=0, epsilon=1, v=-1`。四步后原 Euler/Heun 都得到 **2**，修正后得到 **0**。这证明确定的程序错误，不依赖是否训练充分。`independent` 路径额外乘 `1-sigma_min`，保持其路径端点定义。

## 此前候选与当前主线的关系

此前实现了 `MarkerSpecificNet`：共享形态编码器、四个完整独立解码器。它只完成了结构和小规模流程检查，没有完整性能结果。该网络现在作为 IHC Ultimate v1 的残差精修器，外部保留冻结基准，并以验证数据决定每个标记是否启用精修。

训练、恢复、独立检查及正式推理命令统一见 [ULTIMATE_IHC.md](ULTIMATE_IHC.md)。旧独立实验入口 `scripts/run_marker_context.py` 仍保留，但不等于当前 `scripts/run_final.py` 的两阶段方案，不应混用两者的参数或部署文件。

## 优先复用已有 FM 权重

无需重训即可检验采样修复：

```bash
python scripts/infer_fm_test.py --data-root "DATA_ROOT" --ckpt "YOUR_FM_CHECKPOINT.pt" --marker CD68 --output-dir predictions/fm_fixed --num-steps 50 --solver heun --seed 42
python scripts/package_final.py --input "DATA_ROOT/test/DAPI" --prediction-dir predictions/fm_fixed --output submissions/fm_fixed_CD68.zip --markers CD68
```

将 marker 改为权重实际训练的标记。已有完整训练权重目前不在本机/远程源码中，故本次不能测得修复后的真实旧模型分数。更换 sampler 不需要改变训练目标或权重；纠正标签增强则需要重新训练或显式微调。

## 验证证据与边界

- 自动回归覆盖两种求解器的解析端点、训练目标方向、非法参数、标签强度保持、各 marker 私有分支梯度、奇数尺寸、完整训练/恢复/JPEG 评测/推理/打包，以及 FM 命令行参数与最新 checkpoint 恢复。
- 原始数据从 `08dd1387` 的父提交恢复到本地独立目录：6296 对 DAPI/四 marker、1346 个测试输入；文件名清单与固定 split 匹配。没有把数据重新加入 Git。
- 真实 GPU 子集检查：width=8，128 训练、64 验证，3 轮、192 个优化步，零溢出；仅用于确认可以学习和运行，**不足以证明提分，也不能提交**。公开 launcher 另做了两轮真实数据 smoke。
- 没有使用隐藏测试标签，没有自动向平台提交，没有新模型官方分数。完整训练和与低分权重的重新推理仍是性能验收所必需的步骤。

机器可读检查记录：`docs/experiments/final_ihc_audit_20260920.json`。
