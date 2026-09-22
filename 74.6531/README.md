# AIC 2026 复赛提交快照 — 平台分 **74.6531**

本目录是**唯一取得 74.6531 的那一版模型**的完整、自包含快照：结果文件 + 该模型用到的全部代码 + 全部训练/验证参数。

| 项目 | 值 |
|---|---|
| 平台提交号 | **AIC-2026-51704013** |
| 提交时间 | 2026-09-22 10:29:01 |
| 出分时间 | 2026-09-22 10:31:29 |
| 平台分 | **74.6531** |
| 参赛者 | 小百川 |
| 模型 | 冻结基准（width=48）+ 四标记残差精修（width=24）的固定级联 |
| 部署形态 | 单文件 `final.pt`，`marker_mask = [false, true, false, true]` |

> 74.6531 是**平台自己的评分口径**，与本目录里的本地 SSIM/PSNR **不是同一个量纲**，不能互相换算。

---

## 一、目录结构

```
74.6531/
├── README.md                        本文件：全部参数与复现说明
├── code/                            该模型用到的全部代码（不多不少）
│   ├── requirements.txt
│   ├── configs/
│   │   └── roi_split_semifinal_2026_2380.json      本次使用的固定划分清单
│   ├── scripts/
│   │   ├── run_final.py            正式入口（等价于 run_ultimate）
│   │   ├── run_ultimate.py         三段式驱动：train / evaluate / predict
│   │   └── package_final.py        校验并打包 submission.zip
│   ├── src/
│   │   ├── train_marker_context.py 训练、评测、推理、JPEG 序列化的核心
│   │   ├── select_ultimate.py      验证门控选择 + 导出唯一 final.pt
│   │   ├── evaluate_ultimate.py    holdout 锁定后的一次性对比
│   │   ├── data/roi_manifest.py    ROI 划分、清单校验、MARKERS 定义
│   │   ├── models/
│   │   │   ├── marker_context.py   MarkerContextNet（基准网络）
│   │   │   ├── marker_specific.py  MarkerSpecificNet
│   │   │   └── anchored_ihc.py     AnchoredIHC（冻结基准 + 残差精修级联）
│   │   └── metrics/ssim_psnr.py    局部 SSIM / PSNR 参考实现
│   └── tests/
│       ├── test_ultimate_ihc.py    级联、门控、导出、打包的端到端测试
│       └── test_marker_context.py  与 skimage 交叉核对局部 SSIM
├── model/                           模型与训练/验证审计记录
│   ├── final.pt                     ★ 部署用唯一权重（22,088 KB）
│   ├── recipe.json                  本次训练的完整命令行配方
│   ├── deployment.json              门控决策报告（含 val 逐 marker 明细）
│   ├── holdout_locked.json          锁定后 holdout 一次性对比
│   ├── baseline/                    width=48 基准阶段的记录
│   ├── refiner/                     width=24 残差阶段的记录
│   └── sources/                     训练时 5 个源文件的逐字节快照
└── submission/                      本次提交的产物
    ├── submission.zip               ★ 上传平台的压缩包（29,689 KB，2240 张）
    └── provenance.json              该包的生成溯源（含各项 SHA256）
```

`code/` 只包含**跑这个模型真正需要**的文件。原仓库 `src/models/` 下还有 Pix2Pix、CUT、FlowMatching、StableDiffusion、VSMT 等初赛阶段的模型，与本模型无关，未收录。

---

## 二、模型结构：两阶段固定级联

### 阶段 1 — 冻结基准 `B(x)`
`MarkerContextNet`：共享多尺度形态编码器 + 低分辨率上下文注意力 + **四个 marker 各自独立的完整解码器**。
输入只有 DAPI（单通道，原强度 `/255`），一次前向同时输出 4 个 marker。

### 阶段 2 — 残差精修 `R_m(x, B(x))`
`AnchoredIHC`：把 `[x, B(x)]`（5 通道）送入精修网络，四个输出头权重与偏置**从零初始化**，
因此训练起点严格等于基准。单像素修正幅度被 `tanh` 限制在 ±0.2：

```
candidate_m(x) = clamp( B_m(x) + 0.2 * tanh( R_m(x, B(x)) ), 0, 1 )
```

基准分支不接收梯度、不参与 EMA 更新。

### 部署时到底用了哪一个？

`marker_mask = [false, true, false, true]` —— **不是纯基准，而是逐 marker 混用**：

| 序号 | Marker | 实际输出 | 含义 |
|---:|---|---|---|
| 0 | HLA-DR | **基准** | 精修未通过门控 |
| 1 | CD68 | **精修** | 精修通过门控 |
| 2 | CD45RO | **基准** | 精修未通过门控 |
| 3 | Vimentin | **精修** | 精修通过门控 |

该选择由 [select_ultimate.py](code/src/select_ultimate.py) 的 `choose_markers()` 在**验证集**上决定，
规则写死为：`refined.ssim > base.ssim and refined.psnr >= base.psnr`，平局保留基准。
决策结果被持久化为 buffer 写进 `final.pt`，推理时校验，测试阶段不可再改。

---

## 三、全部超参数

### 3.1 顶层配方（`model/recipe.json`）

| 参数 | 值 |
|---|---|
| `mode` | `train` |
| `seed` | **2026**（复赛强制，CLI 只接受这一个值） |
| `baseline_width` | **48** |
| `width`（精修） | **24** |
| `baseline_epochs` | **60** |
| `refine_epochs` | **20** |
| `batch_size` | 4 |
| `tta` | **4** |
| `device` | `cuda` |
| `baseline_checkpoint` | `null`（基准从零训练，不做暖启） |
| `no_cache` | `false` |
| `split_sha256` | `c050a1378ffa4027d628a24502e5431174a390050d4b6fb9f23f469ab8726444` |

### 3.2 阶段 1：基准（`model/baseline/run.json`）

| 参数 | 值 |
|---|---|
| `architecture` | `context` |
| `width` | 48 |
| `markers` | 4 |
| `context` | `true` |
| `epochs` | 60 |
| `lr` | **5e-4** |
| 参数量 | **3,828,677** |
| 训练样本 | 1540（`train_names`） |
| 选模依据 | 验证集平均 SSIM 最大值 |

### 3.3 阶段 2：残差精修（`model/refiner/run.json`）

| 参数 | 值 |
|---|---|
| `architecture` | `anchored` |
| `width` | 24 |
| `baseline_config` | `{width: 48, markers: 4, context: true}` |
| `residual_limit` | **0.2** |
| `epochs` | 20 |
| `lr` | **1e-4** |
| `baseline_checkpoint` | `.../ultimate_w48_2380/baseline/best.pt` |
| 级联总参数量 | **5,568,154** |
| 其中可训练 | **1,739,477**（基准部分 3,828,677 被冻结） |

### 3.4 损失与增强

监督目标（四个 marker 等权）：

```
1 * (1 - local_SSIM) + 0.5 * L1 + 2 * MSE + 0.1 * L1(pool2) + 0.1 * L1(pool4)
```

- 局部 SSIM：**7×7 有效窗口 + 样本协方差**，已与 `skimage` 数值交叉核对
- 增强：仅**同步**的翻转与直角旋转（DAPI 与 4 个 marker 同一数组一起变换），不改变标记强度分布

### 3.5 运行环境（训练时记录）

| 项 | 值 |
|---|---|
| Python | 3.13.5（Anaconda） |
| PyTorch | 2.6.0+cu124 |
| 操作系统 | Windows 10.0.19045 |
| GPU | NVIDIA GeForce RTX 3090 |
| 训练时 git HEAD | `6d31b9f8a77a536384da255375767e8a0edfcbdb` |

---

## 四、数据划分（`code/configs/roi_split_semifinal_2026_2380.json`）

| 集合 | 样本数 | 用途 |
|---|---:|---|
| train | **1540** | 两个阶段的标签监督 |
| val | **420** | 每轮选 best + 逐 marker 门控决策 |
| holdout | **420** | 锁定后仅执行一次对比 |
| 合计 | 2380 | |

- `seed = 2026`，划分单位是 **ROI**，三个集合 ROI 完全互斥
- val ROI：`ROI005` / `ROI019` / `ROI076`；holdout ROI：`ROI045` / `ROI052` / `ROI078`
- 清单自身 SHA256：`54f1f6e18ae0f306854d482cdc603925a8c15aef3df8e1b3f4cc5d7ca1aca7ac`
- `group_unit` 字段明确写着 `ROI; patient independence unknown` —— **ROI 隔离不等于患者隔离**，本数据集没有患者映射

---

## 五、验证与门控决策（`model/deployment.json`）

`selection_split = val`，`tta = 4`，`seed = 2026`，`smoke_only = false`。
`validation_names_sha256 = f89620b4f8b22e4f94c5e19d7fdc644ffb52290c8142b1cc6143fabfd38dd1b1`

### 逐 marker 门控明细（val，420 张）

| Marker | 基准 SSIM | 基准 PSNR | 精修 SSIM | 精修 PSNR | SSIM↑? | PSNR 不降? | 结论 |
|---|---:|---:|---:|---:|:---:|:---:|---|
| HLA-DR | 0.8207343 | 27.97983 | 0.8218597 | 27.97568 | 是 | **否** | 保留基准 |
| CD68 | 0.7938380 | 36.90582 | 0.7942692 | 36.91659 | 是 | 是 | **启用精修** |
| CD45RO | 0.8616985 | 29.23169 | 0.8615019 | 29.14363 | **否** | 否 | 保留基准 |
| Vimentin | 0.8354730 | 27.48985 | 0.8356330 | 27.53577 | 是 | 是 | **启用精修** |

整体（val 平均）：基准 `SSIM 0.8279359 / PSNR 30.4018`，精修候选 `SSIM 0.8283159 / PSNR 30.3929`。
注意候选整体 PSNR 是**下降**的 —— 门控是逐 marker 判定，这才让 CD68 与 Vimentin 得以启用。

决策报告自身 SHA256（即 `final.pt` 里记录的 `selection_report_sha256`）：
`faf03b01d927e1d7bc8f7bafd6955545fc5a9094f057ba8af13ae0e1d43b547f`

---

## 六、本地指标

### holdout（420 张，锁定后一次性对比，`model/holdout_locked.json`）

| | SSIM | PSNR |
|---|---:|---:|
| 纯基准 | 0.8138270 | 25.36762 |
| **实际部署（混用）** | **0.8138089** | **25.36854** |

逐 marker 差值（部署 − 基准）：

| Marker | ΔSSIM | ΔPSNR |
|---|---:|---:|
| HLA-DR | 0.0 | 0.0 |
| CD68 | **−0.0001808** | **−0.0004113** |
| CD45RO | 0.0 | 0.0 |
| Vimentin | **+0.0001084** | **+0.0040894** |

**需要如实说明**：门控是在 val 上定的，搬到 holdout 后 CD68 反而微降，
导致部署整体 SSIM（0.8138089）比纯基准（0.8138270）低 0.0000181。
差异极小、在噪声量级，但这是真实记录，不修饰。按协议 holdout 不做二次调参。

### 上一版对照

同口径 width=32 的基准在 holdout 上是 `SSIM 0.810559 / PSNR 25.1978`；
提到 width=48 后本地 +0.0033 SSIM，平台分从 67.9114 提升到 **74.6531**。
这说明本地同口径 holdout 差值可用于选型。

---

## 七、输出规格（平台硬性要求）

| 项 | 值 |
|---|---|
| 尺寸 | 256 × 256 |
| 位深/通道 | **8-bit RGB（三通道完全相同）** |
| 格式 | JPEG |
| `quality` | **100** |
| `subsampling` | **0** |
| `optimize` | **false** |
| 文件命名 | `<输入名>_fake.jpg`，放在 `results/test/<marker>/` |
| 数量 | 560 输入 × 4 marker = **2240** |

`final.pt` 的 `deployment.serialization` 里写死了这套参数，推理时校验，不存在测试期可调项。

> 前一版提交曾因存成 2D 灰度（`mode=L`）被判 560 张全部无效。
> 本快照的 `code/src/train_marker_context.py` 已修复为 `np.repeat(channel[:, :, None], 3, axis=2)`，
> 因为 R=G=B，修复不改变任何数值指标，只影响平台能否接收。

### 协议约束（代码层强制，非人工自律）

- 固定种子：CLI 只接受 `seed=2026`
- 单 checkpoint：测试只认 `format=2 / kind=semifinal_final` 的 `final.pt`，普通训练权重会被拒
- 冻结测试：加载后 `eval()`，TTA 预测在 `no_grad()` 下，无优化器、无反向、无 BN 更新、无测试时适配
- TTA：固定几何变换 → 精确逆变换 → 等权平均；配置从 checkpoint 读取并校验
- 无后处理：不做阈值、二值化、形态学、去噪、平滑、锐化、亮度/对比度/Gamma/直方图匹配
- 不用测试集统计量：逐图读取 DAPI 并固定 `/255`，不读测试标签、不做动态归一化

---

## 八、如何复现

依赖：`pip install -r code/requirements.txt`。数据需要放在本机（本目录不含数据集）。

```powershell
cd E:\aic\74.6531\code
$DATA = "E:\aic\复赛数据集(包括训练集和测试集输入)"
```

### 复现提交包（推理 + 打包，只读取 `final.pt`）

```powershell
python scripts/run_final.py --data-root $DATA --run-dir ..\model --mode predict --output ..\rerun
```

`predict` 会先校验 `holdout_locked.json` 与 `final.pt` 的 SHA256 及决策 SHA256 三方一致，
再调用 `src.train_marker_context infer`（TTA=4）生成 2240 张图并自动打包 `..\rerun\submission.zip`。

### 从零重新训练（约需 60 + 20 轮，GPU）

```powershell
python scripts/run_final.py --data-root $DATA --run-dir <空的输出目录> `
  --manifest configs/roi_split_semifinal_2026_2380.json --mode train
```

`--manifest` 必须显式指定，因为脚本默认值指向的是另一个（初赛全量）划分清单，本快照未收录。

### 端到端测试

```powershell
python -m unittest discover -s tests -v
```

---

## 九、文件校验（SHA256）

| 文件 | SHA256 |
|---|---|
| `model/final.pt` | `d4ca3bb6c1b8620482b3581c2613f47798747d73134f1426287e9ffdbf7f5afb` |
| `submission/submission.zip` | `8da0a3e3a461c30a33529897e19575157bb3510c346d54cf80d353a679673953` |
| `submission/provenance.json` | `9e7a7063dbc83383e7c6a633cc6827e7ad18834e990635870633082248f4eff5` |
| `code/configs/roi_split_semifinal_2026_2380.json` | `54f1f6e18ae0f306854d482cdc603925a8c15aef3df8e1b3f4cc5d7ca1aca7ac` |

`final.pt` 内部记录的校验值（可用于交叉核对）：

| 字段 | 值 |
|---|---|
| `deployment.selection_report_sha256` | `faf03b01d927e1d7bc8f7bafd6955545fc5a9094f057ba8af13ae0e1d43b547f` |
| `deployment.source_checkpoint_sha256` | `4af369ba208b8b10e140286705959b08534cbb67bb0f1652b115db98e7504770` |
| `run.split_sha256` | `c050a1378ffa4027d628a24502e5431174a390050d4b6fb9f23f469ab8726444` |
| `submission/provenance.json → deployment_sha256` | 同上 `faf03b01…`（三方一致） |

### 代码与训练时源码的差异（重要且已核验）

`model/sources/` 是训练当时的 5 个源文件逐字节快照，其中：

| 文件 | 与训练时是否一致 |
|---|---|
| `roi_manifest.py` | 一致 |
| `marker_context.py` | 一致 |
| `marker_specific.py` | 一致 |
| `anchored_ihc.py` | 一致 |
| `train_marker_context.py` | **不一致** |

唯一差异就是第七节的 RGB 存盘修复（4 增 2 删），**训练与评测逻辑完全未动**。
因此 `code/` 复现出的是同一模型的同一批预测；由于 GPU 卷积核的非确定性，
重推理结果与提交包并非逐字节相同，差异量级见下一节。
训练时的原始源码哈希为 `769372c1e61b805bc5fb46f2fd8231788c6d8d29ef158ae913a7de691913eaf5`。

---

## 十、本快照的验证记录

以下三项都在**本快照目录内**实测过，不是推断：

### 1. 模块可导入

```powershell
cd E:\aic\74.6531\code
python -c "import src.train_marker_context, src.select_ultimate, src.evaluate_ultimate, scripts.package_final"
```

通过。`code/src/models/__init__.py` 已被精简为只含说明，不再引用初赛模型，导入不受影响。

### 2. 测试全绿

```
python -m unittest discover -s tests
Ran 15 tests in 107.746s
OK
```

### 3. 端到端重推理（最强验证）

```powershell
python scripts/run_final.py --data-root "<复赛数据集>" --run-dir ..\model `
  --manifest configs/roi_split_semifinal_2026_2380.json --mode predict --output <临时目录>
```

结果：`images_per_marker=560`、`total_images=2240`，自动打包成功。
把重推理的 2240 张与 `submission/submission.zip` 内的 2240 张逐张比对：

| 检查项 | 结果 |
|---|---|
| 文件名集合 | 完全一致（缺失 0 / 多余 0） |
| provenance 的 `checkpoint_sha256` | 一致（同一个 `final.pt`） |
| provenance 的 `input_names_sha256` | 一致（同一批输入） |
| provenance 的 `deployment_sha256` / `tta` / `serialization` / `markers` / `input_count` | 全部一致 |
| 逐像素最大灰阶差 | **≤ 3 / 255** |
| 受影响像素占比 | 均值 4.31%，最大 13.27% |
| 两套图互比 PSNR | 最小 56.64 dB，中位 **62.08 dB** |
| 两套图互比 SSIM | 最小 0.998550，中位 **0.999428** |

**结论**：同一模型、同一输入、同一 TTA、同一序列化，预测内容一致；
残余差异来自 GPU（cuDNN）卷积核的非确定性，量级为 1–3 个灰阶。
因此本快照**不能**声称"逐字节复现提交包"，但可以声称"数值等价复现"。

### 4. 关于快照中删除的一个测试

`code/tests/test_marker_context.py` 相比原仓库删除了
`test_legacy_tta_restores_coordinates_and_uses_eight_calls`。
该测试校验的是初赛阶段的 8x TTA（依赖根目录的 `inference_8x_tta` / `tta_eval`，
走 Pix2Pix 的 `model.generator(x, condition)` 接口），与本次提交的模型无关。
本模型的 4x TTA 由 `test_tta_inverse_is_exact_and_unique` 覆盖并通过。
原仓库中该测试仍完整保留。

---

## 十一、已知边界

1. **平台分与本地指标不同量纲**。74.6531 来自平台口径（官方为 `0.7*SSIM + 0.3*Normalize(PSNR)`，
   但公开文本没有给出 PSNR 归一化细节），本地只报告 SSIM 与 PSNR，**没有编造过本地"官方总分"**。
2. **ROI 隔离 ≠ 患者隔离**：划分按 ROI，没有患者映射信息，不能声称患者级独立。
3. **holdout 上 CD68 微降**，见第六节，未做任何事后修补。
4. **未收录中间阶段权重**：`baseline/best.pt`、`baseline/last.pt`、`refiner/best.pt`、`refiner/last.pt`
   共约 231 MB 不在本目录。部署推理只用 `final.pt`；若要从断点续训，需回原仓库路径
   `E:\aic\final-ihc\checkpoints\ultimate_w48_2380\`。
5. **`--mode evaluate` 不可原地重跑**：`holdout_locked.json` 已存在时程序会拒绝覆盖，
   这是刻意设计（避免反复对着 holdout 调参）。确需重跑请先将其移走。
6. **重新做门控选择会被源码哈希拦住**：`deployment.json` 记录的是修复前
   `train_marker_context.py` 的哈希，所以拿它再去跑 `select_ultimate` 会报
   "Deployment source changed"。这是预期的 —— 这份决策报告是 74.6531 这次运行的历史存档。
7. **重推理不是逐字节一致**：GPU（cuDNN）存在非确定性，同一份 `final.pt` 重复推理会得到
   灰阶差 ≤3/255 的结果（互比 PSNR 中位 62.08 dB、SSIM 中位 0.999428）。详见第十节第 3 条。
   平台分的复现应以 `submission/submission.zip` 为准，而非以重推理为准。
8. **本模型未跑过其他种子**：复赛协议只允许 `seed=2026`，因此不存在多种子方差数据，
   无法给出稳定性区间。
