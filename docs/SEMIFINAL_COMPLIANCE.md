# 复赛合规清单

当前唯一正式入口为 `scripts/run_final.py`，固定随机种子 2026，默认划分为 `configs/roi_split_semifinal_2026.json`。

| 规则 | 实现与约束 |
|---|---|
| 固定种子与划分 | CLI 只接受 seed=2026；Train/Val/Holdout 按 ROI 互斥，清单带 SHA256。旧 seed=42 权重禁止作为复赛模型。 |
| 官方数据 | 正式入口只读取指定 `data-root` 的配对 DAPI/四 Marker；不下载或加载外部训练数据。 |
| 配对增强 | DAPI 与四 Marker 组成同一数组后同步旋转/翻转；不改变 Marker 阳性区或强度分布。 |
| 单模型、单 checkpoint | 训练阶段的基准与残差器组成一个固定级联结构；验证选择作为持久 buffer 写入唯一 `final.pt`。测试入口拒绝普通训练 checkpoint。 |
| 冻结测试 | 加载后 `eval()`，TTA 预测处于 `no_grad()`；没有优化器、反向传播、BN 更新、自监督或测试适配。 |
| TTA | 固定几何变换，逐一精确逆变换，同一 `final.pt` 的结果等权平均。TTA 配置从 checkpoint 读取并校验。 |
| 无后处理 | TTA 均值直接转换为灰度 uint8 JPEG；没有阈值、二值化、形态学、去噪、平滑、锐化、亮度、对比度、Gamma、直方图匹配或 CLAHE。 |
| 固定序列化 | 输出固定为 256×256 灰度 JPEG，quality=100、subsampling=0、optimize=false；没有测试时可调压缩参数。 |
| 无 Test GT/统计 | Test 只逐图读取 DAPI 并固定除以 255；不使用 Test Marker、整体统计量、动态归一化或样本筛选。 |
| 决策来源 | epoch、checkpoint、marker gate、TTA 均由 Train/Val 确定并写入审计记录；Holdout 只执行一次锁定检查，Test 只作最终预测。 |

训练产物 `baseline/`、`refiner/` 和 `deployment.json` 是开发证据。复赛测试只使用 `final.pt`；预测前还要求与其 SHA256 匹配的锁定 Holdout 报告。
