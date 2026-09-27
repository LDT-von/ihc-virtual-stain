# ALL-IN Ensemble: 75.2485 (NEW BEST)

**提交时间**：2026-09-27 晚
**成绩**：75.2485 (新历史最高 🥇)
**来源 zip**：`submissions/ensemble_allin_rgb.zip` (30.4 MB)

## 进步幅度

| 对比 | 差值 |
|------|-----:|
| vs v3 (75.1128) | **+0.1357** ✅ |
| vs w96 (75.0876) | **+0.1609** ✅ |
| vs v6 (75.0463) | **+0.2022** ✅ |
| vs 目标 (78.3845) | -3.1360 |

## 集成构成 (5 个模型)

按 `ensemble_all_in.py` 加载顺序：

1. **v3** (fullplus_cd68, 250+ epoch, marker-weighted) — 平台 75.1128 单独冠军
2. **v6** (300 epoch, synthesis noise + aug_strength) — local 之王
3. **v5** (250 epoch, base aug) — 平台 74.9364
4. **v4** (w48 refiner) — 平台 74.6531
5. **w96** (baseline, expanded) — 平台 75.0876

## 为什么 ensemble 突破 75.1 阈值？

每个模型在不同 ROI/marker 上有偏置：
- v3 在 CD45RO/HLA-DR 强
- v6 在 Vimentin/CD68 噪声更鲁棒
- w96 是 expanded split 的"最大量"基线
- v5/v4 提供多样性（不同 epoch / aug）

5 模型平均 = **每张图像多个假设**，消除单模型在某一 marker 上的过拟合。

## 经验

- **集成在最后 0.1 分上有奇效** — 单模型到 ensemble 平均可以 +0.1~0.2
- **多样性 > 个体性能** — v5 单独只有 74.9364，但加入 ensemble 后帮 v6/v3 平滑了边界
- **CD68 提升最明显** — 5 模型对 CD68 marker 的稀疏信号平均掉了离群预测

## 下一步

1. **不动** — 75.2485 是阶段性收官成绩
2. **如果还有提交机会**：尝试 **加权 ensemble**（per-marker 权重）— 比如 CD68 给 v6+v3 高权重
3. **方向**：把 ALL-IN 的推理脚本固化为 `ensemble_all_in.py` 默认入口

## 文件

- `ensemble_all_in.py` — 5 模型推理 + 像素平均
- `submissions/ensemble_allin_rgb.zip` — 提交包（30.4 MB）

