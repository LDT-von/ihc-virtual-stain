# 提高分数的具体方案（基于当前 75.0876 状态）

> 生成时间：2026-09-23  
> 基于：CONTEXT_SUMMARY.md + 当前代码分析

---

## 🎯 核心结论

**不要重头训练！** 当前 `ultimate_w96_expanded/baseline/best.pt` 是基线（75.0876）。

按"投入产出比"排序，**P0 应该是无需重训的方案**：

| 方案 | 预计提分 | 重训? | 风险 | 推荐度 |
|------|---------|-------|------|-------|
| **Ensemble (w48+w96+w96_exp)** | +0.5~1.0 | ❌ 不需要 | 低 | ⭐⭐⭐⭐⭐ |
| **TTA=8 + 翻转增强** | +0.1~0.3 | ❌ 不需要 | 低 | ⭐⭐⭐⭐⭐ |
| **ROI 分组推理** | +0.2~0.5 | ❌ 不需要 | 低 | ⭐⭐⭐⭐ |
| **后处理 (强度归一化)** | +0.3~0.6 | ❌ 不需要 | 低 | ⭐⭐⭐⭐ |
| **CD68-weighted loss 重训** | +1~3 | ✅ 1~2 小时 | 中 | ⭐⭐⭐ |
| **CD68-specific head 加深** | +0.5~1 | ✅ 1~2 小时 | 中 | ⭐⭐ |
| **全量 2800 重训** | +0.2~0.5 | ✅ 2~3 小时 | 中 | ⭐⭐ |
| **W128 / W160** | 未知 | ✅ 3+ 小时 | 高 | ⭐ |

---

## 🚀 立即可做（无需训练，P0 必做）

### 1. Ensemble 多 checkpoint
你已经有 3 个 w96/w48 checkpoint，做 averaging：

```python
# 简单做法：每个 marker channel 独立平均
pred_cd68 = (model_w48(dapi)[...,1] + model_w96(dapi)[...,1] + model_w96_exp(dapi)[...,1]) / 3
```

**已知可以提分**（不同模型多样性 = 减少方差）

### 2. TTA=8 + 翻转 + 旋转增强
当前已是 TTA=8，但 **ensemble + TTA** 还可以叠加。

### 3. 后处理强度归一化（HLA-DR/Vimentin 提分关键）
观察 `platform_scores.json`：
- HLA-DR PSNR 25.484（目标 26.499，低 1.0）
- Vimentin PSNR 25.958（目标 26.499，低 0.5）
- CD45RO PSNR 26.533（已超）

**后处理方案**：每个 marker 用 train 数据集的全局 mean/std 重新归一化预测值。

```python
# 预测值是 [0,1] 的浮点
# 训练集每个 marker 的真实均值/标准差已知
# 把预测值线性变换到训练集分布
pred_calibrated = (pred - pred.mean()) / pred.std() * target_std + target_mean
pred_calibrated = pred_calibrated.clip(0, 1)
```

### 4. ROI-level SSIM 聚合（用于验证）
平台评估可能是按 ROI 聚合而不是按 patch。先确认：
- 看 `infer.py` 输出格式
- 在本地 holdout 验证 ROI-level 聚合 vs patch-level 聚合的差异

---

## 🔬 数据分析（应该先做，避免盲目训练）

### 5. 检查 CD68 标注质量
**关键问题**：CD68 PSNR 已经超目标 (28.1 > 26.5)，但 SSIM 严重不足 (0.773 vs 0.893)。
→ 说明：模型能预测"对的亮度"，但**位置错了**。

**需要排查**：
- CD68 标注是稀疏的（巨噬细胞少）→ 大量 patch 是纯背景
- 平台评估可能对背景区域使用"零假设"，但模型预测的背景可能有微小噪声
- 检查 CD68 真实标注中**空 patch 的比例**

```bash
# 检查 CD68 数据中"几乎全黑"的 patch 比例
python check_cd68_sparsity.py
```

如果 >70% CD68 patch 是"几乎全黑"，那 0.77 SSIM 实际上就是上限了。

### 6. 检查 holdout ROI045/052/078 的 CD68 强度分布
```
如果 holdout 的 CD68 平均亮度 > 训练集 → 模型欠预测
如果 holdout 的 CD68 平均亮度 < 训练集 → 模型过预测
```

---

## 🛠 中等优先级（需要训练，但风险低）

### 7. CD68-weighted Loss 微调
**目标**：保持 CD45RO/Vimentin 性能，**单独提升 CD68**。

**做法**：
- 从 `ultimate_w96_expanded/best.pt` 加载
- 改 `reconstruction_loss`：对 CD68 channel 加权 ×2
- 用更小 LR (5e-5 vs 5e-4) 微调 10~15 epoch
- 预期：CD68 SSIM +0.02~0.05，CD45RO/Vimentin -0.005~0.01

**命令模板**：
```bash
D:/Anaconda3/python.exe -m src.train_marker_context \
  --config configs/roi_split_semifinal_2026_expanded.json \
  --output checkpoints/cd68_finetune_v1 \
  --width 96 \
  --epochs 15 \
  --lr 0.00005 \
  --batch 4 \
  --init-checkpoint checkpoints/ultimate_w96_expanded/baseline/best.pt
```

⚠️ 但这需要修改 `reconstruction_loss` 支持 per-marker weight。详见代码改造。

### 8. CD68-specific 辅助分支
在 `MarkerContextNet` 末尾加一个轻量 "CD68 refiner head"（5x5 conv + sigmoid），只对 CD68 做。

⚠️ 改动较多，需谨慎测试。

---

## 🎯 推荐立即行动顺序

### 今天（先吃免费的提分）：
1. ✅ **实现 3-model ensemble**（w48 + w96 + w96_expanded）→ 预计 +0.3~0.8
2. ✅ **实现后处理强度归一化**（针对 HLA-DR/Vimentin）→ 预计 +0.2~0.5
3. ✅ **检查 CD68 标注稀疏度**（决定是否值得重训）

### 明天（如果免费方案提分 < 1.0）：
4. ⚙️ **CD68-weighted loss 微调**（10~15 epoch）→ 预计 +0.5~1.5
5. ⚙️ **验证 ensemble + 微调的叠加效果**

### 后天（如果还不够）：
6. 🔬 **W=128 重训**（更大容量）→ 未知，可能 +0.3~1.0
7. 🔬 **W=96 + 2800 全量重训** → +0.2~0.5

---

## ⚠️ 重要提醒

1. **平台有提交次数限制** → 每个改动**先验证本地 holdout SSIM** 再提交
2. **Ensemble 已经在跑**（`ensemble_search2.py`）→ 等结果再决定下一步
3. **不要删除数据/模型** → 所有尝试放独立目录
4. **每个实验立即记录**到 `docs/experiments/`

---

## 📌 下一步

请告诉我你想优先做哪个：
- **A**：先实施 Ensemble（最稳，免费）
- **B**：先检查 CD68 标注稀疏度（决定后续方向）
- **C**：直接做 CD68-weighted loss 微调（最可能提分）
- **D**：其他
