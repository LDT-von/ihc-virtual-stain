# IHC 复赛 Mark2 - 总结提示词（新窗口使用）

> 这是给新对话窗口的完整上下文，请仔细阅读后再开始工作。

## 1. 项目背景

**比赛**：复赛 IHC 图像染色预测（marker colorization）
**目标**：根据 DAPI（核染色）灰度图，预测同一 ROI 的 4 种 marker 染色：
- CD68（巨噬细胞，**稀疏标记**）
- CD45RO（T 细胞）
- HLA-DR
- Vimentin（基质细胞）

**输入**：`DAPI.png`（1-channel 灰度，[0,1]）
**输出**：4 个 marker 各自的灰度图（每张 PNG）
**评估**：自动评分（按 marker 分别算 SSIM 和 PSNR 后综合为单一分数）

## 2. 关键文件位置

- 训练代码：`E:\aic\final-ihc\src\train_marker_context.py`
- 模型定义：`E:\aic\final-ihc\src\models\marker_context.py`（核心：MarkerContextNet）
- 数据/ROI 管理：`E:\aic\final-ihc\src\data\roi_manifest.py`
- 推理/打包：`E:\aic\final-ihc\src\infer.py` + `E:\aic\final-ihc\src\package_submission.py`
- 配置文件：`E:\aic\final-ihc\configs\roi_split_semifinal_2026_2380.json`（**2380 ROIs，3 个 holdout ROI: ROI045/052/078**）
- 数据根目录：`E:\aic\复赛数据集(包括训练集和测试集输入)\`
- 平台提交记录：`E:\aic\final-ihc\docs\experiments\platform_scores.json`
- 实验总结：`E:\aic\final-ihc\docs\experiments\`

## 3. 模型架构（一对多 One-to-Many）

**重要**：用户决定保持 **一对多（单模型）架构**。

```
DAPI (1ch) → Shared Backbone (width=96, 4 尺度) → 4 个独立输出 head → 4 marker
```

- 共享 backbone：4 尺度 encoder-decoder + GatedSpatialBlock + PooledContext
- 每个 marker 独立 3×3+1×1 head（head[-1].bias = -2.0 初始化）
- 输出 sigmoid 到 [0,1]
- 关键文件 `marker_context.py`：MarkerContextNet.forward

**为什么重要**：4 个 marker 共享一个 backbone，容量互相竞争。CD45RO 表现很好可能"压制"了 CD68 等稀疏 marker 的学习。

## 4. 关键反推信息（最重要）

### 平台评分公式（反推）
```
score ≈ avg_ssim × 100 + 0.128 × avg_psnr − 12.85
```
- 每提升 0.01 SSIM ≈ **+1.086 分**
- 每提升 1.0 PSNR ≈ **+1.28 分**

### 已知的提交分数（per-marker 拆解）

| 提交 | 总分 | CD68 SSIM/PSNR | CD45RO SSIM/PSNR | HLA-DR SSIM/PSNR | Vimentin SSIM/PSNR |
|------|------|----------------|------------------|------------------|---------------------|
| 74.6531（w48_refiner） | 74.6531 | 0.769 / 27.902 | 0.891 / 26.375 | 0.845 / 25.101 | 0.861 / 25.613 |
| **75.0876（w96_expanded+TTA=4）** | 75.0876 | **0.773 / 28.100** | **0.894 / 26.533** | **0.850 / 25.484** | **0.865 / 25.958** |
| **目标** | **78.3845** | **0.893 / 26.499** | 0.893 / 26.499 | 0.893 / 26.499 | 0.893 / 26.499 |

### Per-marker 差距分析（最大缺口）

| Marker | 当前 SSIM | 目标 SSIM | SSIM 差距 | 当前 PSNR | 目标 PSNR | PSNR 差距 |
|--------|-----------|-----------|-----------|-----------|-----------|-----------|
| **CD68** | 0.773 | 0.893 | **+0.120** 🔴 最大 | 28.100 | 26.499 | **-1.601** ✅ 已超 |
| **CD45RO** | 0.894 | 0.893 | **-0.001** ✅ 已达 | 26.533 | 26.499 | -0.034 ✅ 已达 |
| HLA-DR | 0.850 | 0.893 | +0.043 | 25.484 | 26.499 | +1.015 ⚠️ 低 |
| Vimentin | 0.865 | 0.893 | +0.028 | 25.958 | 26.499 | +0.541 |

### 关键洞察（必读）

1. **CD45RO 已经达标**（0.894 > 0.893）→ **不要浪费资源去优化 CD45RO**
2. **CD68 是最大短板**：SSIM 差 0.120，PSNR 已经超目标（说明模型 CD68 输出"像素值准确但结构错位"）
   - CD68 是稀疏标记（巨噬细胞稀疏分布）
   - 当前 backbone 学到的共享特征对 CD68 不友好
3. **HLA-DR/Vimentin**：PSNR 偏低（-0.5 到 -1），说明整体亮度/对比度不匹配 → 需要后处理或 Refiner
4. **CD68 的特殊性**：稀疏染色 → 背景大、信号少 → 像素误差小（PSNR高），但整体相似度低（SSIM低）

## 5. 已做的尝试 & 结果（必读，避免重复）

### 已训模型清单
| 检查点 | Holdout SSIM (TTA=8) | 平台分数 | 备注 |
|--------|----------------------|----------|------|
| `ultimate_w48_2380/refiner_v2/best.pt` | 0.8157 | 74.6531 | w=48 + refiner |
| `ultimate_w96_2380/baseline/best.pt` | 0.8193 | - | w=96 + 无 refiner |
| `ultimate_w96_2380/refiner/best.pt` | 0.8192 | - | w=96 + refiner |
| **`ultimate_w96_expanded/baseline/best.pt`** | **0.8304** | **75.0876** | w=96 + expanded(2800) |

### Holdout SSIM 局部（w96_expanded + TTA=8）：0.8304

### 已尝试 / 已确认有效
- ✅ **Width=96 优于 Width=48**（+0.014 local SSIM）
- ✅ **expanded 数据集（2800）优于 2380**（+0.011 local SSIM）
- ✅ **TTA=8 略优于 TTA=4**（+0.0004）
- ✅ **Refiner 模型无明显提升**（可能因 backbone 已足够强）

### 已尝试 / 无效或暂未采纳
- ❌ Refiner 提升微小（<0.0005）
- ⚠️ 之前的 w48/w96 训练：受 holdout 选择影响大（ROI078 几乎全是 Vimentin 主导）

## 6. 平台提交限制

- 每天限制提交次数（具体次数未知，建议保守）
- 提交格式：zip，包含每张测试 ROI 的 `{ROI_id}_CD68.png`, `{ROI_id}_CD45RO.png` 等
- 输出 PNG 是 [0,255] 灰度图

## 7. 设备 & 环境

- **GPU**：RTX 3090（24GB VRAM）
- Python 环境：`D:\Anaconda3\python.exe`
- 工作目录：`E:\aic\final-ihc\`
- 数据集根：`E:\aic\复赛数据集(包括训练集和测试集输入)\`
- 操作系统：Windows 10（PowerShell）
- 训练单 epoch 时间（w96 expanded，约 2800 张）：~5-10 分钟（需实测）

## 8. 需要做的事情（按优先级）

### P0：CD68 专项突破（预计 +1~3 分）
CD68 是最大短板，且 SSIM 差 0.120。可能方案：
- **方案 A**：在当前一对多模型中加 CD68-weighted loss（给 CD68 head 更高 SSIM 权重）
- **方案 B**：CD68-specific 数据增强（稀疏标注需要更多 augmentation）
- **方案 C**：CD68 独立辅助分支（与 4 个 head 并行的 CD68 专用 head）
- **方案 D**：检查 CD68 标注质量 → 是否有大量"漏标"或"噪声"

### P1：HLA-DR/Vimentin PSNR 提升（+0.5~1.5 分）
- 后处理：颜色归一化、直方图匹配、ROI 强度归一化
- 检查 refiner 是否对这两个 marker 有帮助
- 输出强度范围是否被截断

### P2：Ensemble / TTA 优化（+0.3~0.8 分）
- 多模型 ensemble
- TTA=8 + 多角度翻转

### P3：扩大训练（更大性能空间）
- 用全量 2800+ 数据训练（已有 ultimate_w96_expanded 是用 2800 训的）

## 9. 用户偏好 / 注意事项

1. **用户希望保持一对多架构**（不要拆成单 marker 模型）
2. 用户在平台有提交次数限制 → **不要轻易重训/重提交**
3. 改动前**必须先验证本地 holdout SSIM 提升**再考虑提交
4. **不要删除原始数据**（已做过一次：初赛 1346 张 DAPI 已重命名为 `DAPI-BACKUP-OLD-1346files`）
5. **不要删除原始 checkpoint**（所有模型都已保留）
6. 每次实验后立即记录到 `docs/experiments/`
7. 用户关注**实际分数提升**，不接受"理论上可能更好"
8. 用户希望**快速试错**，不要陷在单个实验里
9. **不要修改已签入的稳定文件**，新尝试放到独立脚本或新分支

## 10. 推荐的立即行动顺序

1. **先看 CD68 训练数据**：是不是真的稀疏？是否有标注错误？
   - 路径：`E:\aic\复赛数据集(包括训练集和测试集输入)\train\CD68\`
2. **检查当前 w96_expanded 在 holdout 上 CD68 的局部表现**
3. **设计 CD68-weighted loss**：修改 `reconstruction_loss` 让 CD68 head 权重更高
4. **小规模试验**：先在 holdout 3 个 ROI 上验证
5. **再扩到全量训练 + TTA 推理**
6. **新提交前先在 holdout 上预估分数**

## 11. 已有的辅助脚本（可参考）

- `save_scores.py`：保存平台分数
- `score_formula.py`：反推分数公式
- `check_expanded.py`：查看 checkpoint 训练历史
- `ensemble_search2.py`：模型 ensemble 搜索（正在跑）

## 12. 关键命令示例

```bash
# 训练（单 GPU）
cd E:\aic\final-ihc
D:/Anaconda3/python.exe -m src.train_marker_context --config ... --epochs ...

# 推理 & 打包
D:/Anaconda3/python.exe -m src.infer --checkpoint ... --output ...

# 评估本地 holdout
D:/Anaconda3/python.exe -m src.eval --checkpoint ... --manifest configs/roi_split_semifinal_2026_2380.json
```

---

**最重要的两条**：
1. 用户决定保持**一对多架构**
2. **CD68 是最大短板**（SSIM 差 0.120）
