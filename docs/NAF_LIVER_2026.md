# 复赛 Liver：MarkerNAFNet 候选与运行交接

## 证据边界

这是尚未在真实 Liver 数据上训练与验证的候选模型，没有复赛平台成绩，也不能保证提分或排名。当前工作区 `data/recovered_official/` 只核实到初赛 Colon 数据：训练 6296 组、测试 DAPI 1346 张；尚未找到 Liver 复赛数据。不要把初赛 ROI 划分或初赛验证分数当作 Liver 结果。

[官方赛题](https://www.aicomp.cn/tracks/tracks-1/3759.html)规定复赛为 Liver，DAPI 输入、指定 IHC 标记输出，评分结合 SSIM 与 PSNR；PSNR 归一化细节未公开。赛题允许至少完成一个指定标记，多输出成绩取平均。提交前须核对当期指定目标和目录要求。[复赛通知](https://www.aicomp.cn/notice/notice-1/5278.html)称最终以阶段排行榜最高分为准。

## 方法

`src/models/marker_nafnet.py` 是受 [NAFNet](https://github.com/megvii-research/NAFNet) 启发、自行编写的轻量四级编码解码网络。模型只读 DAPI，利用门控块和跳连提取组织结构，以独立标记头输出 IHC；没有将 DAPI 像素直接加到输出。可预测一个标记或同时预测四个标记，不使用外部数据和预训练权重。

训练复用 `train_semifinal_v7.py`：ROI 互斥折分、同步几何增强、不改标签强度、局部 SSIM/L1/MSE 损失、EMA、JPEG 往返后评估。以 OOF 选定轮数，再用全部官方训练标签重训。推理仅输出 checkpoint 中的标记。所有比较须使用同一 Liver 数据、划分、TTA、JPEG 与选模规则。

## 数据布局

将**官方 Liver 复赛数据**放在以下结构，同一文件名须在通道间配对：

```text
LIVER_DATA_ROOT/
  train/DAPI/*.jpg
  train/HLA-DR/*.jpg
  train/CD68/*.jpg
  train/CD45RO/*.jpg
  train/Vimentin/*.jpg
  test/DAPI/*.jpg
```

当前清单生成器会核对四种训练标签和 `ROI数字_行_列.jpg` 命名。若实际 Liver 包缺少某个标签或命名不同，先据实调整读取器和 ROI 规则，再生成清单；不要复制 Colon 清单。

## 单标记运行示例（PowerShell）

将 `$DataRoot` 改为真实 Liver 路径，`$Marker` 改为该阶段指定标记。以下命令供使用者运行；本次未运行。训练期间不要修改源码。每折与最终重训必须使用完全相同的参数。

```powershell
$DataRoot = 'E:\aic\LIVER_DATA_ROOT'
$Marker = 'HLA-DR'
$Manifest = 'configs/roi_split_liver_2026.json'
$RunRoot = 'checkpoints/liver_naf_hladr_v1'
python -m src.train_marker_context split --data-root $DataRoot --output $Manifest --val-rois 2 --holdout-rois 2

$Common = @('--data-root', $DataRoot, '--manifest', $Manifest, '--architecture', 'marker_nafnet', '--target-marker', $Marker, '--width', '32', '--folds', '5', '--epochs', '120', '--eval-every', '10', '--batch-size', '4', '--lr', '0.0003', '--warmup-epochs', '5', '--weaken-start-epoch', '80', '--augmentation', 'geometry', '--loss', 'normalized', '--cd68-weight', '1', '--tta', '4', '--no-cache')
foreach ($Fold in 0..4) {
    python train_semifinal_v7.py fit --stage dev --fold $Fold --output "$RunRoot/fold_$Fold" @Common
}

$FoldRuns = 0..4 | ForEach-Object { "$RunRoot/fold_$_" }
python train_semifinal_v7.py summarize --runs $FoldRuns --output "$RunRoot/selection.json"
python train_semifinal_v7.py fit --stage final --selection "$RunRoot/selection.json" --output "$RunRoot/final" @Common
python train_semifinal_v7.py infer --checkpoint "$RunRoot/final/final.pt" --input "$DataRoot/test/DAPI" --output 'predictions/liver_naf_hladr_v1' --batch-size 4
```

`summarize` 需要全部五折完成。`selection.json` 记录 ROI 覆盖、配方、各评估轮数的 OOF SSIM/PSNR 和选定轮数；`final.pt` 是最终推理权重。推理目录下仅 `results/` 进入提交包，`provenance.json` 留作来源记录。提交前检查 JPG 数量、文件名、目录及官方当期格式要求。

显存不足时，所有折及最终重训统一降低 `--batch-size`；时间不足时，可统一改为三折、较短日程，再建立全套独立运行目录。不同配方的选模文件不可混用。训练时不使用测试集或平台反馈调节图像后处理。

## 提分判定

先在 Liver 的相同 ROI 折上比较新模型与现有候选的**对应标记** SSIM 和 PSNR，再决定是否全量重训与提交。若 SSIM 提高而 PSNR 明显下降，需谨慎判断，因为官方 PSNR 归一化算法未知。单标记模型须符合当期指定目标；多标记模型须看平均成绩。结构检查、初赛结果和论文分数不能替代 Liver OOF 或平台结果。

## 调试记录（2026-09-30）

本机定向回归测试 **37 项全部通过**，包括：

- 单标记/四标记的尺寸、输出范围、梯度和 TTA；标记选择顺序与元数据一致性。
- 通用训练入口的同步几何增强、标签强度保持与缓存不变性；本次修复了该入口误用旧增强的缺陷。
- 合成五个 ROI、两折训练 → OOF 汇总 → 固定轮数全量重训 → RGB JPG 导出；ROI 互斥和完整覆盖、完成后的断点恢复、缺折及不匹配选模拒绝。
- 评估与实际导出共用 RGB JPEG 编码，解码像素一致；非有限预测明确拒绝。
- 本机 RTX 3060 Laptop GPU 强制 FP16，注入一次梯度溢出，确认该批跳过、后续正常更新、模型和 EMA 有限。

另用默认 width=32、256×256 合成输入，在同一显卡完成单标记和四标记各两次 BF16 更新及 4× TTA 检查。上述都是合成数据调试，未执行真实 Liver 训练、测试集推理或平台提交，不能作为提分证据。

复现定向测试：

```powershell
python -m pytest tests/test_marker_context.py tests/test_prototype_marker.py tests/test_ultimate_ihc.py tests/test_final_ihc.py tests/test_marker_nafnet.py tests/test_semifinal_v7.py -q
```

CUDA FP16 测试在无 CUDA 时跳过。运行说明的 PowerShell 语法检查通过；真实数据目录与 GPU 资源仍需由使用者提供。

## 兼容性

本次修改了 V7 训练入口及推理/评估源码，旧版 V7 的 fold 结果和 selection.json 不能直接用于新版 final/infer 的源码哈希校验。旧实验应保留其原代码快照；新实验从新目录重新完成全部折。既有四标记 checkpoint 的通用加载接口仍保持兼容。

通用训练入口现在使用干净几何增强，其旧训练状态也不能在新版源码下直接恢复；保留旧源码快照继续旧实验，或在新目录开启新实验。
