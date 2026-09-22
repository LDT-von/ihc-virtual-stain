"""模型模块（本快照专用）

仅保留产生本次提交（AIC-2026-51704013，平台分 74.6531）的模型：

- MarkerContextNet     共享多尺度编码器 + 上下文注意力 + 四 marker 独立输出头（作基准）
- MarkerSpecificNet    按 marker 独立的形态编码分支
- AnchoredIHC          冻结基准 + 四路残差精修的固定级联（部署形态）

原仓库此文件还从 Pix2Pix / FlowMatching / StableDiffusion 导入初赛阶段的模型，
那些模型与本次提交无关，已按「不多不少」原则排除，不随本快照复制。
因此本文件刻意不导出任何名字，请直接从子模块导入，例如：

    from src.models.anchored_ihc import AnchoredIHC
"""
