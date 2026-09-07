# AIC 比赛项目

> 比赛项目工作目录

## 目录结构

```
aic/
├── data/           # 数据集（已被 .gitignore 忽略）
├── notebooks/      # Jupyter 探索性分析
├── src/            # 源代码
├── models/         # 训练好的模型（已被 .gitignore 忽略）
├── submissions/    # 提交结果
├── logs/           # 训练日志
└── README.md
```

## 快速开始

```bash
# 克隆仓库
git clone <repo-url>

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

## 工作流

1. **数据准备**：`data/` 目录下放入训练/测试数据
2. **探索分析**：在 `notebooks/` 中做 EDA
3. **模型开发**：在 `src/` 中编写训练/推理代码
4. **提交结果**：将预测输出保存到 `submissions/`

## 提交记录

| 时间 | 提交 | 得分 | 备注 |
|------|------|------|------|
|      |      |      |      |
