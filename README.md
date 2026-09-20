# 中文情感分析与观点挖掘系统

> 基于 PyTorch + BERT 的中文评论情感分析系统，包含数据处理、模型训练、服务部署、前端可视化完整链路。

## 📋 项目简介

本项目针对电商/酒店评论场景，构建了一个端到端的中文情感分析系统。系统能够自动识别评论的情感极性（正面/负面），并提供可视化的 Web 交互界面。

## 🏗️ 技术栈

| 模块 | 技术 |
|------|------|
| 深度学习框架 | PyTorch |
| 预训练模型 | BERT-base-chinese |
| 数据处理 | pandas, jieba, scikit-learn |
| 模型训练 | 自定义训练循环 + Transformers |
| Web 服务 | FastAPI + Uvicorn |
| 容器化 | Docker |
| 前端 | Streamlit |

## 📁 项目结构

```
.
├── config/              # 配置文件
│   └── config.yaml
├── data/                # 数据目录
│   ├── raw/             # 原始数据
│   └── processed/       # 处理后数据
├── src/                 # 源代码
│   ├── data/            # 数据处理与EDA
│   ├── models/          # 模型定义
│   ├── train/           # 训练脚本
│   ├── evaluate/        # 评估
│   ├── deploy/          # 部署相关
│   └── utils.py         # 工具函数
├── app/                 # Streamlit 前端
├── notebooks/           # 探索性分析
├── tests/               # 测试
├── logs/                # 训练日志
├── saved_models/        # 保存的模型
├── requirements.txt
└── README.md
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据准备

```bash
python src/data/prepare_data.py
```

### 3. 探索性数据分析

```bash
python src/data/eda.py
```

### 4. 模型训练

```bash
# 基线模型（TextCNN）
python src/train/train_textcnn.py

# BERT 微调
python src/train/train_bert.py
```

### 5. 启动服务

```bash
python src/deploy/app.py
```

### 6. 启动前端

```bash
streamlit run app/main.py
```

## 📊 实验结果

| 模型 | 准确率 | 精确率 | 召回率 | F1 |
|------|--------|--------|--------|-----|
| TF-IDF + LR | 90.42% | 84.25% | 85.37% | 0.8481 |
| TextCNN | 88.42% | 82.29% | 80.32% | 0.8129 |
| BiLSTM | 88.00% | 80.69% | 81.12% | 0.8090 |
| **BERT (Ours)** | **91.17%** | **87.29%** | **84.04%** | **0.8564** |

> 数据集：外卖评论数据集（waimai_10k，约 1.2 万条），类别比约 1:2（正:负）

## 🔧 核心功能

1. **多模型对比**：支持传统 ML、深度学习、预训练模型多种方案
2. **模型服务化**：FastAPI 封装，支持批量预测
3. **可视化界面**：Streamlit 前端，实时展示预测结果
4. **Docker 部署**：一键容器化部署

## 📝 项目亮点

- 完整的工程化结构，配置与代码分离
- 手写训练循环，深入理解 PyTorch 训练流程
- 端到端落地：数据 → 训练 → 部署 → 可视化

## 📄 License

MIT
