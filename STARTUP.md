# 中文情感分析系统 - 启动指南

## 目录

- [环境要求](#环境要求)
- [一、安装依赖](#一安装依赖)
- [二、数据准备](#二数据准备)
- [三、模型训练](#三模型训练)
- [四、启动服务](#四启动服务)
- [五、Docker 部署](#五docker-部署)
- [六、反馈纠错闭环](#六反馈纠错闭环)
- [七、项目结构](#七项目结构)
- [八、常见问题](#常见问题)

---

## 环境要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Windows 10/11（也支持 Linux/macOS） |
| Python | 3.10 ~ 3.12 |
| GPU（可选） | NVIDIA + CUDA 12.6（无 GPU 也可用 CPU 运行） |
| 显存 | 建议 >= 6GB（BERT 微调） |

---

## 一、安装依赖

### 1.1 创建虚拟环境（推荐）

```powershell
conda create -n sentiment python=3.11 -y
conda activate sentiment
```

### 1.2 安装 PyTorch（GPU 版，CUDA 12.6）

```powershell
# 国内镜像加速 + GPU 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> 如果没有 NVIDIA 显卡，改用 CPU 版：
> `pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple`

### 1.3 安装其余依赖

```powershell
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 1.4 配置 HuggingFace 国内镜像

在系统环境变量中添加（或在每次启动前设置）：

```powershell
# PowerShell 临时设置
$env:HF_ENDPOINT = "https://hf-mirror.com"

# 永久设置（推荐）
[Environment]::SetEnvironmentVariable("HF_ENDPOINT", "https://hf-mirror.com", "User")
```

---

## 二、数据准备

### 2.1 二分类数据（外卖评论 ChnSentiCorp）

```powershell
python src/data/prepare_data.py
```

- 自动从 HuggingFace 下载 `seamew/ChnSentiCorp` 数据集
- 清洗后划分 train/val/test，保存到 `data/processed/`
- 生成 EDA 可视化图表到 `data/eda/`

### 2.2 情绪 6 分类数据（SMP2020 + NLPCC2014）

```powershell
python src/data/prepare_emotion_data.py
```

- 合并两个公开情绪数据集，映射为 6 类标签
- 弱标注策略生成"感激"和"焦虑"类样本
- 划分后保存到 `data/processed_emotion/`

> 两套数据相互独立，可按需只运行其中一个。

---

## 三、模型训练

### 3.1 二分类模型（4 个基线 + BERT）

按以下顺序训练（也可只训练 BERT）：

```powershell
# 1. TF-IDF + 逻辑回归（最快，基线）
python src/train/train_tfidf_lr.py

# 2. TextCNN
python src/train/train_textcnn.py

# 3. BiLSTM
python src/train/train_bilstm.py

# 4. BERT 微调（最优模型，需 GPU）
python src/train/train_bert.py
```

- BERT 权重保存为 `saved_models/bert_best.pt`
- 训练配置见 `config/config.yaml` 的 `train` 段

### 3.2 情绪 6 分类模型

```powershell
python src/train/train_bert_emotion.py
```

- 权重保存为 `saved_models/bert_emotion_best.pt`
- Tokenizer 保存为 `saved_models/bert_emotion_tokenizer/`

### 3.3 模型对比

```powershell
# 生成 4 个二分类模型的对比图表
python src/evaluate/compare_models.py

# 误差分析
python src/evaluate/error_analysis.py
```

---

## 四、启动服务

### 4.1 启动 FastAPI 后端

```powershell
# 方式一：使用启动脚本（自动检查模型是否存在）
python src/deploy/start.py

# 方式二：直接 uvicorn 启动
python -m uvicorn src.deploy.app:app --host 0.0.0.0 --port 8000
```

启动后访问：
- API 文档（Swagger UI）：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health

### 4.2 启动 Streamlit 前端

```powershell
# 需要先启动 FastAPI 后端
streamlit run app/main.py --server.port 8501
```

启动后访问：http://localhost:8501

### 4.3 前端功能页签

| 页签 | 功能 |
|------|------|
| 实时情绪识别 | 输入单条文本，输出 6 类情绪卡片 + 概率分布 |
| 批量情绪分析 | 粘贴多条文本或上传 CSV，输出环形图 + 明细表 |
| 长文本情绪曲线 | 输入长文本，逐句分析并绘制情绪走势图 |
| 历史与纠错 | 查看历史记录，纠正错误预测，导出训练格式 CSV |
| 模型档案 | 展示二分类和情绪分类的指标对比 |

### 4.4 主要 API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/predict` | 正/负面二分类预测 |
| POST | `/predict_emotion` | 6 类情绪预测（返回 record_id 自动入历史库） |
| POST | `/analyze_long_text` | 长文本分句情绪分析 |
| GET | `/history?limit=50` | 查询历史记录 |
| POST | `/feedback` | 提交纠正标签 |
| GET | `/feedback/export` | 导出纠错样本 CSV |

---

## 五、Docker 部署

### 5.1 构建并启动

```powershell
docker-compose up -d --build
```

- 服务暴露在 `localhost:8000`
- 使用 CPU 版 PyTorch（如需 GPU 取消 `docker-compose.yml` 中的注释）

### 5.2 停止

```powershell
docker-compose down
```

> 注意：Docker 部署只启动后端 API。前端 Streamlit 需单独运行或另外容器化。

---

## 六、反馈纠错闭环

完整的 MLOps 数据回流流程：

```
用户分析 → SQLite 自动存档 → 前端纠正错误标签 → 导出 CSV → 增量续训 → 模型变准
```

### 6.1 导出纠错样本

在前端「历史与纠错」页签中：
1. 筛选已纠正的记录
2. 点击「导出纠错样本 CSV」按钮
3. 文件保存到 `data/feedback/` 目录

### 6.2 增量续训

```powershell
# 默认参数（1 epoch, lr=1e-5）
python src/train/retrain_from_feedback.py

# 自定义参数
python src/train/retrain_from_feedback.py --epochs 2 --lr 5e-6
```

- 从 `bert_emotion_best.pt` 续训（不覆盖原模型）
- 新权重保存为 `saved_models/bert_emotion_retrained.pt`
- 验证满意后手动替换上线

---

## 七、项目结构

```
Project ladding/
├── config/
│   └── config.yaml                  # 全局配置（数据/模型/训练/部署/应用）
├── src/
│   ├── utils.py                     # 配置加载工具
│   ├── data/
│   │   ├── prepare_data.py          # 二分类数据准备
│   │   ├── prepare_emotion_data.py  # 情绪6分类数据准备
│   │   ├── dataset.py               # PyTorch Dataset
│   │   └── eda.py                   # 探索性数据分析
│   ├── models/
│   │   ├── bert_classifier.py       # BERT 分类模型
│   │   ├── bert_dataset.py           # BERT Dataset
│   │   ├── textcnn.py                # TextCNN 模型
│   │   ├── bilstm.py                 # BiLSTM 模型
│   │   └── text_dataset.py          # 文本向量化 Dataset
│   ├── train/
│   │   ├── train_tfidf_lr.py        # TF-IDF + 逻辑回归
│   │   ├── train_textcnn.py         # TextCNN 训练
│   │   ├── train_bilstm.py          # BiLSTM 训练
│   │   ├── train_bert.py            # BERT 二分类微调
│   │   ├── train_bert_emotion.py    # BERT 情绪6分类微调
│   │   └── retrain_from_feedback.py # 反馈纠错增量续训
│   ├── evaluate/
│   │   ├── metrics.py               # 评估指标计算
│   │   ├── compare_models.py        # 多模型对比
│   │   └── error_analysis.py        # 误差分析
│   ├── deploy/
│   │   ├── app.py                   # FastAPI 应用（v3.0.0）
│   │   ├── predictor.py             # 二分类推理
│   │   ├── emotion_predictor.py     # 情绪6分类推理
│   │   ├── long_text.py             # 长文本分句 + 逐句分析
│   │   ├── start.py                 # API 启动脚本
│   │   ├── export_onnx.py           # ONNX 导出
│   │   └── onnx_predictor.py        # ONNX 推理
│   └── storage/
│       └── history.py               # SQLite 历史记录存储
├── app/
│   ├── main.py                      # Streamlit 前端
│   └── style.py                     # 自定义 CSS 样式
├── data/
│   ├── raw/                         # 原始数据
│   ├── processed/                   # 二分类处理后的数据
│   ├── processed_emotion/           # 情绪分类处理后的数据
│   ├── app/history.db               # SQLite 历史库
│   └── feedback/                    # 纠错样本导出目录
├── saved_models/
│   ├── bert-base-chinese/           # 预训练 BERT 模型
│   ├── bert_best.pt                 # 二分类最优权重
│   ├── bert_emotion_best.pt         # 情绪分类最优权重
│   └── bert_emotion_tokenizer/      # 情绪分类 Tokenizer
├── logs/                            # 训练日志
├── .streamlit/config.toml           # Streamlit 主题配置
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── config/config.yaml
```

---

## 八、常见问题

### Q1: 模型下载超时？

设置 HuggingFace 镜像后再运行：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

### Q2: BERT 训练报 CUDA out of memory？

- 减小 `config/config.yaml` 中的 `batch_size`（如 16 → 8）
- 或使用 CPU 训练（速度较慢但能完成）

### Q3: 启动 API 报 ModuleNotFoundError？

确保在项目根目录下运行，且已激活虚拟环境：

```powershell
conda activate sentiment
cd "c:\Users\30748\Desktop\Project ladding"
```

### Q4: Streamlit 前端打不开？

- 确认 FastAPI 后端已启动（`http://localhost:8000` 可访问）
- 检查 8501 端口是否被占用

### Q5: 历史记录数据库在哪？

SQLite 数据库位于 `data/app/history.db`，可直接用 `sqlite3` 命令查看。

### Q6: 如何重新训练整个流程？

```powershell
# 完整流程（从零开始）
python src/data/prepare_data.py          # 1. 准备二分类数据
python src/data/prepare_emotion_data.py  # 2. 准备情绪分类数据
python src/train/train_bert.py           # 3. 训练二分类模型
python src/train/train_bert_emotion.py   # 4. 训练情绪分类模型
python src/deploy/start.py               # 5. 启动 API
streamlit run app/main.py                # 6. 启动前端
```
