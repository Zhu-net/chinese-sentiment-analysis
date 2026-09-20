---
title: SentimentLens 中文情绪分析
emoji: 🎭
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: "1.32.0"
app_file: app.py
pinned: false
license: mit
tags:
  - nlp
  - sentiment-analysis
  - bert
  - chinese
  - onnx
---

# 🎭 SentimentLens · 中文情绪分析系统

基于 BERT-base-chinese 微调的中文情绪分析系统，支持：

- **6 类细粒度情绪识别**：开心 / 感激 / 悲伤 / 愤怒 / 恐惧 / 焦虑（准确率 81.8%）
- **正/负面二分类**（准确率 91.2%）
- **长文本情绪曲线**：逐句分析情绪走势与转折，输出整段极性总结
- **批量分析**：CSV 上传 / 粘贴多行文本
- **ONNX 推理加速**：无需 PyTorch，onnxruntime CPU 推理

## 技术栈

- 模型：BERT-base-chinese 微调 → ONNX 导出
- 推理：onnxruntime（CPU，无需 GPU）
- 前端：Streamlit + Plotly
- 后端：FastAPI（本地部署时）
