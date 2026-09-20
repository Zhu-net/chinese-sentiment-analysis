# -*- coding: utf-8 -*-
"""
SentimentLens HF Spaces 入口
=============================
纯 ONNX 推理，无需 PyTorch。精简版 Streamlit 应用。
"""
import sys
import os
from pathlib import Path

# HuggingFace 镜像（HF Spaces 服务器在海外，不需要镜像，但保留兼容）
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from transformers import AutoTokenizer
import onnxruntime as ort
import yaml

# 加载配置
with open("config/config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

ecfg = cfg["emotion"]
labels_en = ecfg["labels"]
labels_cn = ecfg["labels_cn"]
emojis = ecfg["emojis"]
polarity_map = ecfg["polarity"]

# 情绪配色
EMOTION_THEME = {
    "开心": {"main": "#F59E0B", "bg": "#FFFBEB", "emoji": "😄"},
    "感激": {"main": "#10B981", "bg": "#ECFDF5", "emoji": "🙏"},
    "悲伤": {"main": "#3B82F6", "bg": "#EFF6FF", "emoji": "😢"},
    "愤怒": {"main": "#EF4444", "bg": "#FFF1F2", "emoji": "😡"},
    "恐惧": {"main": "#8B5CF6", "bg": "#F5F3FF", "emoji": "😱"},
    "焦虑": {"main": "#EC4899", "bg": "#FDF2F8", "emoji": "😰"},
}

MAX_LEN = cfg["data"]["max_length"]
SAVE_DIR = Path("saved_models")


@st.cache_resource
def load_models():
    """加载 ONNX 模型 + tokenizer"""
    tok_dir = SAVE_DIR / ecfg["tokenizer_dir"]
    tok_path = str(tok_dir) if tok_dir.exists() else cfg["model"]["pretrained_model"]
    tokenizer = AutoTokenizer.from_pretrained(tok_path)

    emo_session = ort.InferenceSession(
        str(SAVE_DIR / "bert_emotion.onnx"),
        providers=["CPUExecutionProvider"],
    )
    bin_session = ort.InferenceSession(
        str(SAVE_DIR / "bert_binary.onnx"),
        providers=["CPUExecutionProvider"],
    )
    return tokenizer, emo_session, bin_session


def softmax(logits):
    e = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def predict_emotion(text, tokenizer, session):
    enc = tokenizer(str(text), truncation=True, padding="max_length",
                    max_length=MAX_LEN, return_tensors="np")
    logits = session.run(None, {
        session.get_inputs()[0].name: enc["input_ids"].astype(np.int64),
        session.get_inputs()[1].name: enc["attention_mask"].astype(np.int64),
    })[0]
    probs = softmax(logits)[0]
    pid = int(np.argmax(probs))
    en = labels_en[pid]
    return {
        "emotion": en, "emotion_cn": labels_cn[pid], "emoji": emojis[pid],
        "polarity": polarity_map[en], "confidence": round(float(probs[pid]), 4),
        "probabilities": {labels_cn[i]: round(float(p), 4) for i, p in enumerate(probs)},
    }


def predict_batch(texts, tokenizer, session, batch_size=32):
    results = []
    for start in range(0, len(texts), batch_size):
        batch = [str(t) for t in texts[start:start + batch_size]]
        enc = tokenizer(batch, truncation=True, padding="max_length",
                        max_length=MAX_LEN, return_tensors="np")
        logits = session.run(None, {
            session.get_inputs()[0].name: enc["input_ids"].astype(np.int64),
            session.get_inputs()[1].name: enc["attention_mask"].astype(np.int64),
        })[0]
        probs = softmax(logits)
        pids = np.argmax(probs, axis=1)
        for j, text in enumerate(batch):
            pid = int(pids[j])
            en = labels_en[pid]
            results.append({
                "text": text, "emotion": en, "emotion_cn": labels_cn[pid],
                "emoji": emojis[pid], "polarity": polarity_map[en],
                "confidence": round(float(probs[j][pid]), 4),
            })
    return results


# ===== 页面样式 =====
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #F5F7FA 0%, #E8ECF4 100%); }
    .stApp { background: linear-gradient(135deg, #F0F4FF 0%, #FAF5FF 50%, #FFF5F0 100%); }
    .hero-title {
        font-size: 2rem; font-weight: 800; text-align: center;
        background: linear-gradient(135deg, #6C5CE7, #74B9FF, #00B894);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .emotion-card {
        border-radius: 16px; padding: 1.5rem; margin: 0.5rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    .metric-card {
        background: white; border-radius: 12px; padding: 1rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06); text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ===== 侧边栏 =====
st.sidebar.markdown("## 🎭 SentimentLens")
st.sidebar.caption("中文情绪分析 · ONNX 推理")
st.sidebar.markdown(f"""
**模型指标**
- 情绪 6 分类准确率: **81.8%**
- 二分类准确率: **91.2%**
- 推理引擎: ONNX Runtime (CPU)
""")

# ===== 加载模型 =====
try:
    tokenizer, emo_sess, bin_sess = load_models()
    model_ready = True
except Exception as e:
    st.error(f"模型加载失败: {e}")
    model_ready = False

# ===== Hero =====
st.markdown('<div class="hero-title">🎭 SentimentLens · 中文情绪分析</div>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center;color:#6B7280;">基于 BERT 微调 + ONNX 推理 · '
    '6 类情绪识别 · 长文本曲线 · 批量分析</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🎯 实时情绪识别", "📋 批量分析", "📊 长文本曲线"])

# ===== Tab 1: 实时情绪识别 =====
with tab1:
    text = st.text_area("输入中文文本:", height=100, placeholder="今天收到了录取通知书，特别开心！")
    examples = [
        "今天收到了录取通知书，开心得跳起来了！",
        "谢谢室友在我生病时照顾我，真的非常感激",
        "养了十年的狗狗昨天走了，心里特别难过",
        "快递等了一周还没到，客服态度还差，气死我了",
        "半夜看恐怖片，总觉得身后有人，太吓人了",
    ]
    st.markdown("**💡 快速示例：**")
    cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        if cols[i].button(ex[:10] + "…", key=f"ex_{i}"):
            st.session_state["input_text"] = ex
            st.rerun()

    if st.button("✨ 开始识别", type="primary", disabled=not model_ready) and text.strip():
        result = predict_emotion(text.strip(), tokenizer, emo_sess)
        theme = EMOTION_THEME.get(result["emotion_cn"], {"main": "#6B7280"})
        st.markdown(f"""
        <div class="emotion-card" style="background:{theme['bg']};border:2px solid {theme['main']};">
            <h2 style="color:{theme['main']};text-align:center;">
                {result['emoji']} {result['emotion_cn']} · {result['polarity']}
            </h2>
            <p style="text-align:center;color:#6B7280;">置信度 {result['confidence']*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

        # 概率分布
        probs = result["probabilities"]
        prob_df = pd.DataFrame([
            {"情绪": k, "概率": v, "颜色": EMOTION_THEME.get(k, {}).get("main", "#999")}
            for k, v in sorted(probs.items(), key=lambda x: -x[1])
        ])
        fig = go.Figure(go.Bar(
            x=prob_df["概率"], y=prob_df["情绪"], orientation="h",
            marker_color=prob_df["颜色"], text=prob_df["概率"].apply(lambda v: f"{v*100:.1f}%"),
            textposition="auto",
        ))
        fig.update_layout(title="情绪概率分布", paper_bgcolor="rgba(0,0,0,0)",
                         plot_bgcolor="rgba(0,0,0,0)", height=280)
        st.plotly_chart(fig, use_container_width=True)

# ===== Tab 2: 批量分析 =====
with tab2:
    input_mode = st.radio("输入方式", ["手动输入", "上传文件"], horizontal=True)
    if input_mode == "手动输入":
        texts_input = st.text_area("每行一条文本:", height=150)
        if st.button("🚀 批量识别", type="primary", disabled=not model_ready) and texts_input.strip():
            texts = [t.strip() for t in texts_input.split("\n") if t.strip()]
            results = predict_batch(texts, tokenizer, emo_sess)
            df = pd.DataFrame(results)
            from collections import Counter
            dist = Counter(r["emotion_cn"] for r in results)
            st.markdown(f"**共 {len(results)} 条** · 正面 {sum(1 for r in results if r['polarity']=='正面')} · 负面 {sum(1 for r in results if r['polarity']=='负面')}")

            # 环形图
            fig = px.pie(values=list(dist.values()), names=list(dist.keys()),
                         color=list(dist.keys()),
                         color_discrete_map={k: v["main"] for k, v in EMOTION_THEME.items()},
                         hole=0.5)
            fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df[["text", "emoji", "emotion_cn", "polarity", "confidence"]]
                         .rename(columns={"text": "原文", "emoji": "", "emotion_cn": "情绪",
                                          "polarity": "倾向", "confidence": "置信度"}),
                         use_container_width=True, hide_index=True)
            csv = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("📥 下载 CSV", csv, "results.csv", "text/csv")
    else:
        uploaded = st.file_uploader("上传 txt/csv/xlsx 文件", type=["txt", "csv", "xlsx"])
        if uploaded:
            if uploaded.name.endswith(".txt"):
                content = uploaded.getvalue().decode("utf-8-sig", errors="ignore")
                texts = [l.strip() for l in content.splitlines() if l.strip()]
            elif uploaded.name.endswith(".csv"):
                df = pd.read_csv(uploaded)
                texts = df.iloc[:, 0].dropna().astype(str).tolist()
            else:
                df = pd.read_excel(uploaded)
                texts = df.iloc[:, 0].dropna().astype(str).tolist()
            st.success(f"已提取 {len(texts)} 条文本")
            if st.button("🚀 批量识别", type="primary", disabled=not model_ready):
                results = predict_batch(texts[:200], tokenizer, emo_sess)
                st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

# ===== Tab 3: 长文本情绪曲线 =====
with tab3:
    import re
    long_text = st.text_area("输入段落或长文:", height=150,
                             placeholder="粘贴一段话，系统会逐句分析情绪起伏...")
    if st.button("🌈 分析情绪曲线", type="primary", disabled=not model_ready) and long_text.strip():
        # 分句
        sents = [s.strip() for s in re.split(r"[。！？!?；;\n]+", long_text) if len(s.strip()) >= 2]
        sents = sents[:100]
        if sents:
            results = predict_batch(sents, tokenizer, emo_sess)
            x = [i + 1 for i in range(len(results))]
            y = [r["confidence"] if r["polarity"] == "正面" else -r["confidence"] for r in results]
            colors = [EMOTION_THEME.get(r["emotion_cn"], {"main": "#999"})["main"] for r in results]
            hover = [f"第{i+1}句 {r['emoji']}{r['emotion_cn']} ({r['confidence']*100:.0f}%)<br>{r['text'][:30]}"
                     for i, r in enumerate(results)]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="lines+markers",
                line=dict(color="rgba(108,92,231,.3)", width=2, shape="spline"),
                marker=dict(size=12, color=colors, line=dict(color="#FFF", width=2)),
                text=hover, hovertemplate="%{text}<extra></extra>",
            ))
            fig.add_hline(y=0, line=dict(color="#9CA3AF", width=1, dash="dash"))
            fig.update_layout(
                title="情绪走势曲线（绿区=正面 · 红区=负面）",
                xaxis_title="句子顺序", yaxis_title="极性强度",
                yaxis=dict(range=[-1.15, 1.15], tickvals=[-1, 0, 1],
                           ticktext=["负面", "中性", "正面"]),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=350,
            )
            fig.add_hrect(y0=0, y1=1.15, fillcolor="rgba(16,185,129,.05)", line_width=0, layer="below")
            fig.add_hrect(y0=-1.15, y1=0, fillcolor="rgba(244,63,94,.05)", line_width=0, layer="below")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"**主导情绪**: {max(set(r['emotion_cn'] for r in results), key=[r['emotion_cn'] for r in results].count)}")
            df_sent = pd.DataFrame([
                {"句序": i+1, "": r["emoji"], "情绪": r["emotion_cn"],
                 "倾向": r["polarity"], "置信度": f"{r['confidence']*100:.0f}%", "句子": r["text"]}
                for i, r in enumerate(results)
            ])
            st.dataframe(df_sent, use_container_width=True, hide_index=True)
        else:
            st.warning("未提取到有效句子")
