"""
Streamlit 前端（情绪 6 分类版）
功能：
1. 单条文本情绪预测（6 色情绪卡片 + 全概率分布 + Top3 图）
2. 批量文本预测（6 类环形图 + 明细表）
3. 模型档案（二分类 + 情绪6分类双任务指标）
"""
import sys
import os
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent))

from src.utils import load_config
from src.deploy.emotion_predictor import (
    get_emotion_predictor,
    predict_routed,
    predict_batch_routed,
)
from src.deploy.long_text import analyze_long_text
from src.storage import history as history_store
from style import (
    GLOBAL_CSS,
    SIDEBAR_HEADER_HTML,
    render_hero,
    emotion_card_html,
    emotion_prob_rows_html,
    EMOTION_THEME,
)

st.set_page_config(
    page_title="SentimentLens · 中文情绪分析",
    page_icon="🌈",
    layout="wide",
)
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

CFG = load_config()
ECFG = CFG["emotion"]
EMOTION_CN = ECFG["labels_cn"]          # ['开心','感激','悲伤','愤怒','恐惧','焦虑']
EMOTION_COLORS = ECFG["colors"]         # 6 个主题色
COLOR_MAP = dict(zip(EMOTION_CN, EMOTION_COLORS))

# 推理后端（P1 QLoRA 三路线）
BACKEND_OPTIONS = {
    "🤖 BERT（快 · 35ms）": "bert",
    "🔍 BERT+RAG 精判（慢 · 触发时调用 DeepSeek）": "rag",
    "🧠 QLoRA-1.5B 生成式（带理由 · ~0.6s/条）": "lora",
}
BACKEND_BADGE = {
    "bert": "BERT",
    "rag": "BERT+RAG",
    "lora": "QLoRA-1.5B",
    "lora_fallback_bert": "QLoRA→回退BERT",
}


def backend_selector(key: str) -> str:
    """后端选择控件，返回 model_type 字符串"""
    label = st.radio(
        "选择推理后端", list(BACKEND_OPTIONS.keys()),
        horizontal=True, key=key,
    )
    return BACKEND_OPTIONS[label]


@st.cache_resource(show_spinner="正在加载情绪模型...")
def load_model():
    return get_emotion_predictor()


def top3_fig(top3: list):
    """Top-3 情绪横向条形图（每类专属色）"""
    names = [f"{t['emoji']} {t['emotion_cn']}" for t in top3][::-1]
    vals = [t["probability"] for t in top3][::-1]
    colors = [EMOTION_THEME[t["emotion_cn"]]["main"] for t in top3][::-1]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker=dict(color=colors, cornerradius=8),
        text=[f"{v*100:.1f}%" for v in vals],
        textposition="outside",
        textfont=dict(size=13, color="#2D3142"),
        hovertemplate="%{y}<br>%{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="🏅 Top-3 候选情绪", font=dict(size=15, color="#2D3142"), x=0.02),
        xaxis=dict(range=[0, 1.08], visible=False),
        yaxis=dict(tickfont=dict(size=14, color="#2D3142")),
        margin=dict(t=44, b=8, l=10, r=56),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=210,
    )
    return fig


def emotion_donut_fig(counter: dict):
    """6 类情绪占比环形图（只展示出现过的类）"""
    items = [(cn, counter.get(cn, 0)) for cn in EMOTION_CN if counter.get(cn, 0) > 0]
    labels = [f"{ECFG['emojis'][EMOTION_CN.index(cn)]} {cn}" for cn, _ in items]
    values = [v for _, v in items]
    colors = [COLOR_MAP[cn] for cn, _ in items]
    total = sum(values)
    fig = go.Figure(go.Pie(
        values=values, labels=labels, hole=0.6,
        marker=dict(colors=colors, line=dict(color="#FFFFFF", width=3)),
        textinfo="percent", textfont=dict(size=13, color="#FFFFFF"),
        hovertemplate="%{label}<br>%{value} 条 (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        annotations=[dict(text=f"<b>{total}</b><br><span style='font-size:12px;color:#9CA3AF'>总条数</span>",
                          x=0.5, y=0.5, font=dict(size=20, color="#2D3142"), showarrow=False)],
        legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center", font=dict(size=12)),
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=300,
    )
    return fig


def fill_example(ex_text: str) -> None:
    """示例按钮回调：在 widget 实例化前写入 session_state（Streamlit 合法时机）"""
    st.session_state["single_input"] = ex_text


def fill_long_example(ex_text: str) -> None:
    """长文本示例回调（同理，必须用 on_click）"""
    st.session_state["long_input"] = ex_text


def emotion_curve_fig(sentences: list):
    """长文本情绪走势图：极性强度折线（+正/-负）+ 六色情绪散点"""
    x = [s["idx"] + 1 for s in sentences]
    y = [s["pos_score"] for s in sentences]
    point_colors = [EMOTION_THEME[s["emotion_cn"]]["main"] for s in sentences]
    hover = [f"第{s['idx']+1}句 {s['emoji']}{s['emotion_cn']} "
             f"({s['confidence']*100:.0f}%)<br>{s['text'][:30]}" for s in sentences]

    fig = go.Figure()
    # 灰色平滑折线串联情绪起伏
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines",
        line=dict(color="rgba(108,92,231,.35)", width=2, shape="spline", smoothing=0.6),
        hoverinfo="skip", showlegend=False,
    ))
    # 六色情绪散点（每类专属色）
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(size=15, color=point_colors,
                    line=dict(color="#FFFFFF", width=2)),
        text=hover, hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ))
    # 零轴：正面在上（绿区），负面在下（红区）
    fig.add_hline(y=0, line=dict(color="#9CA3AF", width=1.5, dash="dash"))
    fig.update_layout(
        title=dict(text="🌈 情绪走势曲线（绿区=正面情绪 · 红区=负面情绪）",
                   font=dict(size=15, color="#2D3142"), x=0.02),
        xaxis=dict(title="句子顺序", tickmode="linear", tickfont=dict(size=12),
                   gridcolor="rgba(108,92,231,.08)", zeroline=False),
        yaxis=dict(title="极性强度", range=[-1.15, 1.15],
                   tickvals=[-1, -0.5, 0, 0.5, 1],
                   ticktext=["很负面", "偏负面", "中性", "偏正面", "很正面"],
                   tickfont=dict(size=12), gridcolor="rgba(108,92,231,.08)"),
        margin=dict(t=50, b=10, l=10, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=340,
    )
    # 正负背景色带
    fig.add_hrect(y0=0, y1=1.15, fillcolor="rgba(16,185,129,.05)",
                  line_width=0, layer="below")
    fig.add_hrect(y0=-1.15, y1=0, fillcolor="rgba(244,63,94,.05)",
                  line_width=0, layer="below")
    return fig


LONG_EXAMPLE = (
    "今天看到这家餐厅评分很高，满怀期待地来打卡。店面装修很精致，服务员也特别热情，"
    "刚坐下就有种被重视的感觉。点了招牌烤鱼，味道确实惊艳，外焦里嫩，朋友都赞不绝口，真心感谢老板送的小菜。"
    "不过上菜速度实在太慢了，足足等了五十分钟，中途催了三次还是没动静，饿得肚子咕咕叫，越等越烦躁。"
    "结账时还发现账单多算了一个菜，跟前台沟通了半天才退掉，挺影响心情的。"
    "总体来说味道值得回味，但服务和效率真的让人有点焦虑，下次要不要再来还得再想想。"
)


def main():
    # 模型加载（训练未完成时给出友好提示）
    try:
        predictor = load_model()
        model_ready = True
    except Exception as e:
        model_ready = False
        load_error = str(e)

    # ===== 侧边栏 =====
    with st.sidebar:
        st.markdown(SIDEBAR_HEADER_HTML, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown(
            '<div class="section-title"><span class="dot"></span>🎭 情绪体系</div>',
            unsafe_allow_html=True,
        )
        for cn, color, emoji in zip(EMOTION_CN, EMOTION_COLORS, ECFG["emojis"]):
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;padding:3px 0;">'
                f'<span style="font-size:1.1rem;">{emoji}</span>'
                f'<span style="width:10px;height:10px;border-radius:50%;background:{color};display:inline-block;"></span>'
                f'<span style="color:#4B5563;font-size:.92rem;">{cn}</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown("---")
        st.caption("🧬 模型: BERT-base-chinese 微调\n\n"
                   "📚 语料: SMP2020 + NLPCC2014 微博情绪\n\n"
                   "⚖️ 训练: 类均衡采样 · macro-F1 选模")

    # ===== Hero =====
    st.markdown(render_hero(), unsafe_allow_html=True)

    if not model_ready:
        st.warning(
            "⏳ 情绪模型尚未就绪（可能正在训练中）。请等待 `bert_emotion_best.pt` 生成后刷新页面。\n\n"
            f"加载信息：`{load_error[:120]}`"
        )

    tab1, tab2, tab_long, tab_hist, tab3 = st.tabs(
        ["🎯  实时情绪识别", "📋  批量情绪分析",
         "📊  长文本情绪曲线", "🗂️  历史与纠错", "📈  模型档案"])

    # ===== Tab 1: 单条预测 =====
    with tab1:
        left, right = st.columns([1.15, 1], gap="large")

        with left:
            st.markdown(
                '<div class="section-title"><span class="dot"></span>✍️ 输入文本</div>',
                unsafe_allow_html=True,
            )
            text = st.text_area(
                "请输入要分析的文本：",
                key="single_input",
                height=150,
                placeholder="在这里输入任意中文文本，识别其中的细腻情绪…",
                label_visibility="collapsed",
            )

            st.caption("💡 六种情绪的典型示例：")
            examples = [
                "今天收到了期待已久的录取通知书，开心得跳起来了！",
                "谢谢室友在我生病时照顾我，真的非常感激",
                "养了十年的狗狗昨天走了，心里特别难过",
                "快递等了整整一周还没到，客服态度还差，气死我了",
                "半夜看恐怖片，总觉得身后有人，太吓人了",
                "明天就要面试了，紧张得一晚上没睡着，心里特别慌",
            ]
            chip_cols = st.columns(2, gap="small")
            for i, ex in enumerate(examples):
                with chip_cols[i % 2]:
                    st.markdown('<div class="chip-btn">', unsafe_allow_html=True)
                    st.button(
                        f"{ECFG['emojis'][i]} {ex[:13]}…",
                        key=f"ex_{i}", on_click=fill_example, args=(ex,),
                        use_container_width=True,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

            analyze = st.button("✨ 开始识别", type="primary", use_container_width=True,
                                disabled=not model_ready)
            backend = backend_selector(key="backend_single")

        with right:
            if analyze and text.strip() and model_ready:
                spinner_msg = {
                    "bert": "🤖 BERT 正在分析情绪...",
                    "rag": "🔍 BERT+RAG 正在精判（触发时约 1 秒）...",
                    "lora": "🧠 QLoRA-1.5B 正在生成判断与理由（约 1 秒）...",
                }[backend]
                with st.spinner(spinner_msg):
                    result = predict_routed(text.strip(), model_type=backend)
                # 自动入历史库（反馈闭环数据源）
                result["record_id"] = history_store.add_record(
                    "emotion", text.strip(), result)
                st.session_state["emotion_result"] = result

            result = st.session_state.get("emotion_result")
            if result is not None and model_ready:
                st.markdown(
                    emotion_card_html(result["emotion_cn"], result["emoji"],
                                      result["confidence"], result["polarity"]),
                    unsafe_allow_html=True,
                )
                # 后端徽标 + LoRA 生成理由
                mt = result.get("model_type", "bert")
                badge = BACKEND_BADGE.get(mt, mt)
                extra = []
                if result.get("latency_ms") is not None:
                    extra.append(f"{result['latency_ms']:.0f}ms")
                if result.get("valid_json") is not None:
                    extra.append("JSON✓" if result["valid_json"] else "JSON✗")
                st.caption(f"推理后端：{badge}" + (f" · {' · '.join(extra)}" if extra else ""))
                if result.get("reason"):
                    st.info(f"💬 判定理由：{result['reason']}")
                if mt == "lora_fallback_bert":
                    st.warning("⚠️ LoRA 输出非法 JSON，已自动回退 BERT 结果")
            else:
                st.markdown(
                    """
                    <div class="result-card" style="background:rgba(255,255,255,.6); border:1.5px dashed rgba(108,92,231,.25); box-shadow:none;">
                        <div class="result-emoji" style="filter:grayscale(.35);">🎭</div>
                        <div class="result-label" style="color:#9CA3AF; font-size:1.1rem; letter-spacing:2px;">等待识别</div>
                        <div class="result-conf">输入文本或点击情绪示例，结果将在这里呈现</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # 概率分布 + Top3（LoRA 为生成式 one-hot 输出，无概率分布，跳过）
        result = st.session_state.get("emotion_result")
        if analyze and text.strip() and model_ready and result is not None:
            if result.get("probabilities"):
                st.markdown(
                    '<div class="section-title" style="margin-top:18px;"><span class="dot"></span>🎚️ 六类情绪概率分布</div>',
                    unsafe_allow_html=True,
                )
                c_map, c_bar = st.columns([1.2, 1], gap="large")
                with c_map:
                    st.markdown(emotion_prob_rows_html(result["probabilities"]), unsafe_allow_html=True)
                with c_bar:
                    st.plotly_chart(top3_fig(result["top3"]), use_container_width=True)
            else:
                st.caption("🎚️ 生成式后端输出确定性标签（confidence=1.0），无概率分布")

            with st.expander("🔍 查看原始 JSON 输出"):
                st.json({k: v for k, v in result.items() if k != "text"})

    # ===== Tab 2: 批量预测 =====
    with tab2:
        batch_backend = backend_selector(key="backend_batch")
        if batch_backend != "bert":
            st.caption("⏱️ 非 BERT 后端逐条约 0.6-1s，批量较大时请耐心等待"
                       + ("（RAG 仅对低置信样本触发 API）" if batch_backend == "rag" else ""))
        MODE_PASTE = "📝 粘贴文本"
        input_mode = st.radio(
            "选择输入方式", [MODE_PASTE, "📄 上传 CSV 文件"],
            horizontal=True, label_visibility="collapsed",
        )
        if input_mode == MODE_PASTE:
            texts_input = st.text_area(
                "每行一条文本：", height=180,
                placeholder="今天真开心\n一想到明天考试就紧张得睡不着\n谢谢你们的帮助",
                label_visibility="collapsed",
            )
            if st.button("🚀 批量识别", type="primary", disabled=not model_ready):
                if texts_input.strip():
                    texts = [t.strip() for t in texts_input.split("\n") if t.strip()]
                    with st.spinner(f"正在识别 {len(texts)} 条文本..."):
                        st.session_state["emo_batch_df"] = pd.DataFrame(
                            predict_batch_routed(texts, model_type=batch_backend))
                else:
                    st.warning("请输入文本")
        else:
            uploaded = st.file_uploader("上传 CSV 文件（需包含 text 列）", type=["csv"])
            if uploaded:
                df_up = pd.read_csv(uploaded)
                if "text" in df_up.columns:
                    st.success(f"已加载 {len(df_up)} 条数据")
                    if st.button("🚀 开始批量识别", type="primary", disabled=not model_ready):
                        with st.spinner(f"正在识别 {len(df_up)} 条文本..."):
                            st.session_state["emo_batch_df"] = pd.DataFrame(
                                predict_batch_routed(df_up["text"].tolist(),
                                                     model_type=batch_backend))
                else:
                    st.error("CSV 文件必须包含 'text' 列")

        if "emo_batch_df" in st.session_state:
            df = st.session_state["emo_batch_df"]
            counter = dict(df["emotion_cn"].value_counts())
            total = len(df)

            # 指标卡
            cols = st.columns(4)
            cols[0].metric("📊 总条数", total)
            top_emotion = max(counter, key=counter.get)
            cols[1].metric("🏅 主导情绪", f"{top_emotion}")
            pos_n = int(df["polarity"].eq("正面").sum())
            cols[2].metric("😊 正面占比", f"{pos_n/total*100:.1f}%")
            cols[3].metric("😞 负面占比", f"{(total-pos_n)/total*100:.1f}%")

            c_table, c_chart = st.columns([1.25, 1], gap="medium")
            with c_chart:
                st.plotly_chart(emotion_donut_fig(counter), use_container_width=True)
            with c_table:
                st.markdown(
                    '<div class="section-title"><span class="dot"></span>📑 预测明细</div>',
                    unsafe_allow_html=True,
                )
                show_cols = ["text", "emoji", "emotion_cn", "polarity", "confidence"]
                if "reason" in df.columns:  # LoRA 生成理由
                    show_cols.append("reason")
                st.dataframe(
                    df[show_cols]
                    .style.map(lambda v: f"color:{EMOTION_THEME.get(v, {}).get('main', '#374151')};font-weight:700;",
                               subset=["emotion_cn"]),
                    use_container_width=True, height=320,
                )

            csv = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("📥 下载预测结果 (CSV)", csv,
                               "emotion_results.csv", "text/csv", use_container_width=True)

    # ===== Tab 3: 长文本情绪曲线 =====
    with tab_long:
        st.markdown(
            '<div class="section-title"><span class="dot"></span>📖 输入段落 / 短文（自动按句切分）</div>',
            unsafe_allow_html=True,
        )
        c_in, c_btn = st.columns([5, 1], gap="small")
        with c_in:
            long_text = st.text_area(
                "长文本：", key="long_input", height=190,
                placeholder="粘贴一段话、一条长评或一篇短文，系统会逐句分析情绪起伏…",
                label_visibility="collapsed",
            )
        with c_btn:
            st.markdown('<div class="chip-btn">', unsafe_allow_html=True)
            st.button("📝 示例长评", key="long_ex", on_click=fill_long_example,
                      args=(LONG_EXAMPLE,), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            run_long = st.button("🌈 分析情绪曲线", type="primary",
                                 use_container_width=True, disabled=not model_ready)
            st.caption("最多分析 100 句，批量推理约 1-2 秒")

        if run_long and long_text.strip() and model_ready:
            with st.spinner("正在逐句分析..."):
                long_result = analyze_long_text(long_text.strip(), predictor)
            if not long_result["sentences"]:
                st.warning("未切分出有效句子，请输入更完整的文本。")
            else:
                long_result["record_id"] = history_store.add_record(
                    "long_text", long_text.strip(), long_result)
                st.session_state["long_result"] = long_result

        long_result = st.session_state.get("long_result")
        if long_result and long_result.get("sentences"):
            s = long_result["summary"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("📝 句子总数", s["total"])
            m2.metric("🏅 主导情绪", max(s["distribution"], key=s["distribution"].get),
                      f"{s['dominant_ratio']*100:.0f}% 占比")
            m3.metric("😊 正面句", f"{s['pos_count']} 句", f"{s['pos_ratio']*100:.0f}%")
            m4.metric("😞 负面句", f"{s['neg_count']} 句",
                      f"情绪转折 {s['switches']} 次")

            st.plotly_chart(emotion_curve_fig(long_result["sentences"]),
                            use_container_width=True)

            with st.expander(f"🔍 逐句情绪明细（{s['total']} 句，含情绪转折标记）",
                             expanded=True):
                df_sent = pd.DataFrame(long_result["sentences"])
                df_sent["句序"] = df_sent["idx"] + 1
                st.dataframe(
                    df_sent[["句序", "emoji", "emotion_cn", "polarity",
                             "confidence", "text"]]
                    .rename(columns={"emoji": "", "emotion_cn": "情绪",
                                     "polarity": "倾向", "confidence": "置信度",
                                     "text": "句子"})
                    .style.map(
                        lambda v: f"color:{EMOTION_THEME.get(v, {}).get('main', '#374151')};font-weight:700;",
                        subset=["情绪"]),
                    use_container_width=True, height=330, hide_index=True,
                )

    # ===== Tab 4: 历史与纠错 =====
    with tab_hist:
        st.markdown(
            '<div class="section-title"><span class="dot"></span>🗂️ 分析历史与人工纠错（数据回流闭环）</div>',
            unsafe_allow_html=True)

        stat = history_store.stats()
        k1, k2, k3 = st.columns(3)
        k1.metric("📚 累计分析", stat["total"])
        k2.metric("✏️ 已纠错", stat["corrected"])
        k3.metric("🎯 闭环状态",
                  "可重训" if stat["corrected"] > 0 else "待积累")

        view_mode = st.radio("查看范围", ["全部", "仅已纠正"],
                             horizontal=True, label_visibility="collapsed")
        records = history_store.list_records(
            limit=100, corrected_only=(view_mode == "仅已纠正"))

        if not records:
            st.info("暂无历史记录。去「实时情绪识别」或「长文本情绪曲线」做几次分析吧。")
        else:
            type_cn = {"emotion": "单条情绪", "long_text": "长文本", "binary": "正负面"}
            en2cn = dict(zip(ECFG["labels"], ECFG["labels_cn"]))
            cn2en = dict(zip(ECFG["labels_cn"], ECFG["labels"]))
            NO_CORRECT = "— 未纠正 —"
            hist_df = pd.DataFrame([{
                "id": r["id"],
                "时间": r["created_at"][5:16].replace("T", " "),
                "类型": type_cn.get(r["task_type"], r["task_type"]),
                "原文": (r["input_text"] or "")[:42],
                "原预测": r["predicted_label_cn"] or "-",
                # 编辑列直接用中文选项（兼容旧版 Streamlit，无 format_func）
                "纠正为": en2cn.get(r["corrected_label"], NO_CORRECT),
            } for r in records])

            edited = st.data_editor(
                hist_df,
                column_config={
                    "id": None,  # 隐藏但保留数据
                    "纠正为": st.column_config.SelectboxColumn(
                        "纠正为（如预测有误）",
                        options=[NO_CORRECT] + ECFG["labels_cn"],
                        required=False),
                    "原文": st.column_config.TextColumn(width="large"),
                },
                disabled=["时间", "类型", "原文", "原预测"],
                hide_index=True, use_container_width=True, height=380,
                key=f"hist_editor_{view_mode}",
            )

            c_save, c_export, c_tip = st.columns([1, 1, 2])
            with c_save:
                if st.button("💾 保存纠正", type="primary", use_container_width=True):
                    orig = {r["id"]: (r["corrected_label"] or "") for r in records}
                    n_ok = 0
                    for _, row in edited.iterrows():
                        cn_val = row["纠正为"]
                        new_label = cn2en.get(cn_val, "")  # 中文选项 -> 英文键
                        if new_label and new_label != orig.get(row["id"]):
                            history_store.update_feedback(int(row["id"]), new_label)
                            n_ok += 1
                    if n_ok:
                        st.toast(f"已保存 {n_ok} 条纠正，已进入数据回流池 ✅")
                        st.rerun()
                    else:
                        st.toast("没有检测到新的修改")
            with c_export:
                fb_df = history_store.export_feedback_dataframe()
                if len(fb_df):
                    st.download_button(
                        "📥 导出纠错样本 CSV",
                        data=fb_df.to_csv(index=False).encode("utf-8-sig"),
                        file_name="feedback_samples.csv",
                        mime="text/csv", use_container_width=True,
                        help=f"当前可导出 {len(fb_df)} 条（列格式与训练集一致）")
                else:
                    st.button("📥 导出纠错样本 CSV", disabled=True,
                              use_container_width=True, help="还没有与原预测不同的纠正")
            with c_tip:
                st.caption(
            "🔁 闭环用法：纠正若干条 → 导出 CSV → 运行 "
            "`python src/train/retrain_from_feedback.py` 增量续训 → 替换权重上线")

    # ===== Tab 5: 模型档案 =====
    with tab3:
        st.markdown(
            '<div class="section-title"><span class="dot"></span>🗂️ 双任务模型体系</div>',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.markdown(
                """
                <div class="glass-card">
                    <div style="font-weight:900; font-size:1.05rem; color:#2D3142;">任务一 · 评论正负面二分类</div>
                    <div style="color:#9CA3AF; font-size:.85rem; margin:4px 0 12px;">外卖评论 waimai_10k · 1.2 万条</div>
                    <table style="width:100%; border-collapse:collapse; font-size:.92rem;">
                    <tr><td style="padding:4px 0; color:#6B7280;">模型</td><td style="text-align:right; font-weight:700;">BERT-base-chinese</td></tr>
                    <tr><td style="padding:4px 0; color:#6B7280;">准确率</td><td style="text-align:right; font-weight:700; color:#059669;">91.17%</td></tr>
                    <tr><td style="padding:4px 0; color:#6B7280;">F1</td><td style="text-align:right; font-weight:700; color:#059669;">0.8564</td></tr>
                    <tr><td style="padding:4px 0; color:#6B7280;">服务接口</td><td style="text-align:right; font-weight:700;">/predict</td></tr>
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                """
                <div class="glass-card">
                    <div style="font-weight:900; font-size:1.05rem; color:#2D3142;">任务二 · 细粒度情绪 6 分类</div>
                    <div style="color:#9CA3AF; font-size:.85rem; margin:4px 0 12px;">SMP2020 + NLPCC2014 · 4.4 万条微博</div>
                    <table style="width:100%; border-collapse:collapse; font-size:.92rem;">
                    <tr><td style="padding:4px 0; color:#6B7280;">模型</td><td style="text-align:right; font-weight:700;">BERT-base-chinese</td></tr>
                    <tr><td style="padding:4px 0; color:#6B7280;">准确率</td><td style="text-align:right; font-weight:700; color:#6C5CE7;">81.80%</td></tr>
                    <tr><td style="padding:4px 0; color:#6B7280;">macro-F1</td><td style="text-align:right; font-weight:700; color:#6C5CE7;">0.7993</td></tr>
                    <tr><td style="padding:4px 0; color:#6B7280;">weighted-F1</td><td style="text-align:right; font-weight:700; color:#6C5CE7;">0.8191</td></tr>
                    <tr><td style="padding:4px 0; color:#6B7280;">类别均衡策略</td><td style="text-align:right; font-weight:700;">WeightedRandomSampler</td></tr>
                    <tr><td style="padding:4px 0; color:#6B7280;">服务接口</td><td style="text-align:right; font-weight:700;">/predict_emotion</td></tr>
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown(
            """
            <div class="glass-card" style="margin-top:6px;">
                <div style="font-weight:900; font-size:.98rem; color:#2D3142; margin-bottom:6px;">🎭 情绪各类 F1（测试集）</div>
                <div style="display:flex; gap:10px; flex-wrap:wrap;">
                    <span style="background:#FFFBEB;color:#D97706;font-weight:700;padding:4px 14px;border-radius:999px;">😄 开心 0.89</span>
                    <span style="background:#ECFDF5;color:#059669;font-weight:700;padding:4px 14px;border-radius:999px;">🙏 感激 0.96</span>
                    <span style="background:#EFF6FF;color:#2563EB;font-weight:700;padding:4px 14px;border-radius:999px;">😢 悲伤 0.70</span>
                    <span style="background:#FFF1F2;color:#E11D48;font-weight:700;padding:4px 14px;border-radius:999px;">😠 愤怒 0.82</span>
                    <span style="background:#F5F3FF;color:#7C3AED;font-weight:700;padding:4px 14px;border-radius:999px;">😨 恐惧 0.73</span>
                    <span style="background:#FDF2F8;color:#DB2777;font-weight:700;padding:4px 14px;border-radius:999px;">😰 焦虑 0.70</span>
                </div>
                <div style="color:#9CA3AF; font-size:.8rem; margin-top:8px;">难点：悲伤↔愤怒存在交叉混淆（厌恶情绪归并所致）；焦虑类仅 267 条弱标注样本仍达 F1 0.70</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
