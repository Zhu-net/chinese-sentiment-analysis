"""
自定义 CSS 样式模块
打造明亮、高级的视觉风格：
- 柔和渐变背景（浅蓝紫 → 暖白）
- 毛玻璃卡片 + 柔和阴影
- 渐变主按钮 / 胶囊 Tab
- 渐变色标题文字
"""

# 全局样式
GLOBAL_CSS = """
<style>
/* ===== 字体 ===== */
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700;900&display=swap');

html, body, [class*="css"], .stApp, button, input, textarea {
    font-family: 'Noto Sans SC', 'Microsoft YaHei', sans-serif !important;
}

/* ===== 页面背景：明亮柔和的渐变 ===== */
.stApp {
    background: linear-gradient(135deg, #F3F6FF 0%, #FBF6FF 45%, #EFFAF8 100%) !important;
    background-attachment: fixed !important;
}

/* ===== 主内容区 ===== */
.block-container {
    padding-top: 1.5rem !important;
    max-width: 1180px !important;
}

/* ===== 隐藏 Streamlit 默认菜单/页脚 ===== */
#MainMenu, footer, header { visibility: hidden !important; }

/* ===== Hero 标题区 ===== */
.hero-section {
    background: rgba(255, 255, 255, 0.75);
    backdrop-filter: blur(14px);
    border-radius: 24px;
    padding: 34px 40px 30px 40px;
    margin-bottom: 26px;
    border: 1px solid rgba(108, 92, 231, 0.10);
    box-shadow: 0 10px 34px rgba(108, 92, 231, 0.09);
    text-align: center;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, #EDEBFF 0%, #E0F2FE 100%);
    color: #6C5CE7;
    font-size: 0.85rem;
    font-weight: 700;
    padding: 6px 18px;
    border-radius: 999px;
    letter-spacing: 2px;
    margin-bottom: 14px;
    border: 1px solid rgba(108, 92, 231, 0.15);
}
.hero-title {
    background: linear-gradient(90deg, #6C5CE7 0%, #3B82F6 50%, #10B981 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem;
    font-weight: 900;
    letter-spacing: 2px;
    margin: 0 0 10px 0;
}
.hero-sub {
    color: #6B7280;
    font-size: 1rem;
    margin: 0;
    letter-spacing: 1px;
}
.hero-stats {
    display: flex;
    justify-content: center;
    gap: 42px;
    margin-top: 20px;
    padding-top: 18px;
    border-top: 1px dashed rgba(108, 92, 231, 0.18);
}
.hero-stat { text-align: center; }
.hero-stat-num {
    background: linear-gradient(135deg, #6C5CE7, #3B82F6);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 1.45rem;
    font-weight: 900;
}
.hero-stat-label { color: #9CA3AF; font-size: 0.8rem; margin-top: 2px; }

/* ===== 卡片通用样式 ===== */
.glass-card {
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(12px);
    border-radius: 18px;
    padding: 22px 26px;
    border: 1px solid rgba(108, 92, 231, 0.09);
    box-shadow: 0 6px 22px rgba(108, 92, 231, 0.07);
    margin-bottom: 16px;
}

/* ===== 分区标题 ===== */
.section-title {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 1.15rem;
    font-weight: 700;
    color: #2D3142;
    margin: 4px 0 14px 0;
}
.section-title .dot {
    width: 8px; height: 22px;
    border-radius: 4px;
    background: linear-gradient(180deg, #6C5CE7, #3B82F6);
    display: inline-block;
}

/* ===== 单条预测结果卡 ===== */
.result-card {
    border-radius: 22px;
    padding: 30px 34px;
    text-align: center;
    animation: floatIn .5s ease;
}
.result-card.positive {
    background: linear-gradient(150deg, #FFFFFF 0%, #ECFDF5 100%);
    border: 1px solid rgba(16, 185, 129, 0.18);
    box-shadow: 0 10px 30px rgba(16, 185, 129, 0.13);
}
.result-card.negative {
    background: linear-gradient(150deg, #FFFFFF 0%, #FFF1F2 100%);
    border: 1px solid rgba(244, 63, 94, 0.16);
    box-shadow: 0 10px 30px rgba(244, 63, 94, 0.11);
}
.result-emoji { font-size: 3.6rem; line-height: 1.2; }
.result-label {
    font-size: 1.5rem;
    font-weight: 900;
    margin-top: 6px;
    letter-spacing: 4px;
}
.result-card.positive .result-label { color: #059669; }
.result-card.negative .result-label { color: #E11D48; }
.result-conf { color: #6B7280; font-size: 0.92rem; margin-top: 6px; }
@keyframes floatIn {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ===== 置信度进度条 ===== */
.conf-bar {
    height: 12px;
    background: #EEF0F7;
    border-radius: 8px;
    overflow: hidden;
    margin-top: 16px;
}
.conf-fill {
    height: 100%;
    border-radius: 8px;
    transition: width .8s cubic-bezier(.22,1,.36,1);
}
.conf-fill.positive { background: linear-gradient(90deg, #10B981, #6EE7B7); }
.conf-fill.negative { background: linear-gradient(90deg, #F43F5E, #FDA4AF); }

/* ===== 概率对比条 ===== */
.prob-row { display: flex; align-items: center; gap: 12px; margin: 10px 0; }
.prob-name { width: 52px; font-weight: 700; font-size: 0.92rem; }
.prob-name.pos { color: #059669; }
.prob-name.neg { color: #E11D48; }
.prob-track { flex: 1; height: 10px; background: #EEF0F7; border-radius: 6px; overflow: hidden; }
.prob-fill { height: 100%; border-radius: 6px; }
.prob-fill.pos { background: linear-gradient(90deg, #10B981, #6EE7B7); }
.prob-fill.neg { background: linear-gradient(90deg, #F43F5E, #FDA4AF); }
.prob-val { width: 68px; text-align: right; color: #6B7280; font-size: 0.88rem; font-variant-numeric: tabular-nums; }

/* ===== 按钮 ===== */
.stButton > button {
    border-radius: 12px !important;
    font-weight: 700 !important;
    transition: all .22s ease !important;
    border: none !important;
}
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #6C5CE7 0%, #8B7CF6 60%, #3B82F6 130%) !important;
    color: #FFFFFF !important;
    padding: 10px 34px !important;
    font-size: 1.02rem !important;
    box-shadow: 0 8px 22px rgba(108, 92, 231, 0.32) !important;
}
.stButton > button[kind="secondary"], .stButton > button[data-testid="stBaseButton-secondary"] {
    background: #FFFFFF !important;
    color: #6C5CE7 !important;
    border: 1.5px solid rgba(108, 92, 231, 0.35) !important;
    box-shadow: 0 4px 14px rgba(108, 92, 231, 0.10) !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; filter: brightness(1.04); }
.stButton > button:active { transform: translateY(0) !important; }

/* ===== 示例文本快捷按钮（胶囊） ===== */
.chip-btn button {
    background: rgba(255,255,255,0.9) !important;
    color: #4B5563 !important;
    border: 1px solid rgba(108,92,231,0.22) !important;
    border-radius: 999px !important;
    padding: 6px 18px !important;
    font-size: 0.86rem !important;
    font-weight: 500 !important;
    box-shadow: 0 3px 10px rgba(108,92,231,0.07) !important;
}
.chip-btn button:hover {
    border-color: #6C5CE7 !important;
    color: #6C5CE7 !important;
    transform: translateY(-1px) !important;
}

/* ===== Tabs 胶囊样式 ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    background: rgba(255,255,255,0.8);
    border-radius: 16px;
    padding: 8px;
    box-shadow: 0 4px 16px rgba(108,92,231,0.08);
    border: 1px solid rgba(108,92,231,0.08);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 11px;
    padding: 10px 26px;
    font-weight: 700;
    color: #6B7280;
    background: transparent;
}
.stTabs [data-baseweb="tab"]:hover { color: #6C5CE7; background: rgba(108,92,231,0.06); }
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6C5CE7 0%, #8B7CF6 100%) !important;
    color: #FFFFFF !important;
    box-shadow: 0 6px 16px rgba(108,92,231,0.30);
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* ===== 输入框 ===== */
.stTextArea textarea, .stTextInput input {
    border-radius: 14px !important;
    border: 1.5px solid rgba(108,92,231,0.16) !important;
    background: #FFFFFF !important;
    box-shadow: 0 3px 12px rgba(108,92,231,0.05) !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #6C5CE7 !important;
    box-shadow: 0 0 0 3px rgba(108,92,231,0.12) !important;
}

/* ===== 指标卡 ===== */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.9);
    border-radius: 16px;
    padding: 18px 22px;
    border: 1px solid rgba(108,92,231,0.09);
    box-shadow: 0 5px 18px rgba(108,92,231,0.07);
}
[data-testid="stMetricValue"] { font-weight: 900; }
[data-testid="stMetricLabel"] { color: #6B7280; }

/* ===== 侧边栏 ===== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFFFFF 0%, #F5F3FF 100%) !important;
    border-right: 1px solid rgba(108,92,231,0.08);
}
[data-testid="stSidebar"] .stMetric {
    background: #FFFFFF;
    box-shadow: 0 4px 14px rgba(108,92,231,0.09);
}

/* ===== 数据表格 ===== */
.stDataFrame { border-radius: 14px; overflow: hidden; box-shadow: 0 5px 18px rgba(108,92,231,0.07); }

/* ===== Expander ===== */
.streamlit-expanderHeader, [data-testid="stExpander"] summary {
    border-radius: 12px !important;
    font-weight: 600;
}

/* ===== 微调间距 ===== */
.stMarkdown { line-height: 1.65; }
hr { border: none; border-top: 1px dashed rgba(108,92,231,0.2); margin: 20px 0; }
</style>
"""

# 侧边栏品牌区
SIDEBAR_HEADER_HTML = """
<div style="text-align:center; padding: 6px 0 14px 0;">
    <div style="font-size: 2.4rem;">🧠</div>
    <div style="
        background: linear-gradient(90deg, #6C5CE7, #3B82F6);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.15rem; font-weight: 900; letter-spacing: 2px;
        margin-top: 4px;">SentimentLens</div>
    <div style="color:#9CA3AF; font-size:0.78rem; letter-spacing:3px; margin-top:2px;">情感分析工作台</div>
</div>
"""


def render_hero():
    """渲染 Hero 头部（情绪 6 分类主题）"""
    return f"""
    <div class="hero-section">
        <div class="hero-badge">🌈 细粒度情绪识别 · BERT 微调 · 双任务体系</div>
        <h1 class="hero-title">中文评论情绪分析系统</h1>
        <p class="hero-sub">超越正/负面：识别开心 · 感激 · 悲伤 · 愤怒 · 恐惧 · 焦虑 六种细腻情绪</p>
        <div class="hero-stats">
            <div class="hero-stat">
                <div class="hero-stat-num">6 类</div>
                <div class="hero-stat-label">细粒度情绪</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-num">44K</div>
                <div class="hero-stat-label">训练语料</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-num">Top-3</div>
                <div class="hero-stat-label">情绪候选</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-num">&lt;100ms</div>
                <div class="hero-stat-label">响应延迟</div>
            </div>
        </div>
    </div>
    """


def result_card_html(label_id: int, confidence: float) -> str:
    """生成单条预测结果卡片 HTML"""
    pos = label_id == 1
    emoji = "😊" if pos else "😞"
    label = "正面情感" if pos else "负面情感"
    cls = "positive" if pos else "negative"
    pct = confidence * 100
    return f"""
    <div class="result-card {cls}">
        <div class="result-emoji">{emoji}</div>
        <div class="result-label">{label}</div>
        <div class="result-conf">模型置信度 {pct:.2f}%</div>
        <div class="conf-bar">
            <div class="conf-fill {cls}" style="width: {pct:.1f}%;"></div>
        </div>
    </div>
    """


def prob_rows_html(probabilities: dict) -> str:
    """生成概率分布对比条 HTML"""
    rows = ""
    for name, val in probabilities.items():
        cls = "pos" if name == "正面" else "neg"
        pct = val * 100
        rows += f"""
        <div class="prob-row">
            <div class="prob-name {cls}">{name}</div>
            <div class="prob-track">
                <div class="prob-fill {cls}" style="width: {pct:.1f}%;"></div>
            </div>
            <div class="prob-val">{pct:.2f}%</div>
        </div>
        """
    return f'<div class="glass-card" style="margin-top:14px;">{rows}</div>'


# ===== 6 类情绪视觉系统 =====
# 每种情绪：浅色渐变背景 / 主色（文字、进度条）/ 英文键
EMOTION_THEME = {
    "开心": {"main": "#D97706", "bg_from": "#FFFBEB", "bg_to": "#FEF3C7", "bar": "linear-gradient(90deg,#F59E0B,#FCD34D)"},
    "感激": {"main": "#059669", "bg_from": "#ECFDF5", "bg_to": "#D1FAE5", "bar": "linear-gradient(90deg,#10B981,#6EE7B7)"},
    "悲伤": {"main": "#2563EB", "bg_from": "#EFF6FF", "bg_to": "#DBEAFE", "bar": "linear-gradient(90deg,#3B82F6,#93C5FD)"},
    "愤怒": {"main": "#E11D48", "bg_from": "#FFF1F2", "bg_to": "#FFE4E6", "bar": "linear-gradient(90deg,#EF4444,#FDA4AF)"},
    "恐惧": {"main": "#7C3AED", "bg_from": "#F5F3FF", "bg_to": "#EDE9FE", "bar": "linear-gradient(90deg,#8B5CF6,#C4B5FD)"},
    "焦虑": {"main": "#DB2777", "bg_from": "#FDF2F8", "bg_to": "#FCE7F3", "bar": "linear-gradient(90deg,#EC4899,#F9A8D4)"},
}
POLARITY_STYLE = {"正面": ("#059669", "#ECFDF5"), "负面": ("#E11D48", "#FFF1F2")}


def emotion_card_html(emotion_cn: str, emoji: str, confidence: float, polarity: str) -> str:
    """6 类情绪主结果卡片（每种情绪专属配色）"""
    theme = EMOTION_THEME.get(emotion_cn, EMOTION_THEME["开心"])
    pct = confidence * 100
    pol_color, pol_bg = POLARITY_STYLE.get(polarity, ("#6B7280", "#F3F4F6"))
    return f"""
    <div class="result-card" style="
        background: linear-gradient(150deg, {theme['bg_from']} 0%, {theme['bg_to']} 100%);
        border: 1px solid {theme['main']}2E;
        box-shadow: 0 10px 30px {theme['main']}1F;
    ">
        <div class="result-emoji">{emoji}</div>
        <div class="result-label" style="color: {theme['main']};">{emotion_cn}</div>
        <div style="margin-top:10px;">
            <span style="display:inline-block; background:{pol_bg}; color:{pol_color};
                font-size:.8rem; font-weight:700; padding:3px 14px; border-radius:999px;
                border:1px solid {pol_color}22;">整体倾向 · {polarity}</span>
        </div>
        <div class="result-conf" style="margin-top:10px;">模型置信度 {pct:.2f}%</div>
        <div class="conf-bar">
            <div style="height:100%; width:{pct:.1f}%; border-radius:8px; background:{theme['bar']};
                transition: width .8s cubic-bezier(.22,1,.36,1);"></div>
        </div>
    </div>
    """


def emotion_prob_rows_html(probabilities: dict) -> str:
    """6 类情绪概率分布条（按概率降序，每类专属色）"""
    rows = ""
    for name, val in sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True):
        theme = EMOTION_THEME.get(name, EMOTION_THEME["开心"])
        pct = val * 100
        rows += f"""
        <div class="prob-row">
            <div class="prob-name" style="width:52px; color:{theme['main']};">{name}</div>
            <div class="prob-track">
                <div style="height:100%; width:{pct:.2f}%; border-radius:6px;
                    background:{theme['bar']}; min-width:{4 if pct > 0 else 0}px;"></div>
            </div>
            <div class="prob-val">{pct:.2f}%</div>
        </div>
        """
    return f'<div class="glass-card" style="margin-top:14px;">{rows}</div>'
