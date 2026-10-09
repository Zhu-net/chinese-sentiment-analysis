# -*- coding: utf-8 -*-
"""SFT 训练 / 推理 / 评测共用的 ChatML 模板与标签定义（单点维护，保证训推一致）"""

EMOTIONS_CN = ["开心", "感激", "悲伤", "愤怒", "恐惧", "焦虑"]
# label id <-> 中文标签（与 config.yaml emotion.labels 顺序一致）
LABEL2CN = {i: cn for i, cn in enumerate(EMOTIONS_CN)}
CN2LABEL = {cn: i for i, cn in LABEL2CN.items()}

SYSTEM_PROMPT = (
    "你是中文社交媒体情绪分析专家。只输出 JSON，不要输出任何多余内容。\n"
    "情绪标签限定为：开心、感激、悲伤、愤怒、恐惧、焦虑。"
)

USER_PREFIX = "分析以下文本的主导情绪：\n"


def build_user(text: str) -> str:
    return f"{USER_PREFIX}「{text}」"


def build_assistant(label_cn: str, reason: str) -> str:
    import json
    return json.dumps({"emotion": label_cn, "reason": reason}, ensure_ascii=False)


def build_messages(text: str, label_cn: str | None = None, reason: str | None = None):
    """训练/评测构造消息；label_cn 为 None 时只返回 prompt 侧消息"""
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user(text)},
    ]
    if label_cn is not None:
        msgs.append({"role": "assistant", "content": build_assistant(label_cn, reason or "")})
    return msgs
