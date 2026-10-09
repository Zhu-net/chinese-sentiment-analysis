# -*- coding: utf-8 -*-
"""
消融实验（零训练成本）：在当前合并模型上对比 system prompt 是否含六类定义
========================================================
评测对象：test 全部恐惧+焦虑样本 + 其余四类各抽样，看定义 rubric 能否
缓解 焦虑→恐惧 的边界坍塌，为 v2 重训（定义入 prompt）提供证据。
产物：reports/prompt_ablation.csv
"""
import sys
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import f1_score, accuracy_score

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.sft.prompts import SYSTEM_PROMPT, USER_PREFIX, EMOTIONS_CN
from src.sft.parse_output import parse_emotion_json

# 与 test 金标 LLM 审计 prompt（src/evaluate/llm_annotate_anxiety.py）逐字对齐
DEF_SYSTEM = (
    "你是中文社交媒体情绪分析专家。只输出 JSON，不要输出任何多余内容。\n"
    "六类情绪定义（互斥，按主导情绪标注）：\n"
    "- 开心：高兴、喜悦、满足、期待\n"
    "- 感激：感谢、感恩\n"
    "- 悲伤：难过、失落、无奈、沮丧、哀叹、委屈、孤独\n"
    "- 愤怒：生气、发怒、指责、辱骂、强烈不满、斥责他人/事物\n"
    "- 恐惧：害怕、惊恐、恐慌（对已发生危险的本能害怕）\n"
    "- 焦虑：紧张、担忧、不安、纠结、烦躁（对未发生事件/持续压力的焦灼，"
    "如考试、工作、健康担忧、等结果、睡不着觉、心慌、静不下来）\n"
    "判定要点：\n"
    "- 「担心、害怕、惶恐不安、心慌、静不下来、忐忑、纠结、压力、怕…」"
    "等对未来/未知的担忧→焦虑\n"
    "- 「崩溃、心塞、好累、好难过、好伤心」等对已发生事情的低落→悲伤\n"
    "- 「气死了、太过分、讨厌、骂」等对外指责→愤怒\n"
    "- 对已发生危险的本能害怕（如梦见被杀、地震）→恐惧；"
    "对可能发生坏事的担忧→焦虑\n"
    "情绪标签限定为：开心、感激、悲伤、愤怒、恐惧、焦虑。"
)


def make_prompt(tok, text, system):
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": f"{USER_PREFIX}「{text}」"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def gen_batch(model, tok, texts, system, bs=16):
    preds = []
    tok.padding_side = "left"
    for s in range(0, len(texts), bs):
        chunk = texts[s:s + bs]
        prompts = [make_prompt(tok, t, system) for t in chunk]
        inp = tok(prompts, return_tensors="pt", padding=True,
                  truncation=True, max_length=384).to(model.device)
        out = model.generate(**inp, max_new_tokens=48, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        for ids in out[:, inp["input_ids"].shape[1]:]:
            lid, cn, reason, valid = parse_emotion_json(
                tok.decode(ids, skip_special_tokens=True))
            preds.append(lid if valid else -1)
    return preds


def score(name, gold, pred):
    g, p = np.array(gold), np.array(pred)
    m = g >= 0
    g, p = g[m], p[m]
    valid = (np.array(pred)[m] >= 0)
    mf1 = f1_score(g, p, labels=list(range(6)), average="macro", zero_division=0)
    pc = f1_score(g, p, labels=list(range(6)), average=None, zero_division=0)
    print(f"\n[{name}] n={len(g)} Acc={accuracy_score(g,p):.4f} macroF1={mf1:.4f} "
          f"JSON={valid.mean():.4f}")
    print("  per-class:", {EMOTIONS_CN[i]: round(float(v), 3)
                           for i, v in enumerate(pc)})
    # 焦虑/恐惧专项
    for lab in (4, 5):
        mm = g == lab
        if mm.sum():
            print(f"  {EMOTIONS_CN[lab]} recall={(p[mm]==lab).mean():.3f} (n={mm.sum()})")
    return mf1


def main():
    test = pd.read_csv("data/processed_emotion/test_cleaned_fixed_anx.csv")
    rng = np.random.default_rng(42)
    parts = [test[test.label.isin([4, 5])]]
    for lab in [0, 1, 2, 3]:
        g = test[test.label == lab]
        parts.append(g.sample(min(100, len(g)), random_state=42))
    sub = pd.concat(parts).reset_index(drop=True)
    print(f"消融样本 {len(sub)}（恐惧 {int((sub.label==4).sum())}, "
          f"焦虑 {int((sub.label==5).sum())}）")

    tok = AutoTokenizer.from_pretrained("saved_models/qwen15b_emotion_merged")
    if tok.pad_token is None:
        tok.pad_token = tok.convert_ids_to_tokens(151643)
    model = AutoModelForCausalLM.from_pretrained(
        "saved_models/qwen15b_emotion_merged", dtype=torch.bfloat16,
        device_map="cuda:0").eval()

    texts = sub["text"].tolist()
    print("=== A: 原始 prompt（无定义）===")
    pred_a = gen_batch(model, tok, texts, SYSTEM_PROMPT)
    score("A 原始 prompt", sub.label.tolist(), pred_a)

    print("\n=== B: 增强 prompt（含六类定义）===")
    pred_b = gen_batch(model, tok, texts, DEF_SYSTEM)
    score("B 定义增强 prompt", sub.label.tolist(), pred_b)

    out = sub.copy()
    out["pred_plain"] = pred_a
    out["pred_def"] = pred_b
    out.to_csv("reports/prompt_ablation.csv", index=False, encoding="utf-8-sig")
    # 焦虑样本上的直接对比
    anx = out[out.label == 5]
    print(f"\n焦虑金标 {len(anx)}: 原始对 {int((anx.pred_plain==5).sum())} / "
          f"增强对 {int((anx.pred_def==5).sum())}；"
          f"原始→恐惧 {int((anx.pred_plain==4).sum())} / 增强→恐惧 {int((anx.pred_def==4).sum())}")
    fear = out[out.label == 4]
    print(f"恐惧金标 {len(fear)}: 原始对 {int((fear.pred_plain==4).sum())} / "
          f"增强对 {int((fear.pred_def==4).sum())}")


if __name__ == "__main__":
    main()
