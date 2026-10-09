# -*- coding: utf-8 -*-
"""
诊断：指定 epoch adapter 在 val 上的混淆矩阵（全量焦虑 + 各类分层抽样）
=================================================================
用于定位 macro-F1 短板（预期：均衡 SFT 导致焦虑过判）。
产物：reports/val_diag_epoch{N}.csv
用法: python -m src.sft.analyze_val_errors --epoch 3 [--per-class 250]
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.sft.prompts import build_messages, EMOTIONS_CN
from src.sft.parse_output import parse_emotion_json

BASE = "saved_models/Qwen2.5-1.5B-Instruct"
CKPT = Path("saved_models/qlora/qwen15b_emotion_r16")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--per-class", type=int, default=250)
    args = ap.parse_args()

    val = pd.DataFrame([json.loads(l) for l in
                        open("data/sft/sft_val.jsonl", encoding="utf-8")])
    # 焦虑全取（仅 40 条），其余各类抽 per_class
    rng = np.random.default_rng(42)
    parts = []
    for lab, g in val.groupby("label"):
        if lab == 5:
            parts.append(g)
        else:
            parts.append(g.sample(min(args.per_class, len(g)), random_state=42))
    sub = pd.concat(parts).reset_index(drop=True)
    print(f"诊断样本 {len(sub)}（金标分布 {sub.label.value_counts().sort_index().tolist()}）")

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.convert_ids_to_tokens(151643)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map="cuda:0",
        dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, CKPT / f"epoch{args.epoch}")
    model.eval()

    preds, raws = [], []
    B = 16
    tok.padding_side = "left"
    with torch.no_grad():
        for s in range(0, len(sub), B):
            texts = sub["text"].iloc[s:s + B].tolist()
            prompts = [tok.apply_chat_template(build_messages(t), tokenize=False,
                                               add_generation_prompt=True)
                       for t in texts]
            inp = tok(prompts, return_tensors="pt", padding=True).to(model.device)
            out = model.generate(**inp, max_new_tokens=64, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            for ids in out[:, inp["input_ids"].shape[1]:]:
                raw = tok.decode(ids, skip_special_tokens=True)
                lid, cn, reason, valid = parse_emotion_json(raw)
                preds.append(lid if valid else -1)
                raws.append(raw)
            print(f"{min(s+B, len(sub))}/{len(sub)}", end="\r")

    sub["pred"] = preds
    sub["pred_cn"] = [EMOTIONS_CN[p] if p >= 0 else "非法JSON" for p in preds]
    sub["raw"] = raws
    sub.to_csv(f"reports/val_diag_epoch{args.epoch}.csv", index=False,
               encoding="utf-8-sig")

    d = sub[sub.pred >= 0]
    print("\n分类报告（行=金标）：")
    print(classification_report(d["label"], d["pred"], labels=list(range(6)),
                                target_names=EMOTIONS_CN, zero_division=0))
    cm = confusion_matrix(d["label"], d["pred"], labels=list(range(6)))
    print("混淆矩阵（行=金标，列=预测）：")
    print("金标\\预测 " + " ".join(f"{c:>4s}" for c in EMOTIONS_CN))
    for i, row in enumerate(cm):
        print(f"{EMOTIONS_CN[i]:>6s}   " + " ".join(f"{v:>4d}" for v in row))
    # 焦虑误报来源
    fp = d[(d.label != 5) & (d.pred == 5)]
    print(f"\n焦虑误报 {len(fp)} 条，金标来源：",
          fp["label"].map(EMOTIONS_CN.__getitem__).value_counts().to_dict())


if __name__ == "__main__":
    main()
