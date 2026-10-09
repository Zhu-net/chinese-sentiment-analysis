# -*- coding: utf-8 -*-
"""
M3：LoRA 合并模型在三个评测子集上的评测
======================================
子集：
  full : test_cleaned_fixed_anx.csv 全量 4318
  hard : 278 悲伤/愤怒混淆 + 60 焦虑混淆，norm 对齐现 test 金标后去重
  adv  : data/sft/adv50.csv（34 条有金标 + 16 条无情绪，另算情绪过报率）
  user24: 训练集 user_corpus 24 条（train-seen 诊断，单列不进主表）

指标：Acc / macro-F1 / per-class F1（bootstrap 95% CI）、valid-JSON 率、
      单条流式延迟 P50/P95（full 测前 300 条，其余全测）、峰值显存。
产物：
  reports/eval_subsets/hard_subset.csv
  reports/eval_lora_<subset>.csv（逐条预测）
  reports/eval_metrics.json（key=lora，可与 bert/rag 结果合并）

用法: python -m src.sft.evaluate_lora [--model PATH] [--latency-n 300]
"""
import sys
import re
import json
import time
import argparse
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

from src.sft.prompts import build_messages, EMOTIONS_CN
from src.sft.parse_output import parse_emotion_json

TEST_CSV = Path("data/processed_emotion/test_cleaned_fixed_anx.csv")
TRAIN_CSV = Path("data/processed_emotion/train_augmented_v2_cleaned_fixed2.csv")
ADV_CSV = Path("data/sft/adv50.csv")
SUBSET_DIR = Path("reports/eval_subsets")
METRICS_OUT = Path("reports/eval_metrics.json")
MAX_LEN = 256
MAX_NEW_TOKENS = 64


def normalize(t):
    t = str(t)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"@[\w\u4e00-\u9fa5]+", " ", t)
    t = re.sub(r"#.*?#", " ", t)
    t = re.sub(r"[\s\W_]+", "", t, flags=re.UNICODE)
    return t.lower()


def build_hard_subset(test_df):
    """278 + 60 混淆文件 → norm 对齐现 test 金标 → 去重"""
    test = test_df.copy()
    test["_n"] = test["text"].map(normalize)
    gold = test.set_index("_n")[["text", "label", "label_cn"]]

    parts = []
    c278 = pd.read_csv("reports/confusion_review_llm_annotated.csv")
    c278n = c278["文本"].map(normalize)
    parts.append(pd.DataFrame({"_n": c278n, "src": "sad_angry_278"}))
    c60 = pd.read_csv("reports/anxiety_confusion_llm_annotated.csv")
    c60n = c60["text"].map(normalize)
    parts.append(pd.DataFrame({"_n": c60n, "src": "anxiety_60"}))

    cand = pd.concat(parts).drop_duplicates(subset="_n")
    hit = cand.merge(gold, on="_n", how="inner")
    hit = hit.drop_duplicates(subset="_n").reset_index(drop=True)
    out = hit[["text", "label", "label_cn"]].copy()
    out["src"] = hit["src"]
    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(SUBSET_DIR / "hard_subset.csv", index=False, encoding="utf-8-sig")
    print(f"hard 子集: 候选 {len(cand)}，现存 test 命中 {len(out)} "
          f"(278 存 {int(c278n.isin(set(test['_n'])).sum())}, "
          f"60 存 {int(c60n.isin(set(test['_n'])).sum())})")
    return out


def bootstrap_ci(gold, pred, metric_fn, b=1000, seed=42):
    rng = np.random.default_rng(seed)
    gold, pred = np.array(gold), np.array(pred)
    n = len(gold)
    vals = []
    for _ in range(b):
        idx = rng.integers(0, n, n)
        vals.append(metric_fn(gold[idx], pred[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def compute_metrics(df, label_col="gold", pred_col="pred"):
    """有金标行（label>=0）的指标；pred=-1 表示非法 JSON，按错误计"""
    d = df[df[label_col] >= 0]
    g, p = d[label_col].tolist(), d[pred_col].tolist()
    n = len(d)
    valid = [x for x in df[pred_col] if x >= 0]
    acc = accuracy_score(g, p)
    mf1 = f1_score(g, p, labels=list(range(6)), average="macro", zero_division=0)
    pc = f1_score(g, p, labels=list(range(6)), average=None, zero_division=0)
    ci_acc = bootstrap_ci(g, p, lambda a, b: accuracy_score(a, b))
    ci_f1 = bootstrap_ci(
        g, p, lambda a, b: f1_score(a, b, labels=list(range(6)),
                                    average="macro", zero_division=0))
    return {
        "n_labeled": n, "n_total": len(df),
        "acc": round(float(acc), 4), "acc_ci95": ci_acc,
        "macro_f1": round(float(mf1), 4), "macro_f1_ci95": ci_f1,
        "per_class_f1": {EMOTIONS_CN[i]: round(float(v), 4)
                         for i, v in enumerate(pc)},
        "valid_json_rate": round(len(valid) / len(df), 4),
    }


@torch.no_grad()
def run_inference(model, tokenizer, texts, batch_size=16):
    """批量贪心生成，返回 [(pred_id, cn, reason, raw, valid)]"""
    results = []
    tokenizer.padding_side = "left"
    for start in range(0, len(texts), batch_size):
        chunk = [str(t) for t in texts[start:start + batch_size]]
        prompts = [tokenizer.apply_chat_template(
            build_messages(t), tokenize=False, add_generation_prompt=True)
            for t in chunk]
        inp = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_LEN).to(model.device)
        out = model.generate(**inp, max_new_tokens=MAX_NEW_TOKENS,
                             do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
        gen = out[:, inp["input_ids"].shape[1]:]
        for ids in gen:
            raw = tokenizer.decode(ids, skip_special_tokens=True)
            lid, cn, reason, valid = parse_emotion_json(raw)
            results.append((lid if valid else -1, cn or "", reason, raw, valid))
    tokenizer.padding_side = "right"
    return results


@torch.no_grad()
def latency_benchmark(model, tokenizer, texts, n=300):
    """单条流式延迟（与生产单条接口口径一致）"""
    texts = texts[:n]
    ts = []
    tokenizer.padding_side = "left"
    for t in texts:
        prompt = tokenizer.apply_chat_template(
            build_messages(str(t)), tokenize=False, add_generation_prompt=True)
        inp = tokenizer(prompt, return_tensors="pt").to(model.device)
        t0 = time.perf_counter()
        out = model.generate(**inp, max_new_tokens=MAX_NEW_TOKENS,
                             do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
        _ = out[0][inp["input_ids"].shape[1]:].cpu()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return {"n": len(ts), "p50_ms": round(float(np.percentile(ts, 50)), 1),
            "p95_ms": round(float(np.percentile(ts, 95)), 1),
            "mean_ms": round(float(np.mean(ts)), 1)}


def eval_subset(name, df, model, tokenizer, latency_n):
    print(f"\n===== 评测子集 {name}: {len(df)} 条 =====")
    res = run_inference(model, tokenizer, df["text"].tolist())
    pred = [r[0] for r in res]
    out = df.copy()
    out["pred"] = pred
    out["pred_cn"] = [r[1] for r in res]
    out["reason"] = [r[2] for r in res]
    out["raw"] = [r[3] for r in res]
    out.to_csv(f"reports/eval_lora_{name}.csv", index=False, encoding="utf-8-sig")

    m = compute_metrics(out, label_col="label", pred_col="pred")
    # adv 无情绪行：情绪过报率 = 合法输出任一情绪的比例
    neu = out[out["label"] == -1]
    if len(neu):
        m["neutral_n"] = int(len(neu))
        m["neutral_overfire_rate"] = round(
            float((neu["pred"] >= 0).mean()), 4)
        m["neutral_pred_dist"] = neu["pred_cn"].value_counts().to_dict()

    torch.cuda.reset_peak_memory_stats()
    m["latency_single"] = latency_benchmark(
        model, tokenizer, out["text"].tolist(),
        n=min(latency_n, len(out)))
    m["peak_vram_gb"] = round(
        torch.cuda.max_memory_allocated() / 1024**3, 2)
    print(json.dumps(m, ensure_ascii=False, indent=2))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="saved_models/qwen15b_emotion_merged")
    ap.add_argument("--latency-n", type=int, default=300)
    args = ap.parse_args()

    test = pd.read_csv(TEST_CSV)
    hard = build_hard_subset(test)
    adv = pd.read_csv(ADV_CSV)
    train = pd.read_csv(TRAIN_CSV)
    user24 = train[train["source"].astype(str) == "user_corpus"].copy()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(151643)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    # warmup
    _ = run_inference(model, tokenizer, ["预热模型。"], batch_size=1)

    subsets = {
        "full": test[["text", "label", "label_cn"]],
        "hard": hard[["text", "label", "label_cn"]],
        "adv": adv[["text", "label", "label_cn", "category"]],
        "user24": user24[["text", "label", "label_cn", "source"]],
    }
    all_metrics = {}
    for name, df in subsets.items():
        all_metrics[name] = eval_subset(
            name, df.reset_index(drop=True), model, tokenizer, args.latency_n)

    payload = {}
    if METRICS_OUT.exists():
        payload = json.loads(METRICS_OUT.read_text(encoding="utf-8"))
    payload["lora"] = {"model": args.model, **all_metrics}
    METRICS_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"\n指标已写入 {METRICS_OUT}")


if __name__ == "__main__":
    main()
