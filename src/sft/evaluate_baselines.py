# -*- coding: utf-8 -*-
"""
M3：BERT 与 BERT+RAG 基线在相同评测子集上的评测
================================================
与 evaluate_lora.py 同口径（full / hard / adv / user24），
结果合并进 reports/eval_metrics.json 的 bert / rag 键。

- bert：predict_batch 出指标 + 单条流式延迟（use_rag=False）
- rag ：逐条 predict(use_rag=True)，记录触发率、精判来源、端到端延迟；
        逐行写 CSV 支持断点续跑（API 路由昂贵且可能中途失败）

用法:
  python -m src.sft.evaluate_baselines --backend bert
  python -m src.sft.evaluate_baselines --backend rag [--subset hard]
"""
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.sft.evaluate_lora import build_hard_subset, compute_metrics, METRICS_OUT
from src.utils import load_config

TEST_CSV = Path("data/processed_emotion/test_cleaned_fixed_anx.csv")
TRAIN_CSV = Path("data/processed_emotion/train_augmented_v2_cleaned_fixed2.csv")
ADV_CSV = Path("data/sft/adv50.csv")


def load_subsets():
    test = pd.read_csv(TEST_CSV)
    hard = build_hard_subset(test)
    adv = pd.read_csv(ADV_CSV)
    user24 = pd.read_csv(TRAIN_CSV)
    user24 = user24[user24["source"].astype(str) == "user_corpus"]
    return {
        "full": test[["text", "label", "label_cn"]].reset_index(drop=True),
        "hard": hard[["text", "label", "label_cn"]].reset_index(drop=True),
        "adv": adv[["text", "label", "label_cn", "category"]].reset_index(drop=True),
        "user24": user24[["text", "label", "label_cn", "source"]].reset_index(drop=True),
    }


def finalize_metrics(out, lat_ms=None, vram=None, extra=None):
    m = compute_metrics(out, label_col="label", pred_col="pred")
    neu = out[out["label"] == -1]
    if len(neu):
        m["neutral_n"] = int(len(neu))
        m["neutral_overfire_rate"] = round(float((neu["pred"] >= 0).mean()), 4)
        m["neutral_pred_dist"] = neu["pred_cn"].value_counts().to_dict()
    if lat_ms:
        m["latency_single"] = {
            "n": len(lat_ms),
            "p50_ms": round(float(np.percentile(lat_ms, 50)), 1),
            "p95_ms": round(float(np.percentile(lat_ms, 95)), 1),
            "mean_ms": round(float(np.mean(lat_ms)), 1),
        }
    if vram:
        m["peak_vram_gb"] = vram
    if extra:
        m.update(extra)
    return m


def eval_bert(names):
    from src.deploy.emotion_predictor import get_emotion_predictor
    subsets = load_subsets()
    predictor = get_emotion_predictor()
    en2id = {en: i for i, en in enumerate(load_config()["emotion"]["labels"])}
    all_metrics = {}
    for name in names:
        df = subsets[name]
        print(f"\n===== BERT {name}: {len(df)} =====")
        # 指标（批量，无 RAG）
        batch_res = predictor.predict_batch(df["text"].tolist())
        out = df.copy()
        out["pred"] = [en2id[r["emotion"]] for r in batch_res]
        out["pred_cn"] = [r["emotion_cn"] for r in batch_res]
        out.to_csv(f"reports/eval_bert_{name}.csv", index=False, encoding="utf-8-sig")

        # 单条延迟（full 只测 300）
        lat_n = min(300, len(out))
        torch.cuda.reset_peak_memory_stats()
        lats = []
        for t in out["text"].iloc[:lat_n]:
            t0 = time.perf_counter()
            predictor.predict(str(t), use_rag=False)
            torch.cuda.synchronize()
            lats.append((time.perf_counter() - t0) * 1000)
        vram = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        m = finalize_metrics(out, lats, vram)
        print(json.dumps(m, ensure_ascii=False, indent=2))
        all_metrics[name] = m

    payload = json.loads(METRICS_OUT.read_text(encoding="utf-8")) if METRICS_OUT.exists() else {}
    payload["bert"] = all_metrics
    METRICS_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"指标写入 {METRICS_OUT}")


def eval_rag(names):
    from src.deploy.emotion_predictor import get_emotion_predictor
    subsets = load_subsets()
    predictor = get_emotion_predictor()
    en2id = {en: i for i, en in enumerate(load_config()["emotion"]["labels"])}
    all_metrics = {}
    for name in names:
        df = subsets[name]
        raw_path = Path(f"reports/eval_rag_{name}.csv")
        done_n = 0
        rows = []
        if raw_path.exists():  # 断点续跑
            old = pd.read_csv(raw_path)
            done_n = len(old)
            rows = old.to_dict("records")
            print(f"{name}: 续跑，已有 {done_n}/{len(df)}")

        triggers, lats_all, lats_trig = 0, [], []
        fails = 0
        for i in range(done_n, len(df)):
            t = str(df.iloc[i]["text"])
            t0 = time.perf_counter()
            r = predictor.predict(t, use_rag=True)
            dt = (time.perf_counter() - t0) * 1000
            lats_all.append(dt)
            trig = bool(r.get("refined", False))
            triggers += int(trig)
            if trig:
                lats_trig.append(dt)
            if r.get("refine_source") == "rag_failed":
                fails += 1
            rows.append({
                "text": t, "label": int(df.iloc[i]["label"]),
                "pred": en2id[r["emotion"]], "pred_cn": r["emotion_cn"],
                "refined": trig, "refine_source": r.get("refine_source", ""),
                "reason": r.get("reason", ""), "latency_ms": round(dt, 1),
            })
            if (i + 1) % 20 == 0 or i + 1 == len(df):
                pd.DataFrame(rows).to_csv(raw_path, index=False, encoding="utf-8-sig")
                print(f"  {name} {i+1}/{len(df)} 触发率 {triggers/(i+1-done_n+done_n):.2%} "
                      f"失败 {fails}", flush=True)

        out = pd.DataFrame(rows)
        # 续跑行的延迟也纳入统计
        if "latency_ms" in out.columns:
            all_lat = out["latency_ms"].dropna().tolist()
            trig_lat = out.loc[out["refined"] == True, "latency_ms"].dropna().tolist()
        else:
            all_lat, trig_lat = [], []
        n = len(out)
        trigger_rate = float(out["refined"].mean()) if "refined" in out else 0.0
        vram = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        extra = {
            "rag_trigger_rate": round(trigger_rate, 4),
            "rag_trigger_n": int(out["refined"].sum()) if "refined" in out else 0,
            "rag_failed_n": int((out["refine_source"] == "rag_failed").sum()),
            "latency_triggered": {
                "n": len(trig_lat),
                "p50_ms": round(float(np.percentile(trig_lat, 50)), 1) if trig_lat else None,
                "p95_ms": round(float(np.percentile(trig_lat, 95)), 1) if trig_lat else None,
                "mean_ms": round(float(np.mean(trig_lat)), 1) if trig_lat else None,
            },
        }
        m = finalize_metrics(out, all_lat, vram, extra)
        print(json.dumps(m, ensure_ascii=False, indent=2))
        all_metrics[name] = m

    payload = json.loads(METRICS_OUT.read_text(encoding="utf-8")) if METRICS_OUT.exists() else {}
    payload["rag"] = all_metrics
    METRICS_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"指标写入 {METRICS_OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["bert", "rag"], required=True)
    ap.add_argument("--subset", nargs="+",
                    default=["full", "hard", "adv", "user24"])
    args = ap.parse_args()
    if args.backend == "bert":
        eval_bert(args.subset)
    else:
        eval_rag(args.subset)


if __name__ == "__main__":
    main()
