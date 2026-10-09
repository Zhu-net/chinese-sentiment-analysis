# -*- coding: utf-8 -*-
"""
M3 §6.4：LoRA reason 质量专项审计
================================
从 reports/eval_lora_full.csv 分层抽 100 条合法预测，DeepSeek 当质检员逐条判：
1. support：理由是否支持输出的情绪标签（是/否）
2. hallucination：理由是否引入文本中不存在的事实（是/否）
长度合规率（15-30 字）本地直接统计。
断点续跑：reports/_reason_audit_progress.jsonl
产物：reports/reason_quality_audit.csv；指标写入 eval_metrics.json
"""
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.llm.llm_client import DeepSeekClient
from src.sft.evaluate_lora import METRICS_OUT

PRED_CSV = Path("reports/eval_lora_full.csv")
OUT_CSV = Path("reports/reason_quality_audit.csv")
PROGRESS = Path("reports/_reason_audit_progress.jsonl")
MIN_LEN, MAX_LEN = 15, 30
BATCH = 10

PROMPT = """你是情绪标注质检员。逐条核验模型给出的「情绪标签」和「判定理由」。

判定两个问题：
1. support：理由是否支持该情绪标签？（是=理由能推出该情绪；否=理由与标签无关或推相反结论）
2. hallucination：理由是否包含目标文本中不存在的事实？（是=无中生有；否=只依据文本）

只输出 JSON 数组：[{{"id": 序号, "support": "是/否", "hallucination": "是/否"}}]

待核验：
{items}
"""


def parse(text, expected):
    s = text.strip()
    if "```" in s:
        s = s.split("```")[1]
        if s.startswith("json"):
            s = s[4:]
    l, r = s.find("["), s.rfind("]")
    arr = json.loads(s[l:r + 1])
    out = {}
    for it in arr:
        i = int(it["id"])
        if i in expected:
            out[i] = (str(it["support"]).strip(), str(it["hallucination"]).strip())
    if set(out) != set(expected):
        raise ValueError("id 缺失")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(PRED_CSV)
    df = df[df["pred"] >= 0].copy()  # 只审计合法 JSON 预测
    # 按金标分层等比抽样（固定种子）
    sample = (df.groupby("label", group_keys=False)
                .apply(lambda g: g.sample(
                    min(round(len(g) / len(df) * args.n) or 1, len(g)),
                    random_state=42)))
    sample = sample.head(args.n).reset_index(drop=True)
    print(f"审计样本 {len(sample)} 条")

    if args.fresh and PROGRESS.exists():
        PROGRESS.unlink()
    done = {}
    if PROGRESS.exists():
        for line in PROGRESS.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            done[rec["id"]] = rec["v"]

    key = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))
    client = DeepSeekClient(api_key=key["rag"]["deepseek_api_key"])

    todo = [i for i in range(len(sample)) if i not in done]
    for start in range(0, len(todo), BATCH):
        idxs = todo[start:start + BATCH]
        items = "\n".join(
            f'{k+1}. 文本：{sample.iloc[i]["text"]} ｜ 标签：{sample.iloc[i]["pred_cn"]} '
            f'｜ 理由：{sample.iloc[i]["reason"]}'
            for k, i in enumerate(idxs))
        parsed = None
        for attempt in range(3):
            try:
                resp = client.generate(PROMPT.format(items=items),
                                       temperature=0.0, max_tokens=900)
                parsed = parse(resp, list(range(1, len(idxs) + 1)))
                break
            except Exception as e:
                print(f"批次解析失败({attempt+1}): {e}")
                time.sleep(2 * (attempt + 1))
        if parsed is None:
            print("批次三次失败，跳过（重跑本脚本续跑）")
            continue
        with PROGRESS.open("a", encoding="utf-8") as f:
            for k, i in enumerate(idxs):
                v = parsed[k + 1]
                done[i] = v
                f.write(json.dumps({"id": i, "v": v}, ensure_ascii=False) + "\n")
        print(f"已完成 {len(done)}/{len(sample)}", end="\r")

    if len(done) < len(sample):
        print(f"\n仅完成 {len(done)}/{len(sample)}，请重跑续跑")
        return

    sample["support"] = [done[i][0] for i in range(len(sample))]
    sample["hallucination"] = [done[i][1] for i in range(len(sample))]
    sample["reason_len"] = sample["reason"].map(lambda x: len("".join(str(x).split())))
    sample.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    n = len(sample)
    metrics = {
        "n": n,
        "label_reason_consistency": round(float((sample["support"] == "是").mean()), 4),
        "hallucination_rate": round(float((sample["hallucination"] == "是").mean()), 4),
        "length_compliance": round(float(
            sample["reason_len"].between(MIN_LEN, MAX_LEN).mean()), 4),
    }
    print("\n" + json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"明细: {OUT_CSV}")

    payload = json.loads(METRICS_OUT.read_text(encoding="utf-8")) if METRICS_OUT.exists() else {}
    payload.setdefault("lora", {})["reason_quality"] = metrics
    METRICS_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                           encoding="utf-8")


if __name__ == "__main__":
    main()
