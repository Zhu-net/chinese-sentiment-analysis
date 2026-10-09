# -*- coding: utf-8 -*-
"""
M1 步骤 2：DeepSeek 批量反推 reason（给定文本+金标，生成 15-30 字理由）
=====================================================================
- 断点续跑：data/sft/reason_progress.jsonl，每行 {"id": 行号, "reason": "..."}
- 每批 10 条，要求只依据文本、与金标一致；失败批次自动重试 3 次
- 长度不合规(15-30字)或缺失的条目汇总后进入修复轮，直到合规率>=99%或无法继续
- 最终拼装 data/sft/sft_train_4800.jsonl（含 ChatML messages，训练直接可用）
- 另导出每类 20 条抽检样本 data/sft/reason_review_sample.csv 供人工核验

用法:
  python -m src.sft.gen_reasons.py [--fresh] [--repair-only]
"""
import sys
import json
import time
import argparse
from pathlib import Path

import pandas as pd
import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.llm.llm_client import DeepSeekClient
from src.sft.prompts import LABEL2CN, build_messages

OUT = Path("data/sft")
MANIFEST = OUT / "sample_manifest.csv"
PROGRESS = OUT / "reason_progress.jsonl"
TRAIN_JSONL = OUT / "sft_train_4800.jsonl"
REVIEW_CSV = OUT / "reason_review_sample.csv"
CONFIG = Path("config/config.yaml")

BATCH = 10
MIN_LEN, MAX_LEN = 15, 30

PROMPT_TMPL = """你是中文社交媒体情绪分析专家。下面每条文本都有【已确定】的情绪金标。
请为每条文本写一句判定理由，说明文本中的哪些表达体现了该情绪。

要求：
1. 理由必须与给定金标一致，不得质疑或改判
2. 只依据文本中实际出现的内容，不得添加文本没有的事实或背景
3. 长度严格 {min_len}-{max_len} 个汉字，一句话、陈述句
4. 不要出现"金标/标签/情绪为"这类元话语，直接描述依据
5. 只输出 JSON 数组，不要任何多余文字：[{{"id": 序号, "reason": "理由"}}]

待处理文本：
{items}
"""


def build_prompt(batch):
    items = "\n".join(f'{r["_lid"]}. 【{r["label_cn"]}】{r["text"]}'
                      for _, r in batch.iterrows())
    return PROMPT_TMPL.format(min_len=MIN_LEN, max_len=MAX_LEN, items=items)


def parse_response(text, expected_ids):
    s = text.strip()
    if "```" in s:
        parts = s.split("```")
        s = parts[1]
        if s.startswith("json"):
            s = s[4:]
    l, r = s.find("["), s.rfind("]")
    if l == -1 or r == -1:
        raise ValueError("响应中未找到 JSON 数组")
    arr = json.loads(s[l:r + 1])
    out = {}
    for item in arr:
        lid = int(item["id"])
        reason = str(item.get("reason", "")).strip()
        # 去除常见包裹引号/句号尾差异不处理，保持生成原貌
        if lid in expected_ids and reason:
            out[lid] = reason
    missing = set(expected_ids) - set(out.keys())
    if missing:
        raise ValueError(f"缺少 id: {sorted(missing)}")
    return out


def reason_len(reason: str) -> int:
    """理由长度：去空白后字符数"""
    return len("".join(reason.split()))


def call_batch(client, batch, max_tokens=1200, temperature=0.3):
    prompt = build_prompt(batch)
    last_err = None
    for attempt in range(3):
        try:
            resp = client.generate(prompt, temperature=temperature, max_tokens=max_tokens)
            return parse_response(resp, batch["_lid"].tolist())
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    print(f"\n批次三次失败，跳过: {last_err}")
    return None


def load_progress():
    done = {}
    if PROGRESS.exists():
        for line in PROGRESS.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            done[rec["id"]] = rec["reason"]
    return done


def append_progress(recs):
    with PROGRESS.open("a", encoding="utf-8") as f:
        for idx, reason in recs:
            f.write(json.dumps({"id": int(idx), "reason": reason},
                               ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="忽略进度重跑")
    ap.add_argument("--repair-only", action="store_true", help="只修复长度不合规条目")
    args = ap.parse_args()

    manifest = pd.read_csv(MANIFEST)
    n = len(manifest)
    print(f"manifest: {n} 条")

    if args.fresh and PROGRESS.exists():
        PROGRESS.unlink()
    done = {} if args.fresh else load_progress()
    print(f"已有进度: {len(done)} 条")

    # API key：环境变量优先，其次 config.yaml
    api_key = None
    if CONFIG.exists():
        api_key = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) \
            .get("rag", {}).get("deepseek_api_key")
    client = DeepSeekClient(api_key=api_key)

    # ---- 主生成轮 / 修复轮 ----
    def valid(idx):
        r = done.get(idx)
        return r is not None and MIN_LEN <= reason_len(r) <= MAX_LEN

    rounds = 0
    while True:
        rounds += 1
        if args.repair_only and rounds > 1:
            break
        todo = [i for i in range(n) if not valid(i)]
        if not todo:
            break
        print(f"\n第 {rounds} 轮：待生成/修复 {len(todo)} 条")
        t0 = time.time()
        nb = (len(todo) - 1) // BATCH + 1
        bi = 0
        for start in range(0, len(todo), BATCH):
            bi += 1
            idxs = todo[start:start + BATCH]
            batch = manifest.iloc[idxs].copy().reset_index(drop=True)
            batch["_lid"] = range(1, len(batch) + 1)
            parsed = call_batch(client, batch)
            if parsed is None:
                continue
            recs = [(idxs[k - 1], reason) for k, reason in parsed.items()]
            append_progress(recs)
            for idx, reason in recs:
                done[idx] = reason
            if bi % 10 == 0 or bi == nb:
                print(f"  批次 {bi}/{nb}，累计合规 {sum(valid(i) for i in range(n))}/{n}，"
                      f"用时 {time.time()-t0:.0f}s")

    # ---- 合规统计 ----
    missing = [i for i in range(n) if i not in done]
    bad_len = [i for i in range(n) if i in done and not valid(i)]
    good = n - len(missing) - len(bad_len)
    print("\n" + "=" * 60)
    print(f"合规(15-{MAX_LEN}字): {good}/{n} = {good/n:.2%}")
    print(f"缺失: {len(missing)}；长度不合规: {len(bad_len)}")
    if bad_len:
        lens = [reason_len(done[i]) for i in bad_len]
        print(f"  不合规长度分布: min={min(lens)} max={max(lens)}")

    if missing:
        print("仍有缺失条目，无法拼装，请重跑本脚本续跑")
        return

    # ---- 拼装 SFT jsonl ----
    with TRAIN_JSONL.open("w", encoding="utf-8") as f:
        for i, r in manifest.iterrows():
            reason = done[i]
            label_cn = LABEL2CN[int(r["label"])]
            rec = {
                "text": str(r["text"]),
                "label": int(r["label"]),
                "label_cn": label_cn,
                "reason": reason,
                "source": str(r["source"]),
                "tag": str(r["tag"]),
                "messages": build_messages(str(r["text"]), label_cn, reason),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"写出 {TRAIN_JSONL}")

    # ---- 每类 20 条人工抽检样本 ----
    review = manifest.copy()
    review["reason"] = [reason_len and done[i] for i in range(n)]
    review["reason_len"] = [reason_len(done[i]) for i in range(n)]
    sample = (review.groupby("label_cn", group_keys=False)
              .apply(lambda g: g.sample(min(20, len(g)), random_state=42)))
    sample.to_csv(REVIEW_CSV, index=False, encoding="utf-8-sig")
    print(f"写出 {REVIEW_CSV}（{len(sample)} 条，供人工抽检）")


if __name__ == "__main__":
    main()
