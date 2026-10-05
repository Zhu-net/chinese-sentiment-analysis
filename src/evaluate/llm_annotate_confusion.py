# -*- coding: utf-8 -*-
"""
悲伤↔愤怒 强信号混淆样本 —— LLM 预标注（供人工复核）
====================================================
输入: reports/confusion_sad_angry_review.csv 中
      「双模型一致误判=是 且 模型置信度>0.7」的样本（默认 278 条）
输出: reports/confusion_review_llm_annotated.csv
      - LLM判定: 标签正确 / 标签错误 / 边界模糊
      - LLM建议标签: 六类之一（边界模糊时给主导情绪）
      - LLM理由 / LLM置信
      - 三列空白留给人工：人工核验结论 / 人工标签 / 同意LLM(是/否)
断点续跑: reports/_llm_anno_progress.jsonl，逐批落盘
"""
import sys
import json
import time
import argparse
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.llm.llm_client import DeepSeekClient

API_KEY = "sk-94c2e1d84e6a405fbf4e742341eddd0c"
BATCH = 10
VALID_LABELS = ["开心", "感激", "悲伤", "愤怒", "恐惧", "焦虑"]
VALID_VERDICTS = ["标签正确", "标签错误", "边界模糊"]

PROMPT_TMPL = """你是中文微博情绪标注质检员。任务：核验每条文本的「原标签」是否正确。

六类情绪定义（互斥，按主导情绪标注）：
- 开心：高兴、喜悦、满足、期待
- 感激：感谢、感恩
- 悲伤：难过、失落、无奈、沮丧、哀叹、委屈、孤独
- 愤怒：生气、发怒、指责、辱骂、强烈不满、斥责他人/事物
- 恐惧：害怕、惊恐、恐慌
- 焦虑：紧张、担忧、不安、纠结、烦躁（对未发生或持续压力的焦灼）

核验结论三选一：
- 标签正确：文本主导情绪与原标签一致（即使字面有另一类的词）
- 标签错误：主导情绪明显是另一类，原标签标错
- 边界模糊：①同时包含强度相当的悲伤与愤怒/焦虑，难分主次；②文本太短/无情绪信息，无法判定

判定要点：
- 「烦/郁闷/倒霉/崩溃/狗带/醉了/心塞」等吐槽，若侧重内心受挫难受→悲伤或焦虑；若侧重指责攻击外部对象→愤怒；难分→边界模糊
- 只依据文本本身，不要臆测背景
- 是微博原文，允许有口语、错别字、表情符号

只输出 JSON 数组，不要任何多余文字。每个元素：
{{"id": 序号, "verdict": "标签正确/标签错误/边界模糊", "label": "六类之一", "confidence": "高/中/低", "reason": "20字内中文理由"}}

待核验文本（原标签为{orig_label}，模型误判为{wrong_label}）：
{items}
"""


def build_prompt(batch_df):
    direction = batch_df["混淆方向"].iloc[0]
    orig_label = "悲伤" if direction.startswith("悲伤") else "愤怒"
    wrong_label = "愤怒" if orig_label == "悲伤" else "悲伤"
    items = "\n".join(f"{r['_local_id']}. {r['文本']}" for _, r in batch_df.iterrows())
    return PROMPT_TMPL.format(orig_label=orig_label, wrong_label=wrong_label, items=items)


def parse_response(text, expected_ids):
    """容错解析 LLM 返回的 JSON 数组，返回 {id: result_dict}"""
    s = text.strip()
    if "```" in s:
        s = s.split("```")[1]
        if s.startswith("json"):
            s = s[4:]
    l, r = s.find("["), s.rfind("]")
    if l == -1 or r == -1:
        raise ValueError("响应中未找到 JSON 数组")
    arr = json.loads(s[l:r + 1])
    out = {}
    for item in arr:
        idx = int(item["id"])
        verdict = item["verdict"].strip()
        label = item["label"].strip()
        if verdict not in VALID_VERDICTS or label not in VALID_LABELS:
            continue
        out[idx] = {
            "verdict": verdict,
            "label": label,
            "confidence": str(item.get("confidence", "")).strip(),
            "reason": str(item.get("reason", "")).strip()[:60],
        }
    missing = set(expected_ids) - set(out.keys())
    if missing:
        raise ValueError(f"缺少 id: {sorted(missing)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（调试用，0=全部）")
    ap.add_argument("--fresh", action="store_true", help="忽略进度文件重跑")
    args = ap.parse_args()

    src_csv = Path("reports/confusion_sad_angry_review.csv")
    out_csv = Path("reports/confusion_review_llm_annotated.csv")
    progress = Path("reports/_llm_anno_progress.jsonl")

    df = pd.read_csv(src_csv)
    strong = df[(df["双模型一致误判"] == "是") & (df["模型置信度"] > 0.7)].reset_index(drop=True)
    # 按混淆方向排序，保证每个批次方向一致（prompt 按批次声明原标签）
    strong = strong.sort_values(["混淆方向", "模型置信度"], ascending=[True, False]).reset_index(drop=True)
    if args.limit:
        strong = strong.head(args.limit)
    print(f"强信号样本: {len(strong)} 条，每批 {BATCH} 条")

    # 续跑：读回已完成的 全局编号 -> 结果
    done = {}
    if progress.exists() and not args.fresh:
        for line in progress.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            done[rec["global_id"]] = rec["result"]
        print(f"已完成 {len(done)} 条，续跑剩余部分")

    client = DeepSeekClient(api_key=API_KEY)

    t0 = time.time()
    total_batches = sum((len(g) - 1) // BATCH + 1 for _, g in strong.groupby("混淆方向", sort=False))
    bi = 0
    # 按混淆方向分组，组内分批，保证每批方向一致
    for _, grp in strong.groupby("混淆方向", sort=False):
        for start in range(0, len(grp), BATCH):
            bi += 1
            batch = grp.iloc[start:start + BATCH].copy()
            global_ids = batch.index.tolist()  # strong 中的全局序号
            if all(g in done for g in global_ids):
                continue
            batch["_local_id"] = range(1, len(batch) + 1)
            prompt = build_prompt(batch)

            parsed = None
            for attempt in range(3):
                try:
                    resp = client.generate(prompt, temperature=0.0, max_tokens=1800)
                    parsed = parse_response(resp, list(batch["_local_id"]))
                    break
                except Exception as e:
                    print(f"\n批次 {bi} 第{attempt+1}次解析失败: {e}")
                    time.sleep(2)
            if parsed is None:
                print(f"\n批次 {bi} 三次失败，跳过（稍后可重跑续传）")
                continue

            with progress.open("a", encoding="utf-8") as f:
                for local_id, g in zip(batch["_local_id"], global_ids):
                    rec = {"global_id": int(g), "result": parsed[int(local_id)]}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    done[g] = parsed[int(local_id)]

            print(f"批次 {bi}/{total_batches} 完成，"
                  f"累计 {len(done)}/{len(strong)}，用时 {time.time()-t0:.0f}s", end="\r")

    print("\n全部批次结束，生成复核 CSV ...")

    strong["LLM判定"] = [done.get(i, {}).get("verdict", "") for i in strong.index]
    strong["LLM建议标签"] = [done.get(i, {}).get("label", "") for i in strong.index]
    strong["LLM置信"] = [done.get(i, {}).get("confidence", "") for i in strong.index]
    strong["LLM理由"] = [done.get(i, {}).get("reason", "") for i in strong.index]

    # 一致性修正：判"正确"但建议标签≠原标签，或判"错误"但建议标签=原标签
    # 均属自相矛盾，保守降级为"边界模糊"交人工裁定
    conflict_a = (strong["LLM判定"] == "标签正确") & (strong["LLM建议标签"] != strong["原标签"])
    conflict_b = (strong["LLM判定"] == "标签错误") & (strong["LLM建议标签"] == strong["原标签"])
    strong["LLM判定_修正"] = strong["LLM判定"]
    strong.loc[conflict_a | conflict_b, "LLM判定_修正"] = "边界模糊"

    strong["同意LLM(是/否)"] = ""
    strong["人工核验结论"] = ""
    strong["人工标签"] = ""

    cols = ["核验编号", "混淆方向", "文本", "原标签", "SCL预测", "模型置信度",
            "LLM判定_修正", "LLM建议标签", "LLM置信", "LLM理由",
            "同意LLM(是/否)", "人工核验结论", "人工标签", "baseline预测", "LLM判定"]
    strong[cols].to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 统计（基于修正后的口径）
    print("=" * 70)
    print(f"输出: {out_csv}（{len(strong)} 条，成功标注 {strong['LLM判定'].astype(bool).sum()} 条）")
    print(f"自相矛盾已降级为边界模糊: {int((conflict_a | conflict_b).sum())} 条")
    print("=" * 70)
    print("\nLLM 判定分布（修正后，按混淆方向）：")
    print(pd.crosstab(strong["混淆方向"], strong["LLM判定_修正"], margins=True).to_string())
    print("\nLLM 认为「标签错误」时建议的标签：")
    wrong = strong[strong["LLM判定_修正"] == "标签错误"]
    print(wrong.groupby(["原标签", "LLM建议标签"]).size().to_string())
    print(f"\nLLM 高置信判定占比: {(strong['LLM置信']=='高').mean():.0%}")


if __name__ == "__main__":
    main()
