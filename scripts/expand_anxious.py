# -*- coding: utf-8 -*-
"""
大规模扩充焦虑类样本 + LLM 标签一致性过滤
==========================================
流程：
1. 分批调用 DeepSeek 生成焦虑文本（场景轮换，避免同质化），目标 raw >= 2600 条
2. 去重、长度过滤、与真实训练集去重
3. 用 DeepSeek 逐条判定主导情绪，只保留被判为「焦虑」的样本
4. 输出 data/augmented/anxious_filtered.csv，并合并出 train_augmented_v2.csv

支持断点续跑：中间结果写入 data/augmented/anxious_raw.csv，重复运行会跳过已完成的生成。
"""
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.llm.llm_client import DeepSeekClient

API_KEY = "sk-94c2e1d84e6a405fbf4e742341eddd0c"

# 场景轮换，覆盖真实微博里焦虑的多种来源，避免只集中在考试/面试
SCENARIO_GROUPS = [
    "考试复习、挂科担心、成绩查询",
    "求职面试、等offer、试用期考核",
    "体检报告、等手术、家人住院",
    "航班火车延误、赶不上行程、签证材料",
    "工作deadline、项目上线、被领导追问进度",
    "贷款还款、房租到期、经济压力",
    "等重要消息、消息已读不回、联系不上人",
    "失眠胡思乱想、莫名心慌、对未来不确定",
    "论文答辩、演讲汇报、公开课展示",
    "搬家出国、重要决定前夜、人生转折",
    "孩子考试生病、父母健康、家里出事",
    "竞赛报名截止、材料提交、审核结果",
]

BATCH_SIZE = 40          # 每批生成条数
TARGET_RAW = 2600        # 过滤前的目标量（过滤后预计剩 2000+）
VERIFY_CHUNK = 20        # 每次送去校验的条数
AUG_DIR = Path("data/augmented")
RAW_PATH = AUG_DIR / "anxious_raw.csv"
FILTERED_PATH = AUG_DIR / "anxious_filtered.csv"
TRAIN_DIR = Path("data/processed_emotion")


def clean_line(line: str) -> str:
    line = line.strip().strip('"').strip()
    line = re.sub(r"^\d+[\.、\)）]\s*", "", line)
    line = re.sub(r"^[-\*•]\s*", "", line)
    return line.strip()


def generate_batch(client: DeepSeekClient, scenarios: str, n: int) -> list:
    prompt = f"""请生成 {n} 条表达「焦虑」情绪的中文微博风格短文本。

场景限定：{scenarios}
要求：
1. 主导情绪必须是焦虑（担心、不安、等结果、睡不着、心慌），不要写成纯愤怒、纯悲伤或纯恐惧
2. 口语化、第一人称，像真人发的微博，不要鸡汤、不要书面总结
3. 每条 12-60 字，内容互不重复，具体细节各不相同
4. 每行一条，不要编号，不要任何额外说明

直接输出 {n} 行："""
    result = client.generate(prompt, temperature=1.0, max_tokens=2500)
    lines = [clean_line(l) for l in result.split("\n")]
    return [l for l in lines if 8 <= len(l) <= 80]


def verify_chunk(client: DeepSeekClient, texts: list) -> list:
    """让 DeepSeek 判断每条的主导情绪，返回布尔列表（是否为焦虑）"""
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    labels = "开心、感激、悲伤、愤怒、恐惧、焦虑"
    prompt = f"""你是情感标注员。判断下面每条中文文本的【主导情绪】，只能从以下六个里选一个：{labels}。

判定要点：
- 焦虑：担心尚未发生的事、等结果、不安、心慌、睡不着
- 恐惧：面对具体危险或恐怖事物的害怕
- 悲伤：失落、难过、失去
- 愤怒：生气、抱怨、指责
- 如果几种情绪都有，选最主导的一个

文本：
{numbered}

严格按顺序输出 {len(texts)} 行，每行只有一个情绪词，不要编号、不要解释："""
    result = client.generate(prompt, temperature=0.0, max_tokens=400)
    verdicts = [clean_line(l) for l in result.split("\n") if clean_line(l)]
    # 容错：行数对不上时，逐行找第一个命中的情绪词
    kept = []
    for i in range(len(texts)):
        token = verdicts[i] if i < len(verdicts) else ""
        kept.append("焦虑" in token and not any(w in token for w in ["悲伤", "愤怒", "恐惧", "开心", "感激"]))
    return kept


def main():
    AUG_DIR.mkdir(exist_ok=True, parents=True)
    client = DeepSeekClient(api_key=API_KEY)

    # ---------- 1. 生成 ----------
    existing = []
    if RAW_PATH.exists():
        existing = pd.read_csv(RAW_PATH)["text"].tolist()
        print(f"[resume] raw 已有 {len(existing)} 条")

    seen = set(existing)
    real_train = pd.read_csv(TRAIN_DIR / "train.csv")
    real_texts = set(real_train["text"].astype(str).tolist())

    batch_idx = 0
    while len(seen) < TARGET_RAW:
        scenarios = SCENARIO_GROUPS[batch_idx % len(SCENARIO_GROUPS)]
        print(f"[gen] batch {batch_idx+1} ({scenarios})  当前 {len(seen)}/{TARGET_RAW}")
        try:
            samples = generate_batch(client, scenarios, BATCH_SIZE)
        except Exception as e:
            print(f"[gen] 失败，跳过该批: {e}")
            time.sleep(5)
            batch_idx += 1
            continue
        new = [s for s in samples if s not in seen and s not in real_texts]
        seen.update(new)
        pd.DataFrame({"text": list(seen)}).to_csv(RAW_PATH, index=False, encoding="utf-8-sig")
        print(f"[gen] 本批新增 {len(new)} 条")
        batch_idx += 1

    candidates = [t for t in seen if t not in real_texts]
    print(f"[gen] 生成完成，候选 {len(candidates)} 条")

    # ---------- 2. 过滤 ----------
    already = {}
    if FILTERED_PATH.exists():
        prev = pd.read_csv(FILTERED_PATH)
        already = dict(zip(prev["text"], prev["keep"]))
        print(f"[verify] 已校验 {len(already)} 条，续跑")

    todo = [t for t in candidates if t not in already]
    for start in range(0, len(todo), VERIFY_CHUNK):
        chunk = todo[start:start + VERIFY_CHUNK]
        try:
            flags = verify_chunk(client, chunk)
        except Exception as e:
            print(f"[verify] 失败，稍后重试: {e}")
            time.sleep(5)
            continue
        for t, f in zip(chunk, flags):
            already[t] = int(f)
        pd.DataFrame({"text": list(already.keys()),
                      "keep": list(already.values())}).to_csv(
            FILTERED_PATH, index=False, encoding="utf-8-sig")
        kept_now = sum(already.values())
        print(f"[verify] {len(already)}/{len(candidates)} 已校验，保留 {kept_now}")

    # ---------- 3. 合并训练集 ----------
    kept_texts = [t for t, k in already.items() if k == 1]
    print(f"[done] 过滤后保留 {len(kept_texts)} 条焦虑样本")

    aug = pd.DataFrame({"text": kept_texts, "label": 5, "source": "deepseek_filtered"})
    base = real_train.copy()
    merged = pd.concat([base, aug[["text", "label"]]], ignore_index=True)
    merged = merged.drop_duplicates(subset=["text"])
    out = TRAIN_DIR / "train_augmented_v2.csv"
    merged.to_csv(out, index=False, encoding="utf-8-sig")

    anxious_n = int((merged["label"] == 5).sum())
    print(f"[done] 新训练集: {out}")
    print(f"[done] 总样本 {len(merged)}，其中焦虑类 {anxious_n} "
          f"(原始 {(real_train['label']==5).sum()})")


if __name__ == "__main__":
    main()
