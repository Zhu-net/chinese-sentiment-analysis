# -*- coding: utf-8 -*-
"""
修正 train 集标签：基于 833 条 LLM 预标注结果
- 标签错误(497条): 改为 LLM 建议标签
- 边界模糊(182条): 从 train 删除
- 标签正确(154条): 不动
输出 train_augmented_v2_cleaned_fixed2.csv
"""
import sys
from pathlib import Path
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DATA = Path("data/processed_emotion")
LABEL_MAP = {
    "开心": (0, "happy", "开心"),
    "感激": (1, "grateful", "感激"),
    "悲伤": (2, "sad", "悲伤"),
    "愤怒": (3, "angry", "愤怒"),
    "恐惧": (4, "fear", "恐惧"),
    "焦虑": (5, "anxious", "焦虑"),
}
LABEL_NAMES_CN = ["开心", "感激", "悲伤", "愤怒", "恐惧", "焦虑"]


def main():
    review = pd.read_csv("reports/train_confusion_llm_annotated.csv", encoding="utf-8-sig")
    train = pd.read_csv(DATA / "train_augmented_v2_cleaned_fixed.csv")

    print(f"train 修正前: {len(train)} 条")
    print(f"核验样本: {len(review)} 条")
    print(f"  标签错误: {(review['LLM判定_修正']=='标签错误').sum()}")
    print(f"  边界模糊: {(review['LLM判定_修正']=='边界模糊').sum()}")
    print(f"  标签正确: {(review['LLM判定_修正']=='标签正确').sum()}")

    # 要改标签的文本 → 新标签
    wrong = review[review["LLM判定_修正"] == "标签错误"]
    relabel = {}
    for _, r in wrong.iterrows():
        text = r["text"]
        new_cn = r["LLM建议标签"]
        if new_cn in LABEL_MAP:
            relabel[text] = LABEL_MAP[new_cn]

    # 要删除的文本
    fuzzy = review[review["LLM判定_修正"] == "边界模糊"]
    delete_texts = set(fuzzy["text"])

    # 修正 train
    train_fixed = train.copy()
    n_relabel = 0
    for i, row in train_fixed.iterrows():
        if row["text"] in relabel:
            new_label, new_name, new_cn = relabel[row["text"]]
            train_fixed.at[i, "label"] = new_label
            train_fixed.at[i, "label_name"] = new_name
            train_fixed.at[i, "label_cn"] = new_cn
            n_relabel += 1
    n_delete = train_fixed["text"].isin(delete_texts).sum()
    train_fixed = train_fixed[~train_fixed["text"].isin(delete_texts)].reset_index(drop=True)

    print(f"\ntrain: 改标签 {n_relabel} 条，删除 {n_delete} 条")
    print(f"train 修正后: {len(train_fixed)} 条")

    # 保存
    train_fixed.to_csv(DATA / "train_augmented_v2_cleaned_fixed2.csv", index=False, encoding="utf-8-sig")

    # 类别分布对比
    print(f"\n{'='*70}")
    print("类别分布对比（修正前 → 修正后）")
    print(f"{'='*70}")
    for label in range(6):
        before = (train["label"] == label).sum()
        after = (train_fixed["label"] == label).sum()
        diff = after - before
        sign = "+" if diff >= 0 else ""
        print(f"  {LABEL_NAMES_CN[label]:4s} {before:6d} → {after:6d}  ({sign}{diff})")

    print(f"\n输出: {DATA / 'train_augmented_v2_cleaned_fixed2.csv'}")


if __name__ == "__main__":
    main()
