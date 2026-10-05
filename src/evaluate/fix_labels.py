# -*- coding: utf-8 -*-
"""
标签修正脚本：基于 278 条 LLM 预标注核验结果修正 test_cleaned 标签
- 标签错误(115条): 改为 LLM 建议标签
- 边界模糊(67条):  从数据集删除（标签不确定，是噪声）
- 标签正确(96条):  保持不变
输出 test_cleaned_fixed.csv / val_cleaned_fixed.csv / train_augmented_v2_cleaned_fixed.csv
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


def main():
    review = pd.read_csv("reports/confusion_review_llm_annotated.csv", encoding="utf-8-sig")
    test = pd.read_csv(DATA / "test_cleaned.csv")
    val = pd.read_csv(DATA / "val_cleaned.csv")
    train = pd.read_csv(DATA / "train_augmented_v2_cleaned.csv")

    # 按文本建索引
    test_idx = test.set_index("text")
    val_idx = val.set_index("text")
    train_idx = train.set_index("text")

    # 分三组
    wrong = review[review["LLM判定_修正"] == "标签错误"]      # 115 条 → 改标签
    fuzzy = review[review["LLM判定_修正"] == "边界模糊"]      # 67 条 → 删除
    correct = review[review["LLM判定_修正"] == "标签正确"]     # 96 条 → 不动

    print("=" * 70)
    print("核验结果分组")
    print("=" * 70)
    print(f"  标签错误（改标签）: {len(wrong)}")
    print(f"  边界模糊（删除）:   {len(fuzzy)}")
    print(f"  标签正确（不动）:   {len(correct)}")

    # 收集要删除的文本
    delete_texts = set(fuzzy["文本"])

    # 收集要改标签的文本 → 新标签
    relabel = {}
    for _, r in wrong.iterrows():
        text = r["文本"]
        new_label_cn = r["LLM建议标签"]
        if new_label_cn in LABEL_MAP:
            relabel[text] = LABEL_MAP[new_label_cn]
        else:
            print(f"  [警告] 未知标签 '{new_label_cn}'，跳过: {text[:30]}")

    print(f"\n改标签明细:")
    for label_cn, count in wrong["LLM建议标签"].value_counts().items():
        print(f"  → {label_cn}: {count} 条")

    # ---- 修正 test ----
    test_fixed = test.copy()
    n_relabel_test = 0
    n_delete_test = 0
    for i, row in test_fixed.iterrows():
        if row["text"] in relabel:
            new_label, new_name, new_cn = relabel[row["text"]]
            test_fixed.at[i, "label"] = new_label
            test_fixed.at[i, "label_name"] = new_name
            test_fixed.at[i, "label_cn"] = new_cn
            n_relabel_test += 1
    test_fixed = test_fixed[~test_fixed["text"].isin(delete_texts)].reset_index(drop=True)
    n_delete_test = len(test) - len(test_fixed) - n_relabel_test  # 不太对，算一下
    n_delete_test = test["text"].isin(delete_texts).sum()
    print(f"\ntest: 改标签 {n_relabel_test} 条，删除 {n_delete_test} 条，"
          f"{len(test)} → {len(test_fixed)}")

    # ---- 修正 val（虽然 0 匹配，保险起见走一遍）----
    val_fixed = val.copy()
    n_relabel_val = 0
    for i, row in val_fixed.iterrows():
        if row["text"] in relabel:
            new_label, new_name, new_cn = relabel[row["text"]]
            val_fixed.at[i, "label"] = new_label
            val_fixed.at[i, "label_name"] = new_name
            val_fixed.at[i, "label_cn"] = new_cn
            n_relabel_val += 1
    val_fixed = val_fixed[~val_fixed["text"].isin(delete_texts)].reset_index(drop=True)
    print(f"val: 改标签 {n_relabel_val} 条，删除 {val['text'].isin(delete_texts).sum()} 条，"
          f"{len(val)} → {len(val_fixed)}")

    # ---- 修正 train（同源污染可能存在，保险起见走一遍）----
    train_fixed = train.copy()
    n_relabel_train = 0
    for i, row in train_fixed.iterrows():
        if row["text"] in relabel:
            new_label, new_name, new_cn = relabel[row["text"]]
            train_fixed.at[i, "label"] = new_label
            train_fixed.at[i, "label_name"] = new_name
            train_fixed.at[i, "label_cn"] = new_cn
            n_relabel_train += 1
    train_fixed = train_fixed[~train_fixed["text"].isin(delete_texts)].reset_index(drop=True)
    print(f"train: 改标签 {n_relabel_train} 条，删除 {train['text'].isin(delete_texts).sum()} 条，"
          f"{len(train)} → {len(train_fixed)}")

    # ---- 保存 ----
    test_fixed.to_csv(DATA / "test_cleaned_fixed.csv", index=False, encoding="utf-8-sig")
    val_fixed.to_csv(DATA / "val_cleaned_fixed.csv", index=False, encoding="utf-8-sig")
    train_fixed.to_csv(DATA / "train_augmented_v2_cleaned_fixed.csv", index=False, encoding="utf-8-sig")

    # ---- 汇总 ----
    print("\n" + "=" * 70)
    print("修正后规模")
    print("=" * 70)
    for name, df in [("train", train_fixed), ("val", val_fixed), ("test", test_fixed)]:
        dist = df["label"].value_counts().sort_index()
        LABEL_NAMES_CN = ["开心", "感激", "悲伤", "愤怒", "恐惧", "焦虑"]
        dist_str = " ".join(f"{LABEL_NAMES_CN[k]}={v}" for k, v in dist.items())
        print(f"{name:6s} {len(df):6d}  {dist_str}")

    # 改标前后 test 愤怒类变化
    print(f"\n test 愤怒类: 修正前 {len(test[test['label']==3])} → 修正后 {len(test_fixed[test_fixed['label']==3])}")
    print(f" test 悲伤类: 修正前 {len(test[test['label']==2])} → 修正后 {len(test_fixed[test_fixed['label']==2])}")
    print(f" test 焦虑类: 修正前 {len(test[test['label']==5])} → 修正后 {len(test_fixed[test_fixed['label']==5])}")

    print("\n输出文件:")
    for fn in ["train_augmented_v2_cleaned_fixed.csv",
               "val_cleaned_fixed.csv",
               "test_cleaned_fixed.csv"]:
        print(f"  {DATA / fn}")


if __name__ == "__main__":
    main()
