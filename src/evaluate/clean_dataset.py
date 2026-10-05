# -*- coding: utf-8 -*-
"""
数据清洗（三步）：
1. 删除归一化后长度 < 4 的短文本（无上下文、本质不可判定）
2. 从 train 删除与 val/test 归一化后重复的行（去泄漏，val/test 不动）
3. 删除跨集合标签不一致的近重复文本（从所有集合移除，标签噪声）

不覆盖原始文件，输出 *_cleaned.csv，可回滚。
"""
import re
import sys
from pathlib import Path
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DATA = Path("data/processed_emotion")
LABELS = ["开心", "感激", "悲伤", "愤怒", "恐惧", "焦虑"]
MIN_LEN = 4  # 归一化后最小长度


def normalize(t):
    t = str(t)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"@[\w\u4e00-\u9fa5]+", " ", t)
    t = re.sub(r"#.*?#", " ", t)
    t = re.sub(r"[\s\W_]+", "", t, flags=re.UNICODE)
    return t.lower()


def label_dist(df):
    return df["label"].value_counts().sort_index().rename(lambda x: LABELS[x]).to_dict()


def main():
    train = pd.read_csv(DATA / "train_augmented_v2.csv")
    val = pd.read_csv(DATA / "val.csv")
    test = pd.read_csv(DATA / "test.csv")

    print("=" * 70)
    print("清洗前规模")
    print("=" * 70)
    for name, df in [("train", train), ("val", val), ("test", test)]:
        print(f"{name:6s} {len(df):6d}  标签分布: {label_dist(df)}")

    # 归一化
    train["_norm"] = train["text"].map(normalize)
    val["_norm"] = val["text"].map(normalize)
    test["_norm"] = test["text"].map(normalize)

    # ---- 步骤 1: 短文本（归一化长度 < MIN_LEN）从所有集合删除 ----
    before = {n: len(d) for n, d in [("train", train), ("val", val), ("test", test)]}
    train = train[train["_norm"].str.len() >= MIN_LEN].reset_index(drop=True)
    val = val[val["_norm"].str.len() >= MIN_LEN].reset_index(drop=True)
    test = test[test["_norm"].str.len() >= MIN_LEN].reset_index(drop=True)
    print("\n[步骤1] 删除归一化长度<{}的短文本:".format(MIN_LEN))
    for n in ["train", "val", "test"]:
        print(f"  {n}: {before[n]} -> {len(eval(n))}  (删 {before[n]-len(eval(n))})")

    # ---- 步骤 2: 从 train 删除与 val/test 归一化重复的行 ----
    val_norms = set(val["_norm"])
    test_norms = set(test["_norm"])
    leak_mask = train["_norm"].isin(val_norms | test_norms)
    train = train[~leak_mask].reset_index(drop=True)
    print(f"\n[步骤2] 从 train 删除与 val/test 近重复: 删 {int(leak_mask.sum())} 条")

    # ---- 步骤 3: 删除跨集合标签不一致的近重复文本 ----
    # 重新检查（短文本已删，泄漏已删），找出 val/test 之间以及它们与 train 之间标签不一致的
    all_texts = pd.concat([
        train[["_norm", "label"]].assign(_src="train"),
        val[["_norm", "label"]].assign(_src="val"),
        test[["_norm", "label"]].assign(_src="test"),
    ], ignore_index=True)
    # 对每个 norm，看是否有多种 label
    label_counts = all_texts.groupby("_norm")["label"].nunique()
    bad_norms = set(label_counts[label_counts > 1].index)
    print(f"\n[步骤3] 跨集合标签不一致的归一化串: {len(bad_norms)} 个")

    train = train[~train["_norm"].isin(bad_norms)].reset_index(drop=True)
    val = val[~val["_norm"].isin(bad_norms)].reset_index(drop=True)
    test = test[~test["_norm"].isin(bad_norms)].reset_index(drop=True)
    print(f"  train 删 {train['_norm'].isin(bad_norms).sum() if False else '?'} (已过滤)")

    # ---- 保存 cleaned 版本 ----
    train = train.drop(columns=["_norm"])
    val = val.drop(columns=["_norm"])
    test = test.drop(columns=["_norm"])

    train.to_csv(DATA / "train_augmented_v2_cleaned.csv", index=False, encoding="utf-8-sig")
    val.to_csv(DATA / "val_cleaned.csv", index=False, encoding="utf-8-sig")
    test.to_csv(DATA / "test_cleaned.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("清洗后规模")
    print("=" * 70)
    for name, df in [("train", train), ("val", val), ("test", test)]:
        print(f"{name:6s} {len(df):6d}  标签分布: {label_dist(df)}")

    # ---- 验证：清洗后无泄漏、无短文本、无标签不一致 ----
    print("\n" + "=" * 70)
    print("验证")
    print("=" * 70)
    v_n = set(val["text"].map(normalize))
    t_n = set(test["text"].map(normalize))
    tr_n = set(train["text"].map(normalize))
    print(f"train ∩ val  近重复: {len(tr_n & v_n)}")
    print(f"train ∩ test 近重复: {len(tr_n & t_n)}")
    print(f"val   ∩ test 近重复: {len(v_n & t_n)}")
    short_train = (train["text"].map(normalize).str.len() < MIN_LEN).sum()
    short_val = (val["text"].map(normalize).str.len() < MIN_LEN).sum()
    short_test = (test["text"].map(normalize).str.len() < MIN_LEN).sum()
    print(f"短文本残留: train={short_train} val={short_val} test={short_test}")

    # 标签不一致检查
    all_n = pd.concat([
        train.assign(_n=train["text"].map(normalize))[["_n", "label"]],
        val.assign(_n=val["text"].map(normalize))[["_n", "label"]],
        test.assign(_n=test["text"].map(normalize))[["_n", "label"]],
    ])
    inconsistent = all_n.groupby("_n")["label"].nunique().gt(1).sum()
    print(f"跨集合标签不一致串数: {inconsistent}")

    print("\n输出文件:")
    print(f"  {DATA / 'train_augmented_v2_cleaned.csv'}")
    print(f"  {DATA / 'val_cleaned.csv'}")
    print(f"  {DATA / 'test_cleaned.csv'}")
    print("原始文件未动，可随时回滚。")


if __name__ == "__main__":
    main()
