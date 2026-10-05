# -*- coding: utf-8 -*-
"""数据泄露排查：train/val/test 间文本重复检测"""
import re
import sys
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DATA = Path("data/processed_emotion")
FILES = {
    "train(原始)": "train.csv",
    "train(增强v2)": "train_augmented_v2.csv",
    "val": "val.csv",
    "test": "test.csv",
}


def normalize(t):
    """归一化：去空白、标点、URL、@用户、话题，小写，用于近重复检测"""
    t = str(t)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"@[\w\u4e00-\u9fa5]+", " ", t)
    t = re.sub(r"#.*?#", " ", t)
    t = re.sub(r"[\s\W_]+", "", t, flags=re.UNICODE)
    return t.lower()


def load(name, fn):
    df = pd.read_csv(DATA / fn)
    df["_norm"] = df["text"].map(normalize)
    df["_name"] = name
    return df


def main():
    dfs = {name: load(name, fn) for name, fn in FILES.items()}

    print("=" * 72)
    print("一、各集合规模 + 内部重复")
    print("=" * 72)
    for name, df in dfs.items():
        n = len(df)
        n_exact_dup = n - df["text"].nunique()
        n_norm_dup = n - df["_norm"].nunique()
        print(f"{name:14s} 总数={n:6d}  精确重复={n_exact_dup:4d}  归一化重复={n_norm_dup:4d}")

    print("\n" + "=" * 72)
    print("二、跨集合精确文本交集（泄漏证据）")
    print("=" * 72)
    # 关键：增强训练集 vs val/test
    pairs = [
        ("train(增强v2)", "val"),
        ("train(增强v2)", "test"),
        ("train(原始)", "val"),
        ("train(原始)", "test"),
        ("val", "test"),
    ]
    for a, b in pairs:
        sa = set(dfs[a]["text"])
        sb = set(dfs[b]["text"])
        inter = sa & sb
        print(f"{a} ∩ {b}: 精确重复 {len(inter):4d} 条")

    print("\n" + "=" * 72)
    print("三、跨集合归一化后交集（近重复泄漏）")
    print("=" * 72)
    for a, b in pairs:
        sa = set(dfs[a]["_norm"])
        sb = set(dfs[b]["_norm"])
        inter = sa & sb
        # 过滤空串
        inter.discard("")
        print(f"{a} ∩ {b}: 归一化后重复 {len(inter):4d} 条")

    # 重点：增强v2 与 test 的精确交集，列出若干样例 + 标签是否一致
    print("\n" + "=" * 72)
    print("四、train(增强v2) ∩ test 精确交集样本示例 + 标签一致性")
    print("=" * 72)
    a, b = "train(增强v2)", "test"
    m = dfs[a][["text", "label"]].merge(dfs[b][["text", "label"]], on="text",
                                          suffixes=("_train", "_test"))
    if len(m):
        same_label = (m["label_train"] == m["label_test"]).sum()
        print(f"共 {len(m)} 条交集，其中标签一致 {same_label} 条，标签不一致 {len(m)-same_label} 条")
        print("\n前 10 条交集：")
        for _, r in m.head(10).iterrows():
            t = str(r["text"])[:50]
            print(f"  [train={int(r['label_train'])} test={int(r['label_test'])}] {t}")
    else:
        print("无精确交集。")


if __name__ == "__main__":
    main()
