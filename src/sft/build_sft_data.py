# -*- coding: utf-8 -*-
"""
M1 步骤 1：SFT 训练样本构造（不含 reason，reason 由 gen_reasons.py 生成）
=====================================================================
规则（对应 docs/P1_QLoRA微调技术方案.md §3.2）：
1. 强制纳入：
   - 24 条 user_corpus 用户金标（tag=user_gold）
   - train 愤怒↔悲伤混淆难例：reports/train_confusion_llm_annotated.csv 中
     仍存在于现训练集的样本（tag=hard_confusion，标签以 fixed2 现标签为准）
   注意：reports/confusion_review_llm_annotated.csv 的 278 条取自原始 test.csv，
   经 norm 核验 0 条在训练集、210 条在现 test 集，属评测数据，不纳入训练。
2. 每类抽样上限：开心/愤怒/悲伤/焦虑/恐惧各 900，感激 300，共 4,800；
   强制样本先占位，其余从普通样本随机补足。
3. 文本长度：随机池只取 <=120 字符样本；超长强制样本截断到 120 字符并记录。
4. 泄漏校验：抽样结果与 val/test norm 交集必须为 0。
5. seed=42 固定。

产物：
  data/sft/sample_manifest.csv   # text,label,label_name,label_cn,source,tag,orig_len
  data/sft/sft_val.jsonl         # val 4359 条，仅标签（评测用）
"""
import re
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SEED = 42
MAX_TEXT_CHARS = 120
# label id -> 抽样上限（id 顺序：0开心 1感激 2悲伤 3愤怒 4恐惧 5焦虑）
CAPS = {0: 900, 1: 300, 2: 900, 3: 900, 4: 900, 5: 900}

DATA = Path("data/processed_emotion")
OUT = Path("data/sft")
TRAIN_CSV = DATA / "train_augmented_v2_cleaned_fixed2.csv"
VAL_CSV = DATA / "val_cleaned_fixed.csv"
TEST_CSV = DATA / "test_cleaned_fixed_anx.csv"
TRAIN_HARD_CSV = Path("reports/train_confusion_llm_annotated.csv")


def normalize(t):
    """与 src/evaluate/clean_dataset.py 完全一致的归一化规则"""
    t = str(t)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"@[\w\u4e00-\u9fa5]+", " ", t)
    t = re.sub(r"#.*?#", " ", t)
    t = re.sub(r"[\s\W_]+", "", t, flags=re.UNICODE)
    return t.lower()


def main():
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(TRAIN_CSV)
    val = pd.read_csv(VAL_CSV)
    test = pd.read_csv(TEST_CSV)
    print(f"train={len(train)}  val={len(val)}  test={len(test)}")

    train["_n"] = train["text"].map(normalize)
    val_norms = set(val["text"].map(normalize))
    test_norms = set(test["text"].map(normalize))

    # ---- 强制集合 1：user_corpus 24 条金标 ----
    user_mask = train["source"].astype(str) == "user_corpus"
    user_norms = set(train.loc[user_mask, "_n"])
    print(f"强制纳入 user_gold: {len(train[user_mask])} 条")

    # ---- 强制集合 2：train 混淆难例（833 中现存于训练集的 651 条）----
    hard833 = pd.read_csv(TRAIN_HARD_CSV)
    hard_norms = set(hard833["text"].map(normalize))
    hard_mask = train["_n"].isin(hard_norms) & ~user_mask
    print(f"强制纳入 hard_confusion: {int(hard_mask.sum())} 条 "
          f"(833 核验文件中 {int(train['_n'].isin(hard_norms).sum())} 条仍在训练集)")

    # ---- 278 test 混淆泄漏核验（仅打印证据，不参与抽样）----
    c278 = pd.read_csv("reports/confusion_review_llm_annotated.csv")
    n278 = c278["文本"].map(normalize)
    leak278 = int(n278.isin(set(train["_n"])).sum())
    in_test278 = int(n278.isin(test_norms).sum())
    print(f"[泄漏核验] 278 test混淆: 命中train={leak278} (必须0), 现存test={in_test278}")
    assert leak278 == 0, "278 混淆样本命中训练集，停止！"

    # ---- 标签 -> tag 映射 ----
    tag_by_idx = {}
    for i in train.index[user_mask]:
        tag_by_idx[i] = "user_gold"
    for i in train.index[hard_mask]:
        tag_by_idx[i] = "hard_confusion"

    # ---- 分层抽样 ----
    selected_parts = []
    stats = []
    truncated_hard = 0
    for label, cap in CAPS.items():
        pool = train[train["label"] == label]
        forced_idx = [i for i in pool.index if i in tag_by_idx]
        normal_pool = pool.drop(index=forced_idx)
        # 随机池只取长度合规样本
        normal_pool = normal_pool[normal_pool["text"].str.len() <= MAX_TEXT_CHARS]

        n_need = max(0, cap - len(forced_idx))
        if n_need > len(normal_pool):
            raise RuntimeError(f"label {label} 普通样本不足: 需{n_need} 有{len(normal_pool)}")
        rand_idx = rng.choice(normal_pool.index.to_numpy(), size=n_need, replace=False)

        part_idx = list(forced_idx) + list(rand_idx)
        part = train.loc[part_idx].copy()
        part["tag"] = ["forced_" + tag_by_idx[i] if i in tag_by_idx else "random"
                       for i in part_idx]
        # 超长强制样本截断
        long_mask = part["text"].str.len() > MAX_TEXT_CHARS
        truncated_hard += int(long_mask.sum())
        part.loc[long_mask, "text"] = part.loc[long_mask, "text"].str[:MAX_TEXT_CHARS]
        part["orig_len"] = train.loc[part_idx, "text"].str.len().to_numpy()
        selected_parts.append(part)
        stats.append((label, part["label_cn"].iloc[0], cap, len(forced_idx), n_need))

    manifest = pd.concat(selected_parts, ignore_index=True)

    # ---- 抽样后泄漏终检（截断后重新 norm）----
    sel_norms = set(manifest["text"].map(normalize))
    assert len(sel_norms & val_norms) == 0, "抽样结果与 val 有 norm 重复！"
    assert len(sel_norms & test_norms) == 0, "抽样结果与 test 有 norm 重复！"
    print(f"[泄漏终检] 与 val/test norm 交集 = 0 ；超长强制样本截断 {truncated_hard} 条")

    manifest = manifest[["text", "label", "label_name", "label_cn", "source",
                         "tag", "orig_len"]].reset_index(drop=True)
    manifest.to_csv(OUT / "sample_manifest.csv", index=False, encoding="utf-8-sig")

    print("\n抽样结果：")
    print(f"{'label':6s} {'类别':4s} {'cap':>5s} {'强制':>5s} {'随机':>5s}")
    for lid, cn, cap, nf, nr in stats:
        print(f"{lid:<6d} {cn:<4s} {cap:>5d} {nf:>5d} {nr:>5d}")
    print(f"合计 {len(manifest)} 条")
    print("\ntag 分布：")
    print(manifest["tag"].value_counts().to_string())

    # ---- val jsonl（仅标签，评测用）----
    with (OUT / "sft_val.jsonl").open("w", encoding="utf-8") as f:
        for _, r in val.iterrows():
            f.write(json.dumps(
                {"text": str(r["text"]), "label": int(r["label"]),
                 "label_cn": str(r["label_cn"])},
                ensure_ascii=False) + "\n")
    print(f"\n写出 {OUT / 'sample_manifest.csv'}")
    print(f"写出 {OUT / 'sft_val.jsonl'}（{len(val)} 条）")


if __name__ == "__main__":
    main()
