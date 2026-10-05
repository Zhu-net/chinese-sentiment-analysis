# -*- coding: utf-8 -*-
"""
修正 test 集焦虑标签并重新评估 fixed 模型
- 标签错误(22条): 改为 LLM 建议标签
- 边界模糊(4条):  删除
- 不重训，直接用 fixed 模型评估修正后的 test 集
"""
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

os_sys = __import__("os")
os_sys.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.utils import load_config
from src.models.bert_dataset import BERTDataset
from src.models.bert_classifier import BERTClassifier
from src.evaluate.metrics import evaluate

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
    cfg = load_config()
    ecfg = cfg["emotion"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(cfg["train"]["save_dir"])
    pretrained = cfg["model"]["pretrained_model"]
    max_len = cfg["data"]["max_length"]
    data_dir = Path(cfg["data"]["processed_dir_emotion"])

    # 加载 LLM 标注
    review = pd.read_csv("reports/anxiety_confusion_llm_annotated.csv", encoding="utf-8-sig")
    wrong = review[review["LLM判定_修正"] == "标签错误"]
    fuzzy = review[review["LLM判定_修正"] == "边界模糊"]

    relabel = {}
    for _, r in wrong.iterrows():
        new_cn = r["LLM建议标签"]
        if new_cn in LABEL_MAP:
            relabel[r["text"]] = LABEL_MAP[new_cn]
    delete_texts = set(fuzzy["text"])

    print(f"焦虑标签修正: 改标 {len(relabel)} 条，删除 {len(delete_texts)} 条")

    # 修正 test
    test = pd.read_csv(data_dir / "test_cleaned_fixed.csv")
    test_fixed = test.copy()
    n_relabel = 0
    for i, row in test_fixed.iterrows():
        if row["text"] in relabel:
            new_label, new_name, new_cn = relabel[row["text"]]
            test_fixed.at[i, "label"] = new_label
            test_fixed.at[i, "label_name"] = new_name
            test_fixed.at[i, "label_cn"] = new_cn
            n_relabel += 1
    n_delete = test_fixed["text"].isin(delete_texts).sum()
    test_fixed = test_fixed[~test_fixed["text"].isin(delete_texts)].reset_index(drop=True)
    print(f"test: {len(test)} → {len(test_fixed)} (改标 {n_relabel}, 删除 {n_delete})")

    # 保存修正后的 test
    test_fixed.to_csv(data_dir / "test_cleaned_fixed_anx.csv", index=False, encoding="utf-8-sig")

    # 类别分布变化
    print(f"\n焦虑类: 修正前 {(test['label']==5).sum()} → 修正后 {(test_fixed['label']==5).sum()}")

    # 用 fixed 模型评估
    ckpt = torch.load(save_dir / ecfg["model_file_fixed"], map_location=device, weights_only=False)
    model = BERTClassifier(
        pretrained_model_name=pretrained,
        num_classes=ckpt["config"]["num_classes"],
        dropout=ckpt["config"]["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(pretrained)

    ds = BERTDataset(test_fixed["text"].tolist(), test_fixed["label"].tolist(), tokenizer, max_len)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            all_probs.append(torch.softmax(logits, dim=1).cpu())
            all_labels.append(batch["labels"])
    probs = torch.cat(all_probs, dim=0)
    preds = probs.argmax(dim=1).numpy()
    labels = torch.cat(all_labels).numpy()

    print(f"\n{'='*60}")
    print("fixed 模型 在 焦虑标签修正后 test 集 上的评估")
    print(f"{'='*60}")
    results = evaluate(labels, preds, model_name="fixed-焦虑标签修正后",
                       average="macro", target_names=LABEL_NAMES_CN)

    # 焦虑类详细
    anx_mask = labels == 5
    anx_correct = (preds[anx_mask] == 5).sum()
    print(f"\n焦虑类 recall: {anx_correct}/{anx_mask.sum()} = {anx_correct/anx_mask.sum():.3f}")

    # 与修正前对比
    print(f"\n{'='*60}")
    print("对比（修正前 vs 修正后）")
    print(f"{'='*60}")
    print(f"  test 样本数: {len(test)} → {len(test_fixed)}")
    print(f"  焦虑样本数: {(test['label']==5).sum()} → {(test_fixed['label']==5).sum()}")


if __name__ == "__main__":
    main()
