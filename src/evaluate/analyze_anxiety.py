# -*- coding: utf-8 -*-
"""
分析焦虑类混淆模式：test + train
- 真实焦虑被误判成什么
- 什么被误判成焦虑
- 高频误判样本示例
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

LABELS_CN = ["开心", "感激", "悲伤", "愤怒", "恐惧", "焦虑"]
ANXIOUS = 5


def predict_df(model, df, tokenizer, max_len, device, batch_size=64):
    ds = BERTDataset(df["text"].tolist(), df["label"].tolist(), tokenizer, max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1).cpu()
            all_probs.append(probs)
    probs = torch.cat(all_probs, dim=0).numpy()
    preds = probs.argmax(axis=1)
    df = df.copy()
    df["pred"] = preds
    df["pred_prob"] = probs.max(axis=1)
    return df, probs


def main():
    cfg = load_config()
    ecfg = cfg["emotion"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(cfg["train"]["save_dir"])
    pretrained = cfg["model"]["pretrained_model"]
    max_len = cfg["data"]["max_length"]
    data_dir = Path(cfg["data"]["processed_dir_emotion"])

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

    test = pd.read_csv(data_dir / "test_cleaned_fixed.csv")
    train = pd.read_csv(data_dir / "train_augmented_v2_cleaned_fixed2.csv")

    print("=" * 70)
    print("TEST 集焦虑类分析")
    print("=" * 70)
    test_pred, test_probs = predict_df(model, test, tokenizer, max_len, device)
    test_anx = test_pred[test_pred["label"] == ANXIOUS]
    print(f"焦虑样本数: {len(test_anx)}")
    print(f"焦虑 recall: {(test_anx['pred']==ANXIOUS).sum()}/{len(test_anx)} = {(test_anx['pred']==ANXIOUS).mean():.3f}")
    print(f"\n真实焦虑 → 模型预测分布:")
    print(test_anx["pred"].value_counts().sort_index().rename(lambda x: LABELS_CN[x]))

    # 被误判为焦虑的非焦虑样本
    false_anx = test_pred[(test_pred["pred"] == ANXIOUS) & (test_pred["label"] != ANXIOUS)]
    print(f"\n被误判为焦虑的样本数: {len(false_anx)}")
    print(f"来源分布:")
    print(false_anx["label"].value_counts().sort_index().rename(lambda x: LABELS_CN[x]))

    # 输出焦虑误判样本（真焦虑→其他）
    misclass = test_anx[test_anx["pred"] != ANXIOUS].sort_values("pred_prob", ascending=False)
    print(f"\n--- 真实焦虑被误判的样本（按置信度降序）---")
    for _, r in misclass.iterrows():
        print(f"  真焦虑→{LABELS_CN[r['pred']]} ({r['pred_prob']:.3f}) | {r['text'][:50]}")

    print(f"\n--- 非焦虑被误判为焦虑的样本（按置信度降序）---")
    for _, r in false_anx.sort_values("pred_prob", ascending=False).iterrows():
        print(f"  真{LABELS_CN[r['label']]}→焦虑 ({r['pred_prob']:.3f}) | {r['text'][:50]}")

    # TRAIN 集
    print(f"\n{'='*70}")
    print("TRAIN 集焦虑类分析")
    print("=" * 70)
    train_pred, _ = predict_df(model, train, tokenizer, max_len, device)
    train_anx = train_pred[train_pred["label"] == ANXIOUS]
    print(f"焦虑样本数: {len(train_anx)}")
    print(f"焦虑 recall: {(train_anx['pred']==ANXIOUS).sum()}/{len(train_anx)} = {(train_anx['pred']==ANXIOUS).mean():.3f}")
    print(f"\n真实焦虑 → 模型预测分布:")
    print(train_anx["pred"].value_counts().sort_index().rename(lambda x: LABELS_CN[x]))

    false_anx_train = train_pred[(train_pred["pred"] == ANXIOUS) & (train_pred["label"] != ANXIOUS)]
    print(f"\n被误判为焦虑的样本数: {len(false_anx_train)}")
    print(f"来源分布:")
    print(false_anx_train["label"].value_counts().sort_index().rename(lambda x: LABELS_CN[x]))


if __name__ == "__main__":
    main()
