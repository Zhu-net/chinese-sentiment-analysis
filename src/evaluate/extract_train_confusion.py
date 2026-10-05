# -*- coding: utf-8 -*-
"""
用 fixed 模型对 train 集推理，提取愤怒↔悲伤混淆样本
输出 CSV 供 LLM 预标注
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


def main():
    cfg = load_config()
    ecfg = cfg["emotion"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    save_dir = Path(cfg["train"]["save_dir"])
    model_file = ecfg["model_file_fixed"]
    pretrained = cfg["model"]["pretrained_model"]
    max_len = cfg["data"]["max_length"]
    batch_size = 64  # 推理用大 batch

    # 加载模型
    ckpt = torch.load(save_dir / model_file, map_location=device, weights_only=False)
    model = BERTClassifier(
        pretrained_model_name=pretrained,
        num_classes=ckpt["config"]["num_classes"],
        dropout=ckpt["config"]["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(pretrained)

    # 加载 train 数据
    data_dir = Path(cfg["data"]["processed_dir_emotion"])
    train_df = pd.read_csv(data_dir / "train_augmented_v2_cleaned_fixed.csv")
    print(f"train: {len(train_df)} 条")

    # 推理
    ds = BERTDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, max_len)
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

    # 添加预测列
    train_df["pred"] = preds
    train_df["pred_prob"] = probs.max(axis=1)
    for i in range(6):
        train_df[f"prob_{i}"] = probs[:, i]

    # 提取愤怒(3)↔悲伤(2) 混淆
    LABELS = ["开心", "感激", "悲伤", "愤怒", "恐惧", "焦虑"]
    # 原标愤怒(3) 但模型预测悲伤(2)
    angry_to_sad = train_df[(train_df["label"] == 3) & (train_df["pred"] == 2)].copy()
    # 原标悲伤(2) 但模型预测愤怒(3)
    sad_to_angry = train_df[(train_df["label"] == 2) & (train_df["pred"] == 3)].copy()

    print(f"\n愤怒→悲伤（模型判）: {len(angry_to_sad)} 条")
    print(f"悲伤→愤怒（模型判）: {len(sad_to_angry)} 条")
    print(f"合计: {len(angry_to_sad) + len(sad_to_angry)} 条")

    # 合并并按置信度排序（高置信排前）
    confused = pd.concat([angry_to_sad, sad_to_angry], ignore_index=True)
    confused = confused.sort_values("pred_prob", ascending=False).reset_index(drop=True)
    confused["核验编号"] = range(len(confused), 0, -1)

    # 输出
    out_cols = ["核验编号", "text", "label", "label_cn", "pred", "pred_prob",
                "prob_0", "prob_1", "prob_2", "prob_3", "prob_4", "prob_5"]
    confused_out = confused[out_cols].copy()
    confused_out["混淆方向"] = confused_out.apply(
        lambda r: f"{LABELS[r['label']]}→{LABELS[r['pred']]}" if r['label'] != r['pred'] else "", axis=1)
    confused_out = confused_out[["核验编号", "混淆方向"] + out_cols[1:]]

    out_path = Path("reports/train_confusion_angry_sad.csv")
    confused_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n输出: {out_path} ({len(confused_out)} 条)")

    # 标签分布
    print(f"\n混淆样本原标签分布:")
    print(confused_out["label"].value_counts())


if __name__ == "__main__":
    main()
