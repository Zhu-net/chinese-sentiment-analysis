# -*- coding: utf-8 -*-
"""
反馈纠错样本 -> 增量续训
========================
闭环最后一环：用户在前端纠正的样本（SQLite 中 corrected_label）
→ 导出并与原训练集合并去重
→ 从 bert_emotion_best.pt 续训少量 epoch（小学习率）
→ 保存为独立权重 bert_emotion_retrained.pt（不覆盖原模型，验证满意后手动替换）

用法：
    python src/train/retrain_from_feedback.py
    python src/train/retrain_from_feedback.py --epochs 2 --lr 5e-6
"""
import argparse
import sys
import time
from pathlib import Path
from collections import Counter

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils import load_config
from src.models.bert_dataset import BERTDataset
from src.models.bert_classifier import BERTClassifier
from src.storage.history import export_feedback_dataframe
from src.evaluate.metrics import evaluate


def run_epoch(model, dataloader, criterion, device, optimizer=None, scheduler=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in dataloader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            if train:
                optimizer.zero_grad()
            logits = model(ids, mask)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            total_loss += loss.item()
            all_preds.extend(torch.argmax(logits, 1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return total_loss / len(dataloader), all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description="基于用户纠错样本增量续训情绪模型")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6, help="续训学习率（小于初次微调）")
    args = parser.parse_args()

    cfg = load_config()
    ecfg = cfg["emotion"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(cfg["train"]["save_dir"])
    data_dir = Path(cfg["data"]["processed_dir_emotion"])
    max_len = cfg["data"]["max_length"]

    # 1. 取纠错样本
    fb_df = export_feedback_dataframe()
    if fb_df.empty:
        print("纠错样本为空：请先在前端「历史与纠错」页纠正若干预测再来。")
        return
    print(f"纠错样本: {len(fb_df)} 条，分布: {dict(Counter(fb_df['label_name']))}")

    # 2. 合并原训练集，按 text 去重（纠错样本覆盖同文本旧标签）
    train_df = pd.read_csv(data_dir / "train.csv")
    cols = ["text", "label", "label_name", "label_cn", "source"]
    merged = pd.concat([train_df[cols], fb_df[cols]], ignore_index=True)
    merged = merged.drop_duplicates(subset=["text"], keep="last").reset_index(drop=True)
    print(f"合并去重后训练集: {len(train_df)} -> {len(merged)}")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["pretrained_model"])
    train_ds = BERTDataset(merged["text"].tolist(), merged["label"].tolist(), tokenizer, max_len)
    val_ds = BERTDataset(val_df["text"].tolist(), val_df["label"].tolist(), tokenizer, max_len)
    test_ds = BERTDataset(test_df["text"].tolist(), test_df["label"].tolist(), tokenizer, max_len)

    class_counts = Counter(merged["label"].tolist())
    weights = [1.0 / class_counts[y] for y in merged["label"].tolist()]
    sampler = WeightedRandomSampler(weights, num_samples=len(merged), replacement=True)
    bs = cfg["train"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0)

    # 3. 从最佳权重续训（非破坏性，输出新文件）
    model = BERTClassifier(cfg["model"]["pretrained_model"],
                           num_classes=ecfg["num_labels"], dropout=0.15).to(device)
    ckpt_path = save_dir / ecfg["model_file"]
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"已加载基线权重: {ckpt_path}")

    criterion = nn.CrossEntropyLoss()
    head = list(model.classifier.parameters()) + list(model.dropout.parameters())
    optimizer = torch.optim.AdamW([
        {"params": model.bert.parameters(), "lr": args.lr},
        {"params": head, "lr": args.lr * 50},
    ], weight_decay=cfg["train"]["weight_decay"])
    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * 0.1), total_steps)

    # 4. 续训循环（macro-F1 选模）
    out_file = "bert_emotion_retrained.pt"
    best_f1 = 0
    print(f"\n开始续训 {args.epochs} epoch（lr={args.lr}）...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, _, _ = run_epoch(model, train_loader, criterion, device, optimizer, scheduler)
        _, vp, vl = run_epoch(model, val_loader, criterion, device)
        from sklearn.metrics import f1_score, accuracy_score
        mf1 = f1_score(vl, vp, average="macro", zero_division=0)
        acc = accuracy_score(vl, vp)
        print(f"Epoch {epoch}/{args.epochs} | loss {tr_loss:.4f} | "
              f"val acc {acc:.4f} macroF1 {mf1:.4f} | {time.time()-t0:.0f}s")
        if mf1 > best_f1:
            best_f1 = mf1
            torch.save({
                "model_state_dict": model.state_dict(),
                "pretrained_model": cfg["model"]["pretrained_model"],
                "config": {"num_classes": ecfg["num_labels"], "max_len": max_len, "dropout": 0.15},
            }, save_dir / out_file)
            tokenizer.save_pretrained(save_dir / ecfg["tokenizer_dir"])
            print("  ✓ 已保存")

    # 5. 测试集评估
    best_ckpt = torch.load(save_dir / out_file, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    _, tp, tl = run_epoch(model, test_loader, criterion, device)
    evaluate(tl, tp, model_name="BERT-情绪6分类 (纠错续训后·测试集)",
             average="macro", target_names=ecfg["labels_cn"])
    print(f"\n新权重: {save_dir / out_file}")
    print("验证满意后，将其替换为 config.yaml 中 emotion.model_file 指向的文件即可上线。")


if __name__ == "__main__":
    main()
