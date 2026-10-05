# -*- coding: utf-8 -*-
"""
BERT 情绪 6 分类训练
====================
与二分类 train_bert.py 的关键差异：
1. 数据：data/processed_emotion/（6 类微博情绪，含弱标注的 grateful/anxious）
2. 极端类别不平衡（愤怒 12636 vs 焦虑 267，47:1）：
   -> WeightedRandomSampler 类均衡采样（每个 batch 各类期望曝光均等）
   -> 不再叠加 class weight，避免双重补偿；val/test 保持自然分布
3. 早停/选模指标：macro-F1（各类等权，不受大类主导）
4. 模型权重独立保存：bert_emotion_best.pt
"""
import os
import sys
import time
from pathlib import Path
from collections import Counter

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.utils import load_config
from src.models.bert_dataset import BERTDataset
from src.models.bert_classifier import BERTClassifier
from src.evaluate.metrics import evaluate


def run_epoch(model, dataloader, criterion, device, optimizer=None, scheduler=None):
    """通用 epoch：传了 optimizer 就是训练，否则验证"""
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if train:
                optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

            total_loss += loss.item()
            all_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return total_loss / len(dataloader), all_preds, all_labels


def main():
    cfg = load_config()
    ecfg = cfg["emotion"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    num_classes = ecfg["num_labels"]
    label_names_en = ecfg["labels"]
    label_names_cn = ecfg["labels_cn"]
    pretrained_model = cfg["model"]["pretrained_model"]
    data_dir = Path(cfg["data"]["processed_dir_emotion"])
    max_len = cfg["data"]["max_length"]

    # ===== 1. 数据 =====
    # 使用 train+test 标签修正后数据（清洗 + test 278条 + train 833条核验修正）
    train_df = pd.read_csv(data_dir / "train_augmented_v2_cleaned_fixed2.csv")
    val_df = pd.read_csv(data_dir / "val_cleaned_fixed.csv")
    test_df = pd.read_csv(data_dir / "test_cleaned_fixed_anx.csv")  # 含焦虑标签修正
    print(f"数据: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # ===== 2. Tokenizer / Dataset =====
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    train_ds = BERTDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, max_len)
    val_ds = BERTDataset(val_df["text"].tolist(), val_df["label"].tolist(), tokenizer, max_len)
    test_ds = BERTDataset(test_df["text"].tolist(), test_df["label"].tolist(), tokenizer, max_len)

    # ===== 3. WeightedRandomSampler：类均衡采样 =====
    # 每个训练样本的被抽中权重 = 1 / 该类样本数
    # 替换式采样，保证每个 epoch 六个类别的期望曝光次数相等
    train_labels = train_df["label"].tolist()
    class_counts = Counter(train_labels)
    print("训练集类别频次:", {label_names_cn[k]: v for k, v in sorted(class_counts.items())})
    sample_weights = [1.0 / class_counts[y] for y in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_labels), replacement=True
    )

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=0)

    # ===== 4. 模型（分类头 dropout 略调大，缓解少数类重复采样过拟合）=====
    print("正在初始化模型...")
    model = BERTClassifier(
        pretrained_model_name=pretrained_model,
        num_classes=num_classes,
        dropout=0.15,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    print("模型初始化完成")

    # ===== 5. 优化器 / 调度 / 损失 =====
    # sampler 已做类均衡，loss 不再加权重，避免双重补偿
    print("正在配置优化器...")
    criterion = nn.CrossEntropyLoss()
    head_params = list(model.classifier.parameters()) + list(model.dropout.parameters())
    bert_params = list(model.bert.parameters())
    optimizer = torch.optim.AdamW([
        {"params": bert_params, "lr": cfg["train"]["learning_rate"]},
        {"params": head_params, "lr": cfg["train"]["learning_rate"] * 50},
    ], weight_decay=cfg["train"]["weight_decay"])

    total_steps = len(train_loader) * cfg["train"]["epochs"]
    warmup_steps = int(total_steps * cfg["train"]["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"总步数: {total_steps}, Warmup: {warmup_steps}")
    print("优化器配置完成")

    # ===== 6. 训练循环（以 macro-F1 选模）=====
    save_dir = Path(cfg["train"]["save_dir"])
    save_dir.mkdir(exist_ok=True)
    best_macro_f1 = 0
    patience = 0

    print(f"\n开始训练情绪分类模型（共 {cfg['train']['epochs']} epoch）...")
    print("-" * 78)
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        t0 = time.time()
        train_loss, tp, tl = run_epoch(model, train_loader, criterion, device, optimizer, scheduler)
        val_loss, vp, vl = run_epoch(model, val_loader, criterion, device)

        from sklearn.metrics import f1_score, accuracy_score
        val_macro = f1_score(vl, vp, average="macro", zero_division=0)
        val_weighted = f1_score(vl, vp, average="weighted", zero_division=0)
        val_acc = accuracy_score(vl, vp)
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{cfg['train']['epochs']} | "
              f"Train Loss {train_loss:.4f} | "
              f"Val Acc {val_acc:.4f} macroF1 {val_macro:.4f} weightedF1 {val_weighted:.4f} | "
              f"LR {lr:.2e} | {time.time()-t0:.0f}s")

        if val_macro > best_macro_f1:
            best_macro_f1 = val_macro
            patience = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "pretrained_model": pretrained_model,
                "config": {"num_classes": num_classes, "max_len": max_len, "dropout": 0.15},
            }, save_dir / ecfg["model_file_fixed2"])
            tokenizer.save_pretrained(save_dir / ecfg["tokenizer_dir"])
            print(f"  [OK] macro-F1 提升，模型已保存")
        else:
            patience += 1
            if patience >= cfg["train"]["early_stopping_patience"]:
                print(f"  早停：连续 {patience} epoch macro-F1 未提升")
                break
    print("-" * 78)
    print(f"训练完成，最佳验证 macro-F1: {best_macro_f1:.4f}")

    # ===== 7. 测试集评估 =====
    checkpoint = torch.load(save_dir / ecfg["model_file_fixed2"], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, test_preds, test_labels = run_epoch(model, test_loader, criterion, device)
    results = evaluate(
        test_labels, test_preds,
        model_name="BERT-情绪6分类 (测试集)",
        average="macro",
        target_names=[f"{cn}" for cn in label_names_cn],
    )
    return results


if __name__ == "__main__":
    main()
