# -*- coding: utf-8 -*-
"""
BERT 情绪 6 分类训练（监督对比学习版）
=======================================
在 train_bert_emotion.py 基础上引入 Supervised Contrastive Learning：
  loss = CE(logits, labels) + lambda_scl * SCL(embeddings, labels)

对比损失作用：
  让同类样本在投影空间聚拢、异类分离，直接拉开易混淆类别（悲伤 vs 愤怒）的特征边界。

关键设计：
- 投影头输出经 L2 归一化，保证 SCL 数值稳定
- 验证/测试阶段只算分类指标，不参与 SCL
- 模型独立保存为 bert_emotion_scl_best.pt，便于与 baseline 对比
"""
import os
import sys
import time
import argparse
from pathlib import Path
from collections import Counter

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Windows 控制台编码兜底，避免 gbk 打印中文/emoji 崩溃
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.utils import load_config
from src.models.bert_dataset import BERTDataset
from src.models.contrastive_bert_classifier import ContrastiveBERTClassifier
from src.train.losses import SupervisedContrastiveLoss
from src.evaluate.metrics import evaluate


def run_epoch(model, dataloader, criterion, scl_criterion, device,
              optimizer=None, scheduler=None, lambda_scl=0.1):
    """通用 epoch：传了 optimizer 就是训练，否则验证"""
    train = optimizer is not None
    model.train() if train else model.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_scl = 0.0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if train:
                optimizer.zero_grad()

            logits, embeddings = model(input_ids, attention_mask)

            loss_ce = criterion(logits, labels)
            if train and scl_criterion is not None:
                loss_scl = scl_criterion(embeddings, labels)
                loss = loss_ce + lambda_scl * loss_scl
            else:
                loss_scl = torch.tensor(0.0, device=device)
                loss = loss_ce

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

            total_loss += loss.item()
            total_ce += loss_ce.item()
            total_scl += loss_scl.item() if isinstance(loss_scl, torch.Tensor) else loss_scl
            all_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    n = len(dataloader)
    return total_loss / n, total_ce / n, total_scl / n, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description="SCL 对比学习情绪分类训练")
    parser.add_argument("--fresh", action="store_true",
                        help="忽略已有续训 checkpoint，从头开始训练")
    args = parser.parse_args()

    # 训练期间阻止 Windows 自动休眠/睡眠（长训练夜跑防中断）
    if os.name == "nt":
        import ctypes
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        print("[电源] 已阻止系统休眠，训练结束后自动恢复")

    cfg = load_config()
    ecfg = cfg["emotion"]
    tcfg = cfg["train"]
    scl_cfg = tcfg["scl"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    num_classes = ecfg["num_labels"]
    label_names_cn = ecfg["labels_cn"]
    pretrained_model = cfg["model"]["pretrained_model"]
    data_dir = Path(cfg["data"]["processed_dir_emotion"])
    max_len = cfg["data"]["max_length"]

    lambda_scl = scl_cfg["lambda_scl"]
    projection_dim = scl_cfg["projection_dim"]
    print(f"对比学习: lambda_scl={lambda_scl}, temperature={scl_cfg['temperature']}, "
          f"projection_dim={projection_dim}")

    # ===== 1. 数据 =====
    train_df = pd.read_csv(data_dir / "train_augmented_v2.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    print(f"数据: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # ===== 2. Tokenizer / Dataset =====
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    train_ds = BERTDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, max_len)
    val_ds = BERTDataset(val_df["text"].tolist(), val_df["label"].tolist(), tokenizer, max_len)
    test_ds = BERTDataset(test_df["text"].tolist(), test_df["label"].tolist(), tokenizer, max_len)

    # ===== 3. WeightedRandomSampler：类均衡采样 =====
    train_labels = train_df["label"].tolist()
    class_counts = Counter(train_labels)
    print("训练集类别频次:", {label_names_cn[k]: v for k, v in sorted(class_counts.items())})
    sample_weights = [1.0 / class_counts[y] for y in train_labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_labels), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=tcfg["batch_size"], shuffle=False, num_workers=0)

    # ===== 4. 对比学习模型 =====
    print("正在初始化对比学习模型...")
    model = ContrastiveBERTClassifier(
        pretrained_model_name=pretrained_model,
        num_classes=num_classes,
        dropout=0.15,
        projection_dim=projection_dim,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    print("模型初始化完成")

    # ===== 5. 优化器 / 调度 / 损失 =====
    criterion = nn.CrossEntropyLoss()
    scl_criterion = SupervisedContrastiveLoss(temperature=scl_cfg["temperature"]).to(device)

    head_params = list(model.classifier.parameters()) + list(model.projection.parameters())
    bert_params = list(model.bert.parameters())
    optimizer = torch.optim.AdamW([
        {"params": bert_params, "lr": tcfg["learning_rate"]},
        {"params": head_params, "lr": tcfg["learning_rate"] * 50},
    ], weight_decay=tcfg["weight_decay"])

    total_steps = len(train_loader) * tcfg["epochs"]
    warmup_steps = int(total_steps * tcfg["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"总步数: {total_steps}, Warmup: {warmup_steps}")

    # ===== 6. 训练循环（以 macro-F1 选模，支持断点续训）=====
    save_dir = Path(tcfg["save_dir"])
    save_dir.mkdir(exist_ok=True)
    model_file = ecfg.get("model_file_scl", "bert_emotion_scl_best.pt")
    resume_file = save_dir / "bert_emotion_scl_resume.pt"
    best_macro_f1 = 0
    patience = 0
    start_epoch = 1

    # 恢复上次中断的训练状态（模型/优化器/调度器/epoch/最佳指标）
    if not args.fresh and resume_file.exists():
        ckpt = torch.load(resume_file, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["next_epoch"]
        best_macro_f1 = ckpt["best_macro_f1"]
        patience = ckpt["patience"]
        print(f"[续训] 从 epoch {start_epoch} 恢复，当前最佳 macro-F1: {best_macro_f1:.4f}")

    print(f"\n开始训练情绪分类模型（对比学习版，共 {tcfg['epochs']} epoch）...")
    print("-" * 78)
    finished = False
    for epoch in range(start_epoch, tcfg["epochs"] + 1):
        t0 = time.time()
        train_loss, train_ce, train_scl, tp, tl = run_epoch(
            model, train_loader, criterion, scl_criterion, device,
            optimizer, scheduler, lambda_scl)
        val_loss, val_ce, val_scl, vp, vl = run_epoch(
            model, val_loader, criterion, None, device)

        from sklearn.metrics import f1_score, accuracy_score
        val_macro = f1_score(vl, vp, average="macro", zero_division=0)
        val_weighted = f1_score(vl, vp, average="weighted", zero_division=0)
        val_acc = accuracy_score(vl, vp)
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{tcfg['epochs']} | "
              f"Train Loss {train_loss:.4f} (CE {train_ce:.4f} + SCL {train_scl:.4f}) | "
              f"Val Acc {val_acc:.4f} macroF1 {val_macro:.4f} weightedF1 {val_weighted:.4f} | "
              f"LR {lr:.2e} | {time.time()-t0:.0f}s")

        if val_macro > best_macro_f1:
            best_macro_f1 = val_macro
            patience = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "pretrained_model": pretrained_model,
                "config": {"num_classes": num_classes, "max_len": max_len,
                           "dropout": 0.15, "projection_dim": projection_dim},
            }, save_dir / model_file)
            tokenizer.save_pretrained(save_dir / ecfg["tokenizer_dir"])
            print(f"  [OK] macro-F1 提升，模型已保存 -> {model_file}")
        else:
            patience += 1

        # 每个 epoch 后落盘续训状态（先写临时文件再替换，避免写一半损坏）
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "next_epoch": epoch + 1,
            "best_macro_f1": best_macro_f1,
            "patience": patience,
        }, str(resume_file) + ".tmp")
        os.replace(str(resume_file) + ".tmp", resume_file)

        if patience >= tcfg["early_stopping_patience"]:
            print(f"  早停：连续 {patience} epoch macro-F1 未提升")
            finished = True
            break
    else:
        finished = True
    print("-" * 78)
    print(f"训练完成，最佳验证 macro-F1: {best_macro_f1:.4f}")

    # 全部 epoch 跑完或触发早停后，删除续训文件
    if finished and resume_file.exists():
        resume_file.unlink()
        print("[续训] 训练已全部完成，续训 checkpoint 已清理")

    # ===== 7. 测试集评估 =====
    checkpoint = torch.load(save_dir / model_file, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, _, _, test_preds, test_labels = run_epoch(
        model, test_loader, criterion, None, device)
    results = evaluate(
        test_labels, test_preds,
        model_name="BERT-情绪6分类-SCL (测试集)",
        average="macro",
        target_names=[f"{cn}" for cn in label_names_cn],
    )

    # 恢复系统默认电源策略
    if os.name == "nt":
        import ctypes
        ES_CONTINUOUS = 0x80000000
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    return results


if __name__ == "__main__":
    main()
