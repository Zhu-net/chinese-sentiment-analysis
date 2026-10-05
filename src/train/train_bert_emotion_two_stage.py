# -*- coding: utf-8 -*-
"""
两阶段训练：CE 预训练 -> 冻结底层 SCL 微调
==========================================
阶段 1（复用，不重训）：bert_emotion_best.pt，纯 CE 训练 5 epoch，Val macro-F1 0.7695
阶段 2（本脚本）：
  - 从阶段 1 加载 BERT + 分类头权重（投影头随机初始化）
  - 冻结 BERT embeddings + 前 N 层 encoder，仅微调后 (12-N) 层 + pooler + 双头
  - 增大 batch（32，配合梯度检查点）为 SCL 提供更多同类正样本对
  - CE + λ·SCL 联合微调，小学习率避免破坏已学好的决策面
  - 保存为 bert_emotion_scl2_best.pt，与 baseline / 端到端 SCL 三方对比

支持断点续训（bert_emotion_scl2_resume.pt）。
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
              optimizer=None, scheduler=None, lambda_scl=0.05):
    train = optimizer is not None
    model.train() if train else model.eval()

    total_loss = total_ce = total_scl = 0.0
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
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
                optimizer.step()
                scheduler.step()

            total_loss += loss.item()
            total_ce += loss_ce.item()
            total_scl += float(loss_scl.detach())
            all_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    n = len(dataloader)
    return total_loss / n, total_ce / n, total_scl / n, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description="两阶段 SCL 微调")
    parser.add_argument("--fresh", action="store_true", help="忽略续训 checkpoint 重新开始阶段2")
    args = parser.parse_args()

    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        print("[电源] 已阻止系统休眠，训练结束后自动恢复")

    cfg = load_config()
    ecfg = cfg["emotion"]
    ts_cfg = cfg["train"]["two_stage"]
    scl_proj_dim = cfg["train"]["scl"]["projection_dim"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    num_classes = ecfg["num_labels"]
    label_names_cn = ecfg["labels_cn"]
    pretrained_model = cfg["model"]["pretrained_model"]
    data_dir = Path(cfg["data"]["processed_dir_emotion"])
    max_len = cfg["data"]["max_length"]

    save_dir = Path(cfg["train"]["save_dir"])
    stage1_path = save_dir / ts_cfg["stage1_ckpt"]
    model_file = ts_cfg["model_file"]
    resume_file = save_dir / "bert_emotion_scl2_resume.pt"

    freeze_n = int(ts_cfg["freeze_layers"])
    batch_size = int(ts_cfg["batch_size"])
    epochs = int(ts_cfg["epochs"])
    lr = float(ts_cfg["learning_rate"])
    lambda_scl = float(ts_cfg["lambda_scl"])
    temperature = float(ts_cfg["temperature"])
    use_gc = bool(ts_cfg["gradient_checkpointing"])

    # ===== 1. 数据（与 baseline 完全一致的划分）=====
    train_df = pd.read_csv(data_dir / "train_augmented_v2.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    print(f"数据: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    train_ds = BERTDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, max_len)
    val_ds = BERTDataset(val_df["text"].tolist(), val_df["label"].tolist(), tokenizer, max_len)
    test_ds = BERTDataset(test_df["text"].tolist(), test_df["label"].tolist(), tokenizer, max_len)

    train_labels = train_df["label"].tolist()
    class_counts = Counter(train_labels)
    sample_weights = [1.0 / class_counts[y] for y in train_labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_labels), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # ===== 2. 构建模型 =====
    model = ContrastiveBERTClassifier(
        pretrained_model_name=pretrained_model,
        num_classes=num_classes,
        dropout=0.15,
        projection_dim=scl_proj_dim,
    ).to(device)

    # ===== 3. 阶段 1 -> 阶段 2：加载 CE 预训练权重 =====
    print("-" * 78)
    print(f"[阶段1] 加载 CE 预训练权重: {stage1_path}")
    stage1_ckpt = torch.load(stage1_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(stage1_ckpt["model_state_dict"], strict=False)
    # 预期 missing: projection 投影头参数（随机初始化）；预期 unexpected: 无
    proj_missing = [k for k in missing if k.startswith("projection")]
    other_missing = [k for k in missing if not k.startswith("projection")]
    print(f"  随机初始化的投影头参数: {len(proj_missing)} 个")
    if other_missing:
        print(f"  [警告] 非投影头缺失参数: {other_missing}")
    if unexpected:
        print(f"  [警告] 多余未加载参数: {unexpected}")

    # ===== 4. 阶段 2：冻结 BERT 底层 =====
    print(f"[阶段2] 冻结 embeddings + encoder 第 0~{freeze_n-1} 层，"
          f"微调第 {freeze_n}~11 层 + pooler + classifier + projection")
    for p in model.bert.embeddings.parameters():
        p.requires_grad = False
    for layer in model.bert.encoder.layer[:freeze_n]:
        for p in layer.parameters():
            p.requires_grad = False

    if use_gc:
        # 梯度检查点：冻结部分参数时必须打开 input_require_grads，否则段内无梯度
        model.bert.gradient_checkpointing_enable()
        model.bert.enable_input_require_grads()
        model.bert.config.use_cache = False
        print("  梯度检查点: 已启用（省显存，训练慢约 20-30%）")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = trainable + frozen
    print(f"  可训练参数: {trainable:,} / {total:,} ({trainable/total:.1%})，"
          f"冻结参数: {frozen:,} ({frozen/total:.1%})")

    # 冻结层健全性校验：列出实际 requires_grad=True 的 BERT encoder 层
    trainable_layers = [
        i for i, layer in enumerate(model.bert.encoder.layer)
        if any(p.requires_grad for p in layer.parameters())
    ]
    print(f"  实际可训练 encoder 层索引: {trainable_layers}")
    assert trainable_layers == list(range(freeze_n, 12)), "冻结层与预期不符，中止"
    print("-" * 78)

    # ===== 5. 优化器（只含 trainable 参数，两组学习率）=====
    criterion = nn.CrossEntropyLoss()
    scl_criterion = SupervisedContrastiveLoss(temperature=temperature).to(device)

    # 优化器只包含 requires_grad=True 的参数（冻结层不进参数组）
    head_params = list(model.classifier.parameters()) + list(model.projection.parameters())
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.bert.parameters() if p.requires_grad], "lr": lr},
        {"params": head_params, "lr": lr * 50},
    ], weight_decay=cfg["train"]["weight_decay"])

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * cfg["train"]["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"batch_size={batch_size}, steps/epoch={len(train_loader)}, "
          f"total_steps={total_steps}, warmup={warmup_steps}")
    print(f"lambda_scl={lambda_scl}, temperature={temperature}, lr(bert)={lr:.1e}, lr(head)={lr*50:.1e}")

    # ===== 6. 断点续训恢复 =====
    best_macro_f1 = 0.0
    patience = 0
    start_epoch = 1
    if not args.fresh and resume_file.exists():
        ckpt = torch.load(resume_file, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["next_epoch"]
        best_macro_f1 = ckpt["best_macro_f1"]
        patience = ckpt["patience"]
        print(f"[续训] 从 epoch {start_epoch} 恢复，当前最佳 macro-F1: {best_macro_f1:.4f}")

    # ===== 7. 微调循环 =====
    print(f"\n开始阶段2 SCL 微调（共 {epochs} epoch）...")
    print("-" * 78)
    finished = False
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        tr_loss, tr_ce, tr_scl, _, _ = run_epoch(
            model, train_loader, criterion, scl_criterion, device,
            optimizer, scheduler, lambda_scl)
        va_loss, _, _, vp, vl = run_epoch(
            model, val_loader, criterion, None, device)

        from sklearn.metrics import f1_score, accuracy_score
        val_macro = f1_score(vl, vp, average="macro", zero_division=0)
        val_weighted = f1_score(vl, vp, average="weighted", zero_division=0)
        val_acc = accuracy_score(vl, vp)
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{epochs} | "
              f"Train {tr_loss:.4f} (CE {tr_ce:.4f} + SCL {tr_scl:.4f}) | "
              f"Val Acc {val_acc:.4f} macroF1 {val_macro:.4f} weightedF1 {val_weighted:.4f} | "
              f"LR {cur_lr:.2e} | {time.time()-t0:.0f}s")

        if val_macro > best_macro_f1:
            best_macro_f1 = val_macro
            patience = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "pretrained_model": pretrained_model,
                "config": {"num_classes": num_classes, "max_len": max_len,
                           "dropout": 0.15, "projection_dim": scl_proj_dim},
            }, save_dir / model_file)
            tokenizer.save_pretrained(save_dir / ecfg["tokenizer_dir"])
            print(f"  [OK] macro-F1 提升，模型已保存 -> {model_file}")
        else:
            patience += 1

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "next_epoch": epoch + 1,
            "best_macro_f1": best_macro_f1,
            "patience": patience,
        }, str(resume_file) + ".tmp")
        os.replace(str(resume_file) + ".tmp", resume_file)

        if patience >= cfg["train"]["early_stopping_patience"]:
            print(f"  早停：连续 {patience} epoch macro-F1 未提升")
            finished = True
            break
    else:
        finished = True
    print("-" * 78)
    print(f"阶段2完成，最佳验证 macro-F1: {best_macro_f1:.4f}（阶段1 baseline: 0.7695）")

    if finished and resume_file.exists():
        resume_file.unlink()
        print("[续训] checkpoint 已清理")

    # ===== 8. 测试集评估 =====
    ckpt = torch.load(save_dir / model_file, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    _, _, _, test_preds, test_labels = run_epoch(
        model, test_loader, criterion, None, device)
    results = evaluate(
        test_labels, test_preds,
        model_name="BERT-情绪6分类-两阶段SCL (测试集)",
        average="macro",
        target_names=list(label_names_cn),
    )

    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    return results


if __name__ == "__main__":
    main()
