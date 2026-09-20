"""
BERT 微调训练脚本
========================
核心改进（相比 TextCNN/BiLSTM）：
1. 使用预训练 BERT 权重，而不是从零训练
2. 学习率调度：Warmup + Cosine Decay（BERT 微调的标准做法）
3. 分层学习率：BERT 底层用小学习率，分类头用大学习率
4. 梯度裁剪：防止微调时梯度爆炸
"""
import os
import sys
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# ===== HuggingFace 国内镜像 =====
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils import load_config
from src.data.dataset import load_data, get_texts_labels
from src.models.bert_dataset import BERTDataset
from src.models.bert_classifier import BERTClassifier
from src.evaluate.metrics import evaluate


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()  # 更新学习率

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / len(dataloader), all_preds, all_labels


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return total_loss / len(dataloader), all_preds, all_labels


def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    pretrained_model = cfg["model"]["pretrained_model"]
    print(f"预训练模型: {pretrained_model}")

    # ===== 1. 加载数据 =====
    train_df, val_df, test_df = load_data()
    X_train, y_train = get_texts_labels(train_df)
    X_val, y_val = get_texts_labels(val_df)
    X_test, y_test = get_texts_labels(test_df)

    # ===== 2. 加载 Tokenizer =====
    print("加载 BERT Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    # ===== 3. 创建 Dataset 和 DataLoader =====
    max_len = cfg["data"]["max_length"]
    train_dataset = BERTDataset(X_train, y_train, tokenizer, max_len)
    val_dataset = BERTDataset(X_val, y_val, tokenizer, max_len)
    test_dataset = BERTDataset(X_test, y_test, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=0)

    # ===== 4. 加载模型 =====
    print("加载预训练 BERT 模型...")
    model = BERTClassifier(
        pretrained_model_name=pretrained_model,
        num_classes=cfg["model"]["num_labels"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # ===== 5. 分层学习率 =====
    # BERT 预训练层用小学习率（2e-5），分类头用大学习率（1e-3）
    # 原因：预训练层已经学好了，只需要微调；分类头是随机初始化的，需要大学习率
    bert_params = list(model.bert.parameters())
    head_params = list(model.classifier.parameters()) + list(model.dropout.parameters())

    optimizer = torch.optim.AdamW([
        {"params": bert_params, "lr": cfg["train"]["learning_rate"]},      # 2e-5
        {"params": head_params, "lr": cfg["train"]["learning_rate"] * 50},  # 1e-3
    ], weight_decay=cfg["train"]["weight_decay"])

    # ===== 6. 学习率调度：Warmup + Cosine Decay =====
    # Warmup：前几个 step 线性增加学习率，防止初期不稳定
    # Cosine Decay：之后余弦衰减到 0
    total_steps = len(train_loader) * cfg["train"]["epochs"]
    warmup_steps = int(total_steps * cfg["train"]["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # ===== 7. 损失函数（类别不平衡处理）=====
    # 用平方根缩放权重，避免过度偏向少数类
    pos_weight = math.sqrt(y_train.count(0) / y_train.count(1))
    class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ===== 8. 训练循环 =====
    best_val_f1 = 0
    patience_counter = 0
    save_dir = Path(cfg["train"]["save_dir"])
    save_dir.mkdir(exist_ok=True)

    print(f"\n开始训练 BERT（共 {cfg['train']['epochs']} 个 epoch）...")
    print(f"总步数: {total_steps}, Warmup步数: {warmup_steps}")
    print("-" * 70)

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        start = time.time()

        train_loss, tp, tl = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        train_acc = sum(p == l for p, l in zip(tp, tl)) / len(tl)

        val_loss, vp, vl = validate(model, val_loader, criterion, device)
        val_acc = sum(p == l for p, l in zip(vp, vl)) / len(vl)
        from sklearn.metrics import f1_score
        val_f1 = f1_score(vl, vp, average="binary")

        # 打印当前学习率
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start
        print(f"Epoch {epoch}/{cfg['train']['epochs']} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} | "
              f"LR: {current_lr:.2e} | Time: {elapsed:.1f}s")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            # 保存完整模型（BERT编码器 + 分类头，两者都被微调了）
            torch.save({
                "model_state_dict": model.state_dict(),
                "pretrained_model": pretrained_model,
                "config": {"num_classes": cfg["model"]["num_labels"], "max_len": max_len},
            }, save_dir / "bert_best.pt")
            # 保存 tokenizer
            tokenizer.save_pretrained(save_dir / "bert_tokenizer")
            print(f"  ✓ 验证集 F1 提升，模型已保存！")
        else:
            patience_counter += 1
            if patience_counter >= cfg["train"]["early_stopping_patience"]:
                print(f"  早停：连续 {patience_counter} 个 epoch 未提升")
                break

    print("-" * 70)
    print(f"训练完成！最佳验证集 F1: {best_val_f1:.4f}")

    # ===== 9. 测试集评估 =====
    # 重新加载完整模型（BERT编码器 + 分类头的微调权重）
    print("\n加载最优模型进行测试集评估...")
    model = BERTClassifier(
        pretrained_model_name=pretrained_model,
        num_classes=cfg["model"]["num_labels"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    checkpoint = torch.load(save_dir / "bert_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, test_preds, test_labels = validate(model, test_loader, criterion, device)
    results = evaluate(test_labels, test_preds, "BERT (测试集)")

    return results


if __name__ == "__main__":
    main()
