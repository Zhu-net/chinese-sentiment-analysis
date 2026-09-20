"""
BiLSTM 训练脚本
复用 TextCNN 的训练逻辑，仅替换模型为 BiLSTM
"""
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils import load_config
from src.data.dataset import load_data, get_texts_labels
from src.models.text_dataset import Vocab, TextDataset
from src.models.bilstm import BiLSTM
from src.evaluate.metrics import evaluate


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for input_ids, labels in dataloader:
        input_ids, labels = input_ids.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        # 梯度裁剪：防止梯度爆炸（LSTM 训练中很重要）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
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
        for input_ids, labels in dataloader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            logits = model(input_ids)
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

    # 1. 数据
    train_df, val_df, test_df = load_data()
    X_train, y_train = get_texts_labels(train_df)
    X_val, y_val = get_texts_labels(val_df)
    X_test, y_test = get_texts_labels(test_df)

    # 2. 词表
    max_len = cfg["data"]["max_length"]
    vocab = Vocab(min_freq=2)
    vocab.build(X_train)
    print(f"词表大小: {len(vocab)}")

    # 3. DataLoader
    train_dataset = TextDataset(X_train, y_train, vocab, max_len)
    val_dataset = TextDataset(X_val, y_val, vocab, max_len)
    test_dataset = TextDataset(X_test, y_test, vocab, max_len)
    train_loader = DataLoader(train_dataset, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=0)

    # 4. 模型
    model = BiLSTM(
        vocab_size=len(vocab),
        embed_dim=128,
        hidden_dim=128,
        num_layers=2,
        num_classes=cfg["model"]["num_labels"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 5. 损失 + 优化器
    pos_weight = y_train.count(0) / y_train.count(1)
    class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # 6. 训练
    best_val_f1 = 0
    patience_counter = 0
    save_dir = Path(cfg["train"]["save_dir"])
    save_dir.mkdir(exist_ok=True)

    print(f"\n开始训练 BiLSTM（共 {cfg['train']['epochs']} 个 epoch）...")
    print("-" * 60)

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        start = time.time()
        train_loss, tp, tl = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_acc = sum(p == l for p, l in zip(tp, tl)) / len(tl)

        val_loss, vp, vl = validate(model, val_loader, criterion, device)
        val_acc = sum(p == l for p, l in zip(vp, vl)) / len(vl)
        from sklearn.metrics import f1_score
        val_f1 = f1_score(vl, vp, average="binary")

        print(f"Epoch {epoch}/{cfg['train']['epochs']} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} | "
              f"Time: {time.time()-start:.1f}s")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab": vocab,
                "config": {
                    "vocab_size": len(vocab), "embed_dim": 128, "hidden_dim": 128,
                    "num_layers": 2, "num_classes": cfg["model"]["num_labels"], "max_len": max_len,
                },
            }, save_dir / "bilstm_best.pt")
            print(f"  ✓ 验证集 F1 提升，模型已保存！")
        else:
            patience_counter += 1
            if patience_counter >= cfg["train"]["early_stopping_patience"]:
                print(f"  早停：连续 {patience_counter} 个 epoch 未提升")
                break

    print("-" * 60)
    print(f"训练完成！最佳验证集 F1: {best_val_f1:.4f}")

    # 7. 测试集评估
    checkpoint = torch.load(save_dir / "bilstm_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, test_preds, test_labels = validate(model, test_loader, criterion, device)
    results = evaluate(test_labels, test_preds, "BiLSTM (测试集)")
    return results


if __name__ == "__main__":
    main()
