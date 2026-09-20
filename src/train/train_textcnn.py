"""
TextCNN 训练脚本
========================
PyTorch 训练的核心流程（必会）：
1. 设置设备（GPU/CPU）
2. 准备数据（Dataset + DataLoader）
3. 实例化模型
4. 定义损失函数和优化器
5. 训练循环：
   a. 前向传播（forward）：输入 -> 模型 -> 预测
   b. 计算损失（loss）：预测 vs 真实标签
   c. 反向传播（backward）：计算梯度
   d. 更新参数（optimizer.step）
6. 验证循环（不计算梯度，评估模型）
7. 保存最优模型
"""
import sys
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils import load_config
from src.data.dataset import load_data, get_texts_labels
from src.models.text_dataset import Vocab, TextDataset
from src.models.textcnn import TextCNN
from src.evaluate.metrics import evaluate


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()  # 设置为训练模式（启用 Dropout、BatchNorm 等）
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch_idx, (input_ids, labels) in enumerate(dataloader):
        # 把数据移动到 GPU（如果可用）
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        # ===== a. 前向传播 =====
        # 清空上一次的梯度（PyTorch 梯度默认累加）
        optimizer.zero_grad()
        # 模型前向计算，得到 logits（未归一化的分数）
        logits = model(input_ids)

        # ===== b. 计算损失 =====
        # CrossEntropyLoss 内部会做 Softmax，所以输入是 logits 不是概率
        loss = criterion(logits, labels)

        # ===== c. 反向传播 =====
        # 自动计算所有参数的梯度
        loss.backward()

        # ===== d. 更新参数 =====
        # 根据梯度更新模型参数
        optimizer.step()

        # 记录损失和预测
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)  # 取概率最大的类别作为预测
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(dataloader)
    return avg_loss, all_preds, all_labels


def validate(model, dataloader, criterion, device):
    """验证模型（不计算梯度）"""
    model.eval()  # 设置为评估模式（关闭 Dropout）
    total_loss = 0
    all_preds = []
    all_labels = []

    # torch.no_grad() 上下文：不计算梯度，节省内存和加速
    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits = model(input_ids)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(dataloader)
    return avg_loss, all_preds, all_labels


def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ===== 1. 加载数据 =====
    train_df, val_df, test_df = load_data()
    X_train, y_train = get_texts_labels(train_df)
    X_val, y_val = get_texts_labels(val_df)
    X_test, y_test = get_texts_labels(test_df)

    # ===== 2. 构建词表 =====
    max_len = cfg["data"]["max_length"]
    vocab = Vocab(min_freq=2)
    vocab.build(X_train)
    print(f"词表大小: {len(vocab)}")

    # ===== 3. 创建 Dataset 和 DataLoader =====
    train_dataset = TextDataset(X_train, y_train, vocab, max_len)
    val_dataset = TextDataset(X_val, y_val, vocab, max_len)
    test_dataset = TextDataset(X_test, y_test, vocab, max_len)

    # DataLoader: 批量加载数据
    # batch_size: 每次喂给模型的样本数
    # shuffle: 训练时打乱数据顺序
    # num_workers: 数据加载的进程数（Windows 下设为 0 更稳定）
    train_loader = DataLoader(train_dataset, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=0)

    # ===== 4. 实例化模型 =====
    model = TextCNN(
        vocab_size=len(vocab),
        embed_dim=128,
        num_filters=128,
        kernel_sizes=(2, 3, 4),
        num_classes=cfg["model"]["num_labels"],
        dropout=cfg["model"]["dropout"],
    ).to(device)  # 把模型移动到 GPU

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # ===== 5. 定义损失函数和优化器 =====
    # CrossEntropyLoss: 交叉熵损失，用于分类任务
    # 处理类别不平衡：给少数类（正面=1）更高权重
    pos_weight = (y_train.count(0) / y_train.count(1))  # 负面数/正面数
    class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Adam 优化器：自适应学习率，深度学习最常用
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # ===== 6. 训练循环 =====
    best_val_f1 = 0
    patience_counter = 0
    epochs = cfg["train"]["epochs"]

    print(f"\n开始训练 TextCNN（共 {epochs} 个 epoch）...")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # 训练
        train_loss, train_preds, train_labels = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        train_acc = sum(p == l for p, l in zip(train_preds, train_labels)) / len(train_labels)

        # 验证
        val_loss, val_preds, val_labels = validate(model, val_loader, criterion, device)
        val_acc = sum(p == l for p, l in zip(val_preds, val_labels)) / len(val_labels)

        # 计算验证集 F1
        from sklearn.metrics import f1_score
        val_f1 = f1_score(val_labels, val_preds, average="binary")

        elapsed = time.time() - start_time
        print(f"Epoch {epoch}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} | "
              f"Time: {elapsed:.1f}s")

        # ===== 早停机制 =====
        # 如果验证集 F1 提升，保存最优模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            # 保存模型（保存模型参数 + 词表，方便推理时加载）
            save_dir = Path(cfg["train"]["save_dir"])
            save_dir.mkdir(exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab": vocab,
                "config": {
                    "vocab_size": len(vocab),
                    "embed_dim": 128,
                    "num_filters": 128,
                    "kernel_sizes": (2, 3, 4),
                    "num_classes": cfg["model"]["num_labels"],
                    "max_len": max_len,
                },
            }, save_dir / "textcnn_best.pt")
            print(f"  ✓ 验证集 F1 提升，模型已保存！")
        else:
            patience_counter += 1
            if patience_counter >= cfg["train"]["early_stopping_patience"]:
                print(f"  早停：验证集 F1 连续 {patience_counter} 个 epoch 未提升")
                break

    print("-" * 60)
    print(f"训练完成！最佳验证集 F1: {best_val_f1:.4f}")

    # ===== 7. 测试集评估 =====
    # 加载最优模型
    checkpoint = torch.load(save_dir / "textcnn_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_preds, test_labels = validate(model, test_loader, criterion, device)
    results = evaluate(test_labels, test_preds, "TextCNN (测试集)")

    return results


if __name__ == "__main__":
    main()
