# -*- coding: utf-8 -*-
"""
ONNX 模型导出脚本
==================
将 PyTorch BERT 分类模型导出为 ONNX 格式，推理速度提升 2-3 倍。

导出两个模型：
1. 二分类模型 bert_best.pt -> bert_binary.onnx
2. 情绪6分类 bert_emotion_best.pt -> bert_emotion.onnx

用法：
    python src/deploy/export_onnx.py
    python src/deploy/export_onnx.py --model emotion   # 仅情绪
    python src/deploy/export_onnx.py --model binary     # 仅二分类
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import onnx
import onnxruntime as ort
from transformers import AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils import load_config
from src.models.bert_classifier import BERTClassifier


def export_model(ckpt_path: Path, onnx_path: Path, pretrained_model: str,
                 num_classes: int, max_len: int, tokenizer_dir: str = None):
    """导出单个模型为 ONNX 格式"""
    print(f"\n{'='*50}")
    print(f"导出: {ckpt_path.name} -> {onnx_path.name}")

    device = torch.device("cpu")  # ONNX 导出在 CPU 上进行（兼容性最好）

    # 加载 checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = BERTClassifier(
        pretrained_model_name=pretrained_model,
        num_classes=num_classes,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 加载 tokenizer
    save_dir = ckpt_path.parent
    tok_dir = Path(tokenizer_dir) if tokenizer_dir else save_dir / "bert_tokenizer"
    tok_path = str(tok_dir) if tok_dir.exists() else pretrained_model
    tokenizer = AutoTokenizer.from_pretrained(tok_path)

    # 创建虚拟输入（dummy input）
    dummy = tokenizer(
        "测试文本", truncation=True, padding="max_length",
        max_length=max_len, return_tensors="pt",
    )
    input_ids = dummy["input_ids"].to(device)
    attention_mask = dummy["attention_mask"].to(device)

    # 导出 ONNX
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},       # 序列维动态：
            "attention_mask": {0: "batch_size", 1: "seq_len"},  # 支持变长 padding，
            "logits": {0: "batch_size", 1: "seq_len"},          # 短文本不再固定 128
        },
        opset_version=14,
        do_constant_folding=True,
    )
    print(f"  ✓ 已导出: {onnx_path} ({onnx_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # 验证 ONNX 模型
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print(f"  ✓ ONNX 模型校验通过")

    # 对比 PyTorch vs ONNX 输出
    with torch.no_grad():
        pt_logits = model(input_ids, attention_mask)

    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    onnx_logits = sess.run(None, {
        "input_ids": input_ids.cpu().numpy(),
        "attention_mask": attention_mask.cpu().numpy(),
    })[0]

    max_diff = abs(torch.tensor(onnx_logits) - pt_logits).max().item()
    print(f"  ✓ PyTorch vs ONNX 最大差异: {max_diff:.6f}")
    assert max_diff < 1e-4, f"输出差异过大: {max_diff}"

    # 速度对比
    texts = ["这是一段测试文本用于速度对比"] * 32
    enc = tokenizer(texts, truncation=True, padding="max_length",
                    max_length=max_len, return_tensors="pt")

    # PyTorch
    t0 = time.time()
    for _ in range(10):
        with torch.no_grad():
            model(enc["input_ids"], enc["attention_mask"])
    pt_time = (time.time() - t0) / 10

    # ONNX
    t0 = time.time()
    for _ in range(10):
        sess.run(None, {
            "input_ids": enc["input_ids"].cpu().numpy(),
            "attention_mask": enc["attention_mask"].cpu().numpy(),
        })
    onnx_time = (time.time() - t0) / 10

    speedup = pt_time / onnx_time if onnx_time > 0 else 0
    print(f"  ⏱ PyTorch: {pt_time*1000:.1f}ms | ONNX: {onnx_time*1000:.1f}ms | 加速: {speedup:.2f}x")
    return speedup


def main():
    parser = argparse.ArgumentParser(description="导出 BERT 模型为 ONNX 格式")
    parser.add_argument("--model", choices=["all", "binary", "emotion"],
                        default="all", help="选择导出的模型")
    args = parser.parse_args()

    cfg = load_config()
    save_dir = Path(cfg["train"]["save_dir"])
    max_len = cfg["data"]["max_length"]
    pretrained = cfg["model"]["pretrained_model"]

    speedups = []

    if args.model in ("all", "binary"):
        binary_ckpt = save_dir / "bert_best.pt"
        if binary_ckpt.exists():
            su = export_model(
                binary_ckpt, save_dir / "bert_binary.onnx",
                pretrained, num_classes=2, max_len=max_len,
                tokenizer_dir="bert_tokenizer",
            )
            speedups.append(("binary", su))
        else:
            print(f"[跳过] 二分类模型不存在: {binary_ckpt}")

    if args.model in ("all", "emotion"):
        emo_ckpt = save_dir / cfg["emotion"]["model_file"]
        emo_tok = save_dir / cfg["emotion"]["tokenizer_dir"]
        if emo_ckpt.exists():
            su = export_model(
                emo_ckpt, save_dir / "bert_emotion.onnx",
                pretrained, num_classes=cfg["emotion"]["num_labels"],
                max_len=max_len, tokenizer_dir=str(emo_tok),
            )
            speedups.append(("emotion", su))
        else:
            print(f"[跳过] 情绪模型不存在: {emo_ckpt}")

    print(f"\n{'='*50}")
    print("导出完成！")
    for name, su in speedups:
        print(f"  {name}: 加速 {su:.2f}x")
    print(f"\nONNX 文件位于: {save_dir}/")
    print("下一步: 使用 onnx_predictor.py 进行推理，或部署到 HuggingFace Spaces")


if __name__ == "__main__":
    main()
