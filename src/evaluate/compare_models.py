"""
模型结果对比表
运行所有基线模型并汇总结果，保存为 CSV
"""
import sys
import json
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


def main():
    # 各模型测试集结果（手动填入训练脚本输出的结果）
    results = [
        {"model": "TF-IDF + LR", "accuracy": 0.9042, "precision": 0.8425, "recall": 0.8537, "f1": 0.8481},
        {"model": "TextCNN", "accuracy": 0.8842, "precision": 0.8229, "recall": 0.8032, "f1": 0.8129},
        {"model": "BiLSTM", "accuracy": 0.8800, "precision": 0.8069, "recall": 0.8112, "f1": 0.8090},
        {"model": "BERT", "accuracy": 0.9117, "precision": 0.8729, "recall": 0.8404, "f1": 0.8564},
    ]

    df = pd.DataFrame(results)
    print("\n模型性能对比：")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)

    # 保存
    output_path = Path("saved_models/model_comparison.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
