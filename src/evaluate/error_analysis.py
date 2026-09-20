"""
误差分析脚本
用训练好的 BERT 模型在测试集上预测，找出分类错误的案例
"""
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.data.dataset import load_data, get_texts_labels
from src.deploy.predictor import get_predictor


def main():
    # 加载测试集
    _, _, test_df = load_data()
    X_test, y_test = get_texts_labels(test_df)

    # 加载模型预测
    predictor = get_predictor()
    print(f"对 {len(X_test)} 条测试数据进行预测...")

    results = predictor.predict_batch(X_test)

    # 分析错误案例
    errors = []
    for i, (text, true_label, pred) in enumerate(zip(X_test, y_test, results)):
        pred_label = pred["label_id"]
        if pred_label != true_label:
            errors.append({
                "text": text,
                "true_label": "正面" if true_label == 1 else "负面",
                "pred_label": pred["label"],
                "confidence": pred["confidence"],
            })

    error_df = pd.DataFrame(errors)
    print(f"\n总样本: {len(X_test)}")
    print(f"预测正确: {len(X_test) - len(errors)}")
    print(f"预测错误: {len(errors)} ({len(errors)/len(X_test)*100:.2f}%)")

    # 按真实标签分组统计
    print(f"\n错误类型统计：")
    print(f"  实际正面 -> 预测负面: {(error_df['true_label']=='正面').sum()} 条")
    print(f"  实际负面 -> 预测正面: {(error_df['true_label']=='负面').sum()} 条")

    # 保存错误案例
    output_path = Path("notebooks/error_analysis.csv")
    error_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n错误案例已保存到: {output_path}")

    # 展示部分错误案例
    print("\n" + "=" * 60)
    print("部分错误案例（前10条）：")
    print("=" * 60)
    for _, row in error_df.head(10).iterrows():
        print(f"文本: {row['text'][:60]}")
        print(f"  真实: {row['true_label']} | 预测: {row['pred_label']} (置信度: {row['confidence']:.4f})")
        print("-" * 40)


if __name__ == "__main__":
    main()
