"""
通用评估模块
- 二分类：accuracy / precision / recall / f1(binary) / 2x2 混淆矩阵
- 多分类：accuracy / macro-F1 / weighted-F1 / 每类报告 / NxN 混淆矩阵
通过 average / target_names 参数适配，二分类旧调用方式完全兼容。
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def _print_confusion_matrix(cm: np.ndarray, names):
    """自适应 NxN 混淆矩阵打印"""
    n = cm.shape[0]
    header = "实际＼预测" + "".join(f"{name:>8}" for name in names)
    print("混淆矩阵:")
    print(header)
    for i in range(n):
        row = f"{names[i]:<8}" + "".join(f"{cm[i][j]:>8}" for j in range(n))
        print(row)


def evaluate(y_true, y_pred, model_name="Model", average="binary", target_names=None):
    """
    评估分类结果。

    参数：
        average: "binary"（二分类，默认）| "macro"（多分类宏平均）
        target_names: 类别名称列表；二分类默认 ["负面", "正面"]，多分类必传
    返回：
        指标字典（含 accuracy / f1 / precision / recall，多分类额外含 macro_f1、weighted_f1）
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    is_multiclass = average != "binary"
    if target_names is None:
        target_names = ["负面", "正面"] if not is_multiclass else sorted(set(y_true) | set(y_pred))
    labels = list(range(len(target_names)))

    acc = accuracy_score(y_true, y_pred)

    avg = "macro" if is_multiclass else "binary"
    precision = precision_score(y_true, y_pred, average=avg, zero_division=0)
    recall = recall_score(y_true, y_pred, average=avg, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=avg, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print(f"\n{'=' * 60}")
    print(f"模型: {model_name}")
    print(f"{'=' * 60}")
    print(f"准确率 (Accuracy):  {acc:.4f}")
    print(f"精确率 ({avg}):     {precision:.4f}")
    print(f"召回率 ({avg}):     {recall:.4f}")
    print(f"F1 分数 ({avg}):    {f1:.4f}")
    if is_multiclass:
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        print(f"F1 分数 (weighted): {weighted_f1:.4f}")
    print()
    _print_confusion_matrix(cm, target_names)
    print(f"\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=target_names,
                                labels=labels, zero_division=0))
    print(f"{'=' * 60}")

    result = {
        "model": model_name,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
    }
    if is_multiclass:
        result["weighted_f1"] = weighted_f1
        result["macro_f1"] = f1
    return result
