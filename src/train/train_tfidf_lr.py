"""
基线模型 1：TF-IDF + Logistic Regression
传统机器学习方法，作为性能对比的基准线

原理：
- TF-IDF：将文本转化为词频-逆文档频率向量，捕捉词语的重要性
- Logistic Regression：二分类线性模型，学习特征到标签的映射

这个模型虽然简单，但在很多文本分类任务上是很强的基线。
"""
import sys
import jieba
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.data.dataset import load_data, get_texts_labels
from src.evaluate.metrics import evaluate


def chinese_tokenize(text):
    """中文分词（jieba）"""
    return " ".join(jieba.lcut(text))


def main():
    # 1. 加载数据
    train_df, val_df, test_df = load_data()
    X_train, y_train = get_texts_labels(train_df)
    X_val, y_val = get_texts_labels(val_df)
    X_test, y_test = get_texts_labels(test_df)

    print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")

    # 2. 构建 Pipeline：TF-IDF 向量化 + 逻辑回归
    # Pipeline 可以把多个步骤串起来，避免数据泄露
    model = Pipeline([
        # TF-IDF 向量化
        ("tfidf", TfidfVectorizer(
            tokenizer=chinese_tokenize,    # 中文分词
            max_features=20000,            # 保留最高频的 2 万个词
            ngram_range=(1, 2),            # 一元词 + 二元词组
            min_df=2,                      # 至少出现 2 次才纳入
            sublinear_tf=True,             # 对词频取对数，抑制高频词影响
        )),
        # 逻辑回归分类器
        ("clf", LogisticRegression(
            C=1.0,                          # 正则化强度（越小正则越强）
            max_iter=1000,                  # 最大迭代次数
            class_weight="balanced",        # 自动平衡类别权重（处理类别不平衡）
            n_jobs=-1,                      # 使用所有 CPU 核心
        )),
    ])

    # 3. 训练
    print("\n开始训练 TF-IDF + Logistic Regression...")
    model.fit(X_train, y_train)
    print("训练完成！")

    # 4. 验证集评估
    val_pred = model.predict(X_val)
    evaluate(y_val, val_pred, "TF-IDF + LR (验证集)")

    # 5. 测试集评估
    test_pred = model.predict(X_test)
    results = evaluate(y_test, test_pred, "TF-IDF + LR (测试集)")

    # 6. 保存模型
    save_dir = Path("saved_models")
    save_dir.mkdir(exist_ok=True)
    joblib.dump(model, save_dir / "tfidf_lr.pkl")
    print(f"\n模型已保存到: {save_dir / 'tfidf_lr.pkl'}")

    return results


if __name__ == "__main__":
    main()
