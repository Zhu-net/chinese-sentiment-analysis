"""
数据质量检查工具
检查 LLM 生成样本的质量
"""
import sys
from pathlib import Path
import pandas as pd
import random

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils import load_config


def check_augmented_data(csv_path: str, sample_size: int = 20):
    """
    检查增强数据质量

    Args:
        csv_path: CSV 文件路径
        sample_size: 抽样检查数量
    """
    df = pd.read_csv(csv_path)
    cfg = load_config()

    print("=" * 70)
    print(f"Data Quality Report: {Path(csv_path).name}")
    print("=" * 70)

    # 基础统计
    print(f"\nBasic Statistics:")
    print(f"  Total samples: {len(df)}")
    print(f"  Label distribution: {df['label'].value_counts().to_dict()}")

    # 长度统计
    df['length'] = df['text'].str.len()
    print(f"\nLength Distribution:")
    print(f"  Average: {df['length'].mean():.1f} chars")
    print(f"  Median: {df['length'].median():.0f} chars")
    print(f"  Min: {df['length'].min()} chars")
    print(f"  Max: {df['length'].max()} chars")

    # 异常检测
    print(f"\nWarning - Anomaly Detection:")
    too_short = df[df['length'] < 10]
    too_long = df[df['length'] > 150]
    print(f"  Too short (<10 chars): {len(too_short)} samples")
    print(f"  Too long (>150 chars): {len(too_long)} samples")

    # 重复检测
    duplicates = df[df.duplicated(subset=['text'], keep=False)]
    print(f"  Duplicate texts: {len(duplicates)} samples")

    # 随机抽样展示
    print(f"\nRandom Sampling ({sample_size} samples):")
    print("-" * 70)
    samples = df.sample(min(sample_size, len(df)))

    emotion_labels = cfg["emotion"]["labels_cn"]
    for idx, (_, row) in enumerate(samples.iterrows(), 1):
        label_cn = emotion_labels[row['label']] if row['label'] >= 0 else "未标注"
        print(f"{idx:2d}. [{label_cn}] {row['text']}")

    print("\n" + "=" * 70)
    print("Check completed! Please manually review samples above before merging.")
    print("=" * 70)


def interactive_review(csv_path: str):
    """
    交互式人工审核
    """
    df = pd.read_csv(csv_path)
    cfg = load_config()
    emotion_labels = cfg["emotion"]["labels_cn"]

    print("\n🔍 交互式质量审核")
    print("指令：y=通过, n=删除, q=退出并保存, s=跳过")
    print("-" * 70)

    to_remove = []
    reviewed_count = 0

    for idx, row in df.iterrows():
        if reviewed_count >= 30:  # 最多审核 30 条
            break

        label_cn = emotion_labels[row['label']] if row['label'] >= 0 else "未标注"
        print(f"\n[{reviewed_count + 1}/30] 标签：{label_cn}")
        print(f"文本：{row['text']}")

        choice = input("评价 (y/n/s/q): ").strip().lower()

        if choice == 'q':
            break
        elif choice == 'n':
            to_remove.append(idx)
            print("  ❌ 已标记删除")
        elif choice == 'y':
            print("  ✅ 通过")
        elif choice == 's':
            print("  ⏭️  跳过")

        reviewed_count += 1

    # 删除标记的样本
    if to_remove:
        df_cleaned = df.drop(to_remove)
        output_path = Path(csv_path).parent / f"{Path(csv_path).stem}_cleaned.csv"
        df_cleaned.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ 已删除 {len(to_remove)} 条，保存到：{output_path}")
        print(f"剩余：{len(df_cleaned)} 条")
    else:
        print("\n✅ 无需清洗，数据质量良好")


if __name__ == "__main__":
    # 检查焦虑类生成数据
    anxious_files = [
        "data/augmented/anxious_deepseek_generated.csv",
        "data/augmented/anxious_llm_generated.csv"
    ]

    found_file = None
    for file in anxious_files:
        if Path(file).exists():
            found_file = file
            break

    if found_file:
        print("Quality Check Mode: Auto Statistics + Random Sampling")
        check_augmented_data(found_file, sample_size=20)

        print("\n" + "=" * 70)
        choice = input("\nEnter interactive review mode? (y/n): ").strip().lower()
        if choice == 'y':
            interactive_review(found_file)
    else:
        print(f"File not found: {anxious_files}")
        print("Please run first: python src/llm/data_augmentation.py")
