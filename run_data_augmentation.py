"""
数据增强启动脚本
支持 DeepSeek API 或本地模拟生成（当 API 不可用时）
"""
# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from src.utils import load_config


def test_deepseek_connection():
    """测试 DeepSeek 连接"""
    print("Testing DeepSeek API connection...")
    try:
        from src.llm.llm_client import DeepSeekClient
        client = DeepSeekClient(api_key="sk-94c2e1d84e6a405fbf4e742341eddd0c")
        result = client.generate("Say 'OK'", max_tokens=10)
        print(f"SUCCESS: DeepSeek API is working! Response: {result}")
        return True
    except Exception as e:
        print(f"FAILED: DeepSeek API unavailable - {e}")
        return False


def generate_with_deepseek():
    """使用 DeepSeek 生成数据"""
    from src.llm.data_augmentation import LLMDataAugmentor

    print("\n" + "=" * 70)
    print("Task: Generate anxiety emotion samples (original: 267 samples)")
    print("=" * 70)

    augmentor = LLMDataAugmentor(
        provider="deepseek",
        api_key="sk-94c2e1d84e6a405fbf4e742341eddd0c"
    )

    # 生成焦虑类样本
    print("\nGenerating anxiety samples...")
    anxious_samples = augmentor.generate_samples(
        emotion="anxious",
        num_samples=100,
        scenarios=["考试前", "面试等结果", "医院等报告", "航班延误", "重要演讲前"]
    )

    print(f"\nGenerated {len(anxious_samples)} samples")
    print("\nFirst 10 samples:")
    for i, text in enumerate(anxious_samples[:10], 1):
        print(f"{i:2d}. {text}")

    # 保存
    save_dir = Path("data/augmented")
    save_dir.mkdir(exist_ok=True, parents=True)
    df = pd.DataFrame({
        "text": anxious_samples,
        "label": [5] * len(anxious_samples),  # anxious = 5
        "source": "deepseek_generated"
    })
    save_path = save_dir / "anxious_deepseek_generated.csv"
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved to: {save_path}")

    return anxious_samples


def generate_with_templates():
    """使用模板生成数据（备用方案）"""
    print("\n" + "=" * 70)
    print("Fallback: Generate samples using templates")
    print("=" * 70)

    # 焦虑情绪模板
    templates_anxious = [
        "明天就要{event}了，紧张得{feeling}",
        "{event}的结果还没出来，心里特别{feeling}",
        "一想到{event}就{feeling}，{action}",
        "快{event}了，{feeling}得睡不着觉",
        "等{event}等得{feeling}，真是太煎熬了",
        "{event}在即，心里{feeling}极了",
        "担心{event}{result}，{feeling}得不行",
        "离{event}越来越近，越来越{feeling}",
    ]

    events = [
        "考试", "面试", "手术", "体检出结果", "答辩", "演讲", "比赛",
        "见家长", "签约", "发榜", "开庭", "评审", "路演"
    ]

    feelings = [
        "焦虑", "紧张", "不安", "担心", "慌张", "忐忑", "惶恐", "七上八下"
    ]

    actions = [
        "一直坐立不安", "手心直冒汗", "心跳加速", "脑子一片空白",
        "反复检查准备", "来回踱步", "吃不下饭"
    ]

    results = [
        "不顺利", "搞砸", "出问题", "失败", "不通过", "被拒绝"
    ]

    import random
    samples = []

    for _ in range(500):  # 生成500条
        template = random.choice(templates_anxious)
        text = template.format(
            event=random.choice(events),
            feeling=random.choice(feelings),
            action=random.choice(actions),
            result=random.choice(results)
        )
        samples.append(text)

    # 去重
    samples = list(set(samples))

    print(f"\nGenerated {len(samples)} unique samples")
    print("\nFirst 10 samples:")
    for i, text in enumerate(samples[:10], 1):
        print(f"{i:2d}. {text}")

    # 保存
    save_dir = Path("data/augmented")
    save_dir.mkdir(exist_ok=True, parents=True)
    df = pd.DataFrame({
        "text": samples,
        "label": [5] * len(samples),
        "source": "template_generated"
    })
    save_path = save_dir / "anxious_template_generated.csv"
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved to: {save_path}")

    return samples


def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║        Data Augmentation for Chinese Sentiment Analysis         ║
║        中文情感分析 - 数据增强                                      ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")

    # 测试 DeepSeek
    deepseek_available = test_deepseek_connection()

    if deepseek_available:
        print("\n[Method] Using DeepSeek API for data generation")
        try:
            generate_with_deepseek()
        except Exception as e:
            print(f"\nDeepSeek generation failed: {e}")
            print("Falling back to template method...")
            generate_with_templates()
    else:
        print("\n[Method] Using template-based generation (fallback)")
        generate_with_templates()

    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("1. Check quality: python src/llm/check_augmented_quality.py")
    print("2. Merge data: python src/data/merge_augmented_data.py")
    print("3. Retrain model: python src/train/train_bert_emotion.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
