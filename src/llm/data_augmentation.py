"""
LLM 驱动的数据增强
====================
功能：
1. 生成训练样本（特别是稀缺类别）
2. 改写现有样本（保持情感标签不变）
3. 生成对抗样本（边界 case）
"""
import json
import random
from typing import List, Dict, Tuple
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.llm.llm_client import get_llm_client
from src.utils import load_config


class LLMDataAugmentor:
    """基于 LLM 的数据增强器"""

    def __init__(self, provider: str = "deepseek", api_key: str = None):
        """
        Args:
            provider: LLM 提供商（默认 deepseek）
            api_key: API Key（可选，未提供则从环境变量读取）
        """
        self.client = get_llm_client(provider, api_key=api_key)
        self.cfg = load_config()
        self.emotion_labels = self.cfg["emotion"]["labels"]
        self.emotion_labels_cn = self.cfg["emotion"]["labels_cn"]

    def generate_samples(
        self,
        emotion: str,
        num_samples: int = 100,
        scenarios: List[str] = None
    ) -> List[str]:
        """
        生成指定情绪的训练样本

        Args:
            emotion: 情绪类别（英文），如 "anxious"
            num_samples: 生成数量
            scenarios: 场景列表，如 ["考试", "面试", "看病"]

        Returns:
            生成的文本列表
        """
        emotion_idx = self.emotion_labels.index(emotion)
        emotion_cn = self.emotion_labels_cn[emotion_idx]

        # 默认场景
        if scenarios is None:
            scenarios_map = {
                "happy": ["收到礼物", "考试通过", "升职加薪", "旅游", "美食"],
                "grateful": ["朋友帮助", "陌生人善举", "家人关怀", "老师指导"],
                "sad": ["宠物去世", "分手失恋", "考试失利", "失业", "亲人离世"],
                "angry": ["被欺骗", "服务态度差", "产品质量差", "排队被插队", "被误解"],
                "fear": ["恐怖片", "黑夜独行", "高空", "突发意外", "疾病"],
                "anxious": ["考试前", "面试等结果", "医院等报告", "航班延误", "重要演讲前"]
            }
            scenarios = scenarios_map.get(emotion, ["日常生活"])

        prompt = f"""你是一个数据标注专家。请生成 {num_samples} 条表达「{emotion_cn}」情绪的中文评论或文本。

要求：
1. 情感明确：每条都清晰表达「{emotion_cn}」，不含其他情绪
2. 场景多样：涵盖这些场景：{', '.join(scenarios)}
3. 口语化：像真实用户写的评论，不要书面语
4. 长度适中：每条 15-80 字
5. 格式：每行一条，不加序号，不加标点说明

直接输出 {num_samples} 条文本，每行一条："""

        result = self.client.generate(prompt, temperature=0.9, max_tokens=2000)

        # 解析生成结果
        lines = [line.strip() for line in result.split("\n") if line.strip()]
        # 过滤掉序号（如：1. 2. ）
        samples = []
        for line in lines:
            # 去除可能的序号前缀
            cleaned = line
            if line and line[0].isdigit() and '.' in line[:5]:
                cleaned = line.split('.', 1)[1].strip()
            if len(cleaned) >= 10:  # 过滤太短的
                samples.append(cleaned)

        return samples[:num_samples]

    def paraphrase_samples(
        self,
        texts: List[str],
        emotion: str,
        variants_per_text: int = 2
    ) -> List[Tuple[str, str]]:
        """
        改写样本（保持情感不变）

        Args:
            texts: 原始文本列表
            emotion: 情绪标签（英文）
            variants_per_text: 每条生成几个变体

        Returns:
            [(原文, 改写文本), ...]
        """
        emotion_idx = self.emotion_labels.index(emotion)
        emotion_cn = self.emotion_labels_cn[emotion_idx]

        results = []
        for text in texts:
            prompt = f"""请将以下文本改写 {variants_per_text} 个版本，要求：
1. 保持「{emotion_cn}」情感不变
2. 改变句式、用词，但意思相近
3. 口语化，像真实用户写的
4. 每行一个版本，不加序号

原文：{text}

改写版本："""

            response = self.client.generate(prompt, temperature=0.8, max_tokens=500)
            variants = [line.strip() for line in response.split("\n")
                       if line.strip() and len(line.strip()) >= 10]

            for variant in variants[:variants_per_text]:
                # 简单去除序号
                cleaned = variant
                if variant and variant[0].isdigit() and '.' in variant[:5]:
                    cleaned = variant.split('.', 1)[1].strip()
                results.append((text, cleaned))

        return results

    def generate_adversarial_samples(
        self,
        emotion1: str,
        emotion2: str,
        num_samples: int = 50
    ) -> List[str]:
        """
        生成情绪边界样本（容易混淆的两类）

        Args:
            emotion1: 情绪1（如 "sad"）
            emotion2: 情绪2（如 "angry"）
            num_samples: 生成数量

        Returns:
            生成的边界样本列表
        """
        idx1 = self.emotion_labels.index(emotion1)
        idx2 = self.emotion_labels.index(emotion2)
        emotion1_cn = self.emotion_labels_cn[idx1]
        emotion2_cn = self.emotion_labels_cn[idx2]

        prompt = f"""生成 {num_samples} 条同时包含「{emotion1_cn}」和「{emotion2_cn}」的混合情绪文本。

要求：
1. 两种情绪都明显存在，难以判断主导情绪
2. 口语化，真实场景
3. 每行一条，15-60 字

示例：
- 悲伤+愤怒：辛苦养了十年的狗被邻居毒死了，心碎又愤怒
- 焦虑+恐惧：明天手术，既怕出意外又担心结果不好

直接输出 {num_samples} 条："""

        result = self.client.generate(prompt, temperature=0.9, max_tokens=1500)
        lines = [line.strip() for line in result.split("\n") if line.strip()]

        samples = []
        for line in lines:
            cleaned = line
            if line and line[0].isdigit() and '.' in line[:5]:
                cleaned = line.split('.', 1)[1].strip()
            # 去除可能的标签说明
            if '：' in cleaned or ':' in cleaned:
                cleaned = cleaned.split('：')[-1].split(':')[-1].strip()
            if len(cleaned) >= 10:
                samples.append(cleaned)

        return samples[:num_samples]


def main():
    """示例：生成焦虑类样本（原项目仅 267 条）"""
    import pandas as pd

    # 使用 DeepSeek API（硬编码 API Key）
    augmentor = LLMDataAugmentor(
        provider="deepseek",
        api_key="sk-94c2e1d84e6a405fbf4e742341eddd0c"
    )

    print("=" * 60)
    print("任务 1：生成「焦虑」类样本（原数据仅 267 条）")
    print("=" * 60)

    anxious_samples = augmentor.generate_samples(
        emotion="anxious",
        num_samples=100,
        scenarios=["考试前", "面试等结果", "医院等报告", "航班延误", "重要演讲前"]
    )

    print(f"\n✓ 生成了 {len(anxious_samples)} 条样本，前 10 条：\n")
    for i, text in enumerate(anxious_samples[:10], 1):
        print(f"{i:2d}. {text}")

    # 保存到 CSV
    save_dir = Path("data/augmented")
    save_dir.mkdir(exist_ok=True, parents=True)
    df = pd.DataFrame({
        "text": anxious_samples,
        "label": [5] * len(anxious_samples),  # anxious = 5
        "source": "llm_generated"
    })
    save_path = save_dir / "anxious_llm_generated.csv"
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"\n✓ 已保存到: {save_path}")

    print("\n" + "=" * 60)
    print("任务 2：生成悲伤-愤怒边界样本（易混淆类别）")
    print("=" * 60)

    boundary_samples = augmentor.generate_adversarial_samples(
        emotion1="sad",
        emotion2="angry",
        num_samples=30
    )

    print(f"\n✓ 生成了 {len(boundary_samples)} 条边界样本，前 5 条：\n")
    for i, text in enumerate(boundary_samples[:5], 1):
        print(f"{i}. {text}")

    df_boundary = pd.DataFrame({
        "text": boundary_samples,
        "label": [-1],  # 需要人工标注
        "source": "llm_adversarial"
    })
    boundary_path = save_dir / "sad_angry_boundary.csv"
    df_boundary.to_csv(boundary_path, index=False, encoding="utf-8-sig")
    print(f"\n✓ 已保存到: {boundary_path}（需人工标注后合并训练集）")

    print("\n" + "=" * 60)
    print("✅ 数据增强完成！下一步：")
    print("1. 检查生成质量（随机抽查 20 条）")
    print("2. 合并到训练集重新训练")
    print("3. 对比增强前后的模型效果")
    print("=" * 60)


if __name__ == "__main__":
    main()
