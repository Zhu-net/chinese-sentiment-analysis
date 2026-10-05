"""
大模型客户端封装
支持 DeepSeek API（兼容 OpenAI 格式）
"""
import os
import time
from typing import List, Dict, Optional
from abc import ABC, abstractmethod


class LLMClient(ABC):
    """大模型客户端抽象基类"""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本"""
        pass

    @abstractmethod
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """批量生成"""
        pass


class DeepSeekClient(LLMClient):
    """DeepSeek 客户端（性价比高，推理能力强）"""

    def __init__(self, api_key: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装 openai 库: pip install openai")

        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("请设置环境变量 DEEPSEEK_API_KEY 或传入 api_key 参数")

        # DeepSeek 兼容 OpenAI API
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )
        self.model = "deepseek-chat"  # DeepSeek 主模型

    def generate(self, prompt: str, temperature: float = 0.7,
                 max_tokens: int = 500, max_retries: int = 3) -> str:
        """
        单次生成（带重试机制）

        Args:
            prompt: 输入提示词
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            max_retries: 最大重试次数（遇到 503 等暂时性错误）
        """
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                error_msg = str(e)

                # 503 服务繁忙 - 等待后重试
                if "503" in error_msg or "busy" in error_msg.lower():
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2  # 2秒、4秒、6秒
                        print(f"WARNING: DeepSeek service busy, retry in {wait_time}s ({attempt+1}/{max_retries})...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"ERROR: DeepSeek API failed {max_retries} times")
                        raise Exception("DeepSeek service is persistently busy, please try again later")

                # 其他错误直接抛出
                print(f"ERROR: DeepSeek API call failed: {e}")
                raise

        raise Exception("未知错误")

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """批量生成（串行调用）"""
        results = []
        for i, prompt in enumerate(prompts, 1):
            print(f"正在生成 {i}/{len(prompts)}...", end="\r")
            try:
                result = self.generate(prompt, **kwargs)
                results.append(result)
            except Exception as e:
                print(f"\n⚠️  第 {i} 条生成失败，跳过: {e}")
                results.append("")  # 占位
        print()  # 换行
        return results


def get_llm_client(provider: str = "deepseek", api_key: Optional[str] = None) -> LLMClient:
    """
    工厂函数：返回 LLM 客户端

    Args:
        provider: 提供商（目前仅支持 deepseek）
        api_key: API Key（可选，未提供则从环境变量读取）
    """
    if provider == "deepseek":
        return DeepSeekClient(api_key=api_key)
    else:
        raise ValueError(f"不支持的提供商: {provider}，当前仅支持: deepseek")


if __name__ == "__main__":
    # 测试
    print("=" * 60)
    print("DeepSeek API 连接测试")
    print("=" * 60)

    try:
        # 使用你的 API Key
        client = get_llm_client("deepseek", api_key="sk-94c2e1d84e6a405fbf4e742341eddd0c")

        print("\n测试 1：简单问答")
        result = client.generate("用一句话介绍中文情感分析", max_tokens=100)
        print(f"✅ 生成结果:\n{result}")

        print("\n测试 2：情感样本生成")
        prompt = """生成 3 条表达「焦虑」情绪的中文评论。
要求：口语化，15-50 字，每行一条。

直接输出："""
        result = client.generate(prompt, temperature=0.9, max_tokens=200)
        print(f"✅ 生成样本:\n{result}")

        print("\n" + "=" * 60)
        print("✅ DeepSeek API 测试成功！")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        print("\n请检查：")
        print("1. API Key 是否正确")
        print("2. 网络连接是否正常")
        print("3. 是否已安装 openai 库: pip install openai")
