"""
快速启动脚本：一键完成 LLM 数据增强流程
"""
import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, description: str):
    """运行命令并打印结果"""
    print("\n" + "=" * 70)
    print(f"🚀 {description}")
    print("=" * 70)
    print(f"命令：{cmd}\n")

    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print(f"\n❌ 失败：{description}")
        return False

    print(f"\n✅ 完成：{description}")
    return True


def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║        LLM 数据增强自动化流程                                      ║
║        Chinese Sentiment Analysis - LLM Enhancement              ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")

    # 检查 API Key
    import os
    qwen_key = os.getenv("DASHSCOPE_API_KEY")
    glm_key = os.getenv("ZHIPUAI_API_KEY")

    if not qwen_key and not glm_key:
        print("❌ 错误：未设置 API Key")
        print("\n请设置环境变量（二选一）：")
        print("  export DASHSCOPE_API_KEY='sk-your-qwen-key'  # 通义千问")
        print("  export ZHIPUAI_API_KEY='your-glm-key'        # 智谱 GLM")
        print("\n获取方式：")
        print("  通义千问：https://dashscope.aliyun.com/")
        print("  智谱 GLM：https://open.bigmodel.cn/")
        sys.exit(1)

    provider = "qwen" if qwen_key else "glm"
    print(f"✓ 检测到 API Key：{provider.upper()}")

    # Step 1: 生成增强数据
    if not run_command(
        "python src/llm/data_augmentation.py",
        "生成增强数据（焦虑类 + 边界样本）"
    ):
        return

    # Step 2: 质量检查
    print("\n" + "=" * 70)
    print("📋 步骤 2：质量检查")
    print("=" * 70)
    choice = input("\n是否进行质量检查？(y/n，推荐 y): ").strip().lower()

    if choice == 'y':
        run_command(
            "python src/llm/check_augmented_quality.py",
            "数据质量检查"
        )

    # Step 3: 合并数据
    print("\n" + "=" * 70)
    print("📋 步骤 3：合并到训练集")
    print("=" * 70)
    choice = input("\n确认合并增强数据到训练集？(y/n): ").strip().lower()

    if choice != 'y':
        print("\n⏸️  流程暂停，你可以稍后手动运行：")
        print("  python src/data/merge_augmented_data.py")
        return

    if not run_command(
        "python src/data/merge_augmented_data.py",
        "合并增强数据"
    ):
        return

    # Step 4: 重新训练
    print("\n" + "=" * 70)
    print("📋 步骤 4：重新训练模型")
    print("=" * 70)
    print("⚠️  训练预计耗时：20-40 分钟（取决于 GPU）")
    choice = input("\n是否立即开始训练？(y/n): ").strip().lower()

    if choice == 'y':
        # 修改训练脚本使用增强数据
        train_script = Path("src/train/train_bert_emotion.py")
        content = train_script.read_text(encoding='utf-8')

        if 'train_augmented.csv' not in content:
            print("\n📝 正在修改训练脚本...")
            content = content.replace(
                'train_df = pd.read_csv(data_dir / "train.csv")',
                'train_df = pd.read_csv(data_dir / "train_augmented.csv")  # LLM 增强'
            )
            train_script.write_text(content, encoding='utf-8')
            print("✓ 已修改训练脚本使用增强数据")

        run_command(
            "python src/train/train_bert_emotion.py",
            "训练增强后的模型"
        )
    else:
        print("\n⏸️  训练跳过，你可以稍后手动运行：")
        print("  python src/train/train_bert_emotion.py")

    # 完成
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║        ✅ LLM 数据增强流程完成！                                   ║
║                                                                  ║
║        下一步建议：                                                ║
║        1. 对比增强前后的模型效果                                    ║
║        2. 查看 logs/ 目录的训练日志                                ║
║        3. 启动前端测试：streamlit run app/main.py                 ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
