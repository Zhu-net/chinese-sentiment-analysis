# -*- coding: utf-8 -*-
"""
HuggingFace Spaces 部署准备脚本
================================
自动将 ONNX 模型、tokenizer、config 等拷贝到 deploy/hf_spaces/ 目录，
然后可通过 huggingface_hub 推送到 HF Space。

用法：
    python deploy/prepare_deploy.py              # 准备文件
    python deploy/prepare_deploy.py --push       # 准备 + 创建/推送到 HF Space
"""
import argparse
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEPLOY = ROOT / "deploy" / "hf_spaces"


def prepare_files():
    """拷贝所有必要文件到部署目录"""
    print("📦 准备 HuggingFace Spaces 部署文件...")

    # 1. config
    cfg_src = ROOT / "config" / "config.yaml"
    cfg_dst_dir = DEPLOY / "config"
    cfg_dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_src, cfg_dst_dir / "config.yaml")
    print("  ✓ config.yaml")

    # 2. ONNX 模型
    save_dir = ROOT / "saved_models"
    dst_models = DEPLOY / "saved_models"
    dst_models.mkdir(parents=True, exist_ok=True)

    for onnx_file in ["bert_emotion.onnx", "bert_binary.onnx"]:
        src = save_dir / onnx_file
        if src.exists():
            shutil.copy2(src, dst_models / onnx_file)
            print(f"  ✓ {onnx_file} ({src.stat().st_size / 1024 / 1024:.1f} MB)")
        else:
            print(f"  ✗ {onnx_file} 不存在，请先运行 export_onnx.py")

    # 3. Tokenizer（情绪模型）
    with open(cfg_src, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    emo_tok_dir = save_dir / cfg["emotion"]["tokenizer_dir"]
    if emo_tok_dir.exists():
        dst_tok = dst_models / cfg["emotion"]["tokenizer_dir"]
        if dst_tok.exists():
            shutil.rmtree(dst_tok)
        shutil.copytree(emo_tok_dir, dst_tok)
        print(f"  ✓ emotion tokenizer ({len(list(dst_tok.iterdir()))} files)")
    else:
        print(f"  ✗ emotion tokenizer 不存在: {emo_tok_dir}")

    # 4. 二分类 tokenizer
    bin_tok_dir = save_dir / "bert_tokenizer"
    if bin_tok_dir.exists():
        dst_tok = dst_models / "bert_tokenizer"
        if dst_tok.exists():
            shutil.rmtree(dst_tok)
        shutil.copytree(bin_tok_dir, dst_tok)
        print(f"  ✓ binary tokenizer ({len(list(dst_tok.iterdir()))} files)")
    else:
        print("  ⚠ binary tokenizer 不存在（HF Spaces app 不使用二分类，可忽略）")

    # 5. .gitignore（排除不必要的文件）
    gitignore = DEPLOY / ".gitignore"
    gitignore.write_text(
        "__pycache__/\n*.pyc\n.DS_Store\ndata/\n*.db\n",
        encoding="utf-8",
    )
    print("  ✓ .gitignore")

    # 汇总
    total_size = sum(f.stat().st_size for f in DEPLOY.rglob("*") if f.is_file())
    print(f"\n✅ 部署目录: {DEPLOY}")
    print(f"   总大小: {total_size / 1024 / 1024:.1f} MB")
    print(f"   文件数: {sum(1 for f in DEPLOY.rglob('*') if f.is_file())}")


def push_to_hf(space_name: str = None, token: str = None):
    """创建 HF Space 并推送文件"""
    from huggingface_hub import HfApi

    api = HfApi(token=token)

    # 如果没有指定 Space 名，用默认
    if space_name is None:
        space_name = "sentiment-lens"

    # 获取当前用户
    user = api.whoami()
    username = user["name"]
    repo_id = f"{username}/{space_name}"
    print(f"\n🚀 准备推送到 HuggingFace Space: {repo_id}")

    # 创建 Space（如果不存在）
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="streamlit",
            exist_ok=True,
        )
        print(f"  ✓ Space 已就绪")
    except Exception as e:
        print(f"  ⚠ Space 创建: {e}")

    # 上传所有文件
    print("  上传文件中...")
    api.upload_folder(
        folder_path=str(DEPLOY),
        repo_id=repo_id,
        repo_type="space",
    )
    print(f"\n✅ 推送完成！")
    print(f"   访问地址: https://{username}-{space_name}.hf.space")
    print(f"   或: https://huggingface.co/spaces/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="准备并推送 HuggingFace Space")
    parser.add_argument("--push", action="store_true", help="准备后推送到 HF")
    parser.add_argument("--space-name", default="sentiment-lens", help="Space 名称")
    parser.add_argument("--token", default=None, help="HF token（或通过环境变量 HF_TOKEN）")
    args = parser.parse_args()

    prepare_files()

    if args.push:
        # 检查 token
        token = args.token or __import__("os").environ.get("HF_TOKEN")
        if not token:
            print("\n❌ 未找到 HF Token。请设置环境变量 HF_TOKEN 或使用 --token 参数。")
            print("   获取 Token: https://huggingface.co/settings/tokens")
            print("   设置方法: $env:HF_TOKEN='your_token_here'")
            return
        push_to_hf(args.space_name, token)
    else:
        print("\n💡 推送到 HuggingFace Space:")
        print("   1. 获取 Token: https://huggingface.co/settings/tokens")
        print("   2. 设置环境变量: $env:HF_TOKEN='your_token'")
        print("   3. 运行: python deploy/prepare_deploy.py --push")


if __name__ == "__main__":
    main()
