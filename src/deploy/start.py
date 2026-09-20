"""
启动脚本：检查模型是否存在，不存在则下载
然后启动 FastAPI 服务
"""
import os
import sys
from pathlib import Path

# HuggingFace 国内镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


def ensure_model():
    """确保 BERT 模型和微调后的权重都存在"""
    save_dir = Path("saved_models")
    bert_dir = save_dir / "bert-base-chinese"
    model_path = save_dir / "bert_best.pt"

    # 1. 检查预训练 BERT 模型
    if not (bert_dir / "pytorch_model.bin").exists():
        print("预训练 BERT 模型不存在，开始下载...")
        import requests
        bert_dir.mkdir(parents=True, exist_ok=True)
        base_url = "https://hf-mirror.com/bert-base-chinese/resolve/main/"
        files = ["config.json", "vocab.txt", "tokenizer_config.json", "pytorch_model.bin"]
        for f in files:
            url = base_url + f
            dest = bert_dir / f
            if not dest.exists():
                print(f"  下载 {f}...")
                r = requests.get(url, timeout=120, stream=True)
                r.raise_for_status()
                with open(dest, "wb") as fout:
                    for chunk in r.iter_content(8192):
                        fout.write(chunk)
        print("BERT 模型下载完成！")

    # 2. 检查微调后的模型权重
    if not model_path.exists():
        print("警告：微调后的模型 bert_best.pt 不存在！")
        print("请先运行训练脚本：python src/train/train_bert.py")
        print("或从其他地方复制模型权重到 saved_models/bert_best.pt")
        return False

    return True


def main():
    if ensure_model():
        print("模型检查通过，启动服务...")
        import uvicorn
        uvicorn.run("src.deploy.app:app", host="0.0.0.0", port=8000)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
