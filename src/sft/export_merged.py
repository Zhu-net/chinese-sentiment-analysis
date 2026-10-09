# -*- coding: utf-8 -*-
"""
M2 收尾：将 best epoch 的 LoRA adapter 合并进 bf16 基座并导出完整权重
================================================================
产物：saved_models/qwen15b_emotion_merged/（生产推理不依赖 peft/bnb 量化）
用法：python -m src.sft.export_merged [--epoch N]
不传 --epoch 时自动取 logs/qlora_val_metrics.json 中的 best。
"""
import sys
import json
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_PATH = "saved_models/Qwen2.5-1.5B-Instruct"
CKPT_ROOT = Path("saved_models/qlora/qwen15b_emotion_r16")
METRICS_FILE = Path("logs/qlora_val_metrics.json")
OUT_PATH = Path("saved_models/qwen15b_emotion_merged")


def pick_best_epoch():
    data = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    return int(data["best"]["epoch"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=None)
    args = ap.parse_args()
    epoch = args.epoch or pick_best_epoch()
    adapter_path = CKPT_ROOT / f"epoch{epoch}"
    if not adapter_path.exists():
        raise FileNotFoundError(f"adapter 不存在: {adapter_path}（JSON 率不达标的 epoch 不会保存）")
    print(f"合并 epoch{epoch} adapter: {adapter_path}")

    # CPU 加载 bf16 基座，避免与 4bit 训练残留争抢显存
    base = AutoModelForCausalLM.from_pretrained(BASE_PATH, dtype=torch.bfloat16,
                                                device_map="cpu")
    model = PeftModel.from_pretrained(base, adapter_path)
    print("执行 merge_and_unload ...")
    model = model.merge_and_unload()

    OUT_PATH.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT_PATH, safe_serialization=True)
    AutoTokenizer.from_pretrained(BASE_PATH).save_pretrained(OUT_PATH)
    print(f"合并权重已保存: {OUT_PATH}")

    # 记录合并来源，便于审计复现
    meta = {"base_model": BASE_PATH, "adapter": str(adapter_path),
            "epoch": epoch, "dtype": "bfloat16"}
    (OUT_PATH / "merge_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
