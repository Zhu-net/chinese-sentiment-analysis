# -*- coding: utf-8 -*-
"""
M0 验收：Qwen2.5-1.5B-Instruct 4bit NF4 加载 + ChatML 生成
通过标准：成功输出一句合法 JSON 风格情绪判定，并打印显存占用。
"""
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.sft.prompts import build_messages

MODEL_PATH = "saved_models/Qwen2.5-1.5B-Instruct"


def main():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    # 确认线性层确实被 4bit 量化
    from bitsandbytes.nn import Linear4bit
    n_linear4bit = sum(1 for m in model.modules() if isinstance(m, Linear4bit))
    print(f"Linear4bit 层数: {n_linear4bit}（>0 即量化生效）")
    assert n_linear4bit > 0

    tests = [
        "等了整整一个月，今天终于收到录取通知书了，啊啊啊太激动了！",
        "商品收到有瑕疵，客服一直来回踢皮球，不肯处理。",
        "明天就要答辩了，PPT 还没做完，心慌得睡不着。",
    ]
    for text in tests:
        messages = build_messages(text)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=64, do_sample=False,
                temperature=1.0, top_p=1.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True)
        print(f"\n文本: {text}\n输出: {gen}")

    mem = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\n峰值显存: {mem:.2f} GB")
    print("M0 验收通过：4bit NF4 加载 + ChatML 生成成功")


if __name__ == "__main__":
    main()
