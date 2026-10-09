# -*- coding: utf-8 -*-
"""
M2：Qwen2.5-1.5B-Instruct QLoRA 情绪分类微调
============================================
配置对应 docs/P1_QLoRA微调技术方案.md §4：
- 4bit NF4 + double quant，compute bf16；LoRA r=16/alpha=32/dropout=0.05
- target: q,k,v,o,gate,up,down；3 epoch；micro-batch 4 × accum 8 = 32
- lr 1e-4 cosine warmup 0.03；adamw_torch；wd=0；grad_norm=1.0
- 梯度检查点；max_len=256；仅 assistant token 计 loss
- 每 epoch 末在 val(4359) 批量生成评测，macro-F1 优先、valid-JSON<98% 淘汰
- adapter 每 epoch 存 saved_models/qlora/qwen15b_emotion_r16/epoch{N}
- 指标落 logs/qlora_val_metrics.json，选 best 由 export_merged.py 合并

用法: python -m src.sft.train_qlora
"""
import sys
import json
import time
import random
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    Trainer, TrainingArguments, TrainerCallback, set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import f1_score, accuracy_score

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.sft.prompts import build_messages
from src.sft.parse_output import parse_emotion_json

# ---------------- 超参（方案 §4） ----------------
MODEL_PATH = "saved_models/Qwen2.5-1.5B-Instruct"
TRAIN_FILE = Path("data/sft/sft_train_4800.jsonl")
VAL_FILE = Path("data/sft/sft_val.jsonl")
CKPT_ROOT = Path("saved_models/qlora/qwen15b_emotion_r16")
METRICS_FILE = Path("logs/qlora_val_metrics.json")

MAX_LEN = 256
SEED = 42
EPOCHS = 3
MICRO_BATCH = 8
GRAD_ACCUM = 4
LR = 1e-4
EVAL_GEN_BATCH = 16
MAX_NEW_TOKENS = 64
MIN_JSON_RATE = 0.98


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


class SFTDataset(Dataset):
    """assistant-only loss：system/user token 的 label 置 -100"""

    def __init__(self, rows, tokenizer, max_len=MAX_LEN):
        self.feats = []
        dropped = 0
        for r in rows:
            full = tokenizer.apply_chat_template(
                r["messages"], tokenize=True, add_generation_prompt=False,
                return_dict=True)["input_ids"]
            prompt = tokenizer.apply_chat_template(
                r["messages"][:2], tokenize=True, add_generation_prompt=True,
                return_dict=True)["input_ids"]
            if len(full) > max_len:
                dropped += 1
                continue
            labels = [-100] * len(prompt) + full[len(prompt):]
            labels = labels[:len(full)]
            self.feats.append({"input_ids": full, "labels": labels})
        print(f"数据集样本 {len(self.feats)} 条，超长丢弃 {dropped} 条")

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, i):
        return self.feats[i]


@dataclass
class PadCollator:
    pad_id: int

    def __call__(self, batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, attn, labels = [], [], []
        for b in batch:
            n = len(b["input_ids"])
            pad_n = maxlen - n  # 训练右填充
            input_ids.append(b["input_ids"] + [self.pad_id] * pad_n)
            attn.append([1] * n + [0] * pad_n)
            labels.append(b["labels"] + [-100] * pad_n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


@torch.no_grad()
def evaluate_val(model, tokenizer, val_rows, batch_size=EVAL_GEN_BATCH):
    """在 val 上贪心生成，返回 macro-F1 / Acc / valid-JSON 率"""
    model.eval()
    gold, preds = [], []
    valid_json = 0
    # 左填充适配批量生成
    tokenizer.padding_side = "left"
    for start in range(0, len(val_rows), batch_size):
        chunk = val_rows[start:start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                build_messages(r["text"]),
                tokenize=False, add_generation_prompt=True)
            for r in chunk
        ]
        inp = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_LEN).to(model.device)
        out = model.generate(
            **inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen_ids = out[:, inp["input_ids"].shape[1]:]
        for r, ids in zip(chunk, gen_ids):
            gen = tokenizer.decode(ids, skip_special_tokens=True)
            lid, _, _, valid = parse_emotion_json(gen)
            gold.append(int(r["label"]))
            preds.append(lid if valid else -1)
            valid_json += int(valid)
    tokenizer.padding_side = "right"

    n = len(val_rows)
    json_rate = valid_json / n
    # 非法 JSON 记为错误类别 -1（macro-F1 的 labels 不含 -1，即按错误计）
    acc = accuracy_score(gold, preds)
    macro_f1 = f1_score(gold, preds, labels=list(range(6)), average="macro",
                        zero_division=0)
    per_class = f1_score(gold, preds, labels=list(range(6)), average=None,
                         zero_division=0).tolist()
    model.train()
    return {"acc": round(float(acc), 4), "macro_f1": round(float(macro_f1), 4),
            "valid_json_rate": round(json_rate, 4), "per_class_f1":
                [round(float(x), 4) for x in per_class]}


class ValCallback(TrainerCallback):
    def __init__(self, model, tokenizer, val_rows, ckpt_root: Path):
        self.model = model
        self.tokenizer = tokenizer
        self.val_rows = val_rows
        self.ckpt_root = ckpt_root
        self.results = []

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(round(state.epoch))
        t0 = time.time()
        metrics = evaluate_val(self.model, self.tokenizer, self.val_rows)
        metrics["epoch"] = epoch
        metrics["eval_seconds"] = round(time.time() - t0, 1)
        self.results.append(metrics)
        print(f"\n[val@epoch{epoch}] {metrics}\n", flush=True)

        # valid-JSON 率达标才保存候选
        if metrics["valid_json_rate"] >= MIN_JSON_RATE:
            ep_dir = self.ckpt_root / f"epoch{epoch}"
            self.model.save_pretrained(ep_dir)
            print(f"[保存] adapter -> {ep_dir}", flush=True)

        METRICS_FILE.parent.mkdir(exist_ok=True)
        METRICS_FILE.write_text(
            json.dumps(self.results, ensure_ascii=False, indent=2),
            encoding="utf-8")
        return control


def main():
    set_all_seeds(SEED)
    CKPT_ROOT.mkdir(parents=True, exist_ok=True)

    train_rows = [json.loads(l) for l in TRAIN_FILE.read_text(
        encoding="utf-8").splitlines()]
    val_rows = [json.loads(l) for l in VAL_FILE.read_text(
        encoding="utf-8").splitlines()]
    print(f"train={len(train_rows)} val={len(val_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(151643)  # <|endoftext|>
    tokenizer.padding_side = "right"

    # ------------ 4bit 基座 ------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=False)
    model.enable_input_require_grads()

    # ------------ LoRA ------------
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = SFTDataset(train_rows, tokenizer)
    collator = PadCollator(pad_id=tokenizer.pad_token_id)

    # transformers 5.x 移除 warmup_ratio，按总优化步数换算 warmup_steps
    import math
    steps_per_epoch = math.ceil(len(train_ds) / (MICRO_BATCH * GRAD_ACCUM))
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = math.ceil(total_steps * 0.03)
    print(f"优化步数: {total_steps}（{steps_per_epoch}/epoch × {EPOCHS}），warmup {warmup_steps} 步")

    args = TrainingArguments(
        output_dir=str(CKPT_ROOT / "trainer_ckpt"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=MICRO_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=0.0,
        max_grad_norm=1.0,
        optim="adamw_torch",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        seed=SEED,
        data_seed=SEED,
        dataloader_num_workers=0,
    )

    val_cb = ValCallback(model, tokenizer, val_rows, CKPT_ROOT)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=collator,
        callbacks=[val_cb],
    )
    trainer.train()

    # ------------ 选 best ------------
    results = val_cb.results
    eligible = [r for r in results if r["valid_json_rate"] >= MIN_JSON_RATE]
    if not eligible:
        print("\n所有 epoch JSON 率均不达标，无 best，请检查训练")
        return
    best = max(eligible, key=lambda r: (r["macro_f1"], r["acc"]))
    summary = {"all_epochs": results, "best": best}
    METRICS_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(f"\n[BEST] epoch{best['epoch']}: macro-F1={best['macro_f1']} "
          f"acc={best['acc']} json={best['valid_json_rate']}")
    print(f"下一步: python -m src.sft.export_merged --epoch {best['epoch']}")


if __name__ == "__main__":
    main()
