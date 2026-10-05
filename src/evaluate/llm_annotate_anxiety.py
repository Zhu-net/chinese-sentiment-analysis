# -*- coding: utf-8 -*-
"""
提取 test 集焦虑混淆样本并做 LLM 核验
- 真实焦虑被误判（31条）
- 非焦虑被误判为焦虑（29条）
合计 60 条
"""
import sys
import json
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

os_sys = __import__("os")
os_sys.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.utils import load_config
from src.models.bert_dataset import BERTDataset
from src.models.bert_classifier import BERTClassifier
from src.llm.llm_client import DeepSeekClient

LABELS_CN = ["开心", "感激", "悲伤", "愤怒", "恐惧", "焦虑"]
ANXIOUS = 5
API_KEY = "sk-94c2e1d84e6a405fbf4e742341eddd0c"
BATCH = 10
VALID_LABELS = LABELS_CN
VALID_VERDICTS = ["标签正确", "标签错误", "边界模糊"]

PROMPT_TMPL = """你是中文微博情绪标注质检员。任务：核验每条文本的「原标签」是否正确。

六类情绪定义（互斥，按主导情绪标注）：
- 开心：高兴、喜悦、满足、期待
- 感激：感谢、感恩
- 悲伤：难过、失落、无奈、沮丧、哀叹、委屈、孤独
- 愤怒：生气、发怒、指责、辱骂、强烈不满、斥责他人/事物
- 恐惧：害怕、惊恐、恐慌（对已发生危险的本能害怕）
- 焦虑：紧张、担忧、不安、纠结、烦躁（对未发生事件/持续压力的焦灼，如考试、工作、健康担忧、等结果、睡不着觉、心慌、静不下来）

判定要点：
- 「担心、害怕、惶恐不安、心慌、静不下来、忐忑、纠结、压力、怕...」等对未来/未知的担忧→焦虑
- 「崩溃、心塞、好累、好难过、好伤心」等对已发生事情的低落→悲伤
- 「气死了、太过分、讨厌、骂」等对外指责→愤怒
- 对已发生危险的本能害怕（如梦见被杀、地震）→恐惧；对可能发生坏事的担忧→焦虑
- 难分主次→边界模糊

只输出 JSON 数组，不要任何多余文字。每个元素：
{{"id": 序号, "verdict": "标签正确/标签错误/边界模糊", "label": "六类之一", "confidence": "高/中/低", "reason": "20字内中文理由"}}

待核验文本（原标签为{orig_label}，模型误判为{wrong_label}）：
{items}
"""


def build_prompt(batch_df):
    orig_label = batch_df["orig_cn"].iloc[0]
    wrong_label = batch_df["pred_cn"].iloc[0]
    items = "\n".join(f"{int(r['_local_id'])}. {r['text']}" for _, r in batch_df.iterrows())
    return PROMPT_TMPL.format(orig_label=orig_label, wrong_label=wrong_label, items=items)


def parse_response(text, expected_ids):
    s = text.strip()
    if "```" in s:
        s = s.split("```")[1]
        if s.startswith("json"):
            s = s[4:]
    l, r = s.find("["), s.rfind("]")
    if l == -1 or r == -1:
        raise ValueError("响应中未找到 JSON 数组")
    arr = json.loads(s[l:r + 1])
    out = {}
    for item in arr:
        idx = int(item["id"])
        verdict = item["verdict"].strip()
        label = item["label"].strip()
        if verdict not in VALID_VERDICTS or label not in VALID_LABELS:
            continue
        out[idx] = {
            "verdict": verdict,
            "label": label,
            "confidence": str(item.get("confidence", "")).strip(),
            "reason": str(item.get("reason", "")).strip()[:60],
        }
    missing = set(expected_ids) - set(out.keys())
    if missing:
        raise ValueError(f"缺少 id: {sorted(missing)}")
    return out


def main():
    cfg = load_config()
    ecfg = cfg["emotion"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(cfg["train"]["save_dir"])
    pretrained = cfg["model"]["pretrained_model"]
    max_len = cfg["data"]["max_length"]
    data_dir = Path(cfg["data"]["processed_dir_emotion"])

    ckpt = torch.load(save_dir / ecfg["model_file_fixed"], map_location=device, weights_only=False)
    model = BERTClassifier(
        pretrained_model_name=pretrained,
        num_classes=ckpt["config"]["num_classes"],
        dropout=ckpt["config"]["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(pretrained)

    test = pd.read_csv(data_dir / "test_cleaned_fixed.csv")
    ds = BERTDataset(test["text"].tolist(), test["label"].tolist(), tokenizer, max_len)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            all_probs.append(torch.softmax(logits, dim=1).cpu())
    probs = torch.cat(all_probs, dim=0).numpy()
    preds = probs.argmax(axis=1)
    test = test.copy()
    test["pred"] = preds
    test["pred_prob"] = probs.max(axis=1)
    test["orig_cn"] = test["label_cn"]
    test["pred_cn"] = test["pred"].apply(lambda x: LABELS_CN[x])

    # 提取焦虑双向混淆
    true_anx_mis = test[(test["label"] == ANXIOUS) & (test["pred"] != ANXIOUS)].copy()
    false_anx = test[(test["pred"] == ANXIOUS) & (test["label"] != ANXIOUS)].copy()
    confused = pd.concat([true_anx_mis, false_anx], ignore_index=True)
    confused = confused.sort_values("pred_prob", ascending=False).reset_index(drop=True)
    confused["核验编号"] = range(len(confused), 0, -1)
    print(f"焦虑混淆样本: {len(confused)} 条 (真焦虑误判 {len(true_anx_mis)} + 误判为焦虑 {len(false_anx)})")

    # 按 orig_cn 排序，保证批次原标签一致
    confused = confused.sort_values(["orig_cn", "pred_prob"], ascending=[True, False]).reset_index(drop=True)

    # LLM 标注
    out_csv = Path("reports/anxiety_confusion_llm_annotated.csv")
    progress = Path("reports/_llm_anxiety_anno_progress.jsonl")

    done = {}
    if progress.exists():
        for line in progress.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            done[rec["global_id"]] = rec["result"]
        print(f"已完成 {len(done)} 条，续跑")

    client = DeepSeekClient(api_key=API_KEY)
    t0 = time.time()
    total_batches = (len(confused) - 1) // BATCH + 1
    bi = 0
    for start in range(0, len(confused), BATCH):
        bi += 1
        batch = confused.iloc[start:start + BATCH].copy()
        global_ids = batch.index.tolist()
        if all(g in done for g in global_ids):
            continue
        batch["_local_id"] = range(1, len(batch) + 1)
        prompt = build_prompt(batch)
        parsed = None
        for attempt in range(3):
            try:
                resp = client.generate(prompt, temperature=0.0, max_tokens=1800)
                parsed = parse_response(resp, list(batch["_local_id"]))
                break
            except Exception as e:
                print(f"\n批次 {bi} 第{attempt+1}次失败: {e}")
                time.sleep(2)
        if parsed is None:
            print(f"\n批次 {bi} 三次失败，跳过")
            continue
        with progress.open("a", encoding="utf-8") as f:
            for local_id, g in zip(batch["_local_id"], global_ids):
                rec = {"global_id": int(g), "result": parsed[int(local_id)]}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done[g] = parsed[int(local_id)]
        print(f"批次 {bi}/{total_batches} 完成，累计 {len(done)}/{len(confused)}，用时 {time.time()-t0:.0f}s", end="\r")

    print(f"\n生成 CSV ...")
    confused["LLM判定"] = [done.get(i, {}).get("verdict", "") for i in confused.index]
    confused["LLM建议标签"] = [done.get(i, {}).get("label", "") for i in confused.index]
    confused["LLM置信"] = [done.get(i, {}).get("confidence", "") for i in confused.index]
    confused["LLM理由"] = [done.get(i, {}).get("reason", "") for i in confused.index]

    conflict_a = (confused["LLM判定"] == "标签正确") & (confused["LLM建议标签"] != confused["orig_cn"])
    conflict_b = (confused["LLM判定"] == "标签错误") & (confused["LLM建议标签"] == confused["orig_cn"])
    confused["LLM判定_修正"] = confused["LLM判定"]
    confused.loc[conflict_a | conflict_b, "LLM判定_修正"] = "边界模糊"

    confused.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"输出: {out_csv} ({len(confused)} 条)")
    print(f"\nLLM判定_修正 分布:")
    print(confused["LLM判定_修正"].value_counts())
    wrong = confused[confused["LLM判定_修正"] == "标签错误"]
    if len(wrong):
        print(f"\n标签错误建议标签分布:")
        print(wrong["LLM建议标签"].value_counts())


if __name__ == "__main__":
    main()
