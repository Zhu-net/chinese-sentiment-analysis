# -*- coding: utf-8 -*-
"""
悲伤 ↔ 愤怒 混淆样本人工核验导出
================================
用两个独立训练的模型（baseline CE / 两阶段 SCL）在同一测试集上推理：
  - 两个模型一致、且高置信度地把样本判成另一类 => 标签疑似有误，优先核验
  - 两个模型分歧或低置信度 => 疑似语义天然重叠，判定为"边界模糊"

导出 CSV（utf-8-sig，Excel 可直接打开），预留人工标注列。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.utils import load_config
from src.models.bert_dataset import BERTDataset
from src.models.contrastive_bert_classifier import ContrastiveBERTClassifier

SAD, ANGRY = 2, 3


def predict_probs(model, loader, device):
    """返回全测试集 softmax 概率矩阵 [N, 6]"""
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out = model(ids, mask)
            logits = out[0] if isinstance(out, tuple) else out
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def load_model(ckpt_path, device, pretrained, proj_dim):
    model = ContrastiveBERTClassifier(
        pretrained_model_name=pretrained,
        num_classes=6, dropout=0.15, projection_dim=proj_dim,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model


def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(cfg["train"]["save_dir"])
    data_dir = Path(cfg["data"]["processed_dir_emotion"])
    pretrained = cfg["model"]["pretrained_model"]
    proj_dim = cfg["train"]["scl"]["projection_dim"]
    max_len = cfg["data"]["max_length"]
    labels_cn = cfg["emotion"]["labels_cn"]

    test_df = pd.read_csv(data_dir / "test.csv")
    texts = test_df["text"].tolist()
    y = test_df["label"].tolist()
    print(f"测试集: {len(test_df)} 条；设备: {device}")

    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    ds = BERTDataset(texts, y, tokenizer, max_len)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    # 两个独立模型分别推理
    print("加载模型 1/2: baseline (bert_emotion_best.pt) ...")
    m_base = load_model(save_dir / "bert_emotion_best.pt", device, pretrained, proj_dim)
    p_base = predict_probs(m_base, loader, device)

    print("加载模型 2/2: 两阶段 SCL (bert_emotion_scl2_best.pt) ...")
    m_scl = load_model(save_dir / "bert_emotion_scl2_best.pt", device, pretrained, proj_dim)
    p_scl = predict_probs(m_scl, loader, device)

    pred_base = p_base.argmax(axis=1)
    pred_scl = p_scl.argmax(axis=1)

    # 只保留 悲伤<->愤怒 互判样本（至少一个模型把该样本在这两类间判错）
    rows = []
    for i in range(len(texts)):
        true = y[i]
        cross_pairs = {(SAD, ANGRY), (ANGRY, SAD)}  # (真实, 模型预测)
        in_scope = (true, pred_base[i]) in cross_pairs or (true, pred_scl[i]) in cross_pairs
        if not in_scope:
            continue

        direction = "悲伤→愤怒" if true == SAD else "愤怒→悲伤"
        # 以 SCL 模型（当前最优）的概率为基准
        probs = p_scl[i]
        wrong_target = ANGRY if true == SAD else SAD
        margin = float(abs(probs[true] - probs[wrong_target]))  # 越小越模糊
        agree = (pred_base[i] == pred_scl[i] == wrong_target)   # 双模型一致误判

        rows.append({
            "核验编号": len(rows) + 1,
            "混淆方向": direction,
            "文本": texts[i],
            "原标签": labels_cn[true],
            "baseline预测": labels_cn[pred_base[i]],
            "SCL预测": labels_cn[pred_scl[i]],
            "双模型一致误判": "是" if agree else "否",
            "P(悲伤)": round(float(p_scl[i][SAD]), 4),
            "P(愤怒)": round(float(p_scl[i][ANGRY]), 4),
            "两类概率差margin": round(margin, 4),
            "模型置信度": round(float(probs.max()), 4),
            # —— 以下为人工填写列 ——
            "核验结论": "",        # 标签正确 / 标签错误 / 边界模糊（双重情绪）
            "建议正确标签": "",     # 开心/感激/悲伤/愤怒/恐惧/焦虑
            "备注": "",
        })

    out = pd.DataFrame(rows)
    # 排序：双模型一致误判优先，其次 margin 小（模型在两类间最纠结）的排前面
    out["_agree_rank"] = (out["双模型一致误判"] == "是").astype(int)
    out = out.sort_values(["_agree_rank", "两类概率差margin"], ascending=[False, True])
    out["核验编号"] = range(1, len(out) + 1)
    out = out.drop(columns="_agree_rank")

    out_dir = Path("reports"); out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "confusion_sad_angry_review.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    # ===== 控制台概览 =====
    print("\n" + "=" * 70)
    print(f"导出完成: {out_path}  共 {len(out)} 条")
    print("=" * 70)
    print("\n按方向 × 双模型是否一致：")
    print(pd.crosstab(out["混淆方向"], out["双模型一致误判"], margins=True).to_string())

    # 双模型一致 + 高置信（>0.7）误判 = 标签疑似有误的最强信号
    strong = out[(out["双模型一致误判"] == "是") & (out["模型置信度"] > 0.7)]
    print(f"\n强信号（双模型一致误判且置信度>0.7）: {len(strong)} 条，"
          f"占互判样本 {len(strong)/len(out):.0%}")
    print("  建议优先核验这批样本。")

    print("\n前 15 条强信号样本预览：")
    for _, r in strong.head(15).iterrows():
        print(f"[{r['混淆方向']}] 置信{r['模型置信度']:.2f} "
              f"P悲={r['P(悲伤)']:.2f} P怒={r['P(愤怒)']:.2f} | {str(r['文本'])[:50]}")


if __name__ == "__main__":
    main()
