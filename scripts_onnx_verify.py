# -*- coding: utf-8 -*-
"""ONNX 推理器验证：精度一致性 + 接口兼容"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from src.deploy.onnx_predictor import OnnxEmotionPredictor, OnnxBinaryPredictor
from src.deploy.emotion_predictor import get_emotion_predictor
from src.deploy.predictor import get_predictor

cases = [
    "今天收到了录取通知书，开心得跳起来了！",
    "谢谢室友在我生病时照顾我，真的非常感激",
    "养了十年的狗狗昨天走了，心里特别难过",
    "快递等了一周还没到，客服态度还差，气死我了",
    "半夜看恐怖片，总觉得身后有人，太吓人了",
    "明天就要面试了，紧张得一晚上没睡着",
]

print("===== ONNX vs PyTorch 情绪模型对比 =====")
onnx_emo = OnnxEmotionPredictor()
pt_emo = get_emotion_predictor()

all_match = True
for t in cases:
    r_onnx = onnx_emo.predict(t)
    r_pt = pt_emo.predict(t)
    match = r_onnx["emotion"] == r_pt["emotion"]
    all_match &= match
    print(f"[{'✓' if match else '✗'}] ONNX={r_onnx['emotion_cn']:<3}({r_onnx['confidence']*100:.0f}%) "
          f"PT={r_pt['emotion_cn']:<3}({r_pt['confidence']*100:.0f}%) | {t[:20]}")

# 二分类对比
print("\n===== ONNX vs PyTorch 二分类对比 =====")
onnx_bin = OnnxBinaryPredictor()
pt_bin = get_predictor()
for t in cases[:3]:
    r_onnx = onnx_bin.predict(t)
    r_pt = pt_bin.predict(t)
    match = r_onnx["label"] == r_pt["label"]
    all_match &= match
    print(f"[{'✓' if match else '✗'}] ONNX={r_onnx['label']}({r_onnx['confidence']*100:.0f}%) "
          f"PT={r_pt['label']}({r_pt['confidence']*100:.0f}%) | {t[:20]}")

# 批量接口
batch = onnx_emo.predict_batch(cases, batch_size=3)
assert len(batch) == len(cases) and all("emotion_cn" in b for b in batch)
print(f"\n批量接口 OK: {[(b['emotion_cn'], b['polarity']) for b in batch]}")

print(f"\n{'✅ 全部一致' if all_match else '⚠️ 存在差异'}")
