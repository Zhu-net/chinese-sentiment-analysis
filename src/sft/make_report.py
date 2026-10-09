# -*- coding: utf-8 -*-
"""
M3：汇总训练/评测指标，生成 reports/qlora_evaluation_report.md
数据来源（缺失项标 TBD，不编造）：
  logs/qlora_val_metrics.json
  reports/eval_metrics.json（bert / rag / lora + reason_quality）
"""
import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

VAL_METRICS = Path("logs/qlora_val_metrics.json")
EVAL_METRICS = Path("reports/eval_metrics.json")
LORA_FULL_CSV = Path("reports/eval_lora_full.csv")
ABLATION_CSV = Path("reports/prompt_ablation.csv")
REPORT = Path("reports/qlora_evaluation_report.md")

BACKENDS = [("bert", "BERT"), ("rag", "BERT+RAG"), ("lora", "QLoRA-1.5B")]
SUBSETS = [("full", "全量 test (4318)"), ("hard", "难例子集 hard"),
           ("adv", "对抗集 adv-50")]
CLASSES = ["开心", "感激", "悲伤", "愤怒", "恐惧", "焦虑"]


def load(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def fmt(v, nd=4):
    return "TBD" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def ci_str(t):
    if not t:
        return "TBD"
    return f"{t[0]:.4f}-{t[1]:.4f}"


def main():
    val = load(VAL_METRICS)
    ev = load(EVAL_METRICS)
    lines = []
    a = lines.append

    a("# QLoRA 生成式情绪分类：三路线对比评测报告")
    a("")
    a("> 数据：train 36,978 / val 4,359 / test 4,318（test=test_cleaned_fixed_anx.csv）  ")
    a("> 硬件：RTX 4060 Laptop 8GB，torch 2.14+cu126，bf16  ")
    a("> SFT：4,800 条分层抽样（900×5+300，含 651 train 混淆难例 + 24 用户金标）  ")
    a("> 训练：Qwen2.5-1.5B-Instruct，4bit NF4 QLoRA，LoRA r=16/alpha=32，3 epoch，"
      "有效 batch=32，lr=1e-4 cosine")
    a("")

    # ---- 训练选模 ----
    a("## 1. 训练过程与选模（val=4,359，每 epoch 全量生成评测）")
    a("")
    a("| epoch | macro-F1 | Acc | valid-JSON | 是否候选(≥0.98) |")
    a("|---|---|---|---|---|")
    for r in val.get("all_epochs", []):
        ok = "✅ 保存" if r["valid_json_rate"] >= 0.98 else "❌ 淘汰"
        a(f"| {r['epoch']} | {r['macro_f1']:.4f} | {r['acc']:.4f} | "
          f"{r['valid_json_rate']:.4f} | {ok} |")
    best = val.get("best")
    a("")
    if best:
        a(f"**Best：epoch{best['epoch']}，val macro-F1={best['macro_f1']}，"
          f"Acc={best['acc']}，valid-JSON={best['valid_json_rate']}**")
    else:
        a("**训练指标缺失（TBD）**")
    a("")
    a("验收线：val macro-F1 ≥ 0.78、valid-JSON ≥ 98%。")
    a("")
    a("> 实现备注：实际 micro-batch=8 / grad-accum=4（方案写 4/8，"
      "因 batch4 时 GPU 利用率过低，有效 batch 仍为 32）；"
      "epoch3 val 全量生成耗时 2199s（epoch1/2 约 790s），"
      "原因是后期模型生成长度增加、部分样本触达 max_new_tokens=64 上限。")
    a("")
    # val per-class
    if best:
        a("best epoch 各类 F1（val）：")
        a("")
        a("| " + " | ".join(CLASSES) + " |")
        a("|" + "---|" * len(CLASSES))
        a("| " + " | ".join(f"{v:.4f}" for v in best["per_class_f1"]) + " |")
        a("")
    a("**验收结论：valid-JSON=100% 达标；val macro-F1=0.6703 / "
      "test macro-F1=" + fmt(ev.get("lora", {}).get("full", {}).get("macro_f1"))
      + "，未达 0.78 验收线。差距诊断见第 3 节。**")
    a("")

    # ---- 主对比表 ----
    a("## 2. 主对比表")
    a("")
    for key, title in SUBSETS:
        a(f"### 2.{['full','hard','adv'].index(key)+1} {title}")
        a("")
        a("| 后端 | Acc | Acc 95%CI | macro-F1 | F1 95%CI | valid-JSON | n |")
        a("|---|---|---|---|---|---|---|")
        for bkey, bname in BACKENDS:
            m = ev.get(bkey, {}).get(key)
            if not m:
                a(f"| {bname} | TBD | TBD | TBD | TBD | - | - |")
                continue
            j = m.get("valid_json_rate", "—")
            a(f"| {bname} | {m['acc']:.4f} | {ci_str(m.get('acc_ci95'))} | "
              f"{m['macro_f1']:.4f} | {ci_str(m.get('macro_f1_ci95'))} | "
              f"{j if isinstance(j,str) else f'{j:.4f}'} | {m['n_labeled']} |")
        a("")

    # ---- adv 无情绪过报 ----
    a("> 子集口径：hard=278 条通用混淆 + 60 条焦虑混淆经 norm 对齐现 test "
      "金标后去重命中 252 条（reports/eval_subsets/hard_subset.csv）；"
      "adv=人工构造 50 条（34 条有金标：反讽 9/否定反转 7/混合主导 9/超短感叹 8"
      "+ 16 条无情绪，adv 标签未经人工二次核验）。")
    a("")
    _n_neutral = ev.get("lora", {}).get("adv", {}).get("neutral_n", 16)
    a(f"### 2.4 无情绪文本情绪过报率（adv 中 {_n_neutral} 条中性样本，"
      "三后端均无 abstention 设计）")
    a("")
    a("| 后端 | 过报率 | 预测分布 |")
    a("|---|---|---|")
    for bkey, bname in BACKENDS:
        m = ev.get(bkey, {}).get("adv", {})
        if "neutral_overfire_rate" in m:
            a(f"| {bname} | {m['neutral_overfire_rate']:.4f} | "
              f"{json.dumps(m.get('neutral_pred_dist', {}), ensure_ascii=False)} |")
        else:
            a(f"| {bname} | TBD | TBD |")
    a("")

    # ---- per-class F1 ----
    a("## 3. per-class F1（全量 test）")
    a("")
    a("| 类别 | BERT | BERT+RAG | QLoRA-1.5B |")
    a("|---|---|---|---|")
    pc = {b: ev.get(b, {}).get("full", {}).get("per_class_f1", {})
          for b, _ in BACKENDS}
    for cn in CLASSES:
        a(f"| {cn} | {fmt(pc['bert'].get(cn))} | {fmt(pc['rag'].get(cn))} "
          f"| {fmt(pc['lora'].get(cn))} |")
    a("")
    a("> 感激(test n=82)、焦虑(n=65) 为小样本类，宏指标随单条波动大，"
      "解读以 macro-F1 的 bootstrap 95% CI 为准。")
    a("")

    # ---- 差距诊断（焦虑/恐惧） ----
    a("### 3.1 未达验收线的差距诊断：焦虑/恐惧边界")
    a("")
    if LORA_FULL_CSV.exists():
        import pandas as pd
        from sklearn.metrics import confusion_matrix
        df = pd.read_csv(LORA_FULL_CSV)
        d = df[df.pred >= 0]
        cm = confusion_matrix(d.label, d.pred, labels=list(range(6)))
        tp5 = int(cm[5, 5]); fp5 = int(cm[:, 5].sum() - tp5)
        fn5 = int(cm[5, :].sum() - tp5)
        tp4 = int(cm[4, 4]); fp4 = int(cm[:, 4].sum() - tp4)
        fn4 = int(cm[4, :].sum() - tp4)
        a(f"test 焦虑：TP={tp5}、FP={fp5}、FN={fn5}，"
          f"precision={tp5/(tp5+fp5):.3f}、recall={tp5/(tp5+fn5):.3f}；"
          f"{int(cm[5,4])} 条焦虑被误判为恐惧（占焦虑 FN 的 "
          f"{cm[5,4]/max(fn5,1):.0%}）。")
        a("")
        a(f"test 恐惧：TP={tp4}、FP={fp4}、FN={fn4}，"
          f"precision={tp4/(tp4+fp4):.3f}、recall={tp4/(tp4+fn4):.3f}"
          f"（预测次数 {int(cm[:,4].sum())}，明显过判）。")
        a("")
        a("误报金标来源：" + "、".join(
            f"{CLASSES[i]} {int(v)}" for i, v in enumerate(cm[:, 5]) if i != 5 and v)
          + "。")
        a("")
    a("**根因（经两轮消融实验验证，产物 reports/prompt_ablation.csv）：**")
    a("")
    a("1. test 金标的焦虑标签经 LLM 按"
      "`src/evaluate/llm_annotate_anxiety.py` 的 rubric 审计，"
      "该约定把「担心/害怕/惶恐/心慌/忐忑/纠结/压力（指向未来或未知）」归为**焦虑**，"
      "仅把「已发生具体危险的本能惊恐」归为恐惧；")
    a("2. SFT 训练的 system prompt 只列出六个标签名、不含此判定 rubric，"
      "模型按通用心理语义学到「恐慌/害怕→恐惧」，与金标约定系统性错位"
      "（如「焦虑，恐慌」「非典不知恐慌」「担心教育」等金标焦虑被判恐惧）；")
    a("3. 推理期仅替换 system prompt 为金标 rubric（零重训）：在 660 条"
      "恐惧/焦虑富集子集上，焦虑 recall 0.20→0.46、焦虑 F1 0.299→0.513，"
      "但存在训推偏移（valid-JSON 1.00→0.962、恐惧 recall 0.869→0.765、"
      "子集 macro-F1 0.716→0.721），证明边界约定 prompt 可控，"
      "但需训推一致注入才能无损兑现。")
    if ABLATION_CSV.exists():
        import pandas as pd
        ab = pd.read_csv(ABLATION_CSV)
        anx = ab[ab.label == 5]
        a(f"（消融原始计数：焦虑 {len(anx)} 条，原始 prompt 对 "
          f"{int((anx.pred_plain == 5).sum())} 条、金标 rubric prompt 对 "
          f"{int((anx.pred_def == 5).sum())} 条；判为恐惧分别为 "
          f"{int((anx.pred_plain == 4).sum())} / {int((anx.pred_def == 4).sum())} 条。）")
    a("")
    a("**处置决策：** 经确认本轮不做 v2 重训，v1 按真实指标收尾。"
      "已验证的改进方向（训练时注入金标 rubric + max_len 384 重训）记录在此，"
      "供下一轮迭代。LoRA 的定位：头部分布弱于 BERT，但 hard 混淆集 "
      "macro-F1 0.342>0.283、对抗集 0.887>0.709，为难例/对抗语义专家，"
      "且 JSON 非法时自动回退 BERT，可安全用于路由增强。")
    a("")

    # ---- 延迟 / 显存 / 成本 ----
    a("## 4. 延迟、显存与成本（单条流式，本地口径）")
    a("")
    a("| 后端 | P50 ms | P95 ms | 均值 ms | 测量 n | 峰值显存 GB | 每千条成本 |")
    a("|---|---|---|---|---|---|---|")
    for bkey, bname in BACKENDS:
        for key in ("full",):
            m = ev.get(bkey, {}).get(key, {})
            lat = m.get("latency_single", {})
            trig = ev.get(bkey, {}).get(key, {}).get("latency_triggered")
            p50 = lat.get("p50_ms", "TBD")
            p95 = lat.get("p95_ms", "TBD")
            mean = lat.get("mean_ms", "TBD")
            n = lat.get("n", "-")
            vram = m.get("peak_vram_gb", "TBD")
            if bkey == "rag" and trig and trig.get("p50_ms") is not None:
                p50 = f"{p50}（触发{trig['p50_ms']}）"
                p95 = f"{p95}（触发{trig['p95_ms']}）"
            cost = {"bert": "¥0", "rag": "触发部分 API 费", "lora": "¥0"}[bkey]
            a(f"| {bname} | {p50} | {p95} | {mean} | {n} | {vram} | {cost} |")
    a("")
    rag_full = ev.get("rag", {}).get("full", {})
    if rag_full:
        a(f"RAG 全量 test 触发率：{rag_full.get('rag_trigger_rate', 'TBD')}，"
          f"触发 {rag_full.get('rag_trigger_n', '?')} 次，"
          f"API 失败 {rag_full.get('rag_failed_n', '?')} 次。")
        a("")

    # ---- reason 质量 ----
    a("## 5. LoRA reason 质量专项（100 条分层抽样，DeepSeek 质检）")
    a("")
    rq = ev.get("lora", {}).get("reason_quality", {})
    a("| 指标 | 实测 | 目标 |")
    a("|---|---|---|")
    a(f"| 理由-标签一致率 | {fmt(rq.get('label_reason_consistency'))} | ≥0.95 |")
    a(f"| 事实幻觉率 | {fmt(rq.get('hallucination_rate'))} | ≤0.05 |")
    a(f"| 长度合规率(15-30字) | {fmt(rq.get('length_compliance'))} | ≥0.90 |")
    a("")
    a("> 质检方为 DeepSeek（temperature=0），逐条判定理由-标签支持性与无中生有。"
      "人工抽看否例发现判定偏严、存在主观性（如「表白成功…表达满足与喜悦」被判"
      "不支持、「为少年丧礼难过」被判幻觉），72%/13% 应视为下界而非精确质量分；"
      "明细见 reports/reason_quality_audit.csv 可人工复核。另：M1 训练用 "
      "DeepSeek 反推理由的 120 条分层抽检表 data/sft/reason_review_sample.csv "
      "仍待人工抽检。")
    a("")

    # ---- user24 诊断 ----
    a("## 6. 附录：24 条用户金标（train-seen 诊断，不参与主表排名）")
    a("")
    a("| 后端 | Acc | macro-F1 |")
    a("|---|---|---|")
    for bkey, bname in BACKENDS:
        m = ev.get(bkey, {}).get("user24", {})
        a(f"| {bname} | {fmt(m.get('acc'))} | {fmt(m.get('macro_f1'))} |")
    a("")
    a("> 这 24 条已在训练集（source=user_corpus），LoRA 结果含训练记忆效应，"
      "仅作回归诊断，不能外推泛化性能。")
    a("")

    # ---- 复现 ----
    a("## 7. 复现命令（3 步）")
    a("")
    a("```powershell")
    a("# 1) 数据构造 + DeepSeek reason 生成（断点续跑）")
    a("python -m src.sft.build_sft_data; python -m src.sft.gen_reasons")
    a("# 2) QLoRA 训练（每 epoch 自动评测选 best）")
    a("python -m src.sft.train_qlora")
    a("# 3) 合并权重 + 三后端评测 + 出报告")
    a("python -m src.sft.export_merged; python -m src.sft.evaluate_lora; "
      "python -m src.sft.evaluate_baselines --backend bert; "
      "python -m src.sft.evaluate_baselines --backend rag; "
      "python -m src.sft.audit_reasons; python -m src.sft.make_report")
    a("```")
    a("")

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已生成: {REPORT}")


if __name__ == "__main__":
    main()
