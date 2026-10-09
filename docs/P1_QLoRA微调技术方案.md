# P1 技术方案：QLoRA 微调开源 LLM 实现生成式情绪分类

> 状态：方案待评审（本文档不含代码实现）
> 硬件基线：RTX 4060 Laptop 8GB（sm_89, Ada），torch 2.14+cu126，bf16 支持
> 数据基线：train 36,978 / val 4,359 / test 4,318（test = `test_cleaned_fixed_anx.csv`，已完成标签清洗）

---

## 1. 背景与目标

### 1.1 现状
线上已有两条技术路线：
- **BERT 判别式**：`bert_emotion_fixed_best.pt`，Acc 0.8467 / macro-F1 0.8229，延迟低、零 token 成本
- **BERT + RAG 精判**：低置信/焦虑预测路由到 DeepSeek few-shot，难例准确率 50%→91.7%，但每次触发有 API 成本与 1-2s 延迟，且依赖外网

### 1.2 P1 目标
新增第三条路线：**QLoRA 微调开源小模型，本地完成"情绪标签 + 判定理由"的结构化生成**，形成三路线系统对比。

三个具体交付：
1. 一个在 8GB 单卡上训练完成的 LoRA adapter，输出严格 JSON
2. FastAPI 中可切换的三后端（bert / rag / lora）
3. 一份三维（准确率 / 延迟 / 成本）对比评测报告

### 1.3 非目标（本期不做）
- 不做全量 SFT / 全参数微调
- 不做多模态、不做预训练继续 MLM
- 不追求刷到 SOTA，追求"完整、可复现、可对比"的工程闭环

---

## 2. 模型选型依据

### 2.1 候选对比

| 模型 | 参数量 | bf16 显存 | 4bit 显存 | 8GB QLoRA 可行性 | 中文能力 |
|------|-------|----------|----------|-----------------|---------|
| Qwen2.5-1.5B-Instruct | 1.5B | ~3.1GB | ~1.2GB | ✅ 宽松（micro-batch 可到 8）| 强 |
| **Qwen2.5-3B-Instruct** | 3B | ~6.1GB | ~2.0GB | ✅ 可行（batch 2-4 + 梯度检查点，峰值约 7GB）| 强 |
| ChatGLM3-6B | 6.2B | ~12.5GB | ~4.0GB | ⚠️ 紧张（激活值易 OOM，长文本必挂）| 强 |
| Qwen2.5-7B | 7B | ~14GB | ~4.5GB | ❌ 8GB 不可行 | 强 |

### 2.2 选型结论
- **主力：Qwen2.5-1.5B-Instruct**。微博文本平均 <50 字，任务是 6 分类而非开放生成，1.5B 容量足够；8GB 下训练稳定、推理快（短文本 ~30-60ms），是风险最低的选择
- **拉伸目标：Qwen2.5-3B-Instruct**。M2 完成后若 1.5B macro-F1 与 BERT 差距 >3pp，再升级 3B 对比
- 选 **Instruct 版**而非 Base 版：直接复用 ChatML 模板，JSON 指令遵循能力开箱即用
- 不选 6B/7B：8GB QLoRA 虽有成功案例但 seq_len 稍长即 OOM，复现风险高，简历项目不应把成败押在显存边缘

### 2.3 关键技术选型
- **QLoRA 而非全参 LoRA**：4bit NF4 量化基座冻结训练，8GB 的核心前提
- **transformers + peft + bitsandbytes** 官方链路为主；Windows 上 bitsandbytes 需 ≥0.43（已有官方 win wheel）
- 备选链路 **Unsloth**（训练快 20-50%、省显存）：仅在 bnb 链路踩坑时启用，不预设依赖
- 推理期 adapter 合并为 bf16 权重，避免生产环境带量化推理的兼容性问题

---

## 3. 数据方案

### 3.1 现状与问题
训练集类别极不平衡：

| 类别 | train | val | test |
|------|------:|----:|-----:|
| 开心 | 12,674 | 1,620 | 1,608 |
| 愤怒 | 11,891 | 1,518 | 1,385 |
| 悲伤 | 7,499 | 900 | 965 |
| 焦虑 | 2,686 | 40 | 65 |
| 恐惧 | 1,510 | 188 | 213 |
| 感激 | 718 | 93 | 82 |

直接全量 3.7 万条 SFT：多数类主导、训练时间长、小类学不动；故采用**分层抽样 + 类上限**。

### 3.2 SFT 训练集构造（目标 4,800 条）

| 类别 | 抽样上限 | 策略 |
|------|--------:|------|
| 开心 | 900 | 随机抽 900 |
| 愤怒 | 900 | 随机抽 900 |
| 悲伤 | 900 | 随机抽 900 |
| 焦虑 | 900 | 2,686 中抽 900（含 user_corpus 金标 4 条必入）|
| 恐惧 | 900 | 1,510 中抽 900 |
| 感激 | 300 | 718 中抽 300 + user_corpus 金标 4 条必入；剩余 418 条**全部保留到池外不参与** |
| 难例加权 | +? | 278 条混淆核验样本 + 24 条用户语料（去 test 泄漏后）整体强制纳入，再用普通样本补足上限 |

- 六类实际配比约 900×5 + 300 ≈ 4,800，感激不强凑到 900，避免 LLM 批量合成引入新噪声（数据宁缺毋滥，与项目既有教训一致）
- **test/val 严格不动**：复用现有 `val_cleaned_fixed.csv` / `test_cleaned_fixed_anx.csv`，抽样前用归一化文本去重校验，杜绝泄漏（沿用 `clean_dataset.py` 的 norm 规则）
- 固定 seed=42，抽样脚本可复现

### 3.3 reason 字段生成（DeepSeek 批处理）

SFT 目标输出含判定理由，需要为 4,800 条训练样本生成 `reason`：

- 模型：DeepSeek（现有 API）
- 输入：文本 + 金标情绪，要求 LLM **基于给定标签**反推 15-30 字理由（而非自由判类），保证理由与标签一致
- 输出 schema 约束：仅抽取理由文本，JSON 由我们拼装，不依赖 LLM 输出 JSON
- 成本估算：4,800 × (入 80 + 出 50) tokens ≈ 62 万 tokens，约 ¥2-4
- 断点续跑：写 jsonl 进度文件（沿用标注脚本的成熟模式）
- 质量抽检：每类随机抽 20 条人工看理由是否牵强，记录合格率；理由合格率 <90% 的批次重生成

### 3.4 指令数据格式（ChatML，Qwen 原生）

```
<|im_start|>system
你是中文社交媒体情绪分析专家。只输出 JSON，不要输出任何多余内容。
情绪标签限定为：开心、感激、悲伤、愤怒、恐惧、焦虑。<|im_end|>
<|im_start|>user
分析以下文本的主导情绪：
「商品收到有瑕疵，客服一直来回踢皮球，不肯处理。」<|im_end|>
<|im_start|>assistant
{"emotion": "愤怒", "reason": "商品瑕疵叠加客服推诿，引发对外的不满与恼怒"}<|im_end|>
```

字段约定：
- `emotion`：6 个中文标签之一，枚举非法即记一条坏数据
- `reason`：15-30 字，必须落在文本内可找到依据（不引入文本没有的事实）
- **训推一致**：推理时 system/user 模板逐字相同
- max_len 设 256（system 60 + user 文本上限 128 + assistant 60，覆盖 P99）

### 3.5 数据版本产物
```
data/sft/
  sft_train_4800.jsonl      # 含 reason
  sft_val_4359.jsonl        # val 只需标签，reason 留空（评测时不喂）
  reason_gen_progress.jsonl # 断点
  sample_manifest.csv       # 每条来源（原始/难例/金标）、抽样种子
```

---

## 4. 训练方案（QLoRA）

### 4.1 量化与 LoRA 配置

| 项 | 值 | 依据 |
|----|----|------|
| 量化 | 4bit NF4 + double quant | QLoRA 论文标准配方，compute_dtype=bf16 |
| LoRA rank (r) | 16 | 6 分类任务复杂度有限，16 足够，32 作为对照 |
| LoRA alpha | 32 | 经验值 = 2r，等效缩放 0.5 |
| LoRA dropout | 0.05 | 防小数据过拟合 |
| target_modules | q,k,v,o,gate_proj,up_proj,down_proj | 注意力+MLP 全挂，QLoRA 常规做法 |
| 仅训 adapter | 是 | 基座 requires_grad=False |

### 4.2 训练超参

| 项 | 值 |
|----|----|
| epochs | 3（每 epoch 存盘，按 val macro-F1 选 best）|
| micro-batch | 4（1.5B）/ 2（3B 拉伸时）|
| gradient accumulation | 8（有效 batch 32）|
| learning rate | 1e-4（QLoRA 常规区间 1e-4~2e-4）|
| lr scheduler | cosine，warmup ratio 0.03 |
| 优化器 | adamw_torch（8bit adam 仅在显存不足时启用）|
| weight decay | 0.0（LoRA 场景通常不衰减）|
| max_grad_norm | 1.0 |
| 梯度检查点 | 开启 |
| bf16 | 开启（sm_89 原生支持，禁用 fp16 防数值不稳）|
| 序列长度 | 256 |
| packing | 关闭（短文本分类任务，packing 收益小且增加模板串扰风险）|
| seed | 42（torch/numpy/random/transformers 全固定）|

预估训练时长：4,800 条 × 3 epoch，1.5B 约 25-40 分钟。

### 4.3 训练监控与选模
- 每个 epoch 末在 val（4,359 条）跑全量推理，记录：macro-F1、Acc、valid-JSON 率
- best 标准：**macro-F1 优先**，valid-JSON 率 <98% 的 checkpoint 直接淘汰
- 日志落 `logs/train_qlora_*.log`，adapter 存 `saved_models/qlora/qwen15b_emotion_r16/`
- loss 曲线：train loss 与 val 指标同图记录（matplotlib 出一张 png 存 reports/）

### 4.4 风险与预案

| 风险 | 现象 | 预案 |
|------|------|------|
| bitsandbytes Windows 兼容 | import 报错/量化初始化失败 | 升级 bnb≥0.43；仍失败转 Unsloth 预量化权重 |
| OOM | 3B 拉伸时爆显存 | 降 micro-batch→1、关长样本、退回 1.5B |
| JSON 格式漂移 | 输出带 markdown 代码块/多余字 | 训练数据零污染（assistant 只有 JSON）；推理加解析容错+正则抽取 |
| 过拟合 | train loss 持续降、val F1 epoch2 后掉 | 取 early checkpoint；r 降到 8；加 dropout 到 0.1 |
| 感激类拖后腿 | per-class F1 极低 | 已用上限策略；评测中单列报告，不掩盖 |
| reason 幻觉 | 理由引用文本不存在的信息 | 3.3 的 prompt 约束"只依据文本"；评测加理由一致性抽检指标 |

---

## 5. 推理与系统集成

### 5.1 adapter 合并
训练完成后 `merge_and_unload()` 导出 bf16 完整权重（~3.1GB），保存到 `saved_models/qwen15b_emotion_merged/`，生产环境不依赖 peft/bnb，降低部署复杂度。

### 5.2 FastAPI 三后端路由
- `EmotionPredictor` 增加 `model_type` 维度：`bert` / `rag` / `lora`
- config.yaml 新增 `llm_finetune` 段：模型路径、max_new_tokens=64、temperature=0（分类任务贪心解码，保证可复现）、JSON 解析失败时的回退目标（回退 BERT）
- 接口层：`/predict_emotion` 加可选参数 `model_type`，默认仍走 bert（线上零回归风险）
- LoRA 后端用 transformers 本地推理（vLLM 在 Windows 需 WSL，列为后续部署优化，不进本期）
- 批量接口对 lora 后端做 left-padding 批量生成，控制延迟

### 5.3 返回结构（lora 后端扩展）
在现有返回体上增加：`emotion`、`confidence`（取生成标签的 one-hot 置信，或多次温度采样一致性，本期用前者并在报告注明局限）、`reason`、`valid_json`、`latency_ms`。

---

## 6. 评测表设计

### 6.1 评测数据集（三个子集）
| 子集 | 规模 | 用途 |
|------|-----:|------|
| 全量 test | 4,318 | 主指标，与 BERT 历史结果同口径 |
| 难例子集 hard-300 | ~300 | 278 混淆核验 + 24 用户语料 + 焦虑混淆样本去重构成 |
| 对抗子集 adv-50 | ~50 | 新构造：无情绪文本（"就像图里这位"）、反讽、多情绪混合、超短感叹 |

### 6.2 主对比表（报告核心产出）

| 指标 | BERT | BERT+RAG | **QLoRA-1.5B** | QLoRA-3B(可选) |
|------|------|----------|---------------|----------------|
| Accuracy (全量) | 0.8467 | 待测 | | |
| macro-F1 (全量) | 0.8229 | 待测 | | |
| 难例 Acc (hard-300) | 基线 | 0.917* | | |
| 对抗集 Acc (adv-50) | 待测 | 待测 | | |
| valid-JSON 率 | — | — | | |
| P50 延迟 (单条/本地) | ~20ms | 20ms / 触发 1-2s | | |
| P95 延迟 | 待测 | 待测 | | |
| 峰值显存 | ~1.5GB | ~1.5GB | | |
| 每千条成本 | ¥0 | 触发部分 API 费 | ¥0（本地）| |

\* 0.917 是 24 条小样本数，hard-300 上需重测得到可信数字。

### 6.3 per-class F1 表（六类 × 三模型）
| 类别 | test n | BERT | BERT+RAG | QLoRA-1.5B |
|------|-------:|------|----------|-----------|
| 开心/感激/悲伤/愤怒/恐惧/焦虑 各一行，F1 + recall 双列 |

重点关注：感激（82）、焦虑（65）小样本类的置信区间要在报告中标注（bootstrap 95% CI），避免又拿 1-2 条波动下结论。

### 6.4 reason 质量专项表（QLoRA 独有）
| 指标 | 定义 | 目标 |
|------|------|------|
| 理由-标签一致率 | 抽 100 条，理由是否支持输出标签 | ≥95% |
| 理由事实性 | 理由是否无中生有 | 幻觉率 ≤5% |
| 长度合规率 | 15-30 字占比 | ≥90% |

### 6.5 成本-效果权衡分析
- 给出三路线适用场景结论：高并发常规文本 → BERT；少量难例可接受 API 成本 → RAG；离线/内网/需理由 → QLoRA
- 若做混合路由（BERT 兜底 + LoRA 本地精判替代 RAG），测算相比 RAG 省多少 API 成本

---

## 7. 里程碑与验收

| 里程碑 | 内容 | 工时 | 验收标准 |
|--------|------|-----:|---------|
| M0 环境 | bnb/peft/transformers 版本锁定，Qwen 权重就位（HF 镜像），4bit 加载跑通 | 0.5d | 能加载 1.5B 并生成一句话 |
| M1 数据 | 分层抽样 4,800 条 + reason 批量生成 + 去重校验 | 1d | sft_train_4800.jsonl 产出，抽检合格率≥90% |
| M2 训练 | QLoRA 三 epoch，选 best adapter | 0.5-1d | val macro-F1 ≥0.78（与 BERT 差距可解释即可），JSON 率≥98% |
| M3 集成评测 | 合并权重、三后端路由、跑完三张评测表、出报告 | 1.5d | 接口可切换、报告含全部表格与 CI |

### 最终验收
1. `/predict_emotion?model_type=lora` 返回含 reason 的合法 JSON
2. 三方对比报告落 `reports/qlora_evaluation_report.md`
3. 单卡 8GB 从零复现命令 ≤3 条（数据构造 / 训练 / 评测各一条）

---

## 8. 简历叙事映射（验收后可写）

- 「单卡 8GB 用 QLoRA(NF4) 微调 Qwen2.5-1.5B，输出带理由的结构化情绪，valid-JSON xx%」
- 「设计 BERT / RAG / LoRA 三路线在准确率、P95 延迟、千条成本上的系统评测，给出场景化路由结论」
- 「分层抽样 + 标签条件化理由生成，缓解 6 类 17:1 不平衡下的小类退化」

---

## 9. 待确认决策点
1. 主力模型锁定 Qwen2.5-1.5B，3B 仅作拉伸——是否同意？
2. 感激类不做 LLM 合成补足（上限 300）——是否接受该类指标可能仍弱？
3. reason 生成复用 DeepSeek API（约 ¥2-4）——是否批准？
4. adv-50 对抗集由我构造模板后是否需要你人工过一遍标签？
