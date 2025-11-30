# SemEval 2026 Task 12 – 实验进展记录
> 孙语蔚  
> 仓库：`swy1223/semeval2026-task12`  
> 最近更新日期：2025-11-30

本文件用于记录本周在 **数据增强（docs & questions）** 和 **backbone 替换（BERT → DeBERTa-v3）** 方面的主要实验与结论，方便老师查看进展与复现。

---

## 1. 数据与数据增强

### 1.1 原始数据（official train & dev）

- **Train（原始）**
  - `data/train/questions.jsonl`：约 200 条多标签问题（选项 A–D，可多选）
  - `data/train/docs.json`：181 篇新闻，按 `topic_id` 分组
- **Dev**
  - `data/dev/questions.jsonl`：400 条问题
  - `data/dev/docs.json`：与 dev 对应的新闻集合

### 1.2 本周的数据增强方案

原始数据量较小，BERT 模型在 train 上出现明显过拟合，尝试用大模型做“黑盒蒸馏式”数据合成。

我目前实现了两部分增强：

#### (1) docs 增强（新闻扩写）

- 脚本：`scripts/agument_docs.py`
- 思路：
  - 对每篇原始新闻，构造 prompt，让大模型在**保持主题和核心事实方向不变**的前提下，改写/扩写一篇“风格类似的新新闻”。
  - 同时控制长度、避免直接复制，尽量保证语义相近但表达多样。
- 结果：
  - 在原始 181 篇基础上得到约 **3000 篇新闻**。
  - 保存路径：`data/train/docs_augmented_3000.json`
  - 最终结构与原始 `docs.json` 保持一致：按 `topic_id` 分组，每个 topic 下的 `docs` 列表包含原始 + 合成新闻。

#### (2) questions 增强（问题改写/扩展）

- 在原始 200 条 question 基础上，对每个 `(topic, question)` 使用大模型进行改写或生成相似问题：
  - 保持所属 topic 不变；
  - 保持推理类型不变（例如仍需结合多条证据、仍是多选推理）；
  - 改变措辞、加入轻微背景变化，增加语言多样性。
- 结果：
  - 生成若干新问题，并与原始问题合并，得到约 **1000 条**训练问题。
  - 保存路径：`data/train/questions_augmented_1000.jsonl`
- 目前训练时，**原始样本与合成样本同权**，没有额外的 re-weight 或 filter。

---

## 2. 模型与训练配置

统一训练入口：`scripts/SemEval_Simple_TopK3.py`

- 任务形式：多标签分类（输出 A/B/C/D 四个标签，各自 sigmoid + BCE）
- Loss：
  - Binary Cross-Entropy with `pos_weight`（对正类做一定放大，缓解标签不均衡）
- 文本编码：
  - 使用 backbone encoder（BERT 或 DeBERTa-v3）对新闻滑窗编码
  - 参数：
    - `max_len = 384`
    - `window = 224`
    - `stride = 144`
    - `max_windows = 6`
- 聚合方式（doc-level → question-level）：
  - `--agg topk --topk_k 3`
  - 即对每个问题，从相关 doc 窗口中选 top-k 得分做聚合（simple top-k pooling）
- 主要 backbone：
  - `bert-base-uncased`
  - `microsoft/deberta-v3-base`
- 训练加速：
  - 使用 `--amp` 进行混合精度训练（PyTorch AMP）

---

## 3. 本周关键实验与结果（dev 集）

> 说明：  
> 所有分数均在 `data/dev/questions.jsonl` 上评估。  
> - **F1 指标** 来自：`scripts/evaluate_dev_post_calib.py`  
> - **Official-style score** 来自：`scripts/evaluate_official_score.py`（模仿 SemEval 官方打分方式）

### 3.1 实验总表

| ID  | Backbone                 | Train 数据                            | Epochs | Batch | micro-F1 | macro-F1 | Official score |
|-----|--------------------------|----------------------------------------|--------|-------|----------|----------|----------------|
| A   | **BERT-base**           | 原始 Q (200) + 原始 docs (181)        | 3      | 1     | 0.6365   | 0.6370   | 0.5387         |
| B   | BERT-base               | 增强 Q (1000) + 增强 docs (3000)      | 3      | 4     | 0.6244   | 0.6218   | 0.5387         |
| C   | BERT-base               | 增强 Q (1000) + 增强 docs (3000)      | 3      | 1     | 0.6216   | 0.6201   | 0.5288         |
| D   | **DeBERTa-v3-base**     | 原始 Q (200) + 原始 docs (181)        | 3      | 2     | 0.6401   | 0.6404   | **0.5413**     |
| E   | DeBERTa-v3-base         | 增强 Q (1000) + 增强 docs (3000)      | 3      | 2     | 0.5652   | 0.5645   | 0.4875         |

#### 对应训练命令示例

- **A：BERT + 原始数据（baseline）**

```bash
python3 scripts/SemEval_Simple_TopK3.py \
  --train_jsonl data/train/questions.jsonl \
  --train_docs_json data/train/docs.json \
  --dev_jsonl data/dev/questions.jsonl \
  --dev_docs_json data/dev/docs.json \
  --save_dir outputs_bert_topk3 \
  --backbone bert-base-uncased \
  --epochs 3 \
  --batch_size 1 \
  --max_len 384 --window 224 --stride 144 --max_windows 6 \
  --agg topk --topk_k 3 \
  --pos_weight \
  --amp \
  --seed 42

3.2.2 B：BERT + 增强(Q1000, D3000)
python3 scripts/SemEval_Simple_TopK3.py \
  --train_jsonl data/train/questions_augmented_1000.jsonl \
  --train_docs_json data/train/docs_augmented_3000.json \
  --dev_jsonl data/dev/questions.jsonl \
  --dev_docs_json data/dev/docs.json \
  --save_dir outputs_bert_topk3_augQ1000_D3000_e3_b4 \
  --backbone bert-base-uncased \
  --epochs 3 \
  --batch_size 4 \
  --max_len 384 --window 224 --stride 144 --max_windows 6 \
  --agg topk --topk_k 3 \
  --pos_weight \
  --amp \
  --seed 42

3.2.3 D：DeBERTa-v3 + 原始数据（当前最优配置）
python3 scripts/SemEval_Simple_TopK3.py \
  --train_jsonl data/train/questions.jsonl \
  --train_docs_json data/train/docs.json \
  --dev_jsonl data/dev/questions.jsonl \
  --dev_docs_json data/dev/docs.json \
  --save_dir outputs_deberta_v3_topk3_orig_e3_b2 \
  --backbone microsoft/deberta-v3-base \
  --epochs 3 \
  --batch_size 2 \
  --max_len 384 --window 224 --stride 144 --max_windows 6 \
  --agg topk --topk_k 3 \
  --pos_weight \
  --amp \
  --seed 42

3.2.4 E：DeBERTa-v3 + 增强(Q1000, D3000)
python3 scripts/SemEval_Simple_TopK3.py \
  --train_jsonl data/train/questions_augmented_1000.jsonl \
  --train_docs_json data/train/docs_augmented_3000.json \
  --dev_jsonl data/dev/questions.jsonl \
  --dev_docs_json data/dev/docs.json \
  --save_dir outputs_deberta_v3_topk3_augQ1000_D3000_e3_b2 \
  --backbone microsoft/deberta-v3-base \
  --epochs 3 \
  --batch_size 2 \
  --max_len 384 --window 224 --stride 144 --max_windows 6 \
  --agg topk --topk_k 3 \
  --pos_weight \
  --amp \
  --seed 42

4. 多标签预测行为分析（每题预测集合大小分布）

为观察模型是否“过于保守”或“过于贪心”，统计 dev 集上每道题预测了多少个标签（|预测集合|）。

4.1 BERT + 原始数据（A）

1 个标签：26.5%

2 个标签：33.5%

3 个标签：26.2%

4 个标签：13.8%

行为：多标签预测比较丰富，1～4 个标签都有，和任务设定较匹配。

4.2 DeBERTa-v3 + 原始数据（D）

1 个标签：28.5%

2 个标签：27.8%

3 个标签：24.5%

4 个标签：19.2%

行为：与 BERT+原始类似，但更愿意预测 3–4 个标签，召回略好，因此宏 F1 与官方分略高于 BERT baseline。

4.3 DeBERTa-v3 + 增强数据（E）

1 个标签：71.8%

2 个标签：20.0%

3 个标签：7.5%

4 个标签：0.8%

行为：模型明显偏向单标签预测（超过 70% 的题只选 1 个选项），
多标签能力几乎被“抹平”，导致召回严重下降，macro-F1 和官方分均明显低于原始数据训练的同一 backbone。

5. 结论
5.1 Backbone 替换效果

在完全相同的原始 train/dev 数据和训练脚本下，将 backbone 从 bert-base-uncased 换为 microsoft/deberta-v3-base：

macro-F1：约从 0.637 → 0.640

官方分：约从 0.5387 → 0.5413

因此，目前最优配置为：
DeBERTa-v3-base + 原始 train 数据（ID D）。

5.2 当前版本数据增强的收益有限 / 对 DeBERTa 有负面影响

对 BERT 而言：

使用 Q=1000 + D≈3000 的增强数据，官方分数基本不变（仍在 0.5387 左右），macro-F1 略有下降，说明这版增强对 BERT 是“中性略负”的。

对 DeBERTa 而言：

在相同增强数据上训练时，模型学习到较强的“单标签偏好”，大部分题只预测 1 个选项，导致 F1 和官方分数显著降低。

说明当前合成数据的分布与 dev/test 存在偏差，且 对大容量 backbone（DeBERTa-v3）更敏感。

5.3 整体评价

目前这版“黑盒蒸馏式”数据增强，在现有 prompt 与比例（Q=1000, D≈3000）下，尚未在 dev 上带来稳定的性能提升；

简单堆叠大量合成样本，并不一定优于“较小但高质量的原始数据 + 更强的 backbone”。
