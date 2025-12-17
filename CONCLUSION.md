# SemEval 2026 Task 12 – Progress Log & Paper-Ready Summary

## 1. Abstract 

  该repo记录了SemEval Task 12的渐进建模流程。从类似HuggingFace的BERT多项选择基线（E0）出发，我们将任务重新表述为多标签分类，并通过滑动窗口/全文档分块处理长证据处理，并通过窗口级聚合（TopK）和阈值校准（全局/每标签/主题）以及官方评分导向的后处理提升推断能力。repo里还包含了数据增强试验和使用 Optuna 进行超参数调优。

## 2.Task Formulation
### E0: Multiple-Choice baseline (起点版本)

- 建模：每题 4 个选项（A/B/C/D），模型输出 4 个 logits，用 CrossEntropyLoss 训练单一正确答案。代码里你自定义了 BertForMultipleChoice，把输入 reshape 为 (batch_size*num_choices, seq_len)，再还原为 (batch_size, num_choices) 并用 CE。

- 输入构造：event [SEP] context [SEP] option；context 来自 topic 下文档，先取前 2 篇文档、每篇截断前 200 字符，保证不爆长。

- 数据划分：搭建baseline时验证集暂未发布，所以从训练集中按 0.8/0.2 切分 train/val（200→160/40），并启用早停（20 epochs，patience=5）。

- 结果：val accuracy ≈ 0.5000
 
- 不足：该方法只能进行单项选择，无法多选。

### E1+:多项选择(BCE) 

- 核心转变：把每个选项当成一个二分类，模型输出 4 维 logits，训练用 BCEWithLogitsLoss/BCE（或后续 focal/asl），推理用 sigmoid 得到每个选项概率。

- 阈值推理：在代码里做了 global threshold search（在 0.30~0.80 扫描，选 macro-F1 最大者）。

- 评估指标：subset accuracy、micro-F1、macro-F1（直接基于 y_true/y_pred）。

- 结果：macro-F1 0.5858

## 3. 证据过长的工程化解决
### 3.1 滑动窗口（E1–E4）

引入了窗口数据集：将“事件+文档证据+选项”切成多个窗口（window/stride/max_windows 可调），每个窗口各自前向，再把“同一题同一选项”的多个窗口 logits 聚合回去。

- 聚合（max）：实现了 aggregate_logits_max：先把扁平 logits 按 owners 映射回 (B,4)，聚合阶段用 float32 并兼容 AMP/scatter_reduce_。

- 训练循环支持 AMP：使用 torch.amp.autocast + GradScaler("cuda")。

### 3.2 TopK aggregation（E3–E13）

仅用max容易被噪声窗口“误触发”；后续把聚合升级为TopK：对同一 (样本,选项) 的多个窗口logits取TopK做平均/聚合（实现名为 aggregate_logits_topk，并强调聚合阶段float32、更稳）。

### 3.3 AllDocs chunking（E8–E13，关键跃迁）

在 AllDocs 版本里，不再只用少量 docs，而是把 topic 下所有 docs 拼接/组合，然后按 window/stride 切 chunk（max_windows 可到 16），再走同样的 TopK 聚合。你在注释里明确写了“把所有 docs 内容拼在一起，再切 chunk”。

- 这是后期 macro-F1 大幅提升的结构性原因之一：覆盖证据更全，但代价是计算量上升，需要 topk/阈值来抑制噪声。

## 4.Calibration&Post-processing
### 4.1 Global threshold (E1 起)

直接扫阈值选 macro-F1 最大（你固定扫描区间 0.30~0.80）。

### 4.2 Per-label thresholds（E3+）

- 逐类阈值：每个 label 在一组候选阈值上扫描、选 binary-F1 最好者，然后推理用 [thr_A, thr_B, thr_C, thr_D]。

- 兜底策略：如果 4 类都没过阈值，就取 argmax 确保至少输出 1 个标签。

### 4.3Per-topic calibration（E3_topk起）

实现了 compute_per_topic_delta：先统计每个 topic 的平均预测集合大小 avg_k，再按 (avg_k - target_k)*alpha 生成 delta（avg_k 大于目标就“提高阈值”更严格），并 clip 到区间内。

同时脚本会把 per_label_thr、agg、topk_k 等写入 val_calibration.json，并在 --per_topic_calibration 开启时写回 topic delta。

### 4.4 cap_k（限制最多输出几个标签）

在导出预测时，加入 cap_k：过阈值的候选按概率排序，最多保留前 cap_k 个，同时保留“空集合→argmax”。

### 4.5Official-score-oriented threshold search（E13）

后期新增了专门优化 official score 的阈值搜索脚本：对阈值做搜索，输出官方风格平均得分最好的阈值，并保存新预测文件。
- 结果：日志里显示 best thr=0.40，official avg score 0.6725（cap_k=3）。

## 5.损失函数和训练块
### 5.1 Focal loss（E10–E13）
在AllDocs + DeBERTa上把损失切到focal，dev macro-F1（post-calib）达到 0.7370，official score 0.6675。

### 5.2 pos_weight（类别不均衡尝试）
尝试做过 focal + pos_weight 的版本，在实现里显式构造了 pos_weight_vec 并在训练时传入。
- 结果：post-calib macro-F1=0.7274，official=0.6575。

### 5.3 R-Drop
实现了双forward的对称KL(kl(p1||p2)+kl(p2||p1))，总loss=ce+alpha*kl。
- 结果：post-calib macro-F1=0.6728，official=0.6300（明显掉分）。

## 6. Data augmentation（生成式增强）

写了两个增强脚本：用 gpt2-medium 的 text-generation pipeline 生成 docs/questions，并写回 json/jsonl。
- 结果：简单生成增强并没有稳定带来提升：

  - BERT 原始：macro=0.6370；增强后反而下降到 0.6218/0.6201。
  - DeBERTa 原始：macro=0.6528；增强后下降到0.6396。

## 7. Hyperparameter tuning with Optuna（E12）

新增 tune_with_optuna.py，把搜索空间明确限定在：

- lr：log scale，7e-6~2e-5

- topk_k：2~4

- chunk_idx：从预设 chunk configs 里选 window/stride/max_windows

- batch_size：1~3
并以 dev macro-F1 为 objective，最终把 best trial 写入 optuna_logs/best_params.json。

- 结果：日志里记录 best trial macro-F1=0.7477，且对应参数被保存。

## 8. Experiments (E0–E13) & Results

### 8.1 Baselines (BERT)
| ID   |  Script | Key idea | Dev metric |
| :-----:| :----: | :-----:| :-----:| 
| E0 | semeval_train.py | BERT多项选择，event+2docs+option | val accuracy ≈ 0.5000 |
| E1 | bert_BCE+slipwindow.py |	多标签BCE + 滑窗 + 全局阈值搜索| macro-F1 0.5036|
| E2 | bert_BCE+slipwindow_topk.py | TopK 聚合 +（逐类阈值/可选 per-topic）|macro-F1 0.5858|
| E3 | bert_BCE_win_topk_amp.py | TopK + AMP + 逐类阈值 + argmax |macro-F1 0.6041 |
| E4 |bert_BCE_win_topk_amp_post-calib | 额外后处理校准post-calib | macro-F1 0.6313 |

### 8.2 DeBERTa family 
| ID   |  Script | Key idea | Dev metric |
| :-----:| :----: | :-----:| :-----:| 
| E5 | SemEval_Simple_TopK3.py | backbone改成DeBERTa-v3-base + TopK3 | macro-F1 0.6404；official 0.5413|
| E6 | SemEval_Simple_TopK3_Alldocs.py | AllDocs chunk + TopK3 | macro-F1 0.6755；official 0.6188|
| E7 | SemEval_Simple_TopK3_Alldocs_Focalloss_Rdrop.py | AllDocs + Focal | post-calib macro-F1 0.7370；official 0.6675|
| E8 |SemEval_Simple_TopK3_Alldocs_Focalloss_Rdrop.py |	Focal + pos_weight | macro-F1 0.7274；official 0.6575|
| E9 |SemEval_Simple_TopK3_Alldocs_Focalloss_Rdrop.py|R-drop α=0.5+Focal | macro-F1 0.6728；official 0.6300|
| E10	|tune_with_optuna.py | Optuna 搜索 lr/topk/chunk/bs	|macro-F1 0.7477（trial）|
| E11	|evaluate_dev_post_calib.py	|对 E10 结果做 post-calib	post-calib | macro-F1 0.7393；official 0.6713|
|E12 | search_official_threshold.py |直接搜索 official score最优阈值| best thr=0.40；macro-F1 0.7372；official 0.6725|


## 9. What helped / What didn’t 
### 9.1 helped

- 任务重构：MC → 多标签
允许输出多项成立，推理用 sigmoid+阈值，比强制单选更贴合你后续整体管线（阈值搜索、逐类阈值、cap_k）。

- TopK 聚合（比 max 更抗噪）
把“窗口噪声误触发”降下来，等价于在长证据下做一种轻量的 evidence pooling。

- AllDocs chunking + TopK（结构性增益最大）
覆盖证据显著更全，直接把 macro-F1 推到 ~0.67+，再叠 focal 到 ~0.737。

- Focal loss
在 AllDocs+DeBERTa 下效果最好（你记录的 post-calib macro-F1 0.7370）。

- 逐类阈值 + 空集合兜底 + cap_k
让输出集合大小更可控，尤其适合 official score 或多标签场景。

### 9.1 did not help

- R-drop（α=0.5）：实现正确，但结果明显掉分（macro/official 都降）。

- naive GPT2 augmentation：BERT/DeBERTa 在增强数据上都没有更好，可能有分布漂移/噪声。

## 10. Reproducibility（论文复现实验建议写法）
### 10.1 训练命令模板
python3 scripts/<YOUR_SCRIPT>.py \
  --train_jsonl data/train/questions.jsonl \
  --train_docs_json data/train/docs.json \
  --dev_jsonl data/dev/questions.jsonl \
  --dev_docs_json data/dev/docs.json \
  --save_dir outputs/<EXP_NAME> \
  --backbone microsoft/deberta-v3-base \
  --epochs 6 --batch_size 2 \
  --max_len 512 \
  --window 384 --stride 224 --max_windows 16 \
  --agg topk --topk_k 3 \
  --amp --gpu 0 \
  --cap_k 3 \
  --lr 1e-5 \
  --loss_type focal \
  --rdrop_alpha 0.0

### 10.2 后处理评估（post-calib + official）
python scripts/evaluate_dev_post_calib.py
python scripts/evaluate_official_score.py
python scripts/search_official_threshold.py
