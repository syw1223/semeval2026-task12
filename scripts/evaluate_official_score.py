# -*- coding: utf-8 -*-
"""
按照 AER 官方描述的 1 / 0.5 / 0 规则，对 dev 做整体评分。
- 完全匹配 gold: 1 分
- 有交集但不完全匹配: 0.5 分
- 没交集: 0 分
最后输出平均分（0~1）
"""

import json
import os

PRED = "outputs_deberta_v3_topk3_orig_e3_b2/dev_predictions.json"
GOLD = "data/dev/questions.jsonl"

print("[INFO] PRED =", PRED)
print("[INFO] GOLD =", GOLD)

if not os.path.exists(PRED):
    raise FileNotFoundError(f"Predictions file not found: {PRED}")
if not os.path.exists(GOLD):
    raise FileNotFoundError(f"Gold file not found: {GOLD}")

# 读取 gold：question_id -> set of correct labels
gold = {}
with open(GOLD, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        qid = obj.get("question_id") or obj.get("uuid") or obj.get("id")
        ans = obj.get("golden_answer", "")
        gold_set = set(x.strip() for x in str(ans).split(",") if x.strip())
        gold[qid] = gold_set

print("[INFO] Loaded gold questions:", len(gold))

# 读取预测：question_id -> set of predicted labels
with open(PRED, "r", encoding="utf-8") as f:
    preds = json.load(f)

scores = []
missing = 0

for r in preds:
    qid = r.get("question_id")
    if qid not in gold:
        missing += 1
        continue

    pred_str = r.get("prediction", "")
    pred_set = set(x.strip() for x in str(pred_str).split(",") if x.strip())

    gold_set = gold[qid]

    # 官方 1 / 0.5 / 0 规则（假定：有交集但不完全 = 部分匹配）
    if pred_set == gold_set:
        s = 1.0
    elif pred_set & gold_set:
        s = 0.5
    else:
        s = 0.0

    scores.append(s)

if not scores:
    raise RuntimeError("No aligned predictions, please check ids!")

avg_score = sum(scores) / len(scores)
print(f"[RESULT] Official-style average score = {avg_score:.4f}")
print(f"[INFO] Evaluated {len(scores)} items, skipped {missing} items with no gold.")
