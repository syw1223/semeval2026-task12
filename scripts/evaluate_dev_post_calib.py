# -*- coding: utf-8 -*-
"""
Post-calib 评估脚本（带自检日志）
- 读取 dev 导出的 predictions（已应用 per-label + per-topic + cap_k）
- 与 gold 对齐后，打印 Micro/Macro-F1 和 “每题预测标签数”分布
"""

import os, sys, json, numpy as np
from collections import Counter

PRED = "outputs_deberta_v3_topk3_orig_e3_b2/dev_predictions.json"
GOLD = "data/dev/questions.jsonl"

print("[INFO] Script started.")
print(f"[INFO] PRED path = {PRED}")
print(f"[INFO] GOLD path = {GOLD}")

if not os.path.exists(PRED):
    print(f"[ERROR] Predictions file not found: {PRED}")
    sys.exit(1)
if not os.path.exists(GOLD):
    print(f"[ERROR] Gold file not found: {GOLD}")
    sys.exit(1)

# ==== 工具 ====
LABELS = ["A","B","C","D"]
idx = {c:i for i,c in enumerate(LABELS)}

def to_vec(ans):
    v = np.zeros(4, dtype=int)
    if ans:
        for t in str(ans).split(","):
            t=t.strip()
            if t in idx:
                v[idx[t]] = 1
    return v

# ==== 读文件 ====
try:
    with open(PRED, "r", encoding="utf-8") as f:
        pred_json = json.load(f)
    print(f"[INFO] Loaded predictions: {len(pred_json)} rows.")
except Exception as e:
    print("[ERROR] Failed to read predictions JSON:", repr(e))
    sys.exit(1)

gold = {}
try:
    with open(GOLD, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                qid = obj.get("question_id") or obj.get("uuid") or obj.get("id")
                gold[qid] = obj.get("golden_answer","")
    print(f"[INFO] Loaded gold: {len(gold)} rows.")
except Exception as e:
    print("[ERROR] Failed to read gold JSONL:", repr(e))
    sys.exit(1)

# ==== 对齐 ====
Yt, Yp, sizes = [], [], []
aligned = 0
for r in pred_json:
    qid = r.get("question_id")
    if qid in gold:
        Yt.append(to_vec(gold[qid]))
        Yp.append(to_vec(r.get("prediction","")))
        sizes.append(to_vec(r.get("prediction","")).sum())
        aligned += 1

print(f"[INFO] Aligned pairs = {aligned}")
if aligned == 0:
    print("[ERROR] No aligned question_id between predictions and gold. Check that PRED/GOLD correspond to the same split.")
    sys.exit(1)

Yt = np.vstack(Yt)
Yp = np.vstack(Yp)

# ==== 指标 ====
def prf(y_true, y_pred):
    tp = ((y_true==1)&(y_pred==1)).sum(axis=0)
    fp = ((y_true==0)&(y_pred==1)).sum(axis=0)
    fn = ((y_true==1)&(y_pred==0)).sum(axis=0)

    # per-label
    P = np.divide(tp, tp+fp, out=np.zeros_like(tp,dtype=float), where=(tp+fp)!=0)
    R = np.divide(tp, tp+fn, out=np.zeros_like(tp,dtype=float), where=(tp+fn)!=0)
    F = np.divide(2*P*R, P+R, out=np.zeros_like(P,dtype=float), where=(P+R)!=0)

    pm, rm, fm = P.mean(), R.mean(), F.mean()
    TP, FP, FN = tp.sum(), fp.sum(), fn.sum()
    p_micro = TP/(TP+FP) if (TP+FP)>0 else 0.0
    r_micro = TP/(TP+FN) if (TP+FN)>0 else 0.0
    f_micro = 2*p_micro*r_micro/(p_micro+r_micro) if (p_micro+r_micro)>0 else 0.0
    return (p_micro, r_micro, f_micro), (pm, rm, fm)

mi, ma = prf(Yt, Yp)
dist = Counter(sizes)
n = len(sizes)

print(f"[POST-CALIB] micro-F1={mi[2]:.4f}  macro-F1={ma[2]:.4f}")
print("[POST-CALIB] Pred set size distribution:",
      {k: f"{v} ({v/n:.1%})" for k,v in sorted(dist.items())})
print("[DONE]")
