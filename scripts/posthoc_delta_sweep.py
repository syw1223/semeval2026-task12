# scripts/posthoc_delta_sweep.py (official-partial + robust alignment)
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, os
from typing import List, Dict, Any, Tuple


LABELS = ["A","B","C","D"]


# ----------------------------- Utils -----------------------------


def infer_id_key(example: Dict[str, Any]) -> str:
for k in ("id", "qid", "question_id", "questionId", "questionID"):
if k in example:
return k
# fallback: try to find a key that looks like id
for k in example.keys():
if "id" in k.lower():
return k
raise KeyError("No id-like key found in predictions. Expected one of: id/qid/question_id/questionId/questionID")


# ----------------------------- IO -----------------------------


def load_preds(path: str) -> Tuple[List[Dict[str, Any]], str]:
with open(path, "r", encoding="utf-8") as f:
data = json.load(f)
if not isinstance(data, list) or len(data) == 0:
raise ValueError(f"pred file {path} is empty or not a list")
id_key = infer_id_key(data[0])
return data, id_key




def load_gold(path: str) -> Dict[str, List[str]]:
gold: Dict[str, List[str]] = {}
with open(path, "r", encoding="utf-8") as f:
for line in f:
if not line.strip():
continue
obj = json.loads(line)
qid = str(obj.get("id") or obj.get("question_id") or obj.get("qid") or obj.get("questionId") or obj.get("questionID"))
ans = obj.get("answer")
if isinstance(ans, list):
gold[qid] = [str(x) for x in ans]
else:
gold[qid] = [str(ans)]
if not gold:
raise ValueError(f"gold file {path} yielded 0 items; check format.")
return gold




def load_calib_thr(path: str) -> List[float]:
with open(path, "r", encoding="utf-8") as f:
calib = json.load(f)
thr = calib["per_label_thr"] if isinstance(calib, dict) and "per_label_thr" in calib else calib
assert len(thr) == 4, "per_label_thr 应为 4 维 [tA,tB,tC,tD]"
return [float(x) for x in thr]


# -------------------------- Metrics ---------------------------


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
return 2 * p * r / (p + r) if (p + r) > 0 else 0.0




def micro_macro_f1(all_pred_sets: List[List[str]], all_gold_sets: List[List[str]]) -> Tuple[float, float]:
idx = {l: i for i, l in enumerate(LABELS)}
L = len(LABELS)
tp = [0] * L
fp = [0] * L
fn = [0] * L
for p, g in zip(all_pred_sets, all_gold_sets):