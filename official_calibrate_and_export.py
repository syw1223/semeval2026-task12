#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Official post-hoc calibration + export for SemEval-2026 Task 12 (AER)
--------------------------------------------------------------------
用途：
1) 读取 dev 预测概率（dev_predictions.json）与 gold（questions.jsonl）
2) 以“官方打分”（完全=1、部分=0.5、错误=0）为目标，网格搜索**全局阈值**
3) 应用集合大小约束（cap_k / min_k）与可选的“top3-4 概率差”门控（delta_top3）
4) 导出官方格式 answers_dev_official.jsonl（uuid + "A,B"）
5) （可选）把同样的校准应用到 test_predictions.json，导出 answers_test_official.jsonl

运行示例：
python official_calibrate_and_export.py \
  --pred_dev outputs_bert_topk3/dev_predictions.json \
  --gold_dev data/dev/questions.jsonl \
  --out_dir outputs_bert_topk3 \
  --thr_min 0.20 --thr_max 0.60 --thr_step 0.01 \
  --cap_k 3 --min_k 1 --delta_top3 0.05 \
  --apply_to_test --pred_test outputs_bert_topk3/test_predictions.json

备注：
- 本脚本做“全局阈值”搜索（简单稳妥）。若需要“按标签阈值”，可后续扩展。
- 预测文件需包含 prob_A..prob_D（浮点），以及一个 id 字段（优先 uuid/ question_id）。
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

# -----------------------------
# 读取 / 解析工具
# -----------------------------

def _read_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def _read_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

LABELS = ["A", "B", "C", "D"]


def _extract_id(obj: dict) -> str:
    """从预测 / gold 记录中提取 ID（尽量鲁棒）。优先顺序：uuid > question_id > id > topic_id
    如果都没有，就用内置索引逻辑（调用方保证对齐）。
    """
    for k in ["uuid", "question_id", "id", "qid"]:
        if k in obj:
            return str(obj[k])
    if "topic_id" in obj:
        return f"topic_{obj['topic_id']}"
    # 兜底：调用方应保证用 zip 对齐
    return None


def _extract_probs(obj: dict) -> List[float]:
    return [
        float(obj.get("prob_A", 0.0)),
        float(obj.get("prob_B", 0.0)),
        float(obj.get("prob_C", 0.0)),
        float(obj.get("prob_D", 0.0)),
    ]


def _parse_answer_str(ans: str) -> List[str]:
    """把 "A,B" 解析为 ["A","B"]；容错空白、大小写。"""
    if not ans:
        return []
    parts = [p.strip().upper() for p in ans.split(',') if p.strip()]
    # 只保留合法标签
    return [p for p in parts if p in LABELS]


# -----------------------------
# 官方打分与集合构造
# -----------------------------

def official_score(pred: List[str], gold: List[str]) -> float:
    P, G = set(pred), set(gold)
    if P == G:
        return 1.0
    if len(P & G) > 0:
        return 0.5
    return 0.0


def build_set_from_probs(
    probs: List[float],
    thr: float,
    cap_k: int,
    min_k: int,
    delta_top3: float,
    enable_delta3: bool = True,
) -> List[str]:
    """由概率向量构造预测集合（A-D）。
    流程：阈值筛选 -> 集合大小门控（含“top3-4差值”规则）-> 取前 K。
    """
    assert len(probs) == 4
    # 初始候选：>= thr 的标签
    cand = [i for i, p in enumerate(probs) if p >= thr]
    # 如果空集，先用 Top-1 保底
    if not cand:
        top1 = int(max(range(4), key=lambda i: probs[i]))
        cand = [top1]

    # 排序（按概率降序）
    order = sorted(range(4), key=lambda i: probs[i], reverse=True)
    # 计算动态上限（delta3 规则）
    kmax = cap_k
    if enable_delta3 and cap_k >= 3:
        sorted_probs = [probs[i] for i in order]
        p3 = sorted_probs[2]
        p4 = sorted_probs[3]
        if (p3 - p4) < delta_top3:
            # 不够明显的 top3，就最多允许 2 个
            kmax = min(kmax, 2)

    # 目标 K
    K = max(min_k, min(len(cand), kmax))
    # 最终取 Top-K（全局的前 K，而非仅 cand 的前 K，避免丢高概率标签）
    keep = order[:K]
    keep_labels = [LABELS[i] for i in keep]
    return keep_labels


def evaluate_official(
    dev_pairs: List[Tuple[str, List[float], List[str]]],
    thr: float,
    cap_k: int,
    min_k: int,
    delta_top3: float,
    enable_delta3: bool,
) -> float:
    """返回平均官方得分。dev_pairs: (id, probs[4], gold_labels[])"""
    total = 0.0
    for _id, probs, gold in dev_pairs:
        pred = build_set_from_probs(probs, thr, cap_k, min_k, delta_top3, enable_delta3)
        total += official_score(pred, gold)
    return total / max(1, len(dev_pairs))


# -----------------------------
# 主流程
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dev", required=True, help="path to dev_predictions.json")
    ap.add_argument("--gold_dev", required=True, help="path to dev questions.jsonl (with gold 'answer')")
    ap.add_argument("--out_dir", required=True, help="where to write outputs")

    ap.add_argument("--thr_min", type=float, default=0.20)
    ap.add_argument("--thr_max", type=float, default=0.60)
    ap.add_argument("--thr_step", type=float, default=0.01)

    ap.add_argument("--cap_k", type=int, default=3)
    ap.add_argument("--min_k", type=int, default=1)

    ap.add_argument("--delta_top3", type=float, default=0.05, help="top3 - top4 margin; if < delta, cap at 2")
    ap.add_argument("--no_delta3", action="store_true", help="disable the top3-vs-top4 gating")

    ap.add_argument("--apply_to_test", action="store_true")
    ap.add_argument("--pred_test", type=str, default=None, help="optional: path to test_predictions.json (prob_A..D)")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("[INFO] Load dev predictions:", args.pred_dev)
    dev_pred = _read_json(args.pred_dev)

    print("[INFO] Load dev gold:", args.gold_dev)
    dev_gold_rows = _read_jsonl(args.gold_dev)

    # 构建 id -> gold
    gold_map: Dict[str, List[str]] = {}
    for i, row in enumerate(dev_gold_rows):
        _id = _extract_id(row)
        if _id is None:
            _id = f"row_{i}"
        gold_map[_id] = _parse_answer_str(row.get("answer", ""))

    # 对齐到 pairs
    dev_pairs: List[Tuple[str, List[float], List[str]]] = []
    miss_gold, miss_pred = 0, 0

    # 尝试两种对齐：优先用 id，其次 zip 顺序
    use_zip = False
    ids_pred = [
        _extract_id(x) for x in dev_pred
    ]
    if any(i is None for i in ids_pred) or not all(ip in gold_map for ip in ids_pred if ip is not None):
        # 有缺失，启用 zip 对齐
        use_zip = True
        print("[WARN] Some pred ids missing in gold; falling back to order-based zip alignment.")

    if use_zip:
        N = min(len(dev_pred), len(dev_gold_rows))
        for i in range(N):
            probs = _extract_probs(dev_pred[i])
            gid = _extract_id(dev_gold_rows[i]) or f"row_{i}"
            gold = _parse_answer_str(dev_gold_rows[i].get("answer", ""))
            dev_pairs.append((gid, probs, gold))
    else:
        for obj in dev_pred:
            _id = _extract_id(obj)
            if _id is None:
                miss_pred += 1
                continue
            if _id not in gold_map:
                miss_gold += 1
                continue
            probs = _extract_probs(obj)
            dev_pairs.append((_id, probs, gold_map[_id]))

    print(f"[INFO] Aligned pairs = {len(dev_pairs)}  (miss_gold={miss_gold}, miss_pred={miss_pred})")

    # 网格搜索全局阈值（以官方平均得分为目标）
    best_thr = None
    best_score = -1.0
    thr = args.thr_min
    enable_delta3 = (not args.no_delta3)

    while thr <= args.thr_max + 1e-9:
        score = evaluate_official(
            dev_pairs,
            thr=thr,
            cap_k=args.cap_k,
            min_k=args.min_k,
            delta_top3=args.delta_top3,
            enable_delta3=enable_delta3,
        )
        if score > best_score:
            best_score = score
            best_thr = thr
        thr += args.thr_step

    assert best_thr is not None
    print(f"[CALIB] best_thr={best_thr:.2f} | official_avg={best_score:.4f} | cap_k={args.cap_k} | min_k={args.min_k} | delta_top3={args.delta_top3} | delta3={'on' if enable_delta3 else 'off'}")

    # 保存校准参数
    calib = {
        "objective": "official_avg",  # 完全=1、部分=0.5、错=0
        "best_thr": round(best_thr, 4),
        "cap_k": int(args.cap_k),
        "min_k": int(args.min_k),
        "delta_top3": float(args.delta_top3),
        "enable_delta3": bool(enable_delta3),
        "search": {
            "thr_min": args.thr_min,
            "thr_max": args.thr_max,
            "thr_step": args.thr_step,
        },
        "n_dev_pairs": len(dev_pairs),
        "best_official_avg": round(best_score, 6),
    }
    calib_path = os.path.join(args.out_dir, "val_calibration_official.json")
    with open(calib_path, 'w', encoding='utf-8') as f:
        json.dump(calib, f, ensure_ascii=False, indent=2)
    print("[DONE] wrote:", calib_path)

    # 导出 dev 官方答案格式
    ans_dev_path = os.path.join(args.out_dir, "answers_dev_official.jsonl")
    with open(ans_dev_path, 'w', encoding='utf-8') as fw:
        for _id, probs, gold in dev_pairs:
            pred = build_set_from_probs(
                probs, best_thr, args.cap_k, args.min_k, args.delta_top3, enable_delta3
            )
            fw.write(json.dumps({"uuid": _id, "answer": ",".join(pred)}) + "\n")
    print("[DONE] wrote:", ans_dev_path)

    # （可选）对 test 应用相同校准，导出官方答案
    if args.apply_to_test:
        if not args.pred_test or not os.path.exists(args.pred_test):
            print("[WARN] --apply_to_test 指定了，但未提供有效的 --pred_test。跳过 test 导出。")
        else:
            test_pred = _read_json(args.pred_test)
            # 对齐：test 无 gold，只按顺序写答案。优先使用 id（若存在）。
            ans_test_path = os.path.join(args.out_dir, "answers_test_official.jsonl")
            with open(ans_test_path, 'w', encoding='utf-8') as fw:
                for i, obj in enumerate(test_pred):
                    probs = _extract_probs(obj)
                    _id = _extract_id(obj) or f"row_{i}"
                    pred = build_set_from_probs(
                        probs, best_thr, args.cap_k, args.min_k, args.delta_top3, enable_delta3
                    )
                    fw.write(json.dumps({"uuid": _id, "answer": ",".join(pred)}) + "\n")
            print("[DONE] wrote:", ans_test_path)


if __name__ == "__main__":
    main()
