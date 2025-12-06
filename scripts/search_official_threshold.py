# scripts/search_official_threshold.py
# -*- coding: utf-8 -*-
"""
在 dev 上用官方 1/0.5/0 评分规则，搜索一个“全局概率阈值”，并强制 cap_k=3。
- 输入：
    PRED = dev_predictions.json（里面有 prob_A..D）
    GOLD = data/dev/questions.jsonl
- 输出：
    1）打印最佳阈值及官方平均分
    2）生成一个新的预测文件（只改 prediction 字段），用于 evaluate_official_score.py
"""

import json
import os
import argparse

LABELS = ["A", "B", "C", "D"]

def load_gold(gold_path):
    gold = {}
    with open(gold_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            qid = o.get("question_id") or o.get("uuid") or o.get("id")
            ans = o.get("golden_answer") or ""
            gset = set(s.strip() for s in str(ans).split(",") if s.strip())
            gold[qid] = gset
    return gold

def load_probs(pred_path):
    rows = json.load(open(pred_path, "r", encoding="utf-8"))
    probs = {}
    for r in rows:
        qid = r["question_id"]
        p = [
            float(r["prob_A"]),
            float(r["prob_B"]),
            float(r["prob_C"]),
            float(r["prob_D"]),
        ]
        probs[qid] = p
    return rows, probs

def official_score_at_threshold(probs_map, gold_map, thr, cap_k):
    scores = []
    for qid, gset in gold_map.items():
        if qid not in probs_map:
            continue
        p = probs_map[qid]

        # 根据 thr + cap_k=3 生成预测集合
        candidates = [(j, p[j]) for j in range(4) if p[j] >= thr]
        if not candidates:
            # 一个都没过阈值，就取 argmax 兜底
            j_best = max(range(4), key=lambda j: p[j])
            pred_set = {LABELS[j_best]}
        else:
            # 概率从大到小排序，只取前 cap_k 个
            candidates.sort(key=lambda x: x[1], reverse=True)
            candidates = candidates[:cap_k]
            pred_set = {LABELS[j] for j, _ in candidates}

        # 官方 1/0.5/0 计分
        if pred_set == gset:
            s = 1.0
        elif pred_set & gset:
            s = 0.5
        else:
            s = 0.0
        scores.append(s)

    if not scores:
        return 0.0
    return sum(scores) / len(scores)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str,
                        default="outputs_deberta_top3_topicid_noweight/dev_predictions.json")
    parser.add_argument("--gold_path", type=str,
                        default="data/dev/questions.jsonl")
    parser.add_argument("--out_path", type=str,
                        default="outputs_deberta_top3_topicid_noweight/dev_predictions_official_cap3.json")
    parser.add_argument("--cap_k", type=int, default=3)
    parser.add_argument("--thr_min", type=float, default=0.20)
    parser.add_argument("--thr_max", type=float, default=0.80)
    parser.add_argument("--thr_step", type=float, default=0.01)
    args = parser.parse_args()

    print("[INFO] PRED =", args.pred_path)
    print("[INFO] GOLD =", args.gold_path)

    if not os.path.exists(args.pred_path):
        raise FileNotFoundError(args.pred_path)
    if not os.path.exists(args.gold_path):
        raise FileNotFoundError(args.gold_path)

    gold_map = load_gold(args.gold_path)
    rows, probs_map = load_probs(args.pred_path)

    # 1. 网格搜索最佳阈值
    best_thr, best_score = 0.5, -1.0
    thr = args.thr_min
    while thr <= args.thr_max + 1e-8:
        score = official_score_at_threshold(probs_map, gold_map, thr, args.cap_k)
        if score > best_score:
            best_score = score
            best_thr = thr
        thr += args.thr_step

    print(f"[SEARCH] Best thr = {best_thr:.2f}, Official avg score = {best_score:.4f} (cap_k={args.cap_k})")

    # 2. 用 best_thr + cap_k 重新写 prediction 字段
    new_rows = []
    for r in rows:
        qid = r["question_id"]
        p = probs_map[qid]

        candidates = [(j, p[j]) for j in range(4) if p[j] >= best_thr]
        if not candidates:
            j_best = max(range(4), key=lambda j: p[j])
            pred_set = {LABELS[j_best]}
        else:
            candidates.sort(key=lambda x: x[1], reverse=True)
            candidates = candidates[:args.cap_k]
            pred_set = {LABELS[j] for j, _ in candidates}

        r["prediction"] = ",".join(sorted(pred_set))
        new_rows.append(r)

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(new_rows, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved new predictions to {args.out_path}")

if __name__ == "__main__":
    main()
