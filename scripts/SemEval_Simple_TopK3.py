#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SemEval AER 多标签 —— 最简稳健版（TopK=3，AMP-safe）
- 训练：BCEWithLogits，多标签（A/B/C/D），可选 pos_weight
- 文本打窗：把 docs.json 的所有文档拼接后按 window/stride 切片
- 聚合：按 option 对 chunk 的 logit 做 Top-K(=3) 均值（更稳于 max）
- 阈值：在 dev 上同时搜索「全局阈值」与「逐类阈值（per-label）」
- 导出：dev_predictions.json（含 prob_A..D 与 prediction），val_calibration.json（保存阈值）
- 预测：--predict_only 可对 test 集用保存的 per-label 阈值直接导出

推荐命令（复现你最稳设置）
python scripts/SemEval_Simple_TopK3.py \
  --train_jsonl data/train/questions.jsonl \
  --train_docs_json data/train/docs.json \
  --dev_jsonl   data/dev/questions.jsonl \
  --dev_docs_json   data/dev/docs.json \
  --save_dir outputs_bert_topk3 \
  --backbone bert-base-uncased \
  --epochs 1 --batch_size 1 \
  --max_len 384 --window 224 --stride 144 --max_windows 6 \
  --agg topk --topk_k 3 \
  --pos_weight \
  --amp

预测 test（示例）：
python scripts/SemEval_Simple_TopK3.py \
  --predict_only \
  --test_jsonl data/test/questions.jsonl \
  --test_docs_json data/test/docs.json \
  --save_dir outputs_bert_topk3
"""

from __future__ import annotations
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json, math, argparse, random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score

LABELS = ["A","B","C","D"]
LABEL2IDX = {c:i for i,c in enumerate(LABELS)}
IDX2LABEL = {i:c for i,c in enumerate(LABELS)}

# ------------------------- Utils -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_docs_json(path: str) -> Dict[str, str]:
    with open(path, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        return {str(k): str(v) for k, v in obj.items()}
    if isinstance(obj, list):
        out = {}
        for i, it in enumerate(obj):
            if isinstance(it, dict):
                did = str(it.get("id", i))
                txt = str(it.get("text", it.get("content", "")))
            else:
                did, txt = str(i), str(it)
            out[did] = txt
        return out
    raise ValueError("Unsupported docs.json format")

# ------------------------- Data -------------------------
class QADataset(Dataset):
    def __init__(self, q_jsonl: str, docs_json: str, tokenizer, max_len=384, window=224, stride=144, max_windows=6):
        self.tok = tokenizer
        self.max_len = max_len
        self.window = window
        self.stride = stride
        self.max_windows = max_windows
        self.docs_map = load_docs_json(docs_json)

        self.items = []
        with open(q_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                o = json.loads(line)
                qid = o.get("question_id") or o.get("uuid") or o.get("id")
                event = str(o.get("event", ""))
                opts = [str(o.get("option_A", "")), str(o.get("option_B", "")), str(o.get("option_C", "")), str(o.get("option_D", ""))]
                gold = str(o.get("golden_answer", ""))
                # 简化：用所有 docs（如果你有 per-question 相关 doc_ids，可在此替换）
                all_text = "\n\n".join(self.docs_map.values())
                chunks = self._make_chunks(all_text)
                self.items.append({"qid": qid, "event": event, "options": opts, "chunks": chunks, "gold": gold})

    def _make_chunks(self, text: str) -> List[str]:
        toks = self.tok.tokenize(text)
        chunks, i = [], 0
        while i < len(toks) and len(chunks) < self.max_windows:
            sub = toks[i:i+self.window]
            chunks.append(self.tok.convert_tokens_to_string(sub))
            i += self.stride
        if not chunks:
            chunks = [self.tok.convert_tokens_to_string(toks[:self.window])]
        return chunks

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

class Collator:
    def __init__(self, tokenizer, max_len=384):
        self.tok = tokenizer
        self.max_len = max_len
    def __call__(self, batch):
        input_ids, attn_mask, type_ids = [], [], []
        owner_ex_idx, owner_opt_idx = [], []
        labels = []
        qids = []
        for bi, ex in enumerate(batch):
            qids.append(ex["qid"])
            # gold 多标签向量
            y = np.zeros(4, dtype=np.int64)
            if ex.get("gold"):
                for t in str(ex["gold"]).split(","):
                    t=t.strip()
                    if t in LABEL2IDX:
                        y[LABEL2IDX[t]] = 1
            labels.append(y)
            for oi, opt in enumerate(ex["options"]):
                for chunk in ex["chunks"]:
                    pair_a, pair_b = ex["event"] + "\n\n" + opt, chunk
                    enc = self.tok(
                        pair_a, pair_b,
                        truncation=True, max_length=self.max_len, padding=False,
                        return_tensors=None
                    )
                    input_ids.append(enc["input_ids"])
                    attn_mask.append(enc["attention_mask"])
                    type_ids.append(enc.get("token_type_ids", [0]*len(enc["input_ids"])) )
                    owner_ex_idx.append(bi)
                    owner_opt_idx.append(oi)
        enc = self.tok.pad({
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "token_type_ids": type_ids
        }, padding=True, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "token_type_ids": enc.get("token_type_ids"),
            "owner_ex_idx": torch.tensor(owner_ex_idx, dtype=torch.long),
            "owner_opt_idx": torch.tensor(owner_opt_idx, dtype=torch.long),
            "labels": torch.tensor(np.stack(labels, axis=0), dtype=torch.float),
            "batch_size": len(batch),
            "qids": qids
        }

# ------------------------- Model -------------------------
class MultiLabelHead(nn.Module):
    def __init__(self, hidden, num_labels=4, p=0.2):
        super().__init__()
        self.drop = nn.Dropout(p)
        self.fc = nn.Linear(hidden, num_labels)
    def forward(self, cls):
        return self.fc(self.drop(cls))

class BaselineModel(nn.Module):
    def __init__(self, backbone: str):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone)
        hidden = self.backbone.config.hidden_size
        self.head = MultiLabelHead(hidden)
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls = out.last_hidden_state[:,0,:] if hasattr(out, 'last_hidden_state') else out[0][:,0,:]
        return self.head(cls)  # [N_flat, 4]

# ------------------------- Aggregation -------------------------
def aggregate_logits_topk(logits_flat: torch.Tensor, ex_idx: torch.Tensor, opt_idx: torch.Tensor, B: int, k: int = 3) -> torch.Tensor:
    device = logits_flat.device
    out = torch.zeros((B, 4), dtype=logits_flat.dtype, device=device)
    for b in range(B):
        mask_b = (ex_idx == b)
        for j in range(4):
            vals = logits_flat[mask_b & (opt_idx == j), j]
            if vals.numel() == 0:
                out[b, j] = 0.0
            else:
                kk = min(k, vals.numel())
                topv, _ = torch.topk(vals, kk)
                out[b, j] = topv.mean()
    return out

def aggregate_logits_max(logits_flat: torch.Tensor, ex_idx: torch.Tensor, opt_idx: torch.Tensor, B: int) -> torch.Tensor:
    device = logits_flat.device
    out = torch.zeros((B, 4), dtype=logits_flat.dtype, device=device)
    for b in range(B):
        mask_b = (ex_idx == b)
        for j in range(4):
            vals = logits_flat[mask_b & (opt_idx == j), j]
            out[b, j] = vals.max() if vals.numel() else 0.0
    return out

# ------------------------- Training / Eval -------------------------
@torch.no_grad()
def eval_pass(model, loader, device, agg: str, topk_k: int) -> Tuple[np.ndarray, np.ndarray, float, Dict[str,float]]:
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        labels = batch["labels"].to(device)
        owner_ex_idx = batch["owner_ex_idx"].to(device)
        owner_opt_idx = batch["owner_opt_idx"].to(device)
        B = batch["batch_size"]

        logits_flat = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = aggregate_logits_topk(logits_flat, owner_ex_idx, owner_opt_idx, B, k=topk_k) if agg=="topk" else \
                 aggregate_logits_max(logits_flat, owner_ex_idx, owner_opt_idx, B)
        loss = criterion(logits, labels)
        total_loss += float(loss.cpu())
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    all_logits = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0,4), np.float32)
    all_labels = np.concatenate(all_labels, axis=0) if all_labels else np.zeros((0,4), np.float32)

    # 搜全局阈值
    best_thr, best_macro = 0.5, -1
    def metrics_at(t):
        probs = 1/(1+np.exp(-all_logits))
        pred = (probs >= t).astype(int)
        subset = (pred == all_labels).all(axis=1).mean() if all_labels.size else 0.0
        micro = f1_score(all_labels, pred, average='micro', zero_division=0) if all_labels.size else 0.0
        macro = f1_score(all_labels, pred, average='macro', zero_division=0) if all_labels.size else 0.0
        return subset, micro, macro
    for t in [i/100 for i in range(25, 61)]:
        _, _, macro = metrics_at(t)
        if macro > best_macro:
            best_macro, best_thr = macro, t
    subset, micro, macro = metrics_at(best_thr)
    return all_logits, all_labels, best_thr, {"subset_acc":subset, "micro_f1":micro, "macro_f1":macro}

def run_train(model, train_loader, dev_loader, device, epochs, lr, warmup_ratio, agg, topk_k, amp, pos_weight_vec=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    num_steps = max(1, len(train_loader) * epochs)
    num_warm = int(num_steps * warmup_ratio)
    def lr_lambda(step):
        if step < num_warm:
            return float(step) / float(max(1, num_warm))
        p = (step - num_warm) / float(max(1, num_steps - num_warm))
        return 0.5 * (1.0 + math.cos(math.pi * p))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_vec)
    scaler = torch.amp.GradScaler(device='cuda') if (amp and torch.cuda.is_available()) else None

    best_state, best_dev_macro, best_thr = None, -1.0, 0.5
    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch["labels"].to(device)
            owner_ex_idx = batch["owner_ex_idx"].to(device)
            owner_opt_idx = batch["owner_opt_idx"].to(device)
            B = batch["batch_size"]

            logits_flat = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            logits = aggregate_logits_topk(logits_flat, owner_ex_idx, owner_opt_idx, B, k=topk_k) if agg=="topk" else \
                     aggregate_logits_max(logits_flat, owner_ex_idx, owner_opt_idx, B)
            with torch.amp.autocast("cuda", enabled=(amp and torch.cuda.is_available())):
                loss = criterion(logits, labels)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach().cpu())
            del logits_flat, logits

        # dev 评估
        dev_logits, dev_labels, best_thr_ep, dev_metrics = eval_pass(model, dev_loader, device, agg, topk_k)
        print(f"Epoch {ep}/{epochs} | train loss {total_loss/max(1,len(train_loader)):.4f} | dev loss ~ | best_thr {best_thr_ep:.2f} | subset {dev_metrics['subset_acc']:.4f} | micro {dev_metrics['micro_f1']:.4f} | macro {dev_metrics['macro_f1']:.4f}")
        if dev_metrics['macro_f1'] > best_dev_macro:
            best_dev_macro = dev_metrics['macro_f1']
            best_thr = best_thr_ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_thr

def search_per_label_thresholds(all_logits: np.ndarray, all_labels: np.ndarray, base_thr: float) -> List[float]:
    probs = 1/(1+np.exp(-all_logits))
    per_label = [base_thr]*4
    for j in range(4):
        best_f, best_t = -1.0, base_thr
        yt = all_labels[:, j]
        pj = probs[:, j]
        for t in [i/100 for i in range(20, 71)]:
            yp = (pj >= t).astype(int)
            f = f1_score(yt, yp, average='binary', zero_division=0)
            if f > best_f:
                best_f, best_t = f, t
        per_label[j] = float(best_t)
    return per_label

@torch.no_grad()
def export_predictions(model, loader, device, agg, topk_k, per_label_thr: List[float], out_path: str):
    model.eval()
    rows = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        owner_ex_idx = batch["owner_ex_idx"].to(device)
        owner_opt_idx = batch["owner_opt_idx"].to(device)
        qids = batch["qids"]
        B = batch["batch_size"]

        logits_flat = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = aggregate_logits_topk(logits_flat, owner_ex_idx, owner_opt_idx, B, k=topk_k) if agg=="topk" else \
                 aggregate_logits_max(logits_flat, owner_ex_idx, owner_opt_idx, B)
        probs = torch.sigmoid(logits).cpu().numpy()

        for i in range(B):
            thr = np.array(per_label_thr, dtype=float)
            labs_idx = [j for j in range(4) if probs[i, j] >= thr[j]]
            if not labs_idx:
                labs_idx = [int(np.argmax(probs[i]))]
            rows.append({
                "question_id": qids[i],
                "prob_A": float(probs[i,0]), "prob_B": float(probs[i,1]), "prob_C": float(probs[i,2]), "prob_D": float(probs[i,3]),
                "prediction": ",".join(IDX2LABEL[j] for j in labs_idx)
            })
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out_path}")

# ------------------------- Main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_jsonl')
    ap.add_argument('--train_docs_json')
    ap.add_argument('--dev_jsonl')
    ap.add_argument('--dev_docs_json')
    ap.add_argument('--test_jsonl')
    ap.add_argument('--test_docs_json')
    ap.add_argument('--predict_only', action='store_true', help='仅预测（使用已保存的 per-label 阈值）')

    ap.add_argument('--save_dir', default='outputs')
    ap.add_argument('--backbone', default='bert-base-uncased')
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--warmup_ratio', type=float, default=0.06)

    ap.add_argument('--max_len', type=int, default=384)
    ap.add_argument('--window', type=int, default=224)
    ap.add_argument('--stride', type=int, default=144)
    ap.add_argument('--max_windows', type=int, default=6)

    ap.add_argument('--agg', choices=['topk','max'], default='topk')
    ap.add_argument('--topk_k', type=int, default=3)
    ap.add_argument('--pos_weight', action='store_true')
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    print(f"Loading tokenizer & model: {args.backbone}")
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, use_fast=True)
    model = BaselineModel(args.backbone)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # --- 仅预测 ---
    if args.predict_only:
        assert args.test_jsonl and args.test_docs_json, "predict_only 需要 --test_jsonl 和 --test_docs_json"
        # 读取已保存阈值
        cal_path = os.path.join(args.save_dir, 'val_calibration.json')
        if os.path.exists(cal_path):
            cal = json.load(open(cal_path,'r',encoding='utf-8'))
            per_label_thr = cal.get('per_label_thr') or [cal.get('best_thr',0.5)]*4
        else:
            per_label_thr = [0.5,0.5,0.5,0.5]
        # 构造数据与 DataLoader
        test_ds = QADataset(args.test_jsonl, args.test_docs_json, tokenizer, args.max_len, args.window, args.stride, args.max_windows)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=Collator(tokenizer, args.max_len))
        export_predictions(model, test_loader, device, args.agg, args.topk_k, per_label_thr, os.path.join(args.save_dir, 'test_predictions.json'))
        return

    # --- 训练 + Dev 校准 ---
    assert args.train_jsonl and args.train_docs_json and args.dev_jsonl and args.dev_docs_json, "训练模式需要 train/dev 路径"

    train_ds = QADataset(args.train_jsonl, args.train_docs_json, tokenizer, args.max_len, args.window, args.stride, args.max_windows)
    dev_ds   = QADataset(args.dev_jsonl,   args.dev_docs_json,   tokenizer, args.max_len, args.window, args.stride, args.max_windows)
    collate = Collator(tokenizer, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, collate_fn=collate)
    dev_loader   = DataLoader(dev_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    # pos_weight = neg/pos（简化统计）
    pos_w = None
    if args.pos_weight:
        total = np.zeros(4, dtype=np.int64)
        pos = np.zeros(4, dtype=np.int64)
        for it in train_ds.items:
            y = np.zeros(4, dtype=np.int64)
            if it.get('gold'):
                for t in str(it['gold']).split(','):
                    t=t.strip()
                    if t in LABEL2IDX:
                        y[LABEL2IDX[t]] = 1
            total += 1
            pos += y
        neg = np.maximum(total - pos, 1)
        pos_w = torch.tensor(neg/np.maximum(pos,1), dtype=torch.float, device=device)

    best_thr = run_train(model, train_loader, dev_loader, device, args.epochs, args.lr, args.warmup_ratio, args.agg, args.topk_k, args.amp, pos_weight_vec=pos_w)

    # 用最佳模型在 dev 上再跑一次得到 logits/labels，计算 per-label 阈值
    dev_logits, dev_labels, _, dev_metrics = eval_pass(model, dev_loader, device, args.agg, args.topk_k)
    per_label_thr = search_per_label_thresholds(dev_logits, dev_labels, base_thr=best_thr)

    # 写 calibration
    with open(os.path.join(args.save_dir, 'val_calibration.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'best_thr': float(best_thr),
            'per_label_thr': [float(x) for x in per_label_thr],
            'agg': args.agg,
            'topk_k': int(args.topk_k)
        }, f, ensure_ascii=False, indent=2)

    # 导出 dev 预测
    export_predictions(model, dev_loader, device, args.agg, args.topk_k, per_label_thr, os.path.join(args.save_dir, 'dev_predictions.json'))

if __name__ == '__main__':
    main()
