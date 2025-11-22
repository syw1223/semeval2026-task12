#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SemEval-2026 Task 12 — Simple TopK(窗口) 多标签基线（BCE/ASL 可切换，纯 Torch 聚合，fp16 安全）
- 与“稳定版”命令行参数保持一致；新增 --loss {bce, asl} 与 ASL 超参
- 关键修复：
  1) 聚合前将序列窗口 logits 转回 float32，避免 AMP(fp16) 极小值溢出；
  2) 聚合函数改为纯 PyTorch 实现（不经 numpy、不用 .item() 构造新张量），保持可微；
  3) 使用 torch.amp 的新 API。
- 输出：<save_dir>/dev_predictions.json, <save_dir>/val_calibration.json
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score

# -------------------------- ASL 损失 ---------------------------------
class AsymmetricLossMultiLabel(nn.Module):
    """Ben-Baruch et al., Asymmetric Loss for Multi-Label Classification."""
    def __init__(self, gamma_pos: float = 0.0, gamma_neg: float = 4.0, clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gp = float(gamma_pos)
        self.gn = float(gamma_neg)
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits/targets: [B, C]
        x_sigmoid = torch.sigmoid(logits)
        xs_pos = x_sigmoid
        xs_neg = 1. - x_sigmoid
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.)
        log_pos = torch.log(xs_pos.clamp(min=self.eps))
        log_neg = torch.log(xs_neg.clamp(min=self.eps))
        pos_weight = torch.pow(1. - xs_pos, self.gp) if self.gp > 0 else 1.
        neg_weight = torch.pow(1. - xs_neg, self.gn) if self.gn > 0 else 1.
        loss = - (targets * pos_weight * log_pos + (1. - targets) * neg_weight * log_neg)
        return loss.mean()

# -------------------------- 数据读取/拼接 ----------------------------
LABELS = ["A","B","C","D"]

def load_docs_json(path: str) -> Dict[str, str]:
    with open(path, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    # 支持 {id:text} 或 [{id, text}]
    if isinstance(obj, dict):
        return {str(k): str(v) for k, v in obj.items()}
    out = {}
    for it in obj:
        did = str(it.get('id') or it.get('doc_id') or it.get('uuid'))
        txt = it.get('text') or it.get('content') or ""
        out[did] = str(txt)
    return out

def load_questions_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            qid = str(o.get('question_id') or o.get('uuid') or o.get('id'))
            question = o.get('question') or o.get('query') or ""
            options = {
                'A': o.get('option_A') or o.get('A') or o.get('a') or "",
                'B': o.get('option_B') or o.get('B') or o.get('b') or "",
                'C': o.get('option_C') or o.get('C') or o.get('c') or "",
                'D': o.get('option_D') or o.get('D') or o.get('d') or "",
            }
            ans = o.get('answer') or o.get('golden_answer') or None
            if isinstance(ans, list):
                gold = set([str(x).strip().upper() for x in ans])
            elif isinstance(ans, str) and ans:
                gold = set([t.strip().upper() for t in ans.split(',') if t.strip()])
            else:
                gold = None
            doc_ids = o.get('doc_ids') or o.get('docs') or o.get('evidence_docs') or []
            doc_ids = [str(x) for x in doc_ids]
            rows.append({
                'question_id': qid,
                'question': question,
                'options': options,
                'gold': gold,
                'doc_ids': doc_ids,
            })
    return rows

# -------------------------- 滑窗制作 & 编码 --------------------------
@dataclass
class EncodedWindow:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor

class QADocWinDataset(Dataset):
    def __init__(self, data: List[Dict[str,Any]], docs_map: Dict[str,str], tokenizer, max_len=384, window=224, stride=144, max_windows=6, train=True):
        self.rows = data
        self.docs = docs_map
        self.tok = tokenizer
        self.max_len = max_len
        self.window = window
        self.stride = stride
        self.max_windows = max_windows
        self.train = train

    def _build_context(self, doc_ids: List[str]) -> str:
        texts = [self.docs.get(did, "") for did in doc_ids if did in self.docs]
        return "\n\n".join([t.strip() for t in texts if t])

    def _encode_pair(self, q_with_opt: str, ctx: str) -> List[EncodedWindow]:
        ctx_ids = self.tok(ctx, add_special_tokens=False)['input_ids']
        wins = []
        ctx_ids = ctx_ids or []
        W, S = self.window, self.stride
        starts = list(range(0, max(1, len(ctx_ids) - max(W-1,1) + 1), max(1,S)))
        if not starts:
            starts = [0]
        for st in starts[:self.max_windows]:
            piece = ctx_ids[st: st+W]
            text_chunk = self.tok.decode(piece)
            enc = self.tok(q_with_opt, text_chunk, truncation=True, max_length=self.max_len, return_tensors='pt')
            if 'token_type_ids' not in enc:
                enc['token_type_ids'] = torch.zeros_like(enc['input_ids'])
            wins.append(EncodedWindow(enc['input_ids'][0], enc['attention_mask'][0], enc['token_type_ids'][0]))
        if not wins:
            enc = self.tok(q_with_opt, "", truncation=True, max_length=self.max_len, return_tensors='pt')
            if 'token_type_ids' not in enc:
                enc['token_type_ids'] = torch.zeros_like(enc['input_ids'])
            wins.append(EncodedWindow(enc['input_ids'][0], enc['attention_mask'][0], enc['token_type_ids'][0]))
        return wins

    def __getitem__(self, idx):
        r = self.rows[idx]
        q = r['question']
        ctx = self._build_context(r['doc_ids'])
        qwins = {}
        for lab in LABELS:
            q_with_opt = f"{q}\n\n[Option {lab}] {r['options'][lab]}"
            qwins[lab] = self._encode_pair(q_with_opt, ctx)
        y = np.zeros(4, dtype=np.float32)
        if r['gold'] is not None:
            for i,lab in enumerate(LABELS):
                if lab in r['gold']:
                    y[i] = 1.
        return {
            'question_id': r['question_id'],
            'wins': qwins,
            'target': torch.tensor(y),
        }

    def __len__(self):
        return len(self.rows)

def collate_fn(batch):
    flat_input_ids, flat_attn, flat_type = [], [], []
    owner_ex, owner_opt = [], []
    targets = []
    qids = []
    for ex_idx, ex in enumerate(batch):
        qids.append(ex['question_id'])
        targets.append(ex['target'])
        for opt_idx, lab in enumerate(LABELS):
            wins = ex['wins'][lab]
            for w in wins:
                flat_input_ids.append(w.input_ids)
                flat_attn.append(w.attention_mask)
                flat_type.append(w.token_type_ids)
                owner_ex.append(ex_idx)
                owner_opt.append(opt_idx)
    pad_tok = batch[0]['wins']['A'][0].input_ids.new_zeros(1).item()
    flat_input_ids = nn.utils.rnn.pad_sequence(flat_input_ids, batch_first=True, padding_value=pad_tok)
    flat_attn = nn.utils.rnn.pad_sequence(flat_attn, batch_first=True, padding_value=0)
    flat_type = nn.utils.rnn.pad_sequence(flat_type, batch_first=True, padding_value=0)
    return {
        'input_ids': flat_input_ids,
        'attention_mask': flat_attn,
        'token_type_ids': flat_type,
        'owner_ex': torch.tensor(owner_ex, dtype=torch.long),
        'owner_opt': torch.tensor(owner_opt, dtype=torch.long),
        'targets': torch.stack(targets, dim=0),
        'qids': qids,
    }

# -------------------------- 模型 & 聚合 ------------------------------
class PooledCLS(nn.Module):
    def __init__(self, backbone_name: str):
        super().__init__()
        self.config = AutoConfig.from_pretrained(backbone_name)
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden = self.config.hidden_size
        self.head = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls = out.last_hidden_state[:, 0]
        logit = self.head(cls).squeeze(-1)
        return logit

# fp16 安全的“负无穷”填充值
def _safe_neg_inf_like_dtype(dtype: torch.dtype):
    return -1e4 if dtype == torch.float16 else -1e9

def aggregate_logits_max(logits_flat: torch.Tensor,
                         owner_ex: torch.Tensor,
                         owner_opt: torch.Tensor,
                         B: int) -> torch.Tensor:
    out = torch.full((B, 4), _safe_neg_inf_like_dtype(logits_flat.dtype),
                     dtype=logits_flat.dtype, device=logits_flat.device)
    # 纯 torch 实现：保持计算图可反传（此处 out 是聚合结果本身，不需要梯度）
    for b in range(B):
        for o in range(4):
            mask = (owner_ex == b) & (owner_opt == o)   # [N_flat]
            vals = logits_flat[mask]                    # [n_win]
            if vals.numel() > 0:
                out[b, o] = vals.max()
    return out

def aggregate_logits_topk(logits_flat: torch.Tensor,
                          owner_ex: torch.Tensor,
                          owner_opt: torch.Tensor,
                          B: int,
                          k: int = 3) -> torch.Tensor:
    out = torch.full((B, 4), _safe_neg_inf_like_dtype(logits_flat.dtype),
                     dtype=logits_flat.dtype, device=logits_flat.device)
    for b in range(B):
        for o in range(4):
            mask = (owner_ex == b) & (owner_opt == o)
            vals = logits_flat[mask]
            if vals.numel() > 0:
                kk = k if vals.numel() >= k else vals.numel()
                topk_vals, _ = torch.topk(vals, kk)
                out[b, o] = topk_vals.mean()
    return out

# -------------------------- 评估 & 阈值搜索 --------------------------

def eval_on_loader(model, loader, device, agg: str = 'topk', topk_k: int = 3) -> Tuple[np.ndarray, List[str]]:
    model.eval()
    all_logits, all_qids = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].to(device)
            attn = batch['attention_mask'].to(device)
            ttid = batch['token_type_ids'].to(device)
            owner_ex = batch['owner_ex'].to(device)
            owner_opt = batch['owner_opt'].to(device)
            B = batch['targets'].size(0)
            lf = model(ids, attn, ttid)
            lf = lf.float()  # 聚合前转 fp32，避免 AMP(fp16) 溢出
            if agg == 'max':
                logits = aggregate_logits_max(lf, owner_ex, owner_opt, B)
            else:
                logits = aggregate_logits_topk(lf, owner_ex, owner_opt, B, k=topk_k)
            all_logits.append(logits.cpu().numpy())
            all_qids.extend(batch['qids'])
    return np.concatenate(all_logits, axis=0), all_qids


def search_thresholds_dev(logits: np.ndarray, gold_y: np.ndarray) -> Dict[str, Any]:
    probs = 1 / (1 + np.exp(-logits))
    grid = np.linspace(0.20, 0.60, 41)
    best_thr, best_macro = 0.5, -1
    for t in grid:
        pred = (probs >= t).astype(int)
        for i in range(pred.shape[0]):
            if pred[i].sum() == 0:
                pred[i, np.argmax(probs[i])] = 1
        m = f1_score(gold_y, pred, average='macro', zero_division=0)
        if m > best_macro:
            best_macro, best_thr = m, float(t)
    per_label_thr = []
    for j in range(4):
        best_tj, best_mj = 0.5, -1
        for t in grid:
            pred = (probs[:, j:j+1] >= t).astype(int)
            m = f1_score(gold_y[:, j:j+1], pred, average='macro', zero_division=0)
            if m > best_mj:
                best_mj, best_tj = m, float(t)
        per_label_thr.append(best_tj)
    return {
        'best_thr': best_thr,
        'per_label_thr': per_label_thr,
        'search_grid': [float(x) for x in grid],
        'macro_f1_at_best': best_macro,
    }

# -------------------------- 训练循环 ---------------------------------

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_jsonl', required=True)
    ap.add_argument('--train_docs_json', default=None)
    ap.add_argument('--dev_jsonl', required=True)
    ap.add_argument('--dev_docs_json', default=None)
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
    # ASL
    ap.add_argument('--loss', choices=['bce','asl'], default='bce')
    ap.add_argument('--asl_gamma_pos', type=float, default=0.0)
    ap.add_argument('--asl_gamma_neg', type=float, default=4.0)
    ap.add_argument('--asl_clip', type=float, default=0.05)

    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Loading tokenizer & model: {args.backbone}")

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.backbone, use_fast=True)
    model = PooledCLS(args.backbone)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # data
    train_rows = load_questions_jsonl(args.train_jsonl)
    dev_rows = load_questions_jsonl(args.dev_jsonl)
    train_docs = load_docs_json(args.train_docs_json) if args.train_docs_json else {}
    dev_docs = load_docs_json(args.dev_docs_json) if args.dev_docs_json else train_docs

    train_ds = QADocWinDataset(train_rows, train_docs, tokenizer, max_len=args.max_len, window=args.window, stride=args.stride, max_windows=args.max_windows, train=True)
    dev_ds = QADocWinDataset(dev_rows, dev_docs, tokenizer, max_len=args.max_len, window=args.window, stride=args.stride, max_windows=args.max_windows, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # pos_weight from train
    pos_weight_tensor = None
    if args.pos_weight:
        cnt_pos = np.zeros(4, dtype=np.float64)
        cnt_all = 0
        for r in train_rows:
            if r['gold'] is None:
                continue
            y = np.zeros(4)
            for i,l in enumerate(LABELS):
                if l in r['gold']:
                    y[i] = 1
            cnt_pos += y
            cnt_all += 1
        pos = np.maximum(cnt_pos, 1.0)
        neg = np.maximum(cnt_all - cnt_pos, 1.0)
        w = torch.tensor(neg/pos, dtype=torch.float32, device=device)
        pos_weight_tensor = w

    # criterion
    if args.loss == 'asl':
        criterion = AsymmetricLossMultiLabel(gamma_pos=args.asl_gamma_pos, gamma_neg=args.asl_gamma_neg, clip=args.asl_clip)
        print(f"[LOSS] ASL (gp={args.asl_gamma_pos}, gn={args.asl_gamma_neg}, clip={args.asl_clip})")
    else:
        if pos_weight_tensor is not None:
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
            print("[LOSS] BCEWithLogitsLoss + pos_weight")
        else:
            criterion = nn.BCEWithLogitsLoss()
            print("[LOSS] BCEWithLogitsLoss")

    steps_per_epoch = max(1, len(train_loader))
    t_total = steps_per_epoch * args.epochs
    warmup = int(t_total * args.warmup_ratio)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup, num_training_steps=t_total)

    # AMP（新版API）
    scaler = torch.amp.GradScaler('cuda', enabled=(args.amp and torch.cuda.is_available()))

    # ------------------ 训练 ------------------
    for ep in range(1, args.epochs+1):
        model.train()
        tr_loss = 0.0
        for batch in train_loader:
            ids = batch['input_ids'].to(device)
            attn = batch['attention_mask'].to(device)
            ttid = batch['token_type_ids'].to(device)
            owner_ex = batch['owner_ex'].to(device)
            owner_opt = batch['owner_opt'].to(device)
            y = batch['targets'].to(device)
            B = y.size(0)

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                with torch.amp.autocast('cuda', enabled=True):
                    lf = model(ids, attn, ttid)
                lf = lf.float()  # 聚合前转 fp32
                if args.agg == 'max':
                    logits = aggregate_logits_max(lf, owner_ex, owner_opt, B)
                else:
                    logits = aggregate_logits_topk(lf, owner_ex, owner_opt, B, k=args.topk_k)
                loss = criterion(logits, y.float())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                lf = model(ids, attn, ttid)
                lf = lf.float()
                if args.agg == 'max':
                    logits = aggregate_logits_max(lf, owner_ex, owner_opt, B)
                else:
                    logits = aggregate_logits_topk(lf, owner_ex, owner_opt, B, k=args.topk_k)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
            scheduler.step()
            tr_loss += loss.item()
        tr_loss /= max(1, len(train_loader))

        # quick dev metric（仅参考）
        dev_logits, _ = eval_on_loader(model, dev_loader, device, agg=args.agg, topk_k=args.topk_k)
        probs = 1 / (1 + np.exp(-dev_logits))
        pred = (probs >= 0.3).astype(int)
        for i in range(pred.shape[0]):
            if pred[i].sum() == 0:
                pred[i, np.argmax(probs[i])] = 1
        gold_y = []
        for r in dev_rows:
            yv = np.zeros(4, dtype=int)
            if r['gold'] is not None:
                for i,l in enumerate(LABELS):
                    if l in r['gold']:
                        yv[i] = 1
            gold_y.append(yv)
        gold_y = np.vstack(gold_y)
        micro = f1_score(gold_y, pred, average='micro', zero_division=0)
        macro = f1_score(gold_y, pred, average='macro', zero_division=0)
        print(f"Epoch {ep}/{args.epochs} | train loss {tr_loss:.4f} | dev micro {micro:.4f} | dev macro {macro:.4f}")

    # ------------------ 导出 dev 预测 & 校准 ------------------
    dev_logits, dev_qids = eval_on_loader(model, dev_loader, device, agg=args.agg, topk_k=args.topk_k)
    dev_gold = []
    for r in dev_rows:
        yv = np.zeros(4, dtype=int)
        if r['gold'] is not None:
            for i,l in enumerate(LABELS):
                if l in r['gold']:
                    yv[i] = 1
        dev_gold.append(yv)
    dev_gold = np.vstack(dev_gold)

    calib = search_thresholds_dev(dev_logits, dev_gold)
    with open(os.path.join(args.save_dir, 'val_calibration.json'), 'w', encoding='utf-8') as f:
        json.dump(calib, f, ensure_ascii=False, indent=2)

    probs = 1 / (1 + np.exp(-dev_logits))
    thr = np.array(calib['per_label_thr'], dtype=float)
    out = []
    for i, qid in enumerate(dev_qids):
        p = probs[i]
        pick = [LABELS[j] for j in range(4) if p[j] >= thr[j]]
        if not pick:
            pick = [LABELS[int(np.argmax(p))]]
        out.append({
            'question_id': str(qid),
            'prob_A': float(p[0]), 'prob_B': float(p[1]), 'prob_C': float(p[2]), 'prob_D': float(p[3]),
            'prediction': ','.join(sorted(pick))
        })

    with open(os.path.join(args.save_dir, 'dev_predictions.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print('Saved:', os.path.join(args.save_dir, 'dev_predictions.json'))

if __name__ == '__main__':
    main()
