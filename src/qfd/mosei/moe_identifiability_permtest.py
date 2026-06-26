#!/usr/bin/env python3
"""
qfd.mosei.moe_identifiability_permtest

MOSEI STEP 6 — MoE (M,Q)->w identifiability under leakage-safe TEST-only brokenQ K-permutation bank.

Defensible protocol:
- Experts: fixed unimodal probs (p_l,p_a,p_v) in UNION row space; NaN where missing.
- Gate input ONLY (M,Q): x_mq = [Ml,Ma,Mv,Ql,Qa,Qv] in strict [l,a,v] order.
- Train gate ONCE per (seed,fold) on FULL-only TRAIN using CLEAN Q.
- Evaluate on FULL-only TEST:
    * MoE-NoQ: gate trained on M-only (Q zeros)
    * MoE-cleanQ: trained clean gate + cleanQ test features
    * MoE-brokenQ_k: SAME trained clean gate + brokenQ_k loaded from perm bank perm_{k:03d}

Permutation p-value (one-sided per fold):
    p_perm = (1 + #{k: broken_acc_k >= clean_acc}) / (K + 1)

Outputs:
- CSV: per seed/fold aggregate + diagnostics
- JSON: fold-level mean±std + median p_perm + trace samples

Never permutes in-script. Loads brokenQ_k from perm bank.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:
    _TORCH_IMPORT_ERROR = exc
    torch = None
    F = None

    class _MissingNN:
        Module = object

    nn = _MissingNN()
else:
    _TORCH_IMPORT_ERROR = None


def _require_torch() -> None:
    if torch is None or F is None:
        raise ModuleNotFoundError(
            "qfd.mosei.moe_identifiability_permtest requires the optional torch dependency. "
            "Install the torch extra, for example: python -m pip install '.[torch]'"
        ) from _TORCH_IMPORT_ERROR


# ----------------------------
# Import q_contract_mosei (single source of truth)
# ----------------------------

def _import_qc() -> object:
    """Import the packaged MOSEI contract. Kept as a function for CLI compatibility."""
    from qfd._shared import q_contract_mosei as qc  # type: ignore
    return qc


qc = _import_qc()


# ----------------------------
# Unimodal preds loader
# ----------------------------

def _find_unimodal_preds_file(root: Path, seed: int, fold: int) -> Path:
    cand = [
        root / f"seed_{seed}" / f"fold_{fold}.npz",
        root / f"seed_{seed}" / f"fold{fold}.npz",
        root / f"seed_{seed}" / f"union_unimodal_preds_seed{seed}_fold{fold}.npz",
        root / f"union_unimodal_preds_seed{seed}_fold{fold}.npz",
        root / f"seed_{seed}_fold_{fold}.npz",
    ]
    for p in cand:
        if p.exists():
            return p
    hits = list(root.rglob(f"*seed{seed}*fold{fold}*.npz"))
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise FileNotFoundError(
            f"Ambiguous unimodal preds for seed={seed}, fold={fold} under {root}. "
            f"Matches:\n" + "\n".join(map(str, hits[:30]))
        )
    raise FileNotFoundError(f"Missing unimodal preds for seed={seed}, fold={fold} under {root}")


def _load_unimodal_probs(npz_path: Path, union_ids_str: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(npz_path, allow_pickle=True)
    keys = set(z.files)

    def get_any(cands: List[str]) -> np.ndarray:
        for k in cands:
            if k in keys:
                return z[k]
        raise KeyError(f"Unimodal preds {npz_path} missing {cands}. Found keys: {sorted(keys)}")

    ids = get_any(["ids", "ids_str", "union_ids", "id"]).astype(str).reshape(-1)
    if ids.shape != union_ids_str.shape or np.any(ids != union_ids_str):
        raise ValueError(f"{npz_path}: ids not aligned to UNION order")

    pl = get_any(["p_l", "pl", "prob_l", "probs_l", "p_text", "p_lang", "p_L"]).astype(float).reshape(-1)
    pa = get_any(["p_a", "pa", "prob_a", "probs_a", "p_audio", "pA"]).astype(float).reshape(-1)
    pv = get_any(["p_v", "pv", "prob_v", "probs_v", "p_video", "pV"]).astype(float).reshape(-1)

    for name, arr in [("p_l", pl), ("p_a", pa), ("p_v", pv)]:
        if arr.ndim != 1:
            raise ValueError(f"{name} must be 1D (N,), got {arr.shape} in {npz_path}")
        if arr.shape[0] != union_ids_str.shape[0]:
            raise ValueError(f"{name} length {arr.shape[0]} != N={union_ids_str.shape[0]} in {npz_path}")
        if np.isinf(arr).any():
            raise ValueError(f"{name} contains inf values in {npz_path}")

    def clip_keep_nan(x: np.ndarray) -> np.ndarray:
        out = x.astype(float).copy()
        fin = np.isfinite(out)
        out[fin] = np.clip(out[fin], 0.0, 1.0)
        return out

    return clip_keep_nan(pl), clip_keep_nan(pa), clip_keep_nan(pv)


# ----------------------------
# Q helpers (MOSEI: l,a,v)
# ----------------------------

def _q_to_scalar(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x
    if x.ndim == 2:
        return np.mean(x, axis=1)
    raise ValueError(f"Unsupported Q ndim={x.ndim}")


def _stack_q_lav(q: Any) -> np.ndarray:
    Ql = _q_to_scalar(q.Ql)
    Qa = _q_to_scalar(q.Qa)
    Qv = _q_to_scalar(q.Qv)
    return np.stack([Ql, Qa, Qv], axis=1)  # (N,3) in [l,a,v] order


def _load_broken_q_draws(q_perm_parent: Path, seed: int, fold: int, union: Any, K: int) -> List[Any]:
    """
    Prefer the public precompute_brokenq layout: one bank file with K on axis 1.
    Fall back to the older q_perm_parent/perm_### layout if no direct bank is found.
    """
    if K == 1:
        try:
            return [qc.load_fold_q(q_perm_parent, seed=seed, fold=fold, union=union, require_ids_match=True)]
        except FileNotFoundError:
            pass

    try:
        bank = qc.load_fold_q_bank(q_perm_parent, seed=seed, fold=fold, union=union, require_ids_match=True)
    except FileNotFoundError:
        draws: List[Any] = []
        for k in range(1, K + 1):
            q_root_k = q_perm_parent / f"perm_{k:03d}"
            draws.append(qc.load_fold_q(q_root_k, seed=seed, fold=fold, union=union, require_ids_match=True))
        return draws

    bank_K = int(bank.Ql.shape[1])
    if bank_K < K:
        raise ValueError(f"BrokenQ bank has K={bank_K}, but --K={K} was requested: {bank.q_path}")
    return [qc.select_perm_from_bank(union, bank, k) for k in range(K)]


# ----------------------------
# Probability -> logit (preserve NaNs for missing)
# ----------------------------

def _probs_to_logits(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p2 = np.asarray(p, dtype=float).copy()
    finite = np.isfinite(p2)
    p2[finite] = np.clip(p2[finite], eps, 1.0 - eps)
    out = np.full_like(p2, np.nan, dtype=float)
    out[finite] = np.log(p2[finite] / (1.0 - p2[finite]))
    return out


# ----------------------------
# MoE Gate (uses only M,Q)
# ----------------------------

class MQGate(nn.Module):
    """
    Gate: scores -> (B,3); mask absent modalities; softmax over present.
    x_mq ordering: [Ml,Ma,Mv,Ql,Qa,Qv]
    expert order: [l,a,v]
    """
    def __init__(self, in_dim: int = 6, hidden_dim: int = 0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim <= 0:
            self.fc = nn.Linear(in_dim, 3)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 3),
            )

    def forward(self, x_mq: torch.Tensor, m_mask: torch.Tensor) -> torch.Tensor:
        logits = self.fc(x_mq) if self.hidden_dim <= 0 else self.mlp(x_mq)
        neg_inf = torch.tensor(-1e9, device=logits.device, dtype=logits.dtype)
        logits = torch.where(m_mask > 0.5, logits, neg_inf)
        return F.softmax(logits, dim=1)


# ----------------------------
# Gate training (FULL-only TRAIN)
# ----------------------------

def _train_gate(
    *,
    seed: int,
    gate: MQGate,
    x_mq_train: np.ndarray,          # (n,6)
    m_train: np.ndarray,             # (n,3)
    expert_logits_train: np.ndarray, # (n,3)
    y_train: np.ndarray,             # (n,)
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    epochs: int = 400,
    batch_size: int = 256,
    device: str = "cuda",
) -> MQGate:
    torch.manual_seed(int(seed))
    gate = gate.to(device)

    X = torch.tensor(x_mq_train, dtype=torch.float32, device=device)
    M = torch.tensor(m_train, dtype=torch.float32, device=device)
    L = torch.tensor(expert_logits_train, dtype=torch.float32, device=device)
    y = torch.tensor(np.asarray(y_train, dtype=float), dtype=torch.float32, device=device)

    if torch.isnan(X).any():
        raise ValueError("NaNs in x_mq_train (should not happen on FULL-only TRAIN).")
    if torch.isnan(L).any():
        raise ValueError("NaNs in expert_logits_train (should not happen on FULL-only TRAIN).")

    opt = torch.optim.AdamW(gate.parameters(), lr=lr, weight_decay=weight_decay)

    n = X.shape[0]
    if batch_size <= 0:
        batch_size = n

    for _ in range(int(epochs)):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, mb, lb, yb = X[idx], M[idx], L[idx], y[idx]
            w = gate(xb, mb)                       # (B,3)
            fused_logit = torch.sum(w * lb, dim=1) # (B,)
            loss = F.binary_cross_entropy_with_logits(fused_logit, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    return gate


# ----------------------------
# Inference
# ----------------------------

def _moe_predict_probs(
    *,
    gate: MQGate,
    x_mq: np.ndarray,           # (n,6)
    m_mask: np.ndarray,         # (n,3)
    expert_logits: np.ndarray,  # (n,3)
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray]:
    gate.eval()
    with torch.no_grad():
        X = torch.tensor(x_mq, dtype=torch.float32, device=device)
        M = torch.tensor(m_mask, dtype=torch.float32, device=device)
        L = torch.tensor(expert_logits, dtype=torch.float32, device=device)
        w = gate(X, M)
        fused_logit = torch.sum(w * L, dim=1)
        p = torch.sigmoid(fused_logit).cpu().numpy().astype(float)
        w_np = w.cpu().numpy().astype(float)
    return p, w_np


def _entropy(w: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    w = np.clip(w, eps, 1.0)
    w = w / np.sum(w, axis=1, keepdims=True)
    return -(w * np.log(w)).sum(axis=1)


# ----------------------------
# Permtest helpers
# ----------------------------

def perm_p_value_one_sided(clean_score: float, broken_scores: np.ndarray) -> float:
    broken_scores = np.asarray(broken_scores, dtype=float).reshape(-1)
    return float((1.0 + np.sum(broken_scores >= clean_score)) / (broken_scores.size + 1.0))


def _mean_std(xs: List[float]) -> Dict[str, float]:
    arr = np.asarray(xs, dtype=float)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=0)), "n": int(arr.size)}


def _append_row(rows: List[Dict], **kw) -> None:
    rows.append({k: (v.item() if isinstance(v, np.generic) else v) for k, v in kw.items()})


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--union_npz", type=str, required=True)
    ap.add_argument("--splits_dir", type=str, required=True)
    ap.add_argument("--unimodal_preds_root", type=str, required=True)

    ap.add_argument("--q_clean_root", type=str, required=True)
    ap.add_argument("--q_perm_parent", type=str, required=True)
    ap.add_argument("--K", type=int, default=100)

    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--out_json", type=str, required=True)

    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])

    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--dry_run", action="store_true")

    ap.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--gate_hidden", type=int, default=0)
    ap.add_argument("--gate_lr", type=float, default=1e-2)
    ap.add_argument("--gate_wd", type=float, default=1e-4)
    ap.add_argument("--gate_epochs", type=int, default=400)
    ap.add_argument("--gate_batch", type=int, default=256)

    ap.add_argument("--trace_perms", type=int, nargs="+", default=[1, 2, 3, 100])

    args = ap.parse_args()
    _require_torch()

    union = qc.load_union(args.union_npz)
    N = len(union.ids_str)

    unimodal_root = Path(args.unimodal_preds_root)
    q_clean_root = Path(args.q_clean_root)
    q_perm_parent = Path(args.q_perm_parent)
    K = int(args.K)
    if K < 1:
        raise ValueError("--K must be >= 1")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] --device cuda requested but unavailable; falling back to cpu.")
        device = "cpu"

    rows: List[Dict] = []
    traces: List[Dict] = []

    vals: Dict[str, List[float]] = {
        "moe_noq": [],
        "moe_cleanq": [],
        "moe_broken_mean": [],
        "moe_broken_std": [],
        "moe_clean_minus_broken_mean": [],
        "moe_abs_clean_minus_broken_mean": [],
        "perm_p_one_sided_clean_gt_broken": [],
        "flip_clean_vs_broken_mean": [],
        "flip_clean_vs_broken_max": [],
        "gate_entropy_mean": [],
        "gate_w_mean_l": [],
        "gate_w_mean_a": [],
        "gate_w_mean_v": [],
        "oracle_best": [],
        "oracle_wavg": [],
    }

    for seed in args.seeds:
        for fold in args.folds:
            try:
                split = qc.load_fold_split(args.splits_dir, seed=seed, fold=fold)
                train_mask, test_mask = qc.make_train_test_masks(union, split)

                eval_mask = qc.eval_mask_full_only(union, test_mask)  # FULL-only TEST
                train_full = qc.full_only_mask(union.Ml, union.Ma, union.Mv, base_mask=train_mask)  # FULL-only TRAIN

                n_eval = int(np.sum(eval_mask))
                n_train = int(np.sum(train_full))
                if n_eval == 0:
                    raise RuntimeError(f"seed={seed} fold={fold}: FULL-only TEST empty")
                if n_train == 0:
                    raise RuntimeError(f"seed={seed} fold={fold}: FULL-only TRAIN empty")
                if len(np.unique(union.y[train_full])) < 2:
                    raise RuntimeError(f"seed={seed} fold={fold}: FULL-only TRAIN has single class")

                unimodal_path = _find_unimodal_preds_file(unimodal_root, seed, fold)
                pl, pa, pv = _load_unimodal_probs(unimodal_path, union.ids_str)

                qc.assert_probs_nan_where_missing(union, pl, pa, pv)

                for name, arr in [("p_l", pl), ("p_a", pa), ("p_v", pv)]:
                    bad = eval_mask & ~np.isfinite(arr)
                    if np.any(bad):
                        idx = np.flatnonzero(bad)[:10].tolist()
                        raise ValueError(f"{name} non-finite on FULL-only eval rows. ex_idx={idx}")

                ll = _probs_to_logits(pl)
                la = _probs_to_logits(pa)
                lv = _probs_to_logits(pv)
                L_all = np.stack([ll, la, lv], axis=1).astype(float)  # [l,a,v]
                if not np.isfinite(L_all[train_full]).all():
                    bad = train_full & (~np.isfinite(L_all).all(axis=1))
                    idx = np.flatnonzero(bad)[:10].tolist()
                    raise ValueError(f"Non-finite expert logits on FULL-only TRAIN. ex_idx={idx}")

                q_clean = qc.load_fold_q(
                    q_clean_root, seed=seed, fold=fold,
                    union=union, require_ids_match=True
                )
                qc.assert_q_missing_is_zero(union, q_clean.Ql, q_clean.Qa, q_clean.Qv)
                qc.assert_no_nan_in_present_q(union, q_clean.Ql, q_clean.Qa, q_clean.Qv)

                M_all = np.stack([union.Ml, union.Ma, union.Mv], axis=1).astype(float)
                Q_clean_all = _stack_q_lav(q_clean).astype(float)

                X_clean_all = np.concatenate([M_all, Q_clean_all], axis=1)
                X_noq_all = np.concatenate([M_all, np.zeros_like(Q_clean_all)], axis=1)

                if args.dry_run:
                    print(f"[DRY] seed={seed} fold={fold} train_full={n_train} eval_full_test={n_eval}")
                    print(f"      unimodal={unimodal_path}")
                    print(f"      cleanQ  ={q_clean.q_path}")
                    continue

                gate_noq = MQGate(in_dim=6, hidden_dim=args.gate_hidden)
                gate_noq = _train_gate(
                    seed=seed * 100 + fold + 7,
                    gate=gate_noq,
                    x_mq_train=X_noq_all[train_full],
                    m_train=M_all[train_full],
                    expert_logits_train=L_all[train_full],
                    y_train=union.y[train_full].astype(int),
                    lr=args.gate_lr,
                    weight_decay=args.gate_wd,
                    epochs=args.gate_epochs,
                    batch_size=args.gate_batch,
                    device=device,
                )

                gate = MQGate(in_dim=6, hidden_dim=args.gate_hidden)
                gate = _train_gate(
                    seed=seed * 100 + fold,
                    gate=gate,
                    x_mq_train=X_clean_all[train_full],
                    m_train=M_all[train_full],
                    expert_logits_train=L_all[train_full],
                    y_train=union.y[train_full].astype(int),
                    lr=args.gate_lr,
                    weight_decay=args.gate_wd,
                    epochs=args.gate_epochs,
                    batch_size=args.gate_batch,
                    device=device,
                )

                y_true = union.y[eval_mask].astype(int)

                p_noq, _ = _moe_predict_probs(
                    gate=gate_noq,
                    x_mq=X_noq_all[eval_mask],
                    m_mask=M_all[eval_mask],
                    expert_logits=L_all[eval_mask],
                    device=device,
                )
                p_clean, w_clean = _moe_predict_probs(
                    gate=gate,
                    x_mq=X_clean_all[eval_mask],
                    m_mask=M_all[eval_mask],
                    expert_logits=L_all[eval_mask],
                    device=device,
                )

                y_noq = qc.preds_from_probs(p_noq, thresh=args.thresh)
                y_clean = qc.preds_from_probs(p_clean, thresh=args.thresh)
                noq_acc = qc.balanced_accuracy(y_true, y_noq)
                clean_acc = qc.balanced_accuracy(y_true, y_clean)

                ent = _entropy(w_clean)
                gate_ent = float(np.mean(ent))
                wml = float(np.mean(w_clean[:, 0]))
                wma = float(np.mean(w_clean[:, 1]))
                wmv = float(np.mean(w_clean[:, 2]))

                broken_accs = np.zeros(K, dtype=float)
                flip_scores = np.zeros(K, dtype=float)

                trace = {"seed": seed, "fold": fold, "trace": {}}
                trace_ids = [k for k in args.trace_perms if 1 <= k <= K]

                broken_draws = _load_broken_q_draws(q_perm_parent, seed, fold, union, K)

                for k, q_b in enumerate(broken_draws, start=1):
                    qc.assert_q_missing_is_zero(union, q_b.Ql, q_b.Qa, q_b.Qv)
                    qc.assert_no_nan_in_present_q(union, q_b.Ql, q_b.Qa, q_b.Qv)

                    Q_b = _stack_q_lav(q_b).astype(float)
                    X_b = np.concatenate([M_all, Q_b], axis=1)

                    p_b, _ = _moe_predict_probs(
                        gate=gate,
                        x_mq=X_b[eval_mask],
                        m_mask=M_all[eval_mask],
                        expert_logits=L_all[eval_mask],
                        device=device,
                    )
                    y_b = qc.preds_from_probs(p_b, thresh=args.thresh)
                    b_acc = qc.balanced_accuracy(y_true, y_b)
                    broken_accs[k - 1] = b_acc
                    flip_scores[k - 1] = qc.flip_rate(y_clean, y_b)

                    if k in trace_ids:
                        trace["trace"][f"perm_{k:03d}"] = {"broken_acc": float(b_acc), "flip": float(flip_scores[k - 1])}

                b_mean = float(np.mean(broken_accs))
                b_std = float(np.std(broken_accs, ddof=1)) if K > 1 else float("nan")
                p_perm = perm_p_value_one_sided(clean_acc, broken_accs)

                flip_mean = float(np.mean(flip_scores))
                flip_max = float(np.max(flip_scores))

                probs_t = np.stack([pl[eval_mask], pa[eval_mask], pv[eval_mask]], axis=1)
                eps = 1e-8
                p_true = np.where(y_true[:, None] == 1, probs_t, 1.0 - probs_t)
                loss = -np.log(np.clip(p_true, eps, 1.0))
                best_m = loss.argmin(axis=1)
                p_oracle_best = probs_t[np.arange(best_m.size), best_m]
                oracle_best_acc = qc.balanced_accuracy(y_true, qc.preds_from_probs(p_oracle_best, thresh=args.thresh))

                oracleQ = qc.compute_oracle_q_union(
                    union=union,
                    eval_mask=eval_mask,
                    p_l=pl,
                    p_a=pa,
                    p_v=pv,
                    thresh=args.thresh,
                )
                Q_t = np.stack([oracleQ.Ql[eval_mask], oracleQ.Qa[eval_mask], oracleQ.Qv[eval_mask]], axis=1).astype(float)
                wsum = Q_t.sum(axis=1)
                p_oracle_wavg = np.zeros_like(wsum, dtype=float)
                none = (wsum == 0)
                some = ~none
                p_oracle_wavg[some] = (Q_t[some] * probs_t[some]).sum(axis=1) / wsum[some]
                p_oracle_wavg[none] = probs_t[none].mean(axis=1)
                oracle_wavg_acc = qc.balanced_accuracy(y_true, qc.preds_from_probs(p_oracle_wavg, thresh=args.thresh))

                gap_mean = float(clean_acc - b_mean)
                gap_abs = float(abs(clean_acc - b_mean))

                vals["moe_noq"].append(noq_acc)
                vals["moe_cleanq"].append(clean_acc)
                vals["moe_broken_mean"].append(b_mean)
                vals["moe_broken_std"].append(b_std)
                vals["moe_clean_minus_broken_mean"].append(gap_mean)
                vals["moe_abs_clean_minus_broken_mean"].append(gap_abs)
                vals["perm_p_one_sided_clean_gt_broken"].append(p_perm)
                vals["flip_clean_vs_broken_mean"].append(flip_mean)
                vals["flip_clean_vs_broken_max"].append(flip_max)
                vals["gate_entropy_mean"].append(gate_ent)
                vals["gate_w_mean_l"].append(wml)
                vals["gate_w_mean_a"].append(wma)
                vals["gate_w_mean_v"].append(wmv)
                vals["oracle_best"].append(oracle_best_acc)
                vals["oracle_wavg"].append(oracle_wavg_acc)

                _append_row(
                    rows,
                    seed=seed,
                    fold=fold,
                    fuser="moe",
                    n_train_full=n_train,
                    n_eval_full_test=n_eval,
                    unimodal_path=str(unimodal_path),
                    q_clean_path=str(q_clean.q_path),
                    q_perm_parent=str(q_perm_parent),
                    K=K,
                    moe_noq_acc=noq_acc,
                    moe_clean_acc=clean_acc,
                    moe_broken_mean_acc=b_mean,
                    moe_broken_std_acc=b_std,
                    clean_minus_broken_mean=gap_mean,
                    abs_clean_minus_broken_mean=gap_abs,
                    perm_p_one_sided_clean_gt_broken=p_perm,
                    flip_clean_vs_broken_mean=flip_mean,
                    flip_clean_vs_broken_max=flip_max,
                    gate_entropy_mean=gate_ent,
                    gate_w_mean_l=wml,
                    gate_w_mean_a=wma,
                    gate_w_mean_v=wmv,
                    oracle_best_acc=oracle_best_acc,
                    oracle_wavg_acc=oracle_wavg_acc,
                    gate_arch=("linear" if args.gate_hidden <= 0 else f"mlp(hidden={args.gate_hidden})"),
                    device=device,
                )

                if trace_ids:
                    traces.append(trace)

                print(
                    f"[OK] seed={seed} fold={fold} | "
                    f"noQ/clean/brokenμ±σ={noq_acc:.4f}/{clean_acc:.4f}/{b_mean:.4f}±{b_std:.4f} | "
                    f"clean-brokenμ={gap_mean:+.4f} p_perm={p_perm:.3f} flipμ={flip_mean:.3f} | "
                    f"gate(ent={gate_ent:.3f}, w={wml:.2f}/{wma:.2f}/{wmv:.2f}) | "
                    f"oracle(wavg/best)={oracle_wavg_acc:.4f}/{oracle_best_acc:.4f}"
                )

            except Exception as e:
                tb = traceback.format_exc()
                _append_row(rows, seed=seed, fold=fold, error=str(e), traceback=tb)
                print(f"[FAIL] seed={seed} fold={fold}: {e}")

    if args.dry_run:
        print("[DONE] dry_run complete.")
        return

    if len(rows) == 0:
        raise RuntimeError("No rows produced. All folds failed.")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = sorted({k for r in rows for k in r.keys()})
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    ok_rows = [r for r in rows if "error" not in r]
    summary: Dict[str, Any] = {
        "meta": {
            "union_npz": args.union_npz,
            "splits_dir": args.splits_dir,
            "unimodal_preds_root": str(Path(args.unimodal_preds_root)),
            "q_clean_root": str(q_clean_root),
            "q_perm_parent": str(q_perm_parent),
            "K": K,
            "seeds": args.seeds,
            "folds": args.folds,
            "thresh": float(args.thresh),
            "eval_subset": "FULL-only within TEST (Ml==Ma==Mv==1)",
            "train_subset": "FULL-only within TRAIN (Ml==Ma==Mv==1)",
            "q_effect_isolation": "gate trained once on cleanQ; evaluated on cleanQ vs brokenQ_k with same trained gate",
            "perm_p_value": "one-sided: p=(1+#{broken>=clean})/(K+1) per fold",
            "gate_definition": "w = softmax(g(M,Q)) with absent-masked logits; fused logit = sum_m w_m * logit(p_m)",
            "modality_order": ["l", "a", "v"],
            "gate_arch": ("linear" if args.gate_hidden <= 0 else f"mlp(hidden={args.gate_hidden})"),
            "gate_opt": {
                "device": device,
                "lr": args.gate_lr,
                "weight_decay": args.gate_wd,
                "epochs": args.gate_epochs,
                "batch_size": args.gate_batch,
            },
        },
        "n_rows": int(len(rows)),
        "n_ok": int(len(ok_rows)),
        "balanced_accuracy_foldlevel": {k: _mean_std(v) for k, v in vals.items() if len(v) > 0},
        "perm_p_one_sided_clean_gt_broken_median": float(
            np.median(np.asarray(vals["perm_p_one_sided_clean_gt_broken"], dtype=float))
        ) if vals["perm_p_one_sided_clean_gt_broken"] else float("nan"),
        "trace_samples": traces,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))

    print(f"[DONE] wrote CSV:  {out_csv}")
    print(f"[DONE] wrote JSON: {out_json}")


if __name__ == "__main__":
    main()
