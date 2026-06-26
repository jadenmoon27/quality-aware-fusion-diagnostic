#!/usr/bin/env python3
"""
qfd.mosei.moe_identifiability

MOSEI version (3 experts: l,a,v).

Only necessary changes from StressID:
- modalities are (l,a,v) not (a,v,p)
- masks are (Ml,Ma,Mv)
- Q fields are (Ql,Qa,Qv)
- unimodal probs are (p_l,p_a,p_v)
- ordering is kept consistent everywhere: [l, a, v]

BrokenQ usage:
- Pass --q_broken_root output/final_experiments/mosei/quality_fold_broken
- Script loads brokenQ from that root (no in-script permutation).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import traceback
from typing import Dict, List, Tuple

import numpy as np

from qfd._shared.q_contract_mosei import (
    UnionData,
    FoldSplit,
    FoldQ,
    load_union,
    load_fold_split,
    make_train_test_masks,
    load_fold_q,
    assert_q_missing_is_zero,
    assert_no_nan_in_present_q,
    balanced_accuracy,
    compute_oracle_q_union,
    assert_probs_nan_where_missing,
    preds_from_probs,
    flip_rate,
)

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
            "qfd.mosei.moe_identifiability requires the optional torch dependency. "
            "Install the torch extra, for example: python -m pip install '.[torch]'"
        ) from _TORCH_IMPORT_ERROR


# -----------------------------
# Unimodal preds loader
# -----------------------------

def _find_unimodal_preds_file(root: Path, seed: int, fold: int) -> Path:
    # ONLY NECESSARY CHANGE:
    # include the common MOSEI layout: root/seed_{seed}/fold_{fold}.npz
    cand = [
        root / f"seed_{seed}" / f"fold_{fold}.npz",  # <-- added (most common)
        root / f"seed_{seed}" / f"union_unimodal_preds_seed{seed}_fold{fold}.npz",
        root / f"seed_{seed}" / f"fold{fold}.npz",
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
            f"Ambiguous unimodal preds file for seed={seed}, fold={fold} under {root}. "
            f"Matches:\n" + "\n".join(map(str, hits[:30]))
        )
    raise FileNotFoundError(
        f"Could not find unimodal preds for seed={seed}, fold={fold} under {root}. "
        f"Edit _find_unimodal_preds_file() patterns to match your naming."
    )


def _load_unimodal_probs(npz_path: Path, *, expect_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MOSEI expects 3 experts: (l,a,v).
    Returns: (pl, pa, pv) each shape (N,)
    """
    z = np.load(npz_path, allow_pickle=True)
    keys = set(z.files)

    def get_any(cands: List[str]) -> np.ndarray:
        for k in cands:
            if k in keys:
                return z[k]
        raise KeyError(f"Unimodal preds {npz_path} missing {cands}. Found keys: {sorted(keys)}")

    pl = get_any(["p_l", "pl", "prob_l", "probs_l", "p_text", "p_lang", "p_L", "p_language"])
    pa = get_any(["p_a", "pa", "prob_a", "probs_a", "p_audio", "audio_p", "pA"])
    pv = get_any(["p_v", "pv", "prob_v", "probs_v", "p_video", "video_p", "pV"])

    for name, arr in [("p_l", pl), ("p_a", pa), ("p_v", pv)]:
        if arr.shape[0] != expect_len:
            raise ValueError(f"{name} length {arr.shape[0]} != expected {expect_len} in {npz_path}")
        if arr.ndim != 1:
            raise ValueError(f"{name} must be 1D (N,), got {arr.shape} in {npz_path}")
        if np.isinf(arr).any():
            raise ValueError(f"{name} contains inf values in {npz_path}")

    return pl.astype(float), pa.astype(float), pv.astype(float)


# -----------------------------
# Q stacker (MOSEI: l,a,v)
# -----------------------------

def _stack_q_lav(q: FoldQ) -> np.ndarray:
    def to_scalar(x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            return x.astype(float)
        if x.ndim == 2:
            return x.astype(float).mean(axis=1)
        raise ValueError(f"Unsupported Q ndim={x.ndim}")

    Ql = to_scalar(q.Ql)
    Qa = to_scalar(q.Qa)
    Qv = to_scalar(q.Qv)
    return np.stack([Ql, Qa, Qv], axis=1)  # (N,3) in [l,a,v] order


# -----------------------------
# Probability -> logit (preserve NaNs for missing)
# -----------------------------

def _probs_to_logits(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p2 = p.copy().astype(float)
    finite = np.isfinite(p2)
    p2[finite] = np.clip(p2[finite], eps, 1.0 - eps)
    out = np.full_like(p2, np.nan, dtype=float)
    out[finite] = np.log(p2[finite] / (1.0 - p2[finite]))
    return out


# -----------------------------
# MoE Gate (uses only M,Q)
# -----------------------------

class MQGate(nn.Module):
    """
    Gate: raw scores -> (B,3), then mask absent modalities, softmax over present.
    x_mq ordering is fixed: [Ml,Ma,Mv,Ql,Qa,Qv]
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


# -----------------------------
# Gate training (FULL-only TRAIN)
# -----------------------------

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
    device: str = "cpu",
) -> MQGate:
    torch.manual_seed(int(seed))
    gate = gate.to(device)

    X = torch.tensor(x_mq_train, dtype=torch.float32, device=device)
    M = torch.tensor(m_train, dtype=torch.float32, device=device)
    L = torch.tensor(expert_logits_train, dtype=torch.float32, device=device)
    y = torch.tensor(y_train.astype(float), dtype=torch.float32, device=device)

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


# -----------------------------
# Inference helpers
# -----------------------------

def _moe_predict_probs(
    *,
    gate: MQGate,
    x_mq: np.ndarray,           # (n,6)
    m_mask: np.ndarray,         # (n,3)
    expert_logits: np.ndarray,  # (n,3)
    device: str = "cpu",
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
    w = np.clip(w.astype(float), eps, 1.0)
    w = w / w.sum(axis=1, keepdims=True)
    return -(w * np.log(w)).sum(axis=1)


# -----------------------------
# Reporting helpers
# -----------------------------

def _mean_std(xs: List[float]) -> Dict[str, float]:
    arr = np.array(xs, dtype=float)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=0)), "n": int(arr.size)}


def _append_row(rows: List[Dict], **kw) -> None:
    rows.append({k: (v.item() if isinstance(v, np.generic) else v) for k, v in kw.items()})


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union_npz", type=str, required=True)
    ap.add_argument("--splits_dir", type=str, required=True)
    ap.add_argument("--unimodal_preds_root", type=str, required=True)
    ap.add_argument("--q_clean_root", type=str, required=True)
    ap.add_argument("--q_broken_root", type=str, required=True)  # <- pass quality_fold_broken here
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--out_json", type=str, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--dry_run", action="store_true")

    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--gate_hidden", type=int, default=0)
    ap.add_argument("--gate_lr", type=float, default=1e-2)
    ap.add_argument("--gate_wd", type=float, default=1e-4)
    ap.add_argument("--gate_epochs", type=int, default=400)
    ap.add_argument("--gate_batch", type=int, default=256)

    args = ap.parse_args()
    _require_torch()

    union: UnionData = load_union(args.union_npz)
    N = union.ids.shape[0]
    print(f"[OK] UNION loaded: N={N}")

    unimodal_root = Path(args.unimodal_preds_root)
    q_clean_root = Path(args.q_clean_root)
    q_broken_root = Path(args.q_broken_root)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but torch.cuda.is_available() is False.")

    rows: List[Dict] = []

    acc = {"moe_noq": [], "moe_cleanq": [], "moe_brokenq": [], "oracle_wavg": [], "oracle_best": []}
    flips = {"moe_clean_vs_broken": []}
    gate_diag = {"entropy": [], "w_mean_l": [], "w_mean_a": [], "w_mean_v": []}

    for seed in args.seeds:
        for fold in args.folds:
            try:
                split: FoldSplit = load_fold_split(args.splits_dir, seed, fold)
                train_mask, test_mask = make_train_test_masks(union, split)

                eval_mask = test_mask & (union.Ml == 1) & (union.Ma == 1) & (union.Mv == 1)
                n_eval = int(eval_mask.sum())
                if n_eval == 0:
                    raise RuntimeError(f"seed={seed} fold={fold}: FULL-only test is empty.")

                train_full = train_mask & (union.Ml == 1) & (union.Ma == 1) & (union.Mv == 1)
                n_train_full = int(train_full.sum())
                if n_train_full == 0:
                    raise RuntimeError(f"seed={seed} fold={fold}: FULL-only train is empty.")

                y_all = union.y.astype(int)
                if len(np.unique(y_all[train_full])) < 2:
                    raise RuntimeError(f"seed={seed} fold={fold}: FULL-only train has single class.")

                unimodal_path = _find_unimodal_preds_file(unimodal_root, seed, fold)
                pl, pa, pv = _load_unimodal_probs(unimodal_path, expect_len=N)

                assert_probs_nan_where_missing(union, pl, pa, pv)

                for name, arr in [("p_l", pl), ("p_a", pa), ("p_v", pv)]:
                    bad = eval_mask & ~np.isfinite(arr)
                    if bad.any():
                        idx = np.flatnonzero(bad)[:10]
                        raise ValueError(f"{name} non-finite on FULL-only eval rows. Example idx: {idx}")

                pl = np.where(np.isfinite(pl), np.clip(pl, 0.0, 1.0), pl)
                pa = np.where(np.isfinite(pa), np.clip(pa, 0.0, 1.0), pa)
                pv = np.where(np.isfinite(pv), np.clip(pv, 0.0, 1.0), pv)

                q_clean: FoldQ = load_fold_q(
                    q_clean_root, seed, fold,
                    union=union,
                    require_ids_match=True,
                )
                q_broken: FoldQ = load_fold_q(
                    q_broken_root, seed, fold,
                    union=union,
                    require_ids_match=True,
                )

                assert_q_missing_is_zero(union, q_clean.Ql, q_clean.Qa, q_clean.Qv)
                assert_q_missing_is_zero(union, q_broken.Ql, q_broken.Qa, q_broken.Qv)
                assert_no_nan_in_present_q(union, q_clean.Ql, q_clean.Qa, q_clean.Qv)
                assert_no_nan_in_present_q(union, q_broken.Ql, q_broken.Qa, q_broken.Qv)

                M_all = np.stack([union.Ml, union.Ma, union.Mv], axis=1).astype(float)
                Q_clean_all = _stack_q_lav(q_clean).astype(float)
                Q_broken_all = _stack_q_lav(q_broken).astype(float)

                X_clean_all = np.concatenate([M_all, Q_clean_all], axis=1)
                X_broken_all = np.concatenate([M_all, Q_broken_all], axis=1)
                X_noq_all = np.concatenate([M_all, np.zeros_like(Q_clean_all)], axis=1)

                ll = _probs_to_logits(pl)
                la = _probs_to_logits(pa)
                lv = _probs_to_logits(pv)
                L_all = np.stack([ll, la, lv], axis=1).astype(float)

                if not np.isfinite(L_all[train_full]).all():
                    bad = train_full & ~np.isfinite(L_all).all(axis=1)
                    idx = np.flatnonzero(bad)[:10]
                    raise ValueError(f"Non-finite expert logits on FULL-only TRAIN. Example idx: {idx}")

                if args.dry_run:
                    print(f"[DRY] seed={seed} fold={fold} | train_full={n_train_full} eval_full_test={n_eval}")
                    print(f"      unimodal={unimodal_path}")
                    print(f"      cleanQ  ={q_clean.q_path}")
                    print(f"      brokenQ ={q_broken.q_path}")
                    continue

                gate_noq = MQGate(in_dim=6, hidden_dim=args.gate_hidden)
                gate_noq = _train_gate(
                    seed=seed * 100 + fold + 7,
                    gate=gate_noq,
                    x_mq_train=X_noq_all[train_full],
                    m_train=M_all[train_full],
                    expert_logits_train=L_all[train_full],
                    y_train=y_all[train_full],
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
                    y_train=y_all[train_full],
                    lr=args.gate_lr,
                    weight_decay=args.gate_wd,
                    epochs=args.gate_epochs,
                    batch_size=args.gate_batch,
                    device=device,
                )

                y_true = y_all[eval_mask]

                p_noq, _ = _moe_predict_probs(
                    gate=gate_noq, x_mq=X_noq_all[eval_mask], m_mask=M_all[eval_mask],
                    expert_logits=L_all[eval_mask], device=device
                )
                p_clean, w_clean = _moe_predict_probs(
                    gate=gate, x_mq=X_clean_all[eval_mask], m_mask=M_all[eval_mask],
                    expert_logits=L_all[eval_mask], device=device
                )
                p_broken, _ = _moe_predict_probs(
                    gate=gate, x_mq=X_broken_all[eval_mask], m_mask=M_all[eval_mask],
                    expert_logits=L_all[eval_mask], device=device
                )

                y_noq = preds_from_probs(p_noq, thresh=args.thresh)
                y_clean = preds_from_probs(p_clean, thresh=args.thresh)
                y_broken = preds_from_probs(p_broken, thresh=args.thresh)

                noq_acc = balanced_accuracy(y_true, y_noq)
                clean_acc = balanced_accuracy(y_true, y_clean)
                broken_acc = balanced_accuracy(y_true, y_broken)
                flip = flip_rate(y_clean, y_broken)

                ent = _entropy(w_clean)
                gate_diag["entropy"].append(float(np.mean(ent)))
                gate_diag["w_mean_l"].append(float(np.mean(w_clean[:, 0])))
                gate_diag["w_mean_a"].append(float(np.mean(w_clean[:, 1])))
                gate_diag["w_mean_v"].append(float(np.mean(w_clean[:, 2])))

                oracleQ = compute_oracle_q_union(
                    union=union,
                    eval_mask=eval_mask,
                    p_l=pl,
                    p_a=pa,
                    p_v=pv,
                    thresh=args.thresh,
                )

                Ql_t = oracleQ.Ql[eval_mask].astype(float)
                Qa_t = oracleQ.Qa[eval_mask].astype(float)
                Qv_t = oracleQ.Qv[eval_mask].astype(float)

                probs_t = np.stack([pl[eval_mask], pa[eval_mask], pv[eval_mask]], axis=1)
                Q_t = np.stack([Ql_t, Qa_t, Qv_t], axis=1)

                wsum = Q_t.sum(axis=1)
                p_oracle_wavg = np.zeros_like(wsum, dtype=float)
                none = (wsum == 0)
                some = ~none
                p_oracle_wavg[some] = (Q_t[some] * probs_t[some]).sum(axis=1) / wsum[some]
                p_oracle_wavg[none] = probs_t[none].mean(axis=1)

                eps = 1e-8
                p_true = np.where(y_true[:, None] == 1, probs_t, 1.0 - probs_t)
                loss = -np.log(np.clip(p_true, eps, 1.0))
                best_m = loss.argmin(axis=1)
                p_oracle_best = probs_t[np.arange(len(best_m)), best_m]

                oracle_wavg_acc = balanced_accuracy(y_true, preds_from_probs(p_oracle_wavg, thresh=args.thresh))
                oracle_best_acc = balanced_accuracy(y_true, preds_from_probs(p_oracle_best, thresh=args.thresh))

                acc["moe_noq"].append(noq_acc)
                acc["moe_cleanq"].append(clean_acc)
                acc["moe_brokenq"].append(broken_acc)
                acc["oracle_wavg"].append(oracle_wavg_acc)
                acc["oracle_best"].append(oracle_best_acc)
                flips["moe_clean_vs_broken"].append(flip)

                _append_row(
                    rows,
                    seed=seed, fold=fold, fuser="moe",
                    n_eval_full_test=n_eval, n_train_full=n_train_full,
                    unimodal_path=str(unimodal_path),
                    q_clean_path=str(q_clean.q_path),
                    q_broken_path=str(q_broken.q_path),
                    moe_noq_acc=noq_acc,
                    moe_clean_acc=clean_acc,
                    moe_broken_acc=broken_acc,
                    flip_clean_vs_broken=flip,
                    gate_entropy_mean=float(np.mean(ent)),
                    gate_w_mean_l=float(np.mean(w_clean[:, 0])),
                    gate_w_mean_a=float(np.mean(w_clean[:, 1])),
                    gate_w_mean_v=float(np.mean(w_clean[:, 2])),
                    oracle_wavg=oracle_wavg_acc,
                    oracle_best=oracle_best_acc,
                )

                print(
                    f"[OK] seed={seed} fold={fold} | n_eval={n_eval} | "
                    f"noQ/clean/broken={noq_acc:.4f}/{clean_acc:.4f}/{broken_acc:.4f} "
                    f"flip={flip:.3f} | "
                    f"gate(ent={np.mean(ent):.3f}, w={np.mean(w_clean[:,0]):.2f}/{np.mean(w_clean[:,1]):.2f}/{np.mean(w_clean[:,2]):.2f}) | "
                    f"oracle(wavg/best)={oracle_wavg_acc:.4f}/{oracle_best_acc:.4f}"
                )

            except Exception as e:
                tb = traceback.format_exc()
                _append_row(rows, seed=seed, fold=fold, error=str(e), traceback=tb)
                print(f"[FAIL] seed={seed} fold={fold}: {e}")
                continue

    if args.dry_run:
        print("[DONE] dry_run complete.")
        return

    if len(rows) == 0:
        raise RuntimeError("No rows produced. All folds failed.")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_cols = sorted({k for r in rows for k in r.keys()})
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_cols)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    ok_rows = [r for r in rows if "error" not in r]
    if len(ok_rows) == 0:
        summary = {
            "meta": {
                "union_npz": args.union_npz,
                "splits_dir": args.splits_dir,
                "unimodal_preds_root": str(unimodal_root),
                "q_clean_root": str(q_clean_root),
                "q_broken_root": str(q_broken_root),
                "seeds": args.seeds,
                "folds": args.folds,
                "thresh": args.thresh,
                "eval_subset": "FULL-only within TEST (Ml==Ma==Mv==1)",
                "train_subset": "FULL-only within TRAIN (Ml==Ma==Mv==1)",
                "moe_definition": "gate uses only (M,Q); experts are fixed unimodal logits; fused logit is weighted sum",
                "modality_order": ["l", "a", "v"],
            },
            "status": "all_failed",
            "n_rows": len(rows),
            "n_ok": 0,
        }
    else:
        summary = {
            "meta": {
                "union_npz": args.union_npz,
                "splits_dir": args.splits_dir,
                "unimodal_preds_root": str(unimodal_root),
                "q_clean_root": str(q_clean_root),
                "q_broken_root": str(q_broken_root),
                "seeds": args.seeds,
                "folds": args.folds,
                "thresh": args.thresh,
                "eval_subset": "FULL-only within TEST (Ml==Ma==Mv==1)",
                "train_subset": "FULL-only within TRAIN (Ml==Ma==Mv==1)",
                "q_effect_isolation": "gate trained once on cleanQ; tested on cleanQ vs brokenQ",
                "oracle_best_definition": "best unimodal expert per-sample by log-loss (diagnostic ceiling)",
                "gate_arch": ("linear" if args.gate_hidden <= 0 else f"mlp(hidden={args.gate_hidden})"),
                "gate_opt": {
                    "device": args.device,
                    "lr": args.gate_lr,
                    "weight_decay": args.gate_wd,
                    "epochs": args.gate_epochs,
                    "batch_size": args.gate_batch,
                },
                "modality_order": ["l", "a", "v"],
            },
            "n_ok": len(ok_rows),
            "balanced_accuracy": {k: _mean_std(v) for k, v in acc.items() if len(v) > 0},
            "flip_rates": {k: _mean_std(v) for k, v in flips.items() if len(v) > 0},
            "gate_diagnostics": {k: _mean_std(v) for k, v in gate_diag.items() if len(v) > 0},
            "gaps": {
                "moe_clean_minus_broken": _mean_std([a - b for a, b in zip(acc["moe_cleanq"], acc["moe_brokenq"])]) if len(acc["moe_cleanq"]) else {"mean": None, "std": None, "n": 0},
                "moe_abs_clean_minus_broken": _mean_std([abs(a - b) for a, b in zip(acc["moe_cleanq"], acc["moe_brokenq"])]) if len(acc["moe_cleanq"]) else {"mean": None, "std": None, "n": 0},
                "oraclebest_minus_moe_clean": _mean_std([a - b for a, b in zip(acc["oracle_best"], acc["moe_cleanq"])]) if len(acc["moe_cleanq"]) else {"mean": None, "std": None, "n": 0},
                "oraclebest_minus_moe_broken": _mean_std([a - b for a, b in zip(acc["oracle_best"], acc["moe_brokenq"])]) if len(acc["moe_brokenq"]) else {"mean": None, "std": None, "n": 0},
                "moe_flip_clean_vs_broken": _mean_std(flips["moe_clean_vs_broken"]) if len(flips["moe_clean_vs_broken"]) else {"mean": None, "std": None, "n": 0},
            },
        }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[DONE] wrote CSV:  {out_csv}")
    print(f"[DONE] wrote JSON: {out_json}")


if __name__ == "__main__":
    main()
