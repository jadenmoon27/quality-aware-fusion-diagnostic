#!/usr/bin/env python3
# FINAL_EXPERIMENTS/FINAL/v5_2/03_moe_identifiability_tables.py
#
# Paper-ready, contract-aligned MoE identifiability evaluation + LaTeX tables.
#
# SAME as 02_late_fusion_identifiability_tables.py in:
#  - I/O, folds/seeds loop, masking (FULL-only TEST), Q loading, BrokenQ handling,
#  - OracleQ computed from THIS RUN's unimodal preds on FULL-only TEST,
#  - Structural diagnostics (median competitiveness Δ, mean alignment ρ),
#  - Decision identifiability: Clean−Perm (mean±std), p_perm (median), Oracle−Clean headroom (mean±std),
#  - Baseline table: NoQ and Clean-Q (mean±std),
#  - JSON + CSV + LaTeX printing/writing.
#
# ONLY changes late fusion aggregator -> Conditioning-Aware MoE:
#  - Experts produce logits l_{m,i} = logit(p_{m,i})
#  - Router predicts weights w_{m,i} = softmax( g([M_i, Q_i]) )
#  - Fused logit l_i = sum_m w_{m,i} * l_{m,i}; p_i = sigmoid(l_i)
#  - Router trained once per fold (on TRAIN) under Clean-Q and reused unchanged for Broken-Q/Oracle-Q evaluation.
#
# Example (LR):
# export PYTHONPATH=/path/to/project:$PYTHONPATH
#
# python -m qfd.stressid.moe_identifiability \
#   --family "MoE (gate on M,Q)" \
#   --union_npz /path/to/project/output/fusion_table_noQ/new_fusion_table_union_y2_avp_v1_noQ.npz \
#   --splits_dir /path/to/project/splits \
#   --preds_root /path/to/project/paper_output/unimodal_preds/lr \
#   --q_clean_root /path/to/project/quality \
#   --q_broken_root /path/to/project/paper_output/quality/brokenQ_K200 \
#   --seeds 11 22 33 44 55 \
#   --folds 0 1 2 3 4 \
#   --K 200 \
#   --thresh 0.5 \
#   --require_full_coverage \
#   --router_train_full_only_train \
#   --router_epochs 800 \
#   --router_lr 0.01 \
#   --router_l2 1e-4 \
#   --out_dir /path/to/project/paper_output/reports/moe_lr \
#   --write_tex
#
# Repeat for HGB by pointing preds_root to .../hgb (router still MoE; experts differ only via preds).

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from qfd._shared.q_contract import (
    UnionData,
    FoldQ,
    load_union,
    load_fold_split,
    make_train_test_masks,
    eval_mask_full_only,
    full_only_mask,
    load_fold_q,
    compute_oracle_q_union,
    balanced_accuracy,
    balanced_accuracy_from_probs,
    preds_from_probs,
    assert_probs_nan_where_missing,
)

# -----------------------------
# Utilities (kept aligned)
# -----------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _mean_std(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    return float(np.mean(x)), float(np.std(x, ddof=0))

def _as_1d_prob(x: np.ndarray, name: str, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.shape != (n,):
        raise ValueError(f"{name} must have shape (N,), got {x.shape}")
    return x

def _resolve_pred_keys(z: np.lib.npyio.NpzFile) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    files = set(z.files)

    def pick(keys: List[str], required: bool = True) -> Optional[np.ndarray]:
        for k in keys:
            if k in files:
                return z[k]
        if required:
            raise KeyError(f"Missing required key among {keys}. Found keys={sorted(files)}")
        return None

    y = pick(["y", "y2", "label", "labels"])
    p_a = pick(["p_a", "p_audio", "pa"])
    p_v = pick(["p_v", "p_video", "pv"])
    p_p = pick(["p_p", "p_phys", "pp", "p_physio"])
    return np.asarray(y), np.asarray(p_a), np.asarray(p_v), np.asarray(p_p)

def _load_unimodal_preds_npz(preds_root: Path, seed: int, fold: int, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = preds_root / f"seed_{seed}" / f"fold_{fold}.npz"
    if not p.exists():
        raise FileNotFoundError(f"Missing unimodal preds file: {p}")
    z = np.load(p, allow_pickle=True)
    y, p_a, p_v, p_p = _resolve_pred_keys(z)
    y = np.asarray(y).astype(int)
    p_a = _as_1d_prob(p_a, "p_a", n)
    p_v = _as_1d_prob(p_v, "p_v", n)
    p_p = _as_1d_prob(p_p, "p_p", n)
    return y, p_a, p_v, p_p

def _q_scalar(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        return q
    if q.ndim == 2:
        return np.mean(q, axis=1)
    raise ValueError(f"Q must be 1D or 2D in contract; got {q.shape}")

def _p_trueclass(p_pos: np.ndarray, y: np.ndarray) -> np.ndarray:
    p_pos = np.asarray(p_pos, dtype=float)
    y = np.asarray(y, dtype=int)
    return np.where(y == 1, p_pos, 1.0 - p_pos)

def _competitiveness_delta(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    tc = np.stack([_p_trueclass(P[:, j], y) for j in range(3)], axis=1)  # (n,3)
    srt = np.sort(tc, axis=1)[:, ::-1]
    return srt[:, 0] - srt[:, 1]

def _corr_safe(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mx = np.mean(x); my = np.mean(y)
    vx = np.mean((x - mx) ** 2)
    vy = np.mean((y - my) ** 2)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return float(np.mean((x - mx) * (y - my)) / np.sqrt(vx * vy))

def _alignment_rho_mean(clean_q: FoldQ, y_all: np.ndarray, P_all: np.ndarray, eval_mask: np.ndarray, thresh: float) -> float:
    idx = np.where(eval_mask)[0]
    if idx.size == 0:
        return float("nan")

    y = y_all[idx]
    P = P_all[idx]  # (n,3)
    yhat = (P >= float(thresh)).astype(int)
    c = (yhat == y[:, None]).astype(float)

    Qa = _q_scalar(clean_q.Qa)[idx]
    Qv = _q_scalar(clean_q.Qv)[idx]
    Qp = _q_scalar(clean_q.Qp)[idx]

    r_a = _corr_safe(Qa, c[:, 0])
    r_v = _corr_safe(Qv, c[:, 1])
    r_p = _corr_safe(Qp, c[:, 2])

    rs = [r_a, r_v, r_p]
    rs = [r for r in rs if np.isfinite(r)]
    return float(np.mean(rs)) if rs else float("nan")

def _oracle_best_ba(y: np.ndarray, P: np.ndarray, thresh: float) -> float:
    y = np.asarray(y, dtype=int)
    P = np.asarray(P, dtype=float)
    yhat = (P >= float(thresh)).astype(int)
    correct = (yhat == y[:, None])

    tc = np.stack([_p_trueclass(P[:, j], y) for j in range(3)], axis=1)

    idx_choice = np.zeros(len(y), dtype=int)
    for i in range(len(y)):
        if np.any(correct[i]):
            cand = np.where(correct[i])[0]
            j = cand[np.argmax(tc[i, cand])]
            idx_choice[i] = int(j)
        else:
            idx_choice[i] = int(np.argmax(tc[i]))

    y_oracle = yhat[np.arange(len(y)), idx_choice]
    return float(balanced_accuracy(y, y_oracle))

def _brokenq_iter_columns(Q: np.ndarray, K: int) -> List[np.ndarray]:
    """Return scalar Broken-Q draws from (N,), (N,d), (N,K), or (N,K,d) arrays."""
    Q = np.asarray(Q, dtype=float)
    if Q.ndim == 1:
        return [Q]
    if Q.ndim == 2:
        if Q.shape[1] >= 2 and (K is not None) and (Q.shape[1] == int(K)):
            return [Q[:, k] for k in range(int(K))]
        return [_q_scalar(Q)]
    if Q.ndim == 3:
        if (K is not None) and (Q.shape[1] != int(K)):
            raise ValueError(f"BrokenQ 3D bank has K={Q.shape[1]}, expected {K}: {Q.shape}")
        return [np.mean(Q[:, k, :], axis=1) for k in range(Q.shape[1])]
    raise ValueError(f"BrokenQ must be 1D, 2D, or 3D; got {Q.shape}")

def _perm_pvalue_one_sided(clean_ba: float, broken_bas: np.ndarray) -> float:
    broken_bas = np.asarray(broken_bas, dtype=float)
    K = broken_bas.size
    if K <= 0:
        return float("nan")
    ge = int(np.sum(broken_bas >= float(clean_ba)))
    return float((1 + ge) / (K + 1))

# -----------------------------
# MoE (only new bits vs late fusion)
# -----------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out

def _softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=1, keepdims=True)

def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)

def _logits_from_P(P: np.ndarray, eps: float) -> np.ndarray:
    P = np.asarray(P, dtype=float)  # (n,3)
    return _logit(P, eps=eps)       # (n,3)

def _build_gate_X(M3: np.ndarray, Qa: np.ndarray, Qv: np.ndarray, Qp: np.ndarray) -> np.ndarray:
    """
    Gate features g([M,Q]) with scalar Q per modality:
      X = [M_a, M_v, M_p, Q_a, Q_v, Q_p]  -> (n,6)
    FULL-only eval means M=1, but keep M anyway to match paper definition.
    """
    M3 = np.asarray(M3, dtype=float)
    Qa = np.asarray(Qa, dtype=float)
    Qv = np.asarray(Qv, dtype=float)
    Qp = np.asarray(Qp, dtype=float)
    return np.concatenate([M3, np.stack([Qa, Qv, Qp], axis=1)], axis=1)

class _Router:
    def __init__(self, W: np.ndarray, b: np.ndarray):
        self.W = np.asarray(W, dtype=float)  # (d,3)
        self.b = np.asarray(b, dtype=float)  # (3,)

    def weights(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return _softmax(X @ self.W + self.b[None, :])

def _train_router_adam(
    X: np.ndarray,     # (n,d)
    L: np.ndarray,     # (n,3) fixed expert logits
    y: np.ndarray,     # (n,) {0,1}
    *,
    seed: int,
    epochs: int,
    lr: float,
    l2: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> _Router:
    """
    Train softmax router w=softmax(XW+b) to minimize BCE(sigmoid(sum w*L), y) + l2||W||^2.
    Deterministic given seed.
    """
    X = np.asarray(X, dtype=float)
    L = np.asarray(L, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.shape}")
    if L.shape != (X.shape[0], 3):
        raise ValueError(f"L must be (n,3), got {L.shape}")
    if y.shape != (X.shape[0],):
        raise ValueError(f"y must be (n,), got {y.shape}")

    n, d = X.shape
    rng = np.random.RandomState(int(seed))
    W = 0.01 * rng.randn(d, 3)
    b = np.zeros(3, dtype=float)

    mW = np.zeros_like(W); vW = np.zeros_like(W)
    mb = np.zeros_like(b); vb = np.zeros_like(b)
    t = 0

    for _ in range(int(epochs)):
        t += 1
        z = X @ W + b[None, :]     # (n,3)
        w = _softmax(z)            # (n,3)
        f = np.sum(w * L, axis=1)  # (n,)
        p = _sigmoid(f)            # (n,)

        g_f = (p - y)              # (n,)

        # df/dz_j = w_j * (L_j - f)
        g_z = (g_f[:, None]) * (w * (L - f[:, None]))  # (n,3)

        gW = (X.T @ g_z) / n + 2.0 * l2 * W
        gb = np.sum(g_z, axis=0) / n

        mW = beta1 * mW + (1 - beta1) * gW
        vW = beta2 * vW + (1 - beta2) * (gW * gW)
        mb = beta1 * mb + (1 - beta1) * gb
        vb = beta2 * vb + (1 - beta2) * (gb * gb)

        mW_hat = mW / (1 - beta1 ** t)
        vW_hat = vW / (1 - beta2 ** t)
        mb_hat = mb / (1 - beta1 ** t)
        vb_hat = vb / (1 - beta2 ** t)

        W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
        b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

    return _Router(W=W, b=b)

def _moe_fuse_probs(P: np.ndarray, X_gate: np.ndarray, router: _Router, eps_logit: float) -> np.ndarray:
    """
    MoE fuse:
      L = logit(P)
      w = softmax(XW+b)
      fused_logit = sum_m w_m * L_m
      p = sigmoid(fused_logit)
    """
    P = np.asarray(P, dtype=float)         # (n,3)
    X_gate = np.asarray(X_gate, dtype=float)
    L = _logits_from_P(P, eps=eps_logit)   # (n,3)
    w = router.weights(X_gate)             # (n,3)
    fused = np.sum(w * L, axis=1)          # (n,)
    return _sigmoid(fused)

# -----------------------------
# LaTeX table builders (unchanged)
# -----------------------------

def _fmt_mean_std(x: float, s: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}\\pm{s:.{digits}f}"

def _latex_table_structural(family: str, delta_med: float, rho_mean: float) -> str:
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Structural diagnostics on StressID (FULL-only TEST; 5$\\times$5 CV). "
        "$\\Delta$ is median competitiveness; $\\rho$ is mean quality--correctness alignment.}\n"
        "\\label{tab:struct}\n"
        "\\begin{tabular}{lcc}\n"
        "\\toprule\n"
        "Family & median $\\Delta$ & $\\rho$ \\\\\n"
        "\\midrule\n"
        f"{family} & {delta_med:.3f} & {rho_mean:.3f} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )

def _latex_table_identifiability(family: str, clean_perm_mean: float, clean_perm_std: float, pperm_med: float, oracle_head_mean: float, oracle_head_std: float) -> str:
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Decision-level identifiability on StressID (FULL-only TEST; 5$\\times$5 CV; "
        "K permutations). Columns report Clean--Perm difference, median permutation $p$, and "
        "Oracle--Clean headroom (mean$\\pm$std across folds).}\n"
        "\\label{tab:ident}\n"
        "\\begin{tabular}{lccc}\n"
        "\\toprule\n"
        "Family & Clean--Perm & $p_{\\mathrm{perm}}$ (med) & Oracle--C \\\\\n"
        "\\midrule\n"
        f"{family} & { _fmt_mean_std(clean_perm_mean, clean_perm_std, digits=3) } & {pperm_med:.2f} & { _fmt_mean_std(oracle_head_mean, oracle_head_std, digits=3) } \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )

def _latex_table_baseline(family: str, noq_mean: float, noq_std: float, cleanq_mean: float, cleanq_std: float) -> str:
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Baseline fusion on StressID (FULL-only TEST; 5$\\times$5 CV). "
        "NoQ uses gate inputs without quality; Clean-Q uses fold-scaled quality. Values are mean$\\pm$std.}\n"
        "\\label{tab:baseline}\n"
        "\\begin{tabular}{lcc}\n"
        "\\toprule\n"
        "Family & NoQ & Clean-Q \\\\\n"
        "\\midrule\n"
        f"{family} & { _fmt_mean_std(noq_mean, noq_std, digits=3) } & { _fmt_mean_std(cleanq_mean, cleanq_std, digits=3) } \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )

# -----------------------------
# Main evaluation (same skeleton)
# -----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", type=str, required=True, help="Row label for tables, e.g., MoE (gate on M,Q)")

    ap.add_argument("--union_npz", type=str, required=True)
    ap.add_argument("--splits_dir", type=str, required=True)
    ap.add_argument("--preds_root", type=str, required=True)

    ap.add_argument("--q_clean_root", type=str, required=True)
    ap.add_argument("--q_broken_root", type=str, required=True)

    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--folds", type=int, nargs="+", required=True)

    ap.add_argument("--K", type=int, default=200)
    ap.add_argument("--thresh", type=float, default=0.5)

    ap.add_argument("--require_full_coverage", action="store_true")

    # Router training control (train once per fold; reused unchanged)
    ap.add_argument("--router_train_full_only_train", action="store_true",
                    help="Train router on TRAIN ∩ FULL-only rows (recommended). Else TRAIN only.")
    ap.add_argument("--router_epochs", type=int, default=800)
    ap.add_argument("--router_lr", type=float, default=0.01)
    ap.add_argument("--router_l2", type=float, default=1e-4)
    ap.add_argument("--eps_logit", type=float, default=1e-6)

    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--write_tex", action="store_true")

    return ap.parse_args()

def main() -> None:
    args = parse_args()

    union = load_union(args.union_npz)
    N = len(union.ids_str)

    preds_root = Path(args.preds_root)
    q_clean_root = Path(args.q_clean_root)
    q_broken_root = Path(args.q_broken_root)
    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    rows: List[Dict] = []

    bas_noq: List[float] = []
    bas_cleanq: List[float] = []

    clean_minus_perm: List[float] = []
    pperm_folds: List[float] = []
    oracle_minus_clean: List[float] = []

    delta_medians: List[float] = []
    rho_means: List[float] = []

    # Router cache on disk (same out_dir, deterministic)
    router_dir = out_dir / "routers"
    _ensure_dir(router_dir)

    for seed in args.seeds:
        for fold in args.folds:
            split = load_fold_split(args.splits_dir, seed=seed, fold=fold)
            train_mask, test_mask = make_train_test_masks(union, split, require_full_coverage=bool(args.require_full_coverage))
            eval_mask = eval_mask_full_only(union, test_mask)
            idx_eval = np.where(eval_mask)[0]
            if idx_eval.size == 0:
                raise RuntimeError(f"seed={seed} fold={fold}: FULL-only TEST is empty.")

            # Unimodal preds (UNION-aligned)
            y_all, p_a_all, p_v_all, p_p_all = _load_unimodal_preds_npz(preds_root, seed, fold, N)
            assert_probs_nan_where_missing(union, p_a_all, p_v_all, p_p_all)

            # FULL-only TEST slice probs
            y = y_all[idx_eval]
            P = np.stack([p_a_all[idx_eval], p_v_all[idx_eval], p_p_all[idx_eval]], axis=1)  # (n,3)
            if not np.isfinite(P).all():
                bad = np.where(~np.isfinite(P).any(axis=1))[0][:10]
                raise ValueError(f"seed={seed} fold={fold}: non-finite probs in FULL-only TEST slice. Example rows={bad.tolist()}")

            # Load Q (contract checks inside)
            clean_q = load_fold_q(q_clean_root, seed=seed, fold=fold, union=union, require_ids_match=True, allow_hash_match=True)
            broken_q = load_fold_q(q_broken_root, seed=seed, fold=fold, union=union, require_ids_match=True, allow_hash_match=True)

            # Router training mask: TRAIN (optionally intersect FULL-only)
            if args.router_train_full_only_train:
                train_router_mask = full_only_mask(union.Ma, union.Mv, union.Mp, base_mask=train_mask)
            else:
                train_router_mask = train_mask

            idx_tr = np.where(train_router_mask)[0]
            if idx_tr.size == 0:
                raise RuntimeError(f"seed={seed} fold={fold}: router train subset empty.")

            # Router train data:
            # P_tr: need (ntr,3) probs; FULL-only TRAIN ensures finite; otherwise still safe if present missing -> NaN.
            P_tr_all = np.stack([p_a_all, p_v_all, p_p_all], axis=1)  # (N,3)
            P_tr = P_tr_all[idx_tr]
            if not np.isfinite(P_tr).all():
                # If you didn't train on FULL-only, this would happen. Fail loud.
                raise ValueError(
                    f"seed={seed} fold={fold}: router training subset contains non-finite probs. "
                    f"Use --router_train_full_only_train to match paper and avoid missing probs."
                )

            y_tr = y_all[idx_tr].astype(int)

            # Gate inputs: M + Q (scalar per modality)
            M3_all = np.stack([union.Ma, union.Mv, union.Mp], axis=1).astype(float)  # (N,3)

            Qa_c_all = _q_scalar(clean_q.Qa)
            Qv_c_all = _q_scalar(clean_q.Qv)
            Qp_c_all = _q_scalar(clean_q.Qp)

            X_tr_clean = _build_gate_X(M3_all[idx_tr], Qa_c_all[idx_tr], Qv_c_all[idx_tr], Qp_c_all[idx_tr])
            L_tr = _logits_from_P(P_tr, eps=args.eps_logit)

            # Train (or load) router once per fold under Clean-Q
            router_path = router_dir / f"router_seed{seed}_fold{fold}.npz"
            if router_path.exists():
                rz = np.load(router_path, allow_pickle=True)
                router = _Router(W=rz["W"], b=rz["b"])
                router_status = "LOADED"
            else:
                router_seed = int(seed) * 100 + int(fold)
                router = _train_router_adam(
                    X=X_tr_clean,
                    L=L_tr,
                    y=y_tr,
                    seed=router_seed,
                    epochs=int(args.router_epochs),
                    lr=float(args.router_lr),
                    l2=float(args.router_l2),
                )
                np.savez_compressed(
                    router_path,
                    W=router.W,
                    b=router.b,
                    meta={
                        "seed": int(seed),
                        "fold": int(fold),
                        "router_seed": int(router_seed),
                        "epochs": int(args.router_epochs),
                        "lr": float(args.router_lr),
                        "l2": float(args.router_l2),
                        "router_train_full_only_train": bool(args.router_train_full_only_train),
                        "n_train": int(idx_tr.size),
                        "preds_root": str(preds_root),
                        "q_clean_file": str(clean_q.q_path),
                    },
                    ids=union.ids_str,
                )
                router_status = "TRAINED_SAVED"

            # Build eval gate inputs for NoQ/CleanQ/OracleQ/BrokenQ
            M3_eval = M3_all[idx_eval]

            # NoQ = same router, but set Q inputs to 0 (only M)
            X_eval_noq = _build_gate_X(M3_eval, np.zeros_like(y, dtype=float), np.zeros_like(y, dtype=float), np.zeros_like(y, dtype=float))
            p_noq = _moe_fuse_probs(P, X_eval_noq, router, eps_logit=args.eps_logit)
            ba_noq = balanced_accuracy_from_probs(y, p_noq, thresh=args.thresh)

            # CleanQ
            Qa_c = Qa_c_all[idx_eval]
            Qv_c = Qv_c_all[idx_eval]
            Qp_c = Qp_c_all[idx_eval]
            X_eval_clean = _build_gate_X(M3_eval, Qa_c, Qv_c, Qp_c)
            p_clean = _moe_fuse_probs(P, X_eval_clean, router, eps_logit=args.eps_logit)
            ba_clean = balanced_accuracy_from_probs(y, p_clean, thresh=args.thresh)

            # BrokenQ permutations (must be precomputed offline; here only load + apply)
            Qa_cols = _brokenq_iter_columns(broken_q.Qa, K=int(args.K))
            Qv_cols = _brokenq_iter_columns(broken_q.Qv, K=int(args.K))
            Qp_cols = _brokenq_iter_columns(broken_q.Qp, K=int(args.K))
            K_eff = min(len(Qa_cols), len(Qv_cols), len(Qp_cols))

            broken_bas: List[float] = []
            for k in range(K_eff):
                Qa_b_all = np.asarray(Qa_cols[k], dtype=float)
                Qv_b_all = np.asarray(Qv_cols[k], dtype=float)
                Qp_b_all = np.asarray(Qp_cols[k], dtype=float)

                Qa_b = Qa_b_all[idx_eval]
                Qv_b = Qv_b_all[idx_eval]
                Qp_b = Qp_b_all[idx_eval]

                X_eval_b = _build_gate_X(M3_eval, Qa_b, Qv_b, Qp_b)
                p_b = _moe_fuse_probs(P, X_eval_b, router, eps_logit=args.eps_logit)
                broken_bas.append(balanced_accuracy_from_probs(y, p_b, thresh=args.thresh))

            broken_bas = np.asarray(broken_bas, dtype=float)
            ba_perm_mean = float(np.mean(broken_bas)) if broken_bas.size else float("nan")
            delta_perm_fold = float(ba_clean - ba_perm_mean)
            p_perm_fold = _perm_pvalue_one_sided(ba_clean, broken_bas)

            # OracleQ (from THIS RUN's unimodal preds on FULL-only TEST)
            oracle_q = compute_oracle_q_union(union, eval_mask, p_a_all, p_v_all, p_p_all, thresh=args.thresh)
            Qa_o_all = _q_scalar(oracle_q.Qa)
            Qv_o_all = _q_scalar(oracle_q.Qv)
            Qp_o_all = _q_scalar(oracle_q.Qp)
            X_eval_oracle = _build_gate_X(M3_eval, Qa_o_all[idx_eval], Qv_o_all[idx_eval], Qp_o_all[idx_eval])
            p_oracleq = _moe_fuse_probs(P, X_eval_oracle, router, eps_logit=args.eps_logit)
            ba_oracleq = balanced_accuracy_from_probs(y, p_oracleq, thresh=args.thresh)

            # Keep your paper's oracle-best ceiling for headroom (routing opportunity)
            ba_oracle_best = _oracle_best_ba(y, P, thresh=args.thresh)
            headroom = float(ba_oracle_best - ba_clean)

            # Structural diagnostics (same definitions)
            delta_i = _competitiveness_delta(P, y)
            delta_med = float(np.median(delta_i))
            rho_mean = _alignment_rho_mean(clean_q, y_all, P_tr_all, eval_mask, thresh=args.thresh)

            # collect
            bas_noq.append(float(ba_noq))
            bas_cleanq.append(float(ba_clean))
            clean_minus_perm.append(delta_perm_fold)
            pperm_folds.append(float(p_perm_fold))
            oracle_minus_clean.append(headroom)
            delta_medians.append(delta_med)
            rho_means.append(float(rho_mean))

            rec = {
                "family": args.family,
                "seed": int(seed),
                "fold": int(fold),
                "router_status": router_status,
                "router_path": str(router_path.resolve()),
                "n_full_only_test": int(idx_eval.size),
                "paths": {
                    "preds": str((preds_root / f"seed_{seed}" / f"fold_{fold}.npz").resolve()),
                    "q_clean_root": str(q_clean_root.resolve()),
                    "q_broken_root": str(q_broken_root.resolve()),
                    "q_clean_file": str(clean_q.q_path),
                    "q_broken_file": str(broken_q.q_path),
                },
                "metrics": {
                    "BA_noq": float(ba_noq),
                    "BA_cleanq": float(ba_clean),
                    "BA_broken_mean": float(ba_perm_mean),
                    "BA_oracleq": float(ba_oracleq),
                    "BA_oracle_best": float(ba_oracle_best),
                    "delta_perm": float(delta_perm_fold),
                    "p_perm_one_sided": float(p_perm_fold),
                    "oracle_minus_clean": float(headroom),
                    "median_competitiveness_delta": float(delta_med),
                    "rho_alignment_mean": float(rho_mean),
                    "K_eff_used": int(K_eff),
                },
                "oracleq_audit": {"oracle_q_path": oracle_q.q_path},
            }
            rows.append(rec)

            print(
                f"[OK] {args.family} seed={seed} fold={fold} | router={router_status} | "
                f"NoQ={ba_noq:.3f} CleanQ={ba_clean:.3f} Broken(mean)={ba_perm_mean:.3f} "
                f"Δperm={delta_perm_fold:+.3f} pperm={p_perm_fold:.3f} headroom={headroom:.3f} "
                f"Δmed={delta_med:.3f} rho={rho_mean:.3f} (Keff={K_eff})"
            )

    # Aggregate across folds
    noq_mean, noq_std = _mean_std(np.asarray(bas_noq))
    clean_mean, clean_std = _mean_std(np.asarray(bas_cleanq))

    dperm_mean, dperm_std = _mean_std(np.asarray(clean_minus_perm))
    pperm_med = float(np.median(np.asarray(pperm_folds, dtype=float)))
    head_mean, head_std = _mean_std(np.asarray(oracle_minus_clean))

    delta_med_over_folds = float(np.median(np.asarray(delta_medians, dtype=float)))
    rho_mean_over_folds = float(np.mean(np.asarray(rho_means, dtype=float)))

    # Write JSON + CSV (same schema as late fusion, plus router fields)
    out_json = out_dir / f"moe_identifiability_{args.family}.json"
    with open(out_json, "w") as f:
        json.dump(
            {
                "family": args.family,
                "args": vars(args),
                "n_folds": len(rows),
                "aggregate": {
                    "baseline": {
                        "NoQ_mean": noq_mean, "NoQ_std": noq_std,
                        "CleanQ_mean": clean_mean, "CleanQ_std": clean_std,
                    },
                    "identifiability": {
                        "CleanMinusPerm_mean": dperm_mean,
                        "CleanMinusPerm_std": dperm_std,
                        "p_perm_median": pperm_med,
                        "OracleMinusClean_mean": head_mean,
                        "OracleMinusClean_std": head_std,
                    },
                    "structural": {
                        "medianDelta_medianOverFolds": delta_med_over_folds,
                        "rho_meanOverFolds": rho_mean_over_folds,
                    },
                },
                "fold_rows": rows,
            },
            f,
            indent=2,
        )

    out_csv = out_dir / f"moe_identifiability_{args.family}.csv"
    with open(out_csv, "w") as f:
        cols = [
            "family","seed","fold","n_full_only_test",
            "BA_noq","BA_cleanq","BA_broken_mean","BA_oracleq","BA_oracle_best",
            "delta_perm","p_perm_one_sided","oracle_minus_clean",
            "median_competitiveness_delta","rho_alignment_mean","K_eff_used",
            "router_status","router_path",
        ]
        f.write(",".join(cols) + "\n")
        for r in rows:
            m = r["metrics"]
            vals = [
                r["family"], str(r["seed"]), str(r["fold"]), str(r["n_full_only_test"]),
                f"{m['BA_noq']:.6f}", f"{m['BA_cleanq']:.6f}", f"{m['BA_broken_mean']:.6f}",
                f"{m['BA_oracleq']:.6f}", f"{m['BA_oracle_best']:.6f}",
                f"{m['delta_perm']:.6f}", f"{m['p_perm_one_sided']:.6f}", f"{m['oracle_minus_clean']:.6f}",
                f"{m['median_competitiveness_delta']:.6f}", f"{m['rho_alignment_mean']:.6f}", str(m["K_eff_used"]),
                r["router_status"], r["router_path"],
            ]
            f.write(",".join(vals) + "\n")

    # LaTeX tables
    tex_struct = _latex_table_structural(args.family, delta_med_over_folds, rho_mean_over_folds)
    tex_ident = _latex_table_identifiability(args.family, dperm_mean, dperm_std, pperm_med, head_mean, head_std)
    tex_base = _latex_table_baseline(args.family, noq_mean, noq_std, clean_mean, clean_std)

    print("\n% ===============================")
    print("% LaTeX tables (copy/paste)")
    print("% ===============================\n")
    print(tex_struct)
    print(tex_ident)
    print(tex_base)

    if args.write_tex:
        (out_dir / f"Table_structural_{args.family}.tex").write_text(tex_struct)
        (out_dir / f"Table_identifiability_{args.family}.tex").write_text(tex_ident)
        (out_dir / f"Table_baseline_{args.family}.tex").write_text(tex_base)

    print(f"[DONE] wrote: {out_json}")
    print(f"[DONE] wrote: {out_csv}")

if __name__ == "__main__":
    main()

