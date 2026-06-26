#!/usr/bin/env python3
# FINAL_EXPERIMENTS/FINAL/v5_2/02_late_fusion_identifiability_tables.py
#
# Paper-ready, contract-aligned late-fusion identifiability evaluation + LaTeX tables.
#
# Produces (for one "family" at a time, e.g., LR or HGB):
#  - Table (Structural diagnostics): median competitiveness Δ, mean alignment ρ
#  - Table (Decision identifiability): Clean−Perm (mean±std), p_perm (median), Oracle−Clean headroom (mean±std)
#  - Table (Baseline fusion): NoQ and Clean-Q (mean±std)
#
# Contract alignment:
#  - UNION row order canonical; never reorder.
#  - Splits loaded from disk; never resplit.
#  - FULL-only TEST evaluation only (availability constant).
#  - CleanQ and BrokenQ loaded via q_contract.load_fold_q() (ids/hash checked).
#  - BrokenQ must be precomputed offline and stored as (N,K) (or (N,) for single perm).
#  - OracleQ computed from THIS RUN's unimodal preds on FULL-only TEST.
#
# Example (LR):
# export PYTHONPATH=/path/to/project:$PYTHONPATH

# # 3) Run for LR
# python -m qfd.stressid.late_fusion_identifiability \
#   --family LR \
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
#   --out_dir /path/to/project/paper_output/reports/latefusion_lr \
#   --write_tex

# # 4) Run for HGB
# python -m qfd.stressid.late_fusion_identifiability \
#   --family HGB \
#   --union_npz /path/to/project/output/fusion_table_noQ/new_fusion_table_union_y2_avp_v1_noQ.npz \
#   --splits_dir /path/to/project/splits \
#   --preds_root /path/to/project/paper_output/unimodal_preds/hgb \
#   --q_clean_root /path/to/project/quality \
#   --q_broken_root /path/to/project/paper_output/quality/brokenQ_K200 \
#   --seeds 11 22 33 44 55 \
#   --folds 0 1 2 3 4 \
#   --K 200 \
#   --thresh 0.5 \
#   --require_full_coverage \
#   --out_dir /path/to/project/paper_output/reports/latefusion_hgb \
#   --write_tex
#
# Repeat with --family HGB and preds_root .../hgb (and a different out_dir).

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
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
    load_fold_q,
    compute_oracle_q_union,
    balanced_accuracy,
    balanced_accuracy_from_probs,
    preds_from_probs,
    assert_probs_nan_where_missing,
)

# -----------------------------
# Utilities
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
    """
    Robust key resolution for unimodal dump NPZ.
    Expected keys from your script: p_a, p_v, p_p, y.
    """
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
    """
    Convert Q to a scalar per row for weighting.
    Allowed inputs per contract: (N,) or (N,d) or (N,K) (BrokenQ permutations).
    - (N,)   -> (N,)
    - (N,d)  -> mean over d -> (N,)
    - (N,K)  -> treat as permutations, handled elsewhere
    """
    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        return q
    if q.ndim == 2:
        # Ambiguous between (N,d) and (N,K).
        # Here we use this only for "CleanQ" (where (N,d) is plausible),
        # and we reduce over axis=1.
        return np.mean(q, axis=1)
    raise ValueError(f"Q must be 1D or 2D in contract; got {q.shape}")

def _safe_row_normalize_weights(W: np.ndarray) -> np.ndarray:
    """
    Normalize per-row weights. If a row sum is 0, fallback to uniform weights.
    W shape: (n,3)
    """
    W = np.asarray(W, dtype=float)
    s = np.sum(W, axis=1, keepdims=True)
    out = W.copy()
    bad = (s[:, 0] <= 0) | ~np.isfinite(s[:, 0])
    out[~bad] = out[~bad] / s[~bad]
    if np.any(bad):
        out[bad] = 1.0 / 3.0
    return out

def _late_fusion_noq(P: np.ndarray) -> np.ndarray:
    """
    Uniform averaging of posteriors. P shape: (n,3), finite.
    Returns p_fused shape (n,)
    """
    return np.mean(P, axis=1)

def _late_fusion_quality_weighted(P: np.ndarray, Qa: np.ndarray, Qv: np.ndarray, Qp: np.ndarray) -> np.ndarray:
    """
    Quality-weighted averaging. Inputs are scalar per row. P finite.
    """
    W = np.stack([Qa, Qv, Qp], axis=1)  # (n,3)
    W = _safe_row_normalize_weights(W)
    return np.sum(W * P, axis=1)

def _p_trueclass(p_pos: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    True-class confidence for binary probs.
    """
    p_pos = np.asarray(p_pos, dtype=float)
    y = np.asarray(y, dtype=int)
    return np.where(y == 1, p_pos, 1.0 - p_pos)

def _competitiveness_delta(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Per-sample competitiveness Δ_i = top1(trueclass) - top2(trueclass) over experts.
    P shape (n,3) probs of positive class.
    """
    tc = np.stack([_p_trueclass(P[:, j], y) for j in range(3)], axis=1)  # (n,3)
    # sort descending
    srt = np.sort(tc, axis=1)[:, ::-1]
    return srt[:, 0] - srt[:, 1]

def _corr_safe(x: np.ndarray, y: np.ndarray) -> float:
    """
    Pearson corr with safe fallbacks (returns nan if degenerate).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mx = np.mean(x); my = np.mean(y)
    vx = np.mean((x - mx) ** 2)
    vy = np.mean((y - my) ** 2)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return float(np.mean((x - mx) * (y - my)) / np.sqrt(vx * vy))

def _alignment_rho_mean(clean_q: FoldQ, union: UnionData, y: np.ndarray, P: np.ndarray, eval_mask: np.ndarray, thresh: float) -> float:
    """
    ρ = mean over modalities of corr(Q_m, 1[yhat_m == y]) on FULL-only TEST.
    Q scalarized deterministically.
    """
    idx = np.where(eval_mask)[0]
    if idx.size == 0:
        return float("nan")

    # Predictions per modality on eval subset
    y_eval = y[idx]
    p_eval = P[idx]  # (n,3)
    yhat = (p_eval >= float(thresh)).astype(int)  # (n,3)

    # correctness indicators
    c = (yhat == y_eval[:, None]).astype(float)  # (n,3)

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
    """
    Non-deployable oracle ceiling: per-sample choose a correct expert if any exists,
    else choose the expert with highest true-class confidence.
    Returns Balanced Accuracy using the oracle-chosen hard prediction.
    """
    y = np.asarray(y, dtype=int)
    P = np.asarray(P, dtype=float)  # (n,3)
    yhat = (P >= float(thresh)).astype(int)  # (n,3)
    correct = (yhat == y[:, None])  # (n,3)

    # true-class confidence per expert
    tc = np.stack([_p_trueclass(P[:, j], y) for j in range(3)], axis=1)  # (n,3)

    # Choose index:
    #  - if any correct -> choose the correct expert with max tc (deterministic)
    #  - else -> choose max tc
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
    """
    Return list of scalar-Q vectors (N,) from a BrokenQ array:
      - (N,) or (N,d): single Broken-Q draw
      - (N,K): K scalar draws
      - (N,K,d): K vector-valued draws, scalarized by mean over d
    """
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
    """
    One-sided permutation p-value for H1: Clean > Broken.
    p = (1 + #{broken >= clean}) / (K + 1)
    """
    broken_bas = np.asarray(broken_bas, dtype=float)
    K = broken_bas.size
    if K <= 0:
        return float("nan")
    ge = int(np.sum(broken_bas >= float(clean_ba)))
    return float((1 + ge) / (K + 1))

# -----------------------------
# LaTeX table builders
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
        "\\caption{Baseline late fusion on StressID (FULL-only TEST; 5$\\times$5 CV). "
        "NoQ uses posteriors only; Clean-Q uses fold-scaled quality as weights. Values are mean$\\pm$std.}\n"
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
# Main evaluation
# -----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", type=str, required=True, help="Row label for tables, e.g., LR or HGB")

    ap.add_argument("--union_npz", type=str, required=True)
    ap.add_argument("--splits_dir", type=str, required=True)
    ap.add_argument("--preds_root", type=str, required=True)

    ap.add_argument("--q_clean_root", type=str, required=True)
    ap.add_argument("--q_broken_root", type=str, required=True)

    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--folds", type=int, nargs="+", required=True)

    ap.add_argument("--K", type=int, default=200, help="Expected number of BrokenQ permutations (columns) if stored as (N,K)")
    ap.add_argument("--thresh", type=float, default=0.5)

    ap.add_argument("--require_full_coverage", action="store_true")

    ap.add_argument("--out_dir", type=str, required=True, help="Directory to write tables + per-fold JSON/CSV")
    ap.add_argument("--write_tex", action="store_true", help="Write .tex files in out_dir (also always prints to stdout)")

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

    # Per-fold records
    rows: List[Dict] = []

    # Aggregates (across 25 folds)
    bas_noq: List[float] = []
    bas_cleanq: List[float] = []

    clean_minus_perm: List[float] = []
    pperm_folds: List[float] = []
    oracle_minus_clean: List[float] = []

    delta_medians: List[float] = []
    rho_means: List[float] = []

    for seed in args.seeds:
        for fold in args.folds:
            split = load_fold_split(args.splits_dir, seed=seed, fold=fold)
            _, test_mask = make_train_test_masks(union, split, require_full_coverage=bool(args.require_full_coverage))
            eval_mask = eval_mask_full_only(union, test_mask)
            idx = np.where(eval_mask)[0]
            if idx.size == 0:
                raise RuntimeError(f"seed={seed} fold={fold}: FULL-only TEST is empty.")

            # Unimodal preds (UNION-aligned)
            y_all, p_a_all, p_v_all, p_p_all = _load_unimodal_preds_npz(preds_root, seed, fold, N)
            # Global contract check (even though we evaluate FULL-only)
            assert_probs_nan_where_missing(union, p_a_all, p_v_all, p_p_all)

            # Eval slice (FULL-only TEST)
            y = y_all[idx]
            P = np.stack([p_a_all[idx], p_v_all[idx], p_p_all[idx]], axis=1)  # (n,3)
            if not np.isfinite(P).all():
                bad = np.where(~np.isfinite(P).any(axis=1))[0][:10]
                raise ValueError(f"seed={seed} fold={fold}: non-finite probs in FULL-only TEST slice. Example rows={bad.tolist()}")

            # Load Q (contract checks inside)
            clean_q = load_fold_q(q_clean_root, seed=seed, fold=fold, union=union, require_ids_match=True, allow_hash_match=True)
            broken_q = load_fold_q(q_broken_root, seed=seed, fold=fold, union=union, require_ids_match=True, allow_hash_match=True)

            # Scalarize CleanQ (deterministic)
            Qa_c = _q_scalar(clean_q.Qa)[idx]
            Qv_c = _q_scalar(clean_q.Qv)[idx]
            Qp_c = _q_scalar(clean_q.Qp)[idx]

            # Baselines
            p_noq = _late_fusion_noq(P)
            ba_noq = balanced_accuracy_from_probs(y, p_noq, thresh=args.thresh)

            p_clean = _late_fusion_quality_weighted(P, Qa_c, Qv_c, Qp_c)
            ba_clean = balanced_accuracy_from_probs(y, p_clean, thresh=args.thresh)

            # BrokenQ permutations
            Qa_cols = _brokenq_iter_columns(broken_q.Qa, K=int(args.K))
            Qv_cols = _brokenq_iter_columns(broken_q.Qv, K=int(args.K))
            Qp_cols = _brokenq_iter_columns(broken_q.Qp, K=int(args.K))
            K_eff = min(len(Qa_cols), len(Qv_cols), len(Qp_cols))

            broken_bas: List[float] = []
            for k in range(K_eff):
                Qa_b = np.asarray(Qa_cols[k], dtype=float)[idx]
                Qv_b = np.asarray(Qv_cols[k], dtype=float)[idx]
                Qp_b = np.asarray(Qp_cols[k], dtype=float)[idx]
                p_b = _late_fusion_quality_weighted(P, Qa_b, Qv_b, Qp_b)
                broken_bas.append(balanced_accuracy_from_probs(y, p_b, thresh=args.thresh))

            broken_bas = np.asarray(broken_bas, dtype=float)
            ba_perm_mean = float(np.mean(broken_bas)) if broken_bas.size else float("nan")
            delta_perm_fold = float(ba_clean - ba_perm_mean)
            p_perm_fold = _perm_pvalue_one_sided(ba_clean, broken_bas)

            # Oracle ceiling + headroom
            oracle_q = compute_oracle_q_union(union, eval_mask, p_a_all, p_v_all, p_p_all, thresh=args.thresh)
            ba_oracle = _oracle_best_ba(y, P, thresh=args.thresh)
            headroom = float(ba_oracle - ba_clean)

            # Structural diagnostics
            delta_i = _competitiveness_delta(P, y)
            delta_med = float(np.median(delta_i))
            rho_mean = _alignment_rho_mean(clean_q, union, y_all, np.stack([p_a_all, p_v_all, p_p_all], axis=1), eval_mask, thresh=args.thresh)

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
                "n_full_only_test": int(idx.size),
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
                    "BA_oracle_best": float(ba_oracle),
                    "delta_perm": float(delta_perm_fold),
                    "p_perm_one_sided": float(p_perm_fold),
                    "oracle_minus_clean": float(headroom),
                    "median_competitiveness_delta": float(delta_med),
                    "rho_alignment_mean": float(rho_mean),
                    "K_eff_used": int(K_eff),
                },
                "oracleq_audit": {
                    "oracle_q_path": oracle_q.q_path,
                },
            }
            rows.append(rec)

            print(
                f"[OK] {args.family} seed={seed} fold={fold} | "
                f"NoQ={ba_noq:.3f} CleanQ={ba_clean:.3f} Broken(mean)={ba_perm_mean:.3f} "
                f"Δperm={delta_perm_fold:+.3f} pperm={p_perm_fold:.3f} headroom={headroom:.3f} "
                f"Δmed={delta_med:.3f} rho={rho_mean:.3f} (Keff={K_eff})"
            )

    # Aggregate across folds (25)
    noq_mean, noq_std = _mean_std(np.asarray(bas_noq))
    clean_mean, clean_std = _mean_std(np.asarray(bas_cleanq))

    dperm_mean, dperm_std = _mean_std(np.asarray(clean_minus_perm))
    pperm_med = float(np.median(np.asarray(pperm_folds, dtype=float)))

    head_mean, head_std = _mean_std(np.asarray(oracle_minus_clean))

    # Structural table expects scalar summaries
    delta_med_over_folds = float(np.median(np.asarray(delta_medians, dtype=float)))
    rho_mean_over_folds = float(np.mean(np.asarray(rho_means, dtype=float)))

    # Write machine-readable outputs
    out_json = out_dir / f"latefusion_identifiability_{args.family}.json"
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

    # Also a compact CSV (no pandas dependency)
    out_csv = out_dir / f"latefusion_identifiability_{args.family}.csv"
    with open(out_csv, "w") as f:
        cols = [
            "family","seed","fold","n_full_only_test",
            "BA_noq","BA_cleanq","BA_broken_mean","BA_oracle_best",
            "delta_perm","p_perm_one_sided","oracle_minus_clean",
            "median_competitiveness_delta","rho_alignment_mean","K_eff_used",
        ]
        f.write(",".join(cols) + "\n")
        for r in rows:
            m = r["metrics"]
            vals = [
                r["family"], str(r["seed"]), str(r["fold"]), str(r["n_full_only_test"]),
                f"{m['BA_noq']:.6f}", f"{m['BA_cleanq']:.6f}", f"{m['BA_broken_mean']:.6f}", f"{m['BA_oracle_best']:.6f}",
                f"{m['delta_perm']:.6f}", f"{m['p_perm_one_sided']:.6f}", f"{m['oracle_minus_clean']:.6f}",
                f"{m['median_competitiveness_delta']:.6f}", f"{m['rho_alignment_mean']:.6f}", str(m["K_eff_used"]),
            ]
            f.write(",".join(vals) + "\n")

    # Build LaTeX tables (single-row, per-family)
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