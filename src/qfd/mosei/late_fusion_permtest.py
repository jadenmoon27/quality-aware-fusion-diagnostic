#!/usr/bin/env python3
"""
mosei_step5_late_fusion_permtest_clean_vs_broken.py

MOSEI STEP 5 — Late fusion clean vs K-permutation brokenQ (permtest bank)

Defensible protocol (FULL-only TEST):
- Train fuser ONCE per (seed, fold) using cleanQ features on TRAIN_FULL.
- Evaluate on TEST_FULL:
    * NoQ: arithmetic mean of unimodal probs (no training)
    * cleanQ: trained model + cleanQ features at test time
    * brokenQ_k: SAME trained model + brokenQ features loaded from a precomputed bank
    * oracle-best: per-sample best unimodal by log-loss (diagnostic ceiling)
- Permutation p-value per fold:
    p = (1 + #{k: brokenQ_k >= cleanQ}) / (K + 1)
  (one-sided, tests whether cleanQ > brokenQ under broken controls)

Outputs:
- CSV: per (seed, fold, fuser) cleanQ, NoQ, oracle-best, broken distribution stats, p_perm, flip stats.
- JSON: mean±std across folds + mean gaps + p-value summaries + audit metadata.

Hard requirements:
- Uses q_contract_mosei.py for UNION/splits/Q loading + FULL-only masking + id alignment.
- Does NOT generate brokenQ; loads q_perm_parent as one K-bank, with q_perm_parent/perm_### as a legacy fallback.
- Works with Q arrays 1D or 2D after selecting each permutation; converts to scalar per row via mean over dim if 2D.

Run example:
python python -m qfd.mosei.late_fusion_permtest \
  --union_npz output/final_experiments/mosei/union/mosei_union.npz \
  --splits_dir output/final_experiments/mosei/splits_mosei \
  --unimodal_root output/final_experiments/mosei/unimodal_preds/lr \
  --q_clean_root output/final_experiments/mosei/quality_fold \
  --q_perm_parent output/final_experiments/mosei/quality_broken_permtest_fullonly_mosei \
  --K 100 \
  --out_csv output/final_experiments/mosei/reports/5/late_fusion_permtest.csv \
  --out_json output/final_experiments/mosei/reports/5/late_fusion_permtest.json \
  --seeds 11 22 33 44 55 --folds 0 1 2 3 4 \
  --thresh 0.5 \
  --fusers lr hgb \
  --check_perm_ids 1 2 3 100

Notes:
- broken mean/std are within-fold across K, then you can aggregate across folds.
- flip stats:
    * flip_clean_vs_broken_mean: mean_k flip(clean_pred, broken_pred_k)
    * flip_clean_vs_broken_max: max_k flip(...)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


# ----------------------------
# Import q_contract_mosei (single source of truth)
# ----------------------------

def _import_qc() -> object:
    """Import the packaged MOSEI contract. Kept as a function for CLI compatibility."""
    from qfd._shared import q_contract_mosei as qc  # type: ignore
    return qc


qc = _import_qc()


# ----------------------------
# Metrics
# ----------------------------

def preds_from_probs(p: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    return (np.asarray(p) >= thresh).astype(int)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_pred = np.asarray(y_pred).astype(int).reshape(-1)
    pos = y_true == 1
    neg = y_true == 0
    tpr = float(np.mean(y_pred[pos] == 1)) if np.any(pos) else float("nan")
    tnr = float(np.mean(y_pred[neg] == 0)) if np.any(neg) else float("nan")
    if not np.isfinite(tpr) or not np.isfinite(tnr):
        return float("nan")
    return float(0.5 * (tpr + tnr))


def flip_rate(y1: np.ndarray, y2: np.ndarray) -> float:
    y1 = np.asarray(y1).astype(int).reshape(-1)
    y2 = np.asarray(y2).astype(int).reshape(-1)
    if y1.shape != y2.shape:
        raise ValueError("flip_rate: shape mismatch")
    return float(np.mean(y1 != y2))


def _mean_std(xs: List[float]) -> Dict[str, float]:
    a = np.asarray(xs, dtype=float)
    return {"mean": float(np.mean(a)), "std": float(np.std(a, ddof=0)), "n": int(a.size)}


def _mean_std_ddof1(xs: List[float]) -> Dict[str, float]:
    a = np.asarray(xs, dtype=float)
    if a.size <= 1:
        return {"mean": float(np.mean(a)), "std": float("nan"), "n": int(a.size)}
    return {"mean": float(np.mean(a)), "std": float(np.std(a, ddof=1)), "n": int(a.size)}


# ----------------------------
# Loaders (unimodal probs)
# ----------------------------

def _pick_npz(z: np.lib.npyio.NpzFile, cands: List[str], path: Path) -> np.ndarray:
    for k in cands:
        if k in z.files:
            return z[k]
    raise KeyError(f"{path}: missing any of {cands}. Found {sorted(z.files)}")


def load_unimodal_probs(unimodal_root: Path, seed: int, fold: int, union_ids: np.ndarray) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    p = unimodal_root / f"seed_{seed}" / f"fold_{fold}.npz"
    if not p.exists():
        raise FileNotFoundError(p)
    z = np.load(p, allow_pickle=True)

    ids = _pick_npz(z, ["ids", "id", "union_ids", "ids_str"], p).astype(str).reshape(-1)
    if ids.shape != union_ids.shape or np.any(ids != union_ids):
        raise AssertionError(f"{p}: ids not aligned to UNION")

    pl = _pick_npz(z, ["p_l", "pl", "p_text", "p_lang", "p_L"], p).astype(float).reshape(-1)
    pa = _pick_npz(z, ["p_a", "pa", "p_audio", "p_A"], p).astype(float).reshape(-1)
    pv = _pick_npz(z, ["p_v", "pv", "p_video", "p_V"], p).astype(float).reshape(-1)

    def _clip_keep_nan(x: np.ndarray) -> np.ndarray:
        out = x.astype(float)
        fin = np.isfinite(out)
        out[fin] = np.clip(out[fin], 0.0, 1.0)
        return out

    return str(p), _clip_keep_nan(pl), _clip_keep_nan(pa), _clip_keep_nan(pv)


# ----------------------------
# Q helpers
# ----------------------------

def q_to_scalar(Q: np.ndarray) -> np.ndarray:
    Q = np.asarray(Q, dtype=float)
    if Q.ndim == 1:
        return Q
    if Q.ndim == 2:
        return np.mean(Q, axis=1)
    raise ValueError(f"Q ndim {Q.ndim} unsupported")


def _load_broken_q_draws(q_perm_parent: Path, seed: int, fold: int, union: object, K: int) -> List[object]:
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
        draws: List[object] = []
        for k in range(1, K + 1):
            q_root_k = q_perm_parent / f"perm_{k:03d}"
            draws.append(qc.load_fold_q(q_root_k, seed=seed, fold=fold, union=union, require_ids_match=True))
        return draws

    bank_K = int(bank.Ql.shape[1])
    if bank_K < K:
        raise ValueError(f"BrokenQ bank has K={bank_K}, but --K={K} was requested: {bank.q_path}")
    return [qc.select_perm_from_bank(union, bank, k) for k in range(K)]


def _make_X_q(pl: np.ndarray, pa: np.ndarray, pv: np.ndarray, Ql: np.ndarray, Qa: np.ndarray, Qv: np.ndarray) -> np.ndarray:
    Xp = np.stack([pl, pa, pv], axis=1)
    Xq = np.stack([Ql, Qa, Qv], axis=1)
    return np.concatenate([Xp, Xq], axis=1)  # (N,6)


# ----------------------------
# Models
# ----------------------------

def _fit_lr(X: np.ndarray, y: np.ndarray, seed: int) -> LogisticRegression:
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
        C=1.0,
    )
    clf.fit(X, y)
    return clf


def _fit_hgb(X: np.ndarray, y: np.ndarray, seed: int) -> HistGradientBoostingClassifier:
    clf = HistGradientBoostingClassifier(
        max_depth=3,
        learning_rate=0.05,
        max_iter=300,
        random_state=seed,
    )
    clf.fit(X, y)
    return clf


def _predict_proba_pos(model, X: np.ndarray) -> np.ndarray:
    p = model.predict_proba(X)
    return p[:, 1].astype(float)


# ----------------------------
# Oracle-best (per-sample best unimodal by log-loss)
# ----------------------------

def oracle_best_prob(y_true: np.ndarray, probs_3: np.ndarray) -> np.ndarray:
    eps = 1e-8
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    p_true = np.where(y_true[:, None] == 1, probs_3, 1.0 - probs_3)  # (n,3)
    loss = -np.log(np.clip(p_true, eps, 1.0))                        # (n,3)
    best_m = loss.argmin(axis=1)
    return probs_3[np.arange(best_m.size), best_m].astype(float)


# ----------------------------
# Permtest stats
# ----------------------------

def perm_p_value_one_sided(clean_score: float, broken_scores: np.ndarray) -> float:
    broken_scores = np.asarray(broken_scores, dtype=float).reshape(-1)
    # p = (1 + #{broken >= clean})/(K+1)
    return float((1.0 + np.sum(broken_scores >= clean_score)) / (broken_scores.size + 1.0))


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--union_npz", type=str, required=True)
    ap.add_argument("--splits_dir", type=str, required=True)
    ap.add_argument("--unimodal_root", type=str, required=True)

    ap.add_argument("--q_clean_root", type=str, required=True)
    ap.add_argument("--q_perm_parent", type=str, required=True)
    ap.add_argument("--K", type=int, default=100)

    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--out_json", type=str, required=True)

    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])

    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--fusers", type=str, nargs="+", default=["lr"], choices=["lr", "hgb"])

    ap.add_argument("--check_perm_ids", type=int, nargs="+", default=[1, 2, 3, 100],
                    help="Perm indices to log/audit per fold (does not change computation).")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    union = qc.load_union(args.union_npz)
    n = len(union.ids_str)

    splits_dir = Path(args.splits_dir)
    unimodal_root = Path(args.unimodal_root)
    q_clean_root = Path(args.q_clean_root)
    q_perm_parent = Path(args.q_perm_parent)

    K = int(args.K)
    if K < 1:
        raise ValueError("--K must be >= 1")

    # per-row outputs
    rows: List[Dict] = []

    # summary accumulators
    per_fuser_clean: Dict[str, List[float]] = {f: [] for f in args.fusers}
    per_fuser_noq: Dict[str, List[float]] = {f: [] for f in args.fusers}
    per_fuser_oracle: Dict[str, List[float]] = {f: [] for f in args.fusers}

    per_fuser_broken_mean: Dict[str, List[float]] = {f: [] for f in args.fusers}
    per_fuser_broken_std: Dict[str, List[float]] = {f: [] for f in args.fusers}
    per_fuser_gap_clean_minus_broken_mean: Dict[str, List[float]] = {f: [] for f in args.fusers}
    per_fuser_gap_clean_minus_broken_abs: Dict[str, List[float]] = {f: [] for f in args.fusers}

    per_fuser_p_perm: Dict[str, List[float]] = {f: [] for f in args.fusers}

    per_fuser_flip_mean: Dict[str, List[float]] = {f: [] for f in args.fusers}
    per_fuser_flip_max: Dict[str, List[float]] = {f: [] for f in args.fusers}

    # optional audit traces
    audit_traces: List[Dict] = []

    for seed in args.seeds:
        for fold in args.folds:
            split = qc.load_fold_split(splits_dir, seed=seed, fold=fold)
            train_mask, test_mask = qc.make_train_test_masks(union, split)

            eval_mask = qc.eval_mask_full_only(union, test_mask)   # FULL-only TEST
            train_full = qc.full_only_mask(union.Ml, union.Ma, union.Mv, base_mask=train_mask)  # FULL-only TRAIN

            n_eval = int(np.sum(eval_mask))
            n_train = int(np.sum(train_full))
            if n_eval == 0:
                raise RuntimeError(f"seed={seed} fold={fold}: FULL-only TEST empty")
            if n_train == 0:
                raise RuntimeError(f"seed={seed} fold={fold}: FULL-only TRAIN empty")
            if len(np.unique(union.y[train_full])) < 2:
                raise RuntimeError(f"seed={seed} fold={fold}: FULL-only TRAIN has single class")

            unimodal_path, pl, pa, pv = load_unimodal_probs(unimodal_root, seed, fold, union.ids_str)

            # contract: probs NaN where missing, finite where present
            qc.assert_probs_nan_where_missing(union, pl, pa, pv)

            # cleanQ
            q_clean = qc.load_fold_q(q_clean_root, seed=seed, fold=fold, union=union, require_ids_match=True)
            qc.assert_q_missing_is_zero(union, q_clean.Ql, q_clean.Qa, q_clean.Qv)
            qc.assert_no_nan_in_present_q(union, q_clean.Ql, q_clean.Qa, q_clean.Qv)

            Ql = q_to_scalar(q_clean.Ql)
            Qa = q_to_scalar(q_clean.Qa)
            Qv = q_to_scalar(q_clean.Qv)

            # FULL-only implies probs finite on eval
            for name, arr in [("p_l", pl), ("p_a", pa), ("p_v", pv)]:
                bad = eval_mask & (~np.isfinite(arr))
                if np.any(bad):
                    ex = np.flatnonzero(bad)[:10].tolist()
                    raise AssertionError(f"seed={seed} fold={fold}: {name} non-finite on FULL-only TEST. ex={ex}")

            y_true = union.y[eval_mask].astype(int)

            # NoQ baseline
            probs_eval = np.stack([pl[eval_mask], pa[eval_mask], pv[eval_mask]], axis=1)
            p_noq = probs_eval.mean(axis=1)
            noq_acc = balanced_accuracy(y_true, preds_from_probs(p_noq, thresh=args.thresh))

            # oracle-best
            p_oracle = oracle_best_prob(y_true, probs_eval)
            oracle_acc = balanced_accuracy(y_true, preds_from_probs(p_oracle, thresh=args.thresh))

            if args.dry_run:
                print(f"[DRY] seed={seed} fold={fold} train_full={n_train} eval_full_test={n_eval}")
                print(f"      unimodal  ={unimodal_path}")
                print(f"      cleanQ    ={q_clean.q_path}")
                continue

            # train model on cleanQ features
            X_clean = _make_X_q(pl, pa, pv, Ql, Qa, Qv)
            Xtr = X_clean[train_full]
            ytr = union.y[train_full].astype(int)

            for fuser in args.fusers:
                rng_seed = seed * 100 + fold
                if fuser == "lr":
                    model = _fit_lr(Xtr, ytr, rng_seed)
                else:
                    model = _fit_hgb(Xtr, ytr, rng_seed)

                # cleanQ test
                p_clean = _predict_proba_pos(model, X_clean[eval_mask])
                y_clean = preds_from_probs(p_clean, thresh=args.thresh)
                clean_acc = balanced_accuracy(y_true, y_clean)

                # brokenQ distribution (K perms)
                broken_scores = np.zeros(K, dtype=float)
                flip_scores = np.zeros(K, dtype=float)

                broken_draws = _load_broken_q_draws(q_perm_parent, seed, fold, union, K)

                # For speed: preallocate prob feature matrix; only Q columns change
                # We'll build X_broken in UNION-space per k (cheap enough at N~MOSEI).
                for k, q_b in enumerate(broken_draws, start=1):
                    # contract invariants
                    qc.assert_q_missing_is_zero(union, q_b.Ql, q_b.Qa, q_b.Qv)
                    qc.assert_no_nan_in_present_q(union, q_b.Ql, q_b.Qa, q_b.Qv)

                    Ql_b = q_to_scalar(q_b.Ql)
                    Qa_b = q_to_scalar(q_b.Qa)
                    Qv_b = q_to_scalar(q_b.Qv)

                    X_brok = _make_X_q(pl, pa, pv, Ql_b, Qa_b, Qv_b)
                    p_b = _predict_proba_pos(model, X_brok[eval_mask])
                    y_b = preds_from_probs(p_b, thresh=args.thresh)

                    broken_scores[k - 1] = balanced_accuracy(y_true, y_b)
                    flip_scores[k - 1] = flip_rate(y_clean, y_b)

                b_mean = float(np.mean(broken_scores))
                b_std = float(np.std(broken_scores, ddof=1)) if K > 1 else float("nan")
                p_perm = perm_p_value_one_sided(clean_acc, broken_scores)

                gap_mean = float(clean_acc - b_mean)
                gap_abs = float(abs(clean_acc - b_mean))
                flip_mean = float(np.mean(flip_scores))
                flip_max = float(np.max(flip_scores))

                per_fuser_noq[fuser].append(noq_acc)
                per_fuser_oracle[fuser].append(oracle_acc)
                per_fuser_clean[fuser].append(clean_acc)
                per_fuser_broken_mean[fuser].append(b_mean)
                per_fuser_broken_std[fuser].append(b_std)
                per_fuser_gap_clean_minus_broken_mean[fuser].append(gap_mean)
                per_fuser_gap_clean_minus_broken_abs[fuser].append(gap_abs)
                per_fuser_p_perm[fuser].append(p_perm)
                per_fuser_flip_mean[fuser].append(flip_mean)
                per_fuser_flip_max[fuser].append(flip_max)

                row = {
                    "seed": seed,
                    "fold": fold,
                    "fuser": fuser,
                    "n_train_full": n_train,
                    "n_eval_full_test": n_eval,
                    "unimodal_path": unimodal_path,
                    "q_clean_path": q_clean.q_path,
                    "q_perm_parent": str(q_perm_parent),
                    "K": K,
                    "noq_mean_acc": noq_acc,
                    "cleanq_acc": clean_acc,
                    "brokenq_mean_acc": b_mean,
                    "brokenq_std_acc": b_std,
                    "oracle_best_acc": oracle_acc,
                    "clean_minus_broken_mean": gap_mean,
                    "abs_clean_minus_broken_mean": gap_abs,
                    "perm_p_one_sided_clean_gt_broken": p_perm,
                    "flip_clean_vs_broken_mean": flip_mean,
                    "flip_clean_vs_broken_max": flip_max,
                }
                rows.append(row)

                print(
                    f"[OK] fuser={fuser} seed={seed} fold={fold} | "
                    f"NoQ/clean/brokenμ±σ/oracle={noq_acc:.4f}/{clean_acc:.4f}/{b_mean:.4f}±{b_std:.4f}/{oracle_acc:.4f} | "
                    f"clean-brokenμ={gap_mean:+.4f} p_perm={p_perm:.3f} flipμ={flip_mean:.3f}"
                )

                # optional trace: check a few perm indices for reporting
                trace_ks = [k for k in args.check_perm_ids if 1 <= k <= K]
                if trace_ks:
                    audit_traces.append({
                        "seed": seed,
                        "fold": fold,
                        "fuser": fuser,
                        "trace_perm_ids": trace_ks,
                        "trace_broken_acc": {f"perm_{k:03d}": float(broken_scores[k-1]) for k in trace_ks},
                        "trace_flip": {f"perm_{k:03d}": float(flip_scores[k-1]) for k in trace_ks},
                    })

    # Write CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = sorted({k for r in rows for k in r.keys()})
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Summary JSON
    table: Dict[str, Dict[str, Dict[str, float]]] = {}
    for fuser in args.fusers:
        table[fuser] = {
            "NoQ": _mean_std(per_fuser_noq[fuser]),
            "cleanQ": _mean_std(per_fuser_clean[fuser]),
            "brokenQ_mean_over_K": _mean_std(per_fuser_broken_mean[fuser]),
            "brokenQ_std_over_K": _mean_std(per_fuser_broken_std[fuser]),
            "oracle-best": _mean_std(per_fuser_oracle[fuser]),
        }

    perm_p_summary = {f: _mean_std(per_fuser_p_perm[f]) for f in args.fusers}
    # also report median p across folds (often clearer than mean)
    perm_p_median = {f: float(np.median(np.asarray(per_fuser_p_perm[f], dtype=float))) for f in args.fusers}

    summary = {
        "meta": {
            "union_npz": args.union_npz,
            "splits_dir": args.splits_dir,
            "unimodal_root": str(Path(args.unimodal_root)),
            "q_clean_root": str(Path(args.q_clean_root)),
            "q_perm_parent": str(Path(args.q_perm_parent)),
            "K": int(args.K),
            "seeds": args.seeds,
            "folds": args.folds,
            "thresh": float(args.thresh),
            "eval_subset": "FULL-only within TEST (Ml==Ma==Mv==1)",
            "train_subset": "FULL-only within TRAIN (Ml==Ma==Mv==1)",
            "NoQ_definition": "arithmetic mean of unimodal probs",
            "brokenQ_definition": "brokenQ_k loaded from precomputed perm bank; same trained model; permutes within FULL-only TEST among present",
            "perm_p_value": "one-sided: p=(1+#{broken>=clean})/(K+1) per fold",
            "oracle_best_definition": "best unimodal expert per-sample by log-loss (diagnostic ceiling)",
        },
        "table_balanced_accuracy": table,
        "gap_clean_minus_broken_mean": {f: _mean_std(per_fuser_gap_clean_minus_broken_mean[f]) for f in args.fusers},
        "abs_gap_clean_minus_broken_mean": {f: _mean_std(per_fuser_gap_clean_minus_broken_abs[f]) for f in args.fusers},
        "flip_clean_vs_broken_mean": {f: _mean_std(per_fuser_flip_mean[f]) for f in args.fusers},
        "flip_clean_vs_broken_max": {f: _mean_std(per_fuser_flip_max[f]) for f in args.fusers},
        "perm_p_one_sided_clean_gt_broken_meanstd": perm_p_summary,
        "perm_p_one_sided_clean_gt_broken_median": perm_p_median,
        "audit_traces": audit_traces,
        "n_rows_csv": int(len(rows)),
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))

    print(f"[DONE] wrote CSV:  {out_csv}")
    print(f"[DONE] wrote JSON: {out_json}")


if __name__ == "__main__":
    main()
