#!/usr/bin/env python3
# FINAL_EXPERIMENTS/FINAL/v5_3/precompute_qcal_loss_predictor.py
#
# Precompute Qcal via TRAIN-only supervised "loss-quantile" correctness-likelihood predictor.
#
# - UNION order canonical; never reorder.
# - Splits loaded via q_contract (subject-safe).
# - Target uses TRAIN labels only:
#     loss_m,i = -y log p - (1-y) log(1-p)
#     t_m,i = 1[ loss_m,i <= median(loss_m on TRAIN for modality-present rows) ]
# - Features per modality:
#     x = [Qraw_vector (use Q_*_2d if available; else Q_*) , |p-0.5| ]
# - Fit logistic per modality (L2-regularized, simple GD; no sklearn).
# - Output Qcal_clean per fold: Qa,Qv,Qp shape (N,)
# - Output Qcal_broken per fold: Qa,Qv,Qp shape (N,K), TEST-only present-only permutation.
#
# Expected qraw layout (your screenshot):
#   qraw_root/seed_{seed}/union_quality_*seed{seed}_fold{fold}.npz
# keys include: ids, Q_a, Q_v, Q_p, and preferably Q_a_2d, Q_v_2d, Q_p_2d
#
# Writes:
#   out_clean_root/seed_{seed}/fold_{fold}.npz   (ids_str, Qa,Qv,Qp)
#   out_broken_root/seed_{seed}/fold_{fold}.npz  (ids_str, Qa,Qv,Qp) where each is (N,K)

# python -m qfd.stressid.precompute_qcal_loss_predictor \
#   --union_npz /path/to/project/output/fusion_table_noQ/new_fusion_table_union_y2_avp_v1_noQ.npz \
#   --splits_dir /path/to/project/splits \
#   --preds_root /path/to/project/paper_output/unimodal_preds/lr \
#   --qraw_root /path/to/project/quality \
#   --out_clean_root /path/to/project/paper_output/quality_qcal_loss/clean \
#   --out_broken_root /path/to/project/paper_output/quality_qcal_loss/broken_K200 \
#   --seeds 11 22 33 44 55 \
#   --folds 0 1 2 3 4 \
#   --K 200 \
#   --thresh 0.5 \
#   --base_seed 12345 \
#   --lam 1e-2 \
#   --iters 2000 \
#   --lr 0.1 \
#   --require_full_coverage

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from qfd._shared.q_contract import (
    load_union,
    load_fold_split,
    make_train_test_masks,
    eval_mask_full_only,
    assert_probs_nan_where_missing,
)

# ----------------------------
# IO
# ----------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _save_npz(path: Path, **kwargs) -> None:
    _ensure_dir(path.parent)
    np.savez(path, **kwargs)

def _load_unimodal_preds_npz(preds_root: Path, seed: int, fold: int) -> Dict[str, np.ndarray]:
    p = preds_root / f"seed_{seed}" / f"fold_{fold}.npz"
    if not p.exists():
        raise FileNotFoundError(f"Missing unimodal preds: {p}")
    z = np.load(p, allow_pickle=True)
    need = ["ids", "y", "p_a", "p_v", "p_p"]
    for k in need:
        if k not in z.files:
            raise KeyError(f"{p} missing '{k}'. Found: {sorted(z.files)}")
    out = {k: z[k] for k in z.files}
    out["_path"] = str(p)
    return out

def _load_qraw_fold_npz(qraw_root: Path, seed: int, fold: int) -> Dict[str, np.ndarray]:
    seed_dir = qraw_root / f"seed_{seed}"
    if not seed_dir.exists():
        raise FileNotFoundError(f"Missing seed dir: {seed_dir}")
    pat = f"*seed{seed}*fold{fold}*.npz"
    matches = sorted(seed_dir.glob(pat))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly 1 Qraw match for pattern '{pat}' in {seed_dir}, found {len(matches)}: "
            f"{[m.name for m in matches]}"
        )
    p = matches[0]
    z = np.load(p, allow_pickle=True)

    if "ids_str" in z.files:
        ids = z["ids_str"]
    elif "ids" in z.files:
        ids = z["ids"]
    else:
        raise KeyError(f"{p} missing ids_str or ids. Found: {sorted(z.files)}")

    # prefer vector raw
    def pick(mod: str) -> np.ndarray:
        k2 = f"Q_{mod}_2d"
        k1 = f"Q_{mod}"
        if k2 in z.files:
            return z[k2]
        if k1 in z.files:
            return z[k1]
        raise KeyError(f"{p} missing {k2} or {k1}. Found: {sorted(z.files)}")

    Qa_raw = pick("a")
    Qv_raw = pick("v")
    Qp_raw = pick("p")

    return {"ids_str": ids, "Qa_raw": Qa_raw, "Qv_raw": Qv_raw, "Qp_raw": Qp_raw, "_path": str(p)}

# ----------------------------
# math helpers
# ----------------------------

def _as_2d(Q: np.ndarray) -> np.ndarray:
    Q = np.asarray(Q, dtype=float)
    if Q.ndim == 1:
        return Q[:, None]
    if Q.ndim == 2:
        return Q
    raise ValueError(f"Q must be 1D or 2D. Got {Q.shape}")

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def _log_loss_binary(y: np.ndarray, p: np.ndarray, clip: float = 1e-6) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    p = np.clip(p, clip, 1.0 - clip)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))

def _fit_logistic_l2(X: np.ndarray, t: np.ndarray, lam: float, iters: int, lr: float) -> np.ndarray:
    """
    L2-regularized logistic regression, bias included.
    X: (n,d), t: (n,) in {0,1}
    returns w: (d+1,)
    """
    X = np.asarray(X, dtype=float)
    t = np.asarray(t, dtype=float)
    n, d = X.shape
    Xb = np.concatenate([np.ones((n, 1), dtype=float), X], axis=1)  # (n,d+1)
    w = np.zeros(d + 1, dtype=float)

    for _ in range(int(iters)):
        z = Xb @ w
        p = _sigmoid(z)
        g = (Xb.T @ (p - t)) / n
        g[1:] += lam * w[1:]
        w -= lr * g
    return w

def _predict_logistic(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    Xb = np.concatenate([np.ones((n, 1), dtype=float), X], axis=1)
    return _sigmoid(Xb @ w)

def _make_features(Qraw_2d: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    concat [Qraw_vector, |p-0.5|]
    """
    Qraw_2d = _as_2d(Qraw_2d)
    conf = np.abs(np.asarray(p, dtype=float) - 0.5)[:, None]
    return np.concatenate([Qraw_2d.astype(float), conf], axis=1)

# ----------------------------
# broken permutation builder
# ----------------------------

def _permute_test_present_only(
    Q: np.ndarray,
    test_mask: np.ndarray,
    present_mask: np.ndarray,
    K: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Q: (N,) clean
    returns Qb: (N,K) where only rows in (test & present) are permuted; others unchanged.
    """
    Q = np.asarray(Q, dtype=float)
    N = Q.shape[0]
    Qb = np.repeat(Q[:, None], int(K), axis=1)

    idx = np.where((test_mask.astype(bool)) & (present_mask.astype(bool)))[0]
    if idx.size == 0:
        return Qb

    base = Q[idx].copy()
    for k in range(int(K)):
        perm = rng.permutation(idx.size)
        Qb[idx, k] = base[perm]
    return Qb

# ----------------------------
# main
# ----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--union_npz", type=str, required=True)
    ap.add_argument("--splits_dir", type=str, required=True)
    ap.add_argument("--preds_root", type=str, required=True)
    ap.add_argument("--qraw_root", type=str, required=True)

    ap.add_argument("--out_clean_root", type=str, required=True)
    ap.add_argument("--out_broken_root", type=str, required=True)

    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--folds", type=int, nargs="+", required=True)

    ap.add_argument("--K", type=int, default=200)
    ap.add_argument("--thresh", type=float, default=0.5)

    ap.add_argument("--base_seed", type=int, default=12345)
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.1)

    ap.add_argument("--require_full_coverage", action="store_true")
    return ap.parse_args()

def main() -> None:
    args = parse_args()

    union = load_union(args.union_npz)
    preds_root = Path(args.preds_root)
    qraw_root = Path(args.qraw_root)

    out_clean_root = Path(args.out_clean_root)
    out_broken_root = Path(args.out_broken_root)

    for seed in args.seeds:
        for fold in args.folds:
            split = load_fold_split(args.splits_dir, seed=int(seed), fold=int(fold))
            train_mask, test_mask = make_train_test_masks(
                union, split, require_full_coverage=bool(args.require_full_coverage)
            )

            # unimodal preds (UNION aligned)
            pred = _load_unimodal_preds_npz(preds_root, int(seed), int(fold))
            ids_pred = np.asarray([str(x) for x in pred["ids"]], dtype=object)
            if not np.array_equal(ids_pred, union.ids_str):
                raise ValueError(f"Preds ids mismatch vs UNION seed={seed} fold={fold}")

            p_a = np.asarray(pred["p_a"], dtype=float)
            p_v = np.asarray(pred["p_v"], dtype=float)
            p_p = np.asarray(pred["p_p"], dtype=float)

            assert_probs_nan_where_missing(union, p_a, p_v, p_p)

            y = np.asarray(union.y, dtype=int)

            # qraw fold pack (UNION aligned)
            qraw = _load_qraw_fold_npz(qraw_root, int(seed), int(fold))
            ids_q = np.asarray([str(x) for x in qraw["ids_str"]], dtype=object)
            if not np.array_equal(ids_q, union.ids_str):
                raise ValueError(f"Qraw ids mismatch vs UNION seed={seed} fold={fold}")

            Qa_raw = _as_2d(qraw["Qa_raw"])
            Qv_raw = _as_2d(qraw["Qv_raw"])
            Qp_raw = _as_2d(qraw["Qp_raw"])

            # TRAIN rows for fitting must be modality-present and finite preds
            Ma = np.asarray(union.Ma, dtype=int)
            Mv = np.asarray(union.Mv, dtype=int)
            Mp = np.asarray(union.Mp, dtype=int)

            # We fit on TRAIN only, present-only, finite-only.
            def fit_mod(p: np.ndarray, Qraw2d: np.ndarray, M: np.ndarray) -> Tuple[np.ndarray, float]:
                tr = train_mask.astype(bool) & (M.astype(bool)) & np.isfinite(p)
                if np.sum(tr) < 10:
                    # degenerate fallback: constant 0.5
                    w = np.zeros(Qraw2d.shape[1] + 1 + 1, dtype=float)  # +conf plus bias
                    med = np.nan
                    return w, med

                loss = _log_loss_binary(y[tr], p[tr])
                med = float(np.median(loss))
                t = (loss <= med).astype(np.float32)

                X = _make_features(Qraw2d[tr], p[tr])
                w = _fit_logistic_l2(X, t, lam=float(args.lam), iters=int(args.iters), lr=float(args.lr))
                return w, med

            w_a, med_a = fit_mod(p_a, Qa_raw, Ma)
            w_v, med_v = fit_mod(p_v, Qv_raw, Mv)
            w_p, med_p = fit_mod(p_p, Qp_raw, Mp)

            # Predict Qcal for all UNION rows, but enforce missing => 0
            def pred_mod(p: np.ndarray, Qraw2d: np.ndarray, M: np.ndarray, w: np.ndarray) -> np.ndarray:
                out = np.zeros(union.ids_str.shape[0], dtype=float)
                ok = (M.astype(bool)) & np.isfinite(p)
                if np.any(ok):
                    X = _make_features(Qraw2d[ok], p[ok])
                    out[ok] = _predict_logistic(X, w)
                out[~ok] = 0.0
                return out

            Qa_cal = pred_mod(p_a, Qa_raw, Ma, w_a)
            Qv_cal = pred_mod(p_v, Qv_raw, Mv, w_v)
            Qp_cal = pred_mod(p_p, Qp_raw, Mp, w_p)

            # Broken permutations: TEST-only, present-only
            rng = np.random.default_rng(int(args.base_seed) + 1000 * int(seed) + int(fold))

            Qa_b = _permute_test_present_only(Qa_cal, test_mask, Ma, int(args.K), rng)
            Qv_b = _permute_test_present_only(Qv_cal, test_mask, Mv, int(args.K), rng)
            Qp_b = _permute_test_present_only(Qp_cal, test_mask, Mp, int(args.K), rng)

            # Save
            out_clean = out_clean_root / f"seed_{seed}" / f"fold_{fold}.npz"
            out_brok = out_broken_root / f"seed_{seed}" / f"fold_{fold}.npz"

            _save_npz(out_clean, ids_str=union.ids_str, Qa=Qa_cal, Qv=Qv_cal, Qp=Qp_cal,
                      meta=np.array({"seed": int(seed), "fold": int(fold), "med_loss": (med_a, med_v, med_p)}, dtype=object))

            _save_npz(out_brok, ids_str=union.ids_str, Qa=Qa_b, Qv=Qv_b, Qp=Qp_b,
                      meta=np.array({"seed": int(seed), "fold": int(fold), "K": int(args.K)}, dtype=object))

            # quick sanity: non-trivial variance on TEST present rows
            test_full = eval_mask_full_only(union, test_mask)
            idx = np.where(test_full)[0]
            qmean = (float(np.mean(Qa_cal[idx])), float(np.mean(Qv_cal[idx])), float(np.mean(Qp_cal[idx])))
            qstd  = (float(np.std(Qa_cal[idx])),  float(np.std(Qv_cal[idx])),  float(np.std(Qp_cal[idx])))

            print(
                f"[OK] seed={seed} fold={fold} "
                f"Qcal_test_mean(a,v,p)={qmean[0]:.3f},{qmean[1]:.3f},{qmean[2]:.3f} "
                f"std={qstd[0]:.3f},{qstd[1]:.3f},{qstd[2]:.3f}"
            )

if __name__ == "__main__":
    main()