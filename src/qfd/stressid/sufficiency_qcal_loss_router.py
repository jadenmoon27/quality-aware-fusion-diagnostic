#!/usr/bin/env python3
# FINAL_EXPERIMENTS/FINAL/v5_3/experiment3_sufficiency_qcal_loss_router.py
#
# Real sufficiency test (non-oracle at TEST):
# - Qcal learned on TRAIN only (precomputed clean + broken K perms).
# - Competition enforced on TRAIN using labels only on TRAIN:
#     choose eps so that BA_a(eps) matches target BA (best-other or mean-other) OR
#     if audio is not dominant by BA, enforce a fixed eps_min > 0.
# - Evaluate on FULL-only TEST.
# - Fusion policies:
#   (A) late: normalized Q-weighted avg (baseline)
#   (B) softmax: weights = softmax(Q / tau)
#   (C) hard: pick modality argmax Q per row (routing)
# - Clean vs Broken comparison (K perms) with Δ_perm + one-sided permutation p.
# - Oracle headroom reported on the same eval subset (upper bound, label-using).

# python -m qfd.stressid.sufficiency_qcal_loss_router \
#   --union_npz /path/to/project/output/fusion_table_noQ/new_fusion_table_union_y2_avp_v1_noQ.npz \
#   --splits_dir /path/to/project/splits \
#   --preds_root /path/to/project/paper_output/unimodal_preds/lr \
#   --qcal_clean_root /path/to/project/paper_output/quality_qcal_loss/clean \
#   --qcal_broken_root /path/to/project/paper_output/quality_qcal_loss/broken_K200 \
#   --out_root /path/to/project/paper_output/reports \
#   --family lr \
#   --seeds 11 22 33 44 55 \
#   --folds 0 1 2 3 4 \
#   --K 200 \
#   --thresh 0.5 \
#   --require_full_coverage \
#   --fusion hard \
#   --target_mode best_other \
#   --eps_min_if_nondominant 0.7 \
#   --eps_grid 0.0 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.60 0.70 0.80 0.90 1.0 \
#
# Outputs:
#   out_root/{family}/exp3_sufficiency_qcal_loss_router/{late|softmax|hard}/fold_rows.json
#   out_root/{family}/exp3_sufficiency_qcal_loss_router/{late|softmax|hard}/aggregate.json

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from qfd._shared.q_contract import (
    load_union,
    load_fold_split,
    make_train_test_masks,
    eval_mask_full_only,
    assert_probs_nan_where_missing,
    balanced_accuracy_from_probs,
)

# ----------------------------
# IO
# ----------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _write_json(path: Path, obj: Dict) -> None:
    _ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

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

def _load_qcal_clean_npz(root: Path, seed: int, fold: int) -> Dict[str, np.ndarray]:
    p = root / f"seed_{seed}" / f"fold_{fold}.npz"
    if not p.exists():
        raise FileNotFoundError(f"Missing Qcal clean: {p}")
    z = np.load(p, allow_pickle=True)
    need = ["ids_str", "Qa", "Qv", "Qp"]
    for k in need:
        if k not in z.files:
            raise KeyError(f"{p} missing '{k}'. Found: {sorted(z.files)}")
    return {"ids_str": z["ids_str"], "Qa": z["Qa"], "Qv": z["Qv"], "Qp": z["Qp"], "_path": str(p)}

def _load_qcal_broken_npz(root: Path, seed: int, fold: int) -> Dict[str, np.ndarray]:
    p = root / f"seed_{seed}" / f"fold_{fold}.npz"
    if not p.exists():
        raise FileNotFoundError(f"Missing Qcal broken: {p}")
    z = np.load(p, allow_pickle=True)
    need = ["ids_str", "Qa", "Qv", "Qp"]
    for k in need:
        if k not in z.files:
            raise KeyError(f"{p} missing '{k}'. Found: {sorted(z.files)}")
    return {"ids_str": z["ids_str"], "Qa": z["Qa"], "Qv": z["Qv"], "Qp": z["Qp"], "_path": str(p)}

# ----------------------------
# helpers
# ----------------------------

def _infer_K(Q: np.ndarray) -> int:
    Q = np.asarray(Q)
    if Q.ndim != 2:
        raise ValueError(f"Broken Q expected (N,K). Got {Q.shape}")
    return int(Q.shape[1])

def _get_k(Q: np.ndarray, k: int) -> np.ndarray:
    return np.asarray(Q, dtype=float)[:, int(k)]

def _degrade_toward_uniform(pa: np.ndarray, eps: float) -> np.ndarray:
    pa = np.asarray(pa, dtype=float)
    return (1.0 - eps) * pa + eps * 0.5

def _perm_stats(ba_clean: float, ba_broken: np.ndarray) -> Dict[str, float]:
    ba_broken = np.asarray(ba_broken, dtype=float)
    K = int(ba_broken.shape[0])
    delta = float(ba_clean - float(np.mean(ba_broken)))
    p = float((1.0 + float(np.sum(ba_broken >= ba_clean))) / float(K + 1))
    return {"delta_perm_fold": delta, "p_perm_fold": p, "ba_broken_mean": float(np.mean(ba_broken)), "K": K}

def _oracle_best(P: np.ndarray, y: np.ndarray, thresh: float) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    yhat = (P >= float(thresh)).astype(int)      # (n,3)
    correct = (yhat == y[:, None])               # (n,3)
    conf = np.abs(P - 0.5)                       # (n,3)
    n = P.shape[0]
    chosen = np.zeros(n, dtype=int)
    any_correct = np.any(correct, axis=1)
    for i in range(n):
        if any_correct[i]:
            cands = np.where(correct[i])[0]
            chosen[i] = cands[int(np.argmax(conf[i, cands]))]
        else:
            chosen[i] = int(np.argmax(conf[i]))
    return P[np.arange(n), chosen]

def _rho(q: np.ndarray, c: np.ndarray) -> float:
    q = np.asarray(q, dtype=float)
    c = np.asarray(c, dtype=float)
    if np.std(q) < 1e-12 or np.std(c) < 1e-12:
        return 0.0
    return float(np.corrcoef(q, c)[0, 1])

def _late(P: np.ndarray, Q: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, int]:
    Q = np.asarray(Q, dtype=float)
    denom = np.sum(Q, axis=1, keepdims=True)
    bad = int(np.sum(denom <= eps))
    w = np.where(denom > eps, Q / denom, 1.0 / 3.0)
    p = np.sum(w * P, axis=1)
    return p, bad

def _softmax(P: np.ndarray, Q: np.ndarray, tau: float) -> np.ndarray:
    Q = np.asarray(Q, dtype=float)
    z = Q / max(float(tau), 1e-6)
    z = z - np.max(z, axis=1, keepdims=True)
    ex = np.exp(z)
    w = ex / np.sum(ex, axis=1, keepdims=True)
    return np.sum(w * P, axis=1)

def _hard(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    Q = np.asarray(Q, dtype=float)
    j = np.argmax(Q, axis=1)
    return P[np.arange(P.shape[0]), j]

def _choose_eps_train(
    y_tr: np.ndarray,
    Pa_tr: np.ndarray,
    Pv_tr: np.ndarray,
    Pp_tr: np.ndarray,
    eps_grid: List[float],
    thresh: float,
    target_mode: str,
    eps_min_if_nondominant: float,
) -> Tuple[float, Dict[str, float]]:
    """
    TRAIN-only eps selection.

    - Compute BA_a(0), BA_v, BA_p.
    - If audio is dominant by BA (BA_a(0) > target), pick eps minimizing |BA_a(eps)-target|.
    - Else enforce eps = eps_min_if_nondominant (guarantees competition manipulation is non-trivial).
    """
    ba_a0 = float(balanced_accuracy_from_probs(y_tr, Pa_tr, thresh=float(thresh)))
    ba_v  = float(balanced_accuracy_from_probs(y_tr, Pv_tr, thresh=float(thresh)))
    ba_p  = float(balanced_accuracy_from_probs(y_tr, Pp_tr, thresh=float(thresh)))

    if target_mode == "best_other":
        target = max(ba_v, ba_p)
    elif target_mode == "mean_other":
        target = 0.5 * (ba_v + ba_p)
    else:
        raise ValueError(f"bad target_mode: {target_mode}")

    diag = {"ba_a0": ba_a0, "ba_v": ba_v, "ba_p": ba_p, "target": float(target)}

    if ba_a0 > target + 1e-6:
        best_eps = float(eps_grid[0])
        best_obj = float("inf")
        for eps in eps_grid:
            pa = _degrade_toward_uniform(Pa_tr, float(eps))
            ba_a = float(balanced_accuracy_from_probs(y_tr, pa, thresh=float(thresh)))
            obj = abs(ba_a - target)
            if obj < best_obj:
                best_obj = obj
                best_eps = float(eps)
        diag["mode"] = "match_target"
        diag["obj"] = float(best_obj)
        return best_eps, diag

    diag["mode"] = "forced_min"
    diag["obj"] = float("nan")
    return float(eps_min_if_nondominant), diag

# ----------------------------
# main
# ----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--union_npz", type=str, required=True)
    ap.add_argument("--splits_dir", type=str, required=True)
    ap.add_argument("--preds_root", type=str, required=True)

    ap.add_argument("--qcal_clean_root", type=str, required=True)
    ap.add_argument("--qcal_broken_root", type=str, required=True)

    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--family", type=str, choices=["lr", "hgb"], required=True)

    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--folds", type=int, nargs="+", required=True)

    ap.add_argument("--K", type=int, default=200)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--require_full_coverage", action="store_true")

    ap.add_argument("--eps_grid", type=float, nargs="+", required=True)
    ap.add_argument("--target_mode", type=str, default="best_other", choices=["best_other", "mean_other"])
    ap.add_argument("--eps_min_if_nondominant", type=float, default=0.35)

    ap.add_argument("--fusion", type=str, default="hard", choices=["late", "softmax", "hard"])
    ap.add_argument("--tau", type=float, default=0.10, help="softmax temperature (only used if fusion=softmax)")
    return ap.parse_args()

def main() -> None:
    args = parse_args()

    union = load_union(args.union_npz)
    preds_root = Path(args.preds_root)

    qcal_clean_root = Path(args.qcal_clean_root)
    qcal_broken_root = Path(args.qcal_broken_root)

    out_dir = Path(args.out_root) / args.family / "exp3_sufficiency_qcal_loss_router" / args.fusion
    _ensure_dir(out_dir)

    fold_rows: List[Dict] = []

    for seed in args.seeds:
        for fold in args.folds:
            split = load_fold_split(args.splits_dir, seed=int(seed), fold=int(fold))
            train_mask, test_mask = make_train_test_masks(
                union, split, require_full_coverage=bool(args.require_full_coverage)
            )

            tr_full = eval_mask_full_only(union, train_mask)
            te_full = eval_mask_full_only(union, test_mask)

            tr_idx = np.where(tr_full)[0]
            te_idx = np.where(te_full)[0]
            if te_idx.size == 0 or tr_idx.size == 0:
                raise RuntimeError(f"Empty FULL-only TRAIN/TEST seed={seed} fold={fold}")

            pred = _load_unimodal_preds_npz(preds_root, int(seed), int(fold))
            ids_pred = np.asarray([str(x) for x in pred["ids"]], dtype=object)
            if not np.array_equal(ids_pred, union.ids_str):
                raise ValueError(f"Preds ids mismatch vs UNION seed={seed} fold={fold}")

            p_a = np.asarray(pred["p_a"], dtype=float)
            p_v = np.asarray(pred["p_v"], dtype=float)
            p_p = np.asarray(pred["p_p"], dtype=float)
            assert_probs_nan_where_missing(union, p_a, p_v, p_p)

            qc = _load_qcal_clean_npz(qcal_clean_root, int(seed), int(fold))
            qb = _load_qcal_broken_npz(qcal_broken_root, int(seed), int(fold))

            ids_qc = np.asarray([str(x) for x in qc["ids_str"]], dtype=object)
            ids_qb = np.asarray([str(x) for x in qb["ids_str"]], dtype=object)
            if not np.array_equal(ids_qc, union.ids_str) or not np.array_equal(ids_qb, union.ids_str):
                raise ValueError(f"Qcal ids mismatch vs UNION seed={seed} fold={fold}")

            K_found = _infer_K(np.asarray(qb["Qa"]))
            if int(K_found) != int(args.K):
                raise ValueError(f"Expected K={args.K}, got K={K_found} seed={seed} fold={fold}")

            # TRAIN-only eps selection
            y_tr = union.y[tr_idx].astype(int)
            Pa_tr = p_a[tr_idx]
            Pv_tr = p_v[tr_idx]
            Pp_tr = p_p[tr_idx]
            if not (np.isfinite(Pa_tr).all() and np.isfinite(Pv_tr).all() and np.isfinite(Pp_tr).all()):
                raise ValueError("Non-finite preds on FULL-only TRAIN (unexpected).")

            eps_star, eps_diag = _choose_eps_train(
                y_tr=y_tr,
                Pa_tr=Pa_tr,
                Pv_tr=Pv_tr,
                Pp_tr=Pp_tr,
                eps_grid=[float(x) for x in args.eps_grid],
                thresh=float(args.thresh),
                target_mode=str(args.target_mode),
                eps_min_if_nondominant=float(args.eps_min_if_nondominant),
            )

            # Build eval P (FULL-only TEST), then apply eps*
            y_te = union.y[te_idx].astype(int)
            P = np.stack([p_a[te_idx], p_v[te_idx], p_p[te_idx]], axis=1)
            if not np.isfinite(P).all():
                raise ValueError("Non-finite preds on FULL-only TEST (unexpected).")
            P[:, 0] = _degrade_toward_uniform(P[:, 0], eps_star)

            # Clean Q
            Q_eval = np.stack(
                [
                    np.asarray(qc["Qa"], dtype=float)[te_idx],
                    np.asarray(qc["Qv"], dtype=float)[te_idx],
                    np.asarray(qc["Qp"], dtype=float)[te_idx],
                ],
                axis=1,
            )

            # Evaluate fusion for clean/broken
            def fuse(P_: np.ndarray, Q_: np.ndarray) -> Tuple[np.ndarray, int]:
                if args.fusion == "late":
                    return _late(P_, Q_)
                if args.fusion == "softmax":
                    return _softmax(P_, Q_, tau=float(args.tau)), 0
                if args.fusion == "hard":
                    return _hard(P_, Q_), 0
                raise ValueError("bad fusion")

            p_clean, bad_clean = fuse(P, Q_eval)
            ba_clean = float(balanced_accuracy_from_probs(y_te, p_clean, thresh=float(args.thresh)))

            ba_b = np.zeros(int(args.K), dtype=float)
            bad_max = int(bad_clean)
            for k in range(int(args.K)):
                Qk = np.stack(
                    [
                        _get_k(np.asarray(qb["Qa"]), k)[te_idx],
                        _get_k(np.asarray(qb["Qv"]), k)[te_idx],
                        _get_k(np.asarray(qb["Qp"]), k)[te_idx],
                    ],
                    axis=1,
                )
                p_k, bad = fuse(P, Qk)
                bad_max = max(bad_max, int(bad))
                ba_b[k] = float(balanced_accuracy_from_probs(y_te, p_k, thresh=float(args.thresh)))

            ps = _perm_stats(ba_clean, ba_b)

            # Oracle headroom
            p_oracle = _oracle_best(P, y_te, thresh=float(args.thresh))
            ba_oracle = float(balanced_accuracy_from_probs(y_te, p_oracle, thresh=float(args.thresh)))
            headroom = float(ba_oracle - ba_clean)

            # alignment diagnostic on eval
            yhat = (P >= float(args.thresh)).astype(int)
            corr_vec = (yhat == y_te[:, None]).astype(float)
            rho_a = _rho(Q_eval[:, 0], corr_vec[:, 0])
            rho_v = _rho(Q_eval[:, 1], corr_vec[:, 1])
            rho_p = _rho(Q_eval[:, 2], corr_vec[:, 2])

            fold_rows.append(
                {
                    "seed": int(seed),
                    "fold": int(fold),
                    "n_train_full": int(tr_idx.size),
                    "n_eval_full": int(te_idx.size),
                    "eps_star": float(eps_star),
                    "eps_diag": eps_diag,
                    "ba_clean": float(ba_clean),
                    "ba_broken_mean": float(ps["ba_broken_mean"]),
                    "delta_perm_fold": float(ps["delta_perm_fold"]),
                    "p_perm_fold": float(ps["p_perm_fold"]),
                    "oracle_ba": float(ba_oracle),
                    "oracle_headroom": float(headroom),
                    "bad_denom_clean": int(bad_clean),
                    "bad_denom_broken_max": int(bad_max),
                    "rho_a": float(rho_a),
                    "rho_v": float(rho_v),
                    "rho_p": float(rho_p),
                    "preds_path": str(pred["_path"]),
                    "qcal_clean_path": str(qc["_path"]),
                    "qcal_broken_path": str(qb["_path"]),
                    "K": int(args.K),
                    "thresh": float(args.thresh),
                    "fusion": str(args.fusion),
                    "tau": float(args.tau),
                    "target_mode": str(args.target_mode),
                    "eps_min_if_nondominant": float(args.eps_min_if_nondominant),
                }
            )

            print(
                f"[OK] seed={seed} fold={fold} eps*={eps_star:.3f} ({eps_diag['mode']}) "
                f"Δ={ps['delta_perm_fold']:+.4f} p={ps['p_perm_fold']:.4f} "
                f"rho(a,v,p)=({rho_a:+.3f},{rho_v:+.3f},{rho_p:+.3f}) "
                f"headroom={headroom:+.4f}"
            )

    deltas = np.asarray([r["delta_perm_fold"] for r in fold_rows], dtype=float)
    pvals  = np.asarray([r["p_perm_fold"] for r in fold_rows], dtype=float)
    head   = np.asarray([r["oracle_headroom"] for r in fold_rows], dtype=float)
    epss   = np.asarray([r["eps_star"] for r in fold_rows], dtype=float)

    agg = {
        "experiment": "exp3_sufficiency_qcal_loss_router",
        "mode": str(args.fusion),
        "family": args.family,
        "n_folds": int(len(fold_rows)),
        "delta_perm_mean": float(np.mean(deltas)),
        "delta_perm_std": float(np.std(deltas)),
        "p_perm_median": float(np.median(pvals)),
        "oracle_headroom_mean": float(np.mean(head)),
        "oracle_headroom_std": float(np.std(head)),
        "eps_star_mean": float(np.mean(epss)),
        "eps_star_std": float(np.std(epss)),
        "K": int(args.K),
        "thresh": float(args.thresh),
        "tau": float(args.tau),
        "target_mode": str(args.target_mode),
        "eps_min_if_nondominant": float(args.eps_min_if_nondominant),
    }

    _write_json(out_dir / "fold_rows.json", {"fold_rows": fold_rows})
    _write_json(out_dir / "aggregate.json", agg)

    print(
        f"[DONE] fusion={args.fusion} Δ={agg['delta_perm_mean']:+.4f}±{agg['delta_perm_std']:.4f} "
        f"p50={agg['p_perm_median']:.4f} headroom={agg['oracle_headroom_mean']:+.4f}±{agg['oracle_headroom_std']:.4f} "
        f"eps*={agg['eps_star_mean']:.3f}±{agg['eps_star_std']:.3f}"
    )

if __name__ == "__main__":
    main()