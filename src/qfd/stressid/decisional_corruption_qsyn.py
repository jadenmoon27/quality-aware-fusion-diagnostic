#!/usr/bin/env python3
# FINAL_EXPERIMENTS/FINAL/v5_3/experiment2_decisional_corruption_qsyn.py
#
# Experiment 2 — Decisional corruption + synthetic Qsyn (diagnostic power)
#
# Defensible protocol:
# - Apply decisional corruption ONLY to audio posterior on FULL-only TEST, leaving other posteriors fixed.
# - Load Qsyn_clean and Qsyn_broken (K perms) via q_contract.load_fold_q() (no on-the-fly Broken).
# - Derive corruption mask from Qsyn_clean (Qa = 1 - c on eval subset) to guarantee consistency.
# - Evaluate late fusion under Clean vs Broken, compute Δ_perm + one-sided permutation p per fold.
# - Report mean±std Δ across folds and median p across folds; include oracle headroom.
#
# Example:
# python -m qfd.stressid.precompute_qsyn_decisional_corruption \
#   --union_npz /path/to/project/output/fusion_table_noQ/new_fusion_table_union_y2_avp_v1_noQ.npz \
#   --splits_dir /path/to/project/splits \
#   --out_qsyn_clean_root /path/to/project/paper_output/quality_qsyn/clean_k0p5_L8_vp1_present \
#   --out_qsyn_broken_root /path/to/project/paper_output/quality_qsyn/broken_k0p5_L8_vp1_present_K200 \
#   --seeds 11 22 33 44 55 \
#   --folds 0 1 2 3 4 \
#   --k 0.5 \
#   --L 8.0 \
#   --K 200 \
#   --base_seed 12345 \
#   --require_full_coverage

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from qfd._shared.q_contract import (
    balanced_accuracy_from_probs,
    eval_mask_full_only,
    load_fold_q,
    load_fold_split,
    load_union,
    make_train_test_masks,
    assert_probs_nan_where_missing,
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
            raise KeyError(f"Preds file {p} missing key '{k}'. Found keys: {sorted(z.files)}")
    out = {k: z[k] for k in z.files}
    out["_path"] = str(p)
    return out

# ----------------------------
# Helpers
# ----------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def _logit(p: np.ndarray, clip: float = 1e-6) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, clip, 1.0 - clip)
    return np.log(p / (1.0 - p))

def _q_scalar(Q: np.ndarray) -> np.ndarray:
    Q = np.asarray(Q, dtype=float)
    if Q.ndim == 1:
        return Q
    if Q.ndim == 2:
        return np.mean(Q, axis=1)
    raise ValueError(f"Q must be 1D or 2D to scalarize. Got {Q.shape}")

def _get_broken_k(Q: np.ndarray, k: int) -> np.ndarray:
    Q = np.asarray(Q, dtype=float)
    if Q.ndim == 2:
        return Q[:, k]          # (N,)
    if Q.ndim == 3:
        return Q[:, k, :]       # (N,d)
    raise ValueError(f"Broken-Q must be (N,K) or (N,K,d). Got {Q.shape}")

def _infer_K(Q: np.ndarray) -> int:
    Q = np.asarray(Q)
    if Q.ndim == 2:
        return int(Q.shape[1])
    if Q.ndim == 3:
        return int(Q.shape[1])
    raise ValueError(f"Broken-Q must be (N,K) or (N,K,d). Got {Q.shape}")

def _quality_weighted_avg(P: np.ndarray, Qs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    Qs = np.asarray(Qs, dtype=float)
    denom = np.sum(Qs, axis=1, keepdims=True)
    w = np.where(denom > eps, Qs / denom, 1.0 / 3.0)
    return np.sum(w * P, axis=1)

def _oracle_best(P: np.ndarray, y: np.ndarray, thresh: float) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    yhat = (P >= float(thresh)).astype(int)
    correct = (yhat == y[:, None])
    conf = np.abs(P - 0.5)
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

def _perm_stats(ba_clean: float, ba_broken: np.ndarray) -> Dict[str, float]:
    ba_broken = np.asarray(ba_broken, dtype=float)
    K = int(ba_broken.shape[0])
    delta = float(ba_clean - float(np.mean(ba_broken)))
    p = float((1.0 + float(np.sum(ba_broken >= ba_clean))) / float(K + 1))
    return {"delta_perm_fold": delta, "p_perm_fold": p, "ba_broken_mean": float(np.mean(ba_broken))}


# ----------------------------
# Main
# ----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--union_npz", type=str, required=True)
    ap.add_argument("--splits_dir", type=str, required=True)
    ap.add_argument("--preds_root", type=str, required=True, help=".../unimodal_preds/{lr|hgb}")

    ap.add_argument("--qsyn_clean_root", type=str, required=True)
    ap.add_argument("--qsyn_broken_root", type=str, required=True)

    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--family", type=str, choices=["lr", "hgb"], required=True)

    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--folds", type=int, nargs="+", required=True)

    ap.add_argument("--K", type=int, default=200)
    ap.add_argument("--L", type=float, default=8.0, help="corruption logit magnitude (must match precompute audit)")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--require_full_coverage", action="store_true")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    union = load_union(args.union_npz)
    preds_root = Path(args.preds_root)
    out_dir = Path(args.out_root) / args.family / "exp2_decisional_corruption_qsyn" / "late"

    fold_logs: List[Dict] = []

    for seed in args.seeds:
        for fold in args.folds:
            # canonical split + eval subset
            split = load_fold_split(args.splits_dir, seed=seed, fold=fold)
            _, test_mask = make_train_test_masks(
                union, split, require_full_coverage=bool(args.require_full_coverage)
            )
            eval_mask = eval_mask_full_only(union, test_mask)
            idx = np.where(eval_mask)[0]
            if idx.size == 0:
                raise RuntimeError(f"Empty FULL-only TEST for seed={seed} fold={fold}")

            # load unimodal preds (UNION-aligned)
            pred_pack = _load_unimodal_preds_npz(preds_root, seed=seed, fold=fold)
            ids_pred = np.asarray([str(x) for x in pred_pack["ids"]], dtype=object)
            if not np.array_equal(ids_pred, union.ids_str):
                raise ValueError(f"Preds ids order mismatch vs UNION for seed={seed} fold={fold}")

            p_a = np.asarray(pred_pack["p_a"], dtype=float)
            p_v = np.asarray(pred_pack["p_v"], dtype=float)
            p_p = np.asarray(pred_pack["p_p"], dtype=float)

            # contract sanity: missing probs must be NaN (even though FULL-only eval)
            assert_probs_nan_where_missing(union, p_a, p_v, p_p)

            y = union.y[idx].astype(int)
            P_base = np.stack([p_a[idx], p_v[idx], p_p[idx]], axis=1)
            if not np.isfinite(P_base).all():
                raise ValueError("Non-finite probs on FULL-only TEST (unexpected).")

            # load Qsyn clean + broken (K perms) via contract
            q_clean = load_fold_q(args.qsyn_clean_root, seed=seed, fold=fold, union=union, require_ids_match=True)
            q_brok = load_fold_q(args.qsyn_broken_root, seed=seed, fold=fold, union=union, require_ids_match=True)

            K_found = _infer_K(np.asarray(q_brok.Qa))
            if int(K_found) != int(args.K):
                raise ValueError(f"Expected K={args.K}, got K={K_found} in broken Qsyn seed={seed} fold={fold}")

            # derive corruption mask from Qsyn_clean on eval subset:
            # Qa = 1 - c on eval subset (by construction)
            Qa_eval = _q_scalar(np.asarray(q_clean.Qa))[idx]
            c = (1.0 - Qa_eval)  # should be 0/1
            c = (c >= 0.5).astype(np.uint8)

            # apply decisional corruption ONLY to audio probs on eval subset where c==1
            pa_eval = np.asarray(p_a[idx], dtype=float)
            la = _logit(pa_eval)
            s = np.where(y == 1, 1.0, -1.0)  # sign(2*y-1)
            corrupt_idx = np.where(c == 1)[0]
            if corrupt_idx.size > 0:
                la[corrupt_idx] = -s[corrupt_idx] * float(args.L)
            pa_corrupt = _sigmoid(la)

            # build corrupted P'
            P = P_base.copy()
            P[:, 0] = pa_corrupt  # only audio changes

            # evaluate CLEAN under Qsyn
            Q_eval = np.stack(
                [
                    _q_scalar(np.asarray(q_clean.Qa))[idx],
                    _q_scalar(np.asarray(q_clean.Qv))[idx],
                    _q_scalar(np.asarray(q_clean.Qp))[idx],
                ],
                axis=1,
            )
            p_fuse_clean = _quality_weighted_avg(P, Q_eval)
            ba_clean = float(balanced_accuracy_from_probs(y, p_fuse_clean, thresh=float(args.thresh)))

            # evaluate BROKEN_k under Qsyn_broken
            ba_broken = np.zeros(int(args.K), dtype=float)
            for k in range(int(args.K)):
                Qa_k = _get_broken_k(np.asarray(q_brok.Qa), k)
                Qv_k = _get_broken_k(np.asarray(q_brok.Qv), k)
                Qp_k = _get_broken_k(np.asarray(q_brok.Qp), k)
                Qk = np.stack([_q_scalar(Qa_k)[idx], _q_scalar(Qv_k)[idx], _q_scalar(Qp_k)[idx]], axis=1)
                p_fuse_k = _quality_weighted_avg(P, Qk)
                ba_broken[k] = float(balanced_accuracy_from_probs(y, p_fuse_k, thresh=float(args.thresh)))

            ps = _perm_stats(ba_clean, ba_broken)

            # oracle headroom under corrupted P'
            p_oracle = _oracle_best(P, y, thresh=float(args.thresh))
            ba_oracle = float(balanced_accuracy_from_probs(y, p_oracle, thresh=float(args.thresh)))
            headroom = float(ba_oracle - ba_clean)

            fold_logs.append(
                {
                    "experiment": "exp2_decisional_corruption_qsyn",
                    "mode": "late",
                    "family": args.family,
                    "seed": int(seed),
                    "fold": int(fold),
                    "K": int(args.K),
                    "L": float(args.L),
                    "n_eval": int(idx.size),
                    "n_corrupted_eval": int(c.sum()),
                    "ba_clean": ba_clean,
                    "ba_broken_mean": ps["ba_broken_mean"],
                    "delta_perm_fold": ps["delta_perm_fold"],
                    "p_perm_fold": ps["p_perm_fold"],
                    "oracle_ba": ba_oracle,
                    "oracle_headroom": headroom,
                    "qsyn_clean_path": str(q_clean.q_path),
                    "qsyn_broken_path": str(q_brok.q_path),
                    "preds_path": str(pred_pack["_path"]),
                }
            )

            print(
                f"[OK] seed={seed} fold={fold} "
                f"Δ={ps['delta_perm_fold']:+.4f} p={ps['p_perm_fold']:.4f} "
                f"headroom={headroom:+.4f} n_corrupt={int(c.sum())}"
            )

    deltas = np.asarray([r["delta_perm_fold"] for r in fold_logs], dtype=float)
    pvals = np.asarray([r["p_perm_fold"] for r in fold_logs], dtype=float)
    head = np.asarray([r["oracle_headroom"] for r in fold_logs], dtype=float)

    agg = {
        "experiment": "exp2_decisional_corruption_qsyn",
        "mode": "late",
        "family": args.family,
        "n_folds": int(len(fold_logs)),
        "delta_perm_mean": float(np.mean(deltas)),
        "delta_perm_std": float(np.std(deltas)),
        "p_perm_median": float(np.median(pvals)),
        "oracle_headroom_mean": float(np.mean(head)),
        "oracle_headroom_std": float(np.std(head)),
        "K": int(args.K),
        "L": float(args.L),
        "thresh": float(args.thresh),
    }

    _write_json(out_dir / "fold_logs.json", {"fold_logs": fold_logs})
    _write_json(out_dir / "aggregate.json", agg)

    print(
        f"[DONE] Δ={agg['delta_perm_mean']:+.4f}±{agg['delta_perm_std']:.4f} "
        f"p50={agg['p_perm_median']:.4f} headroom={agg['oracle_headroom_mean']:+.4f}"
    )


if __name__ == "__main__":
    main()