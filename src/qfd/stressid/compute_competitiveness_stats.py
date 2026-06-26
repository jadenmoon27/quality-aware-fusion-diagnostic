#!/usr/bin/env python3
"""
02_compute_competitiveness_stats.py

Compute inter-expert competitiveness on StressID (or any UNION with {a,v,p})
STRICTLY following q_contract:

Inputs
- UNION: ids_str, y/y2, M_a/M_v/M_p
- unimodal preds (per seed×fold): ids, test_mask, p_a, p_v, p_p (UNION-aligned)
  (NaN where modality missing; finite on TEST∩PRESENT; per your unimodal dump contract)

Evaluation subset (paper):
- FULL-only TEST: test_mask ∩ (Ma=Mv=Mp=1)

Per sample i in subset:
- true-class confidence for modality m:
    c_m(i) = p_m(i)        if y_i=1
           = 1 - p_m(i)    if y_i=0
- Δ_i = c_best(i) - c_second(i)

Per fold, report:
- median Δ
- Competitive mass %: P(Δ < 0.05)
- Inter-expert correctness disagreement %:
    P( not all(ŷ_a=ŷ_v=ŷ_p) )   on FULL-only TEST

Aggregate across 25 folds:
- mean ± std across folds for each metric

Writes:
- CSV with 25 per-fold rows
- JSON with aggregate summary

python -m qfd.stressid.compute_competitiveness_stats \
  --union_npz /path/to/project/output/fusion_table_noQ/new_fusion_table_union_y2_avp_v1_noQ.npz \
  --splits_dir /path/to/project/splits \
  --preds_root /path/to/project/paper_output/unimodal_preds/lr \
  --seeds 11 22 33 44 55 \
  --folds 0 1 2 3 4 \
  --thresh 0.5 \
  --delta_thresh 0.05 \
  --out_csv  /path/to/project/paper_output/reports/competitiveness_lr.csv \
  --out_json /path/to/project/paper_output/reports/competitiveness_lr.json

# HGB
python -m qfd.stressid.compute_competitiveness_stats \
  --union_npz /path/to/project/output/fusion_table_noQ/new_fusion_table_union_y2_avp_v1_noQ.npz \
  --splits_dir /path/to/project/splits \
  --preds_root /path/to/project/paper_output/unimodal_preds/hgb \
  --seeds 11 22 33 44 55 \
  --folds 0 1 2 3 4 \
  --thresh 0.5 \
  --delta_thresh 0.05 \
  --out_csv  /path/to/project/paper_output/reports/competitiveness_hgb.csv \
  --out_json /path/to/project/paper_output/reports/competitiveness_hgb.json

Run it twice: once for LR preds_root, once for HGB preds_root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from qfd._shared.q_contract import (
    load_union,
    load_fold_split,
    make_train_test_masks,
    eval_mask_full_only,
    preds_from_probs,
    assert_probs_nan_where_missing,
)

# -----------------------------
# Core
# -----------------------------

def true_class_conf(p_pos: np.ndarray, y: np.ndarray) -> np.ndarray:
    """True-class confidence given P(y=1)."""
    p_pos = np.asarray(p_pos, dtype=float)
    y = np.asarray(y, dtype=int)
    return np.where(y == 1, p_pos, 1.0 - p_pos)

def best_second_from_three(ca: np.ndarray, cv: np.ndarray, cp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = np.stack([ca, cv, cp], axis=1)  # (n,3)
    srt = np.sort(X, axis=1)[:, ::-1]
    return srt[:, 0], srt[:, 1]

def summarize_fold(delta: np.ndarray, disagree: np.ndarray, delta_thresh: float) -> Dict[str, float]:
    if delta.size == 0:
        return {
            "median_delta": float("nan"),
            "competitive_mass_pct": float("nan"),
            "disagreement_pct": float("nan"),
        }
    return {
        "median_delta": float(np.median(delta)),
        "competitive_mass_pct": float(np.mean(delta < delta_thresh)),
        "disagreement_pct": float(np.mean(disagree)),
    }

def _require_keys(z: np.lib.npyio.NpzFile, fp: Path, keys: List[str]) -> None:
    files = set(z.files)
    missing = [k for k in keys if k not in files]
    if missing:
        raise KeyError(f"{fp}: missing keys {missing}. Found: {sorted(files)}")

def _assert_ids_aligned(ids_file: np.ndarray, ids_union: np.ndarray, fp: Path) -> None:
    ids_file = np.asarray([str(x) for x in ids_file], dtype=object)
    if ids_file.shape != ids_union.shape:
        raise ValueError(f"{fp}: ids length {ids_file.shape[0]} != UNION {ids_union.shape[0]}")
    if not np.array_equal(ids_file, ids_union):
        bad = np.where(ids_file != ids_union)[0][:10]
        sample = [(int(i), ids_union[i], ids_file[i]) for i in bad]
        raise ValueError(f"{fp}: ids not aligned to UNION order. Sample mismatches: {sample}")

def _load_fold_preds(fp: Path, union) -> Dict[str, np.ndarray]:
    z = np.load(fp, allow_pickle=True)
    _require_keys(z, fp, ["ids", "test_mask", "p_a", "p_v", "p_p"])
    _assert_ids_aligned(z["ids"], union.ids_str, fp)

    test_mask = np.asarray(z["test_mask"]).astype(bool).reshape(-1)
    p_a = np.asarray(z["p_a"], dtype=float).reshape(-1)
    p_v = np.asarray(z["p_v"], dtype=float).reshape(-1)
    p_p = np.asarray(z["p_p"], dtype=float).reshape(-1)

    # contract checks (strict + defensive)
    if test_mask.shape != (len(union.ids_str),):
        raise ValueError(f"{fp}: test_mask shape {test_mask.shape} != (N,)")

    assert_probs_nan_where_missing(union, p_a, p_v, p_p)

    return {"test_mask": test_mask, "p_a": p_a, "p_v": p_v, "p_p": p_p}

# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--union_npz", required=True)
    ap.add_argument("--splits_dir", required=True, help="Used to recompute test_mask and check it matches preds.")
    ap.add_argument("--preds_root", required=True, help="Root containing seed_{seed}/fold_{fold}.npz")
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--delta_thresh", type=float, default=0.05)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_json", required=True)
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    union = load_union(args.union_npz)
    N = len(union.ids_str)

    preds_root = Path(args.preds_root)
    rows: List[Dict] = []

    for seed in args.seeds:
        for fold in args.folds:
            fp = preds_root / f"seed_{seed}" / f"fold_{fold}.npz"
            if not fp.exists():
                raise FileNotFoundError(fp)

            # recompute masks from canonical splits (defensible)
            split = load_fold_split(args.splits_dir, seed=seed, fold=fold)
            train_mask, test_mask_contract = make_train_test_masks(union, split)
            full_only_test = eval_mask_full_only(union, test_mask_contract)

            d = _load_fold_preds(fp, union)
            test_mask_file = d["test_mask"]

            # strict: the stored test_mask must match the contract split exactly
            if not np.array_equal(test_mask_file, test_mask_contract):
                bad = np.where(test_mask_file != test_mask_contract)[0][:10]
                sample = [(int(i), union.ids_str[i], int(test_mask_contract[i]), int(test_mask_file[i])) for i in bad]
                raise ValueError(
                    f"{fp}: test_mask does not match split-derived mask. "
                    f"Sample (row,id,contract,file): {sample}"
                )

            idx = np.where(full_only_test)[0]
            if idx.size == 0:
                raise ValueError(f"{fp}: FULL-only TEST empty under contract.")

            y = union.y[idx].astype(int)
            p_a = d["p_a"][idx]
            p_v = d["p_v"][idx]
            p_p = d["p_p"][idx]

            # FULL-only => present => must be finite
            if not (np.isfinite(p_a).all() and np.isfinite(p_v).all() and np.isfinite(p_p).all()):
                raise ValueError(f"{fp}: non-finite probs on FULL-only TEST (should be finite).")

            c_a = true_class_conf(p_a, y)
            c_v = true_class_conf(p_v, y)
            c_p = true_class_conf(p_p, y)

            best, second = best_second_from_three(c_a, c_v, c_p)
            delta = best - second

            yhat_a = preds_from_probs(p_a, thresh=args.thresh)
            yhat_v = preds_from_probs(p_v, thresh=args.thresh)
            yhat_p = preds_from_probs(p_p, thresh=args.thresh)

            # disagreement: not(all equal)
            disagree = ~((yhat_a == yhat_v) & (yhat_a == yhat_p))

            s = summarize_fold(delta, disagree, delta_thresh=args.delta_thresh)

            rows.append(
                {
                    "seed": int(seed),
                    "fold": int(fold),
                    "n_full_only_test": int(idx.size),
                    "median_delta": s["median_delta"],
                    "competitive_mass_pct": s["competitive_mass_pct"],
                    "disagreement_pct": s["disagreement_pct"],
                    "preds_file": str(fp),
                }
            )

            print(
                f"[OK] seed={seed} fold={fold} n_full={idx.size} "
                f"medΔ={s['median_delta']:.4f} "
                f"mass(Δ<{args.delta_thresh:.2f})={100*s['competitive_mass_pct']:.2f}% "
                f"disagree={100*s['disagreement_pct']:.2f}%"
            )

    df = pd.DataFrame(rows)
    if len(df) != len(args.seeds) * len(args.folds):
        raise AssertionError("Unexpected fold count mismatch.")

    def mean_std(x: pd.Series) -> Dict[str, float]:
        return {"mean": float(x.mean()), "std": float(x.std())}

    summary = {
        "median_delta": mean_std(df["median_delta"]),
        "competitive_mass_pct": mean_std(df["competitive_mass_pct"]),
        "disagreement_pct": mean_std(df["disagreement_pct"]),
        "n_full_only_test": mean_std(df["n_full_only_test"]),
        "n_folds": int(len(df)),
        "delta_thresh": float(args.delta_thresh),
        "thresh": float(args.thresh),
    }

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(
            {"args": vars(args), "summary_mean_std_across_folds": summary, "per_fold_csv": str(out_csv)},
            f,
            indent=2,
        )

    print("\n=== Aggregate (mean ± std across folds) ===")
    print(f"Median Δ              : {summary['median_delta']['mean']:.4f} ± {summary['median_delta']['std']:.4f}")
    print(f"Competitive mass      : {100*summary['competitive_mass_pct']['mean']:.1f}% ± {100*summary['competitive_mass_pct']['std']:.1f}%")
    print(f"Correctness disagreement: {100*summary['disagreement_pct']['mean']:.1f}% ± {100*summary['disagreement_pct']['std']:.1f}%")
    print(f"N FULL-only TEST      : {summary['n_full_only_test']['mean']:.1f} ± {summary['n_full_only_test']['std']:.1f}")
    print(f"\n[WROTE] {out_csv}")
    print(f"[WROTE] {out_json}")

if __name__ == "__main__":
    main()


