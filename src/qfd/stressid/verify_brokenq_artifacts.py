#!/usr/bin/env python3
# experiments/_shared/verify_brokenq_artifacts.py
#
# Sanity verifier: loads CLEAN Q and BROKEN Q via q_contract.load_fold_q()
# and checks invariants + reports useful audit stats.

# export PYTHONPATH=/path/to/project:$PYTHONPATH

# for seed in 11 22 33 44 55; do
#   for fold in 0 1 2 3 4; do
#     echo "=== seed=${seed} fold=${fold} ==="
#     python -m qfd.stressid.verify_brokenq_artifacts \
#       --union_npz /path/to/project/output/fusion_table_noQ/new_fusion_table_union_y2_avp_v1_noQ.npz \
#       --splits_dir /path/to/project/splits \
#       --q_clean_root /path/to/project/quality \
#       --q_broken_root /path/to/project/paper_output/quality/brokenQ_K200 \
#       --seed ${seed} \
#       --fold ${fold} \
#       --require_full_coverage || echo "FAILED seed=${seed} fold=${fold}"
#   done
# done

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from qfd._shared.q_contract import (
    load_union,
    load_fold_split,
    make_train_test_masks,
    load_fold_q,
    eval_mask_full_only,
    q_signature,
    assert_q_missing_is_zero,
    assert_no_nan_in_present_q,
)

def _identical_frac_1d(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    a = np.asarray(a); b = np.asarray(b); mask = np.asarray(mask, dtype=bool)
    return float(np.mean(a[mask] == b[mask])) if np.any(mask) else float("nan")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--union_npz", type=str, required=True)
    ap.add_argument("--splits_dir", type=str, required=True)
    ap.add_argument("--q_clean_root", type=str, required=True)
    ap.add_argument("--q_broken_root", type=str, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--require_full_coverage", action="store_true")
    args = ap.parse_args()

    union = load_union(args.union_npz)
    split = load_fold_split(args.splits_dir, seed=args.seed, fold=args.fold)
    _, test_mask = make_train_test_masks(union, split, require_full_coverage=bool(args.require_full_coverage))
    full_test = eval_mask_full_only(union, test_mask)

    clean = load_fold_q(args.q_clean_root, seed=args.seed, fold=args.fold, union=union)
    broken = load_fold_q(args.q_broken_root, seed=args.seed, fold=args.fold, union=union)

    # Must both satisfy contract
    assert_q_missing_is_zero(union, clean.Qa, clean.Qv, clean.Qp)
    assert_no_nan_in_present_q(union, clean.Qa, clean.Qv, clean.Qp)
    assert_q_missing_is_zero(union, broken.Qa, broken.Qv, broken.Qp)
    assert_no_nan_in_present_q(union, broken.Qa, broken.Qv, broken.Qp)

    # Report signatures on FULL-only TEST (useful to paste into logs)
    sig_clean = q_signature(union, clean, full_test)
    sig_broken = q_signature(union, broken, full_test)

    print("== Q signature (FULL-only TEST) ==")
    for k in sorted(sig_clean.keys()):
        if k.endswith("_n") or k.endswith("_mean") or k.endswith("_std") or k.endswith("_frac0"):
            print(f"{k:18s}  clean={sig_clean[k]:.6f}  broken={sig_broken[k]:.6f}")

    # If broken stores K permutations as 2D, check K and identical fractions against clean for first few perms.
    def firstcol(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q)
        if q.ndim == 1: return q
        if q.ndim == 2: return q[:, 0]
        if q.ndim == 3: return q[:, 0, :].mean(axis=1)
        raise ValueError(q.shape)

    # Present-only within TEST per modality: check identical fraction on TEST∩PRESENT.
    test_present_a = test_mask & (union.Ma == 1)
    test_present_v = test_mask & (union.Mv == 1)
    test_present_p = test_mask & (union.Mp == 1)

    print("\n== identical fraction clean vs broken (perm[0]) on TEST∩PRESENT ==")
    print("Qa:", _identical_frac_1d(firstcol(clean.Qa), firstcol(broken.Qa), test_present_a))
    print("Qv:", _identical_frac_1d(firstcol(clean.Qv), firstcol(broken.Qv), test_present_v))
    print("Qp:", _identical_frac_1d(firstcol(clean.Qp), firstcol(broken.Qp), test_present_p))

    print("\n[OK] verifier completed")

if __name__ == "__main__":
    main()


