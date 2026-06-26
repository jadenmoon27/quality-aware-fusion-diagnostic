#!/usr/bin/env python3
# FINAL_EXPERIMENTS/_shared/precompute_brokenq_mosei.py
#
# Precompute leakage-safe Broken-Q artifacts for every (seed, fold) on MOSEI.
# This is the ONLY place Broken-Q is constructed. All experiments must LOAD these files.
#
# Contract (aligned to qfd/_shared/q_contract_mosei.py):
# - UNION row order is canonical; we never reorder.
# - Split ids are loaded from disk; never resplit.
# - Broken-Q is TEST-only and present-only per modality:
#     permute Q_m within { i in TEST : M_m(i)=1 }.
# - E and M are unchanged.
# - Missing rows must remain exactly zero in Q (all dims).
# - Output is UNION-aligned arrays with ids_str + union_ids_hash for alignment verification.
#
# Example:
# python -m qfd.mosei.precompute_brokenq \
#   --union_npz /path/to/project/output/final_experiments/mosei/union/mosei_union.npz \
#   --splits_dir /path/to/project/output/final_experiments/mosei/splits_mosei \
#   --q_clean_root /path/to/project/output/final_experiments/mosei/quality_fold \
#   --out_broken_root /path/to/project/output/final_experiments/mosei/quality_fold_broken_K200 \
#   --seeds 11 22 33 44 55 \
#   --folds 0 1 2 3 4 \
#   --K 200 \
#   --base_seed 12345 \
#   --require_full_coverage
#
# Output:
#   {out_root}/seed_{seed}/fold_{fold}.npz
# containing:
#   Ql, Qa, Qv: shape (N, K) if Q is 1D and K>1
#              shape (N, K, d) if Q is 2D and K>1
#              else (N,) or (N,d) when K<=1
#   ids_str: (N,) object
#   union_ids_hash: str
#   meta: dict (seed, fold, K, base_seed, test_present_counts, source_q_path, etc.)

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# IMPORTANT: import MOSEI contract (NOT StressID)
from qfd._shared.q_contract_mosei import (
    UnionData,
    FoldSplit,
    FoldQ,
    load_union,
    load_fold_split,
    make_train_test_masks,
    load_fold_q,
    union_ids_hash,
    assert_q_missing_is_zero,
    assert_no_nan_in_present_q,
)

# ----------------------------
# Core permutation logic
# ----------------------------

def _permute_within_indices(Q: np.ndarray, idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Return a permuted copy of Q where only rows at idx are permuted among themselves.
    Q can be (N,) or (N,d). idx is 1D int array.
    """
    Q = np.asarray(Q, dtype=float)
    out = Q.copy()
    if idx.size <= 1:
        return out

    perm = rng.permutation(idx.size)
    if out.ndim == 1:
        out[idx] = out[idx][perm]
    elif out.ndim == 2:
        out[idx, :] = out[idx, :][perm, :]
    else:
        raise ValueError(f"Q must be 1D or 2D, got shape {out.shape}")
    return out


def build_broken_q(
    *,
    union: UnionData,
    test_mask: np.ndarray,
    clean: FoldQ,
    K: int,
    base_seed: int,
    seed: int,
    fold: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Build Broken-Q arrays for (seed, fold) on MOSEI.

    Returns:
      Ql_b, Qa_b, Qv_b each shaped:
        - if K<=1: same shape as clean.Qm -> (N,) or (N,d)
        - if K>1:
            * clean.Qm is (N,)   -> (N,K)
            * clean.Qm is (N,d)  -> (N,K,d)
      meta: audit dict
    """
    N = union.ids_str.shape[0]
    test_mask = np.asarray(test_mask, dtype=bool)
    if test_mask.shape != (N,):
        raise ValueError(f"test_mask must be (N,), got {test_mask.shape}")

    # Eligible indices per modality: TEST ∩ PRESENT(modality)
    idx_l = np.where(test_mask & (union.Ml == 1))[0]
    idx_a = np.where(test_mask & (union.Ma == 1))[0]
    idx_v = np.where(test_mask & (union.Mv == 1))[0]

    # Ensure clean Q respects contract pre-permutation.
    assert_q_missing_is_zero(union, clean.Ql, clean.Qa, clean.Qv)
    assert_no_nan_in_present_q(union, clean.Ql, clean.Qa, clean.Qv)

    def alloc(Qm: np.ndarray) -> np.ndarray:
        Qm = np.asarray(Qm)
        if K <= 1:
            return np.zeros_like(Qm, dtype=float)
        if Qm.ndim == 1:
            return np.zeros((N, K), dtype=float)
        if Qm.ndim == 2:
            d = Qm.shape[1]
            return np.zeros((N, K, d), dtype=float)
        raise ValueError(f"Q must be 1D or 2D, got shape {Qm.shape}")

    Ql_out = alloc(clean.Ql)
    Qa_out = alloc(clean.Qa)
    Qv_out = alloc(clean.Qv)

    # Deterministic, fold-specific RNG stream (same philosophy as StressID).
    root_seed = (int(base_seed) * 1000003 + int(seed) * 1009 + int(fold) * 9176) & 0xFFFFFFFF

    for k in range(max(K, 1)):
        rng_l = np.random.default_rng((root_seed + 10_000_000 + k) & 0xFFFFFFFF)
        rng_a = np.random.default_rng((root_seed + 20_000_000 + k) & 0xFFFFFFFF)
        rng_v = np.random.default_rng((root_seed + 30_000_000 + k) & 0xFFFFFFFF)

        Ql_k = _permute_within_indices(clean.Ql, idx_l, rng_l)
        Qa_k = _permute_within_indices(clean.Qa, idx_a, rng_a)
        Qv_k = _permute_within_indices(clean.Qv, idx_v, rng_v)

        # Re-enforce invariants post-permutation.
        assert_q_missing_is_zero(union, Ql_k, Qa_k, Qv_k)
        assert_no_nan_in_present_q(union, Ql_k, Qa_k, Qv_k)

        if K <= 1:
            Ql_out = Ql_k
            Qa_out = Qa_k
            Qv_out = Qv_k
        else:
            if Ql_k.ndim == 1:
                Ql_out[:, k] = Ql_k
                Qa_out[:, k] = Qa_k
                Qv_out[:, k] = Qv_k
            else:
                Ql_out[:, k, :] = Ql_k
                Qa_out[:, k, :] = Qa_k
                Qv_out[:, k, :] = Qv_k

    meta = {
        "kind": "Broken-Q",
        "dataset": "MOSEI",
        "seed": int(seed),
        "fold": int(fold),
        "K": int(K),
        "base_seed": int(base_seed),
        "root_seed": int(root_seed),
        "permute_sets": {
            "l_n_test_present": int(idx_l.size),
            "a_n_test_present": int(idx_a.size),
            "v_n_test_present": int(idx_v.size),
        },
        "source_q_path": str(clean.q_path),
        "note": "Permutation is TEST-only and present-only per modality; no scaling/recompute.",
    }
    return Ql_out, Qa_out, Qv_out, meta


# ----------------------------
# I/O helpers
# ----------------------------

def save_broken_q_npz(
    out_path: Path,
    *,
    union: UnionData,
    Ql: np.ndarray,
    Qa: np.ndarray,
    Qv: np.ndarray,
    meta: Dict,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    uhash = union_ids_hash(union.ids_str)

    # Must be loadable by q_contract_mosei.load_fold_q(): use canonical keys Ql/Qa/Qv.
    np.savez_compressed(
        out_path,
        Ql=np.asarray(Ql, dtype=float),
        Qa=np.asarray(Qa, dtype=float),
        Qv=np.asarray(Qv, dtype=float),
        ids_str=np.asarray(union.ids_str, dtype=object),
        union_ids_hash=uhash,
        meta=meta,
    )


def out_file_for(out_root: Path, seed: int, fold: int) -> Path:
    # Keep it trivially discoverable by q_contract_mosei.find_fold_q_file()
    return out_root / f"seed_{seed}" / f"fold_{fold}.npz"


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--union_npz", type=str, required=True)
    ap.add_argument("--splits_dir", type=str, required=True)
    ap.add_argument("--q_clean_root", type=str, required=True)
    ap.add_argument("--out_broken_root", type=str, required=True)

    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])

    ap.add_argument("--K", type=int, default=200, help="number of Broken-Q permutations stored per fold")
    ap.add_argument("--base_seed", type=int, default=12345)
    ap.add_argument(
        "--require_full_coverage",
        action="store_true",
        help="enforce train∪test covers UNION for each fold (recommended if that is your split contract)",
    )
    args = ap.parse_args()

    union = load_union(args.union_npz)

    out_root = Path(args.out_broken_root)
    q_root = Path(args.q_clean_root)

    for seed in args.seeds:
        for fold in args.folds:
            split = load_fold_split(args.splits_dir, seed=seed, fold=fold)
            train_mask, test_mask = make_train_test_masks(
                union, split, require_full_coverage=bool(args.require_full_coverage)
            )

            # Load CLEAN fold Q (must already satisfy the contract).
            clean_q = load_fold_q(
                q_root, seed=seed, fold=fold, union=union, require_ids_match=True, allow_hash_match=True
            )

            Ql_b, Qa_b, Qv_b, meta = build_broken_q(
                union=union,
                test_mask=test_mask,
                clean=clean_q,
                K=int(args.K),
                base_seed=int(args.base_seed),
                seed=int(seed),
                fold=int(fold),
            )

            meta = dict(meta)
            meta.update(
                {
                    "union_npz": str(Path(args.union_npz).resolve()),
                    "splits_dir": str(Path(args.splits_dir).resolve()),
                    "q_clean_root": str(q_root.resolve()),
                    "out_broken_root": str(out_root.resolve()),
                    "n_union": int(len(union.ids_str)),
                    "n_train": int(train_mask.sum()),
                    "n_test": int(test_mask.sum()),
                }
            )

            out_path = out_file_for(out_root, seed=seed, fold=fold)
            save_broken_q_npz(out_path, union=union, Ql=Ql_b, Qa=Qa_b, Qv=Qv_b, meta=meta)

            print(f"[OK] wrote {out_path}  (K={int(args.K)})")

    print("[DONE]")


if __name__ == "__main__":
    main()