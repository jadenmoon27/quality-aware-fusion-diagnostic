#!/usr/bin/env python3
# experiments/_shared/precompute_brokenq.py
#
# Precompute leakage-safe Broken-Q artifacts for every (seed, fold).
# This is the ONLY place Broken-Q is constructed. All experiments must LOAD these files.
#
# Contract (aligned to experiments/_shared/q_contract.py):
# - UNION row order is canonical; we never reorder.
# - Split ids are loaded from disk; never resplit.
# - Broken-Q is TEST-only and present-only per modality:
#     permute Q_m within { i in TEST : M_m(i)=1 }.
# - E and M are unchanged.
# - Missing rows must remain exactly zero in Q (all dims).
# - Output is UNION-aligned arrays with ids_str + union_ids_hash for alignment verification.
# python -m qfd.stressid.precompute_brokenq \
#   --union_npz /path/to/project/output/fusion_table_noQ/new_fusion_table_union_y2_avp_v1_noQ.npz \
#   --splits_dir /path/to/project/splits \
#   --q_clean_root /path/to/project/quality \
#   --out_broken_root /path/to/project/paper_output/quality/brokenQ_K200 \
#   --seeds 11 22 33 44 55 \
#   --folds 0 1 2 3 4 \
#   --K 200 \
#   --base_seed 12345 \
#   --require_full_coverage
#
# Output:
#   {out_root}/seed_{seed}/union_quality_BROKENQ_avp_seed{seed}_fold{fold}.npz
# containing:
#   Qa, Qv, Qp: shape (N, K) if K>1 else (N,)
#   ids_str: (N,) object
#   union_ids_hash: str
#   meta: dict (seed, fold, K, base_seed, test_present_counts, source_q_path, etc.)

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# IMPORTANT: this must import YOUR canonical contract module
from qfd._shared.q_contract import (
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

def _permute_within_indices(
    Q: np.ndarray,
    idx: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Return a permuted copy of Q where only rows at idx are permuted among themselves.
    Q can be (N,) or (N,d). idx is 1D int array.
    """
    out = Q.copy()
    if idx.size <= 1:
        return out

    perm = rng.permutation(idx.size)
    if Q.ndim == 1:
        out[idx] = out[idx][perm]
    elif Q.ndim == 2:
        out[idx, :] = out[idx, :][perm, :]
    else:
        raise ValueError(f"Q must be 1D or 2D, got shape {Q.shape}")
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
    Build Broken-Q arrays for (seed, fold).

    Returns:
      Qa_b, Qv_b, Qp_b each shaped (N,K) if K>1 else (N,)
      meta: audit dict
    """
    N = union.ids_str.shape[0]
    test_mask = np.asarray(test_mask, dtype=bool)
    if test_mask.shape != (N,):
        raise ValueError(f"test_mask must be (N,), got {test_mask.shape}")

    # Indices eligible for permutation per modality: TEST ∩ PRESENT(modality)
    idx_a = np.where(test_mask & (union.Ma == 1))[0]
    idx_v = np.where(test_mask & (union.Mv == 1))[0]
    idx_p = np.where(test_mask & (union.Mp == 1))[0]

    # Ensure clean Q already respects contract before permuting.
    assert_q_missing_is_zero(union, clean.Qa, clean.Qv, clean.Qp)
    assert_no_nan_in_present_q(union, clean.Qa, clean.Qv, clean.Qp)

    def alloc(Qm: np.ndarray) -> np.ndarray:
        if K <= 1:
            return np.zeros_like(Qm, dtype=float)
        if Qm.ndim == 1:
            return np.zeros((N, K), dtype=float)
        if Qm.ndim == 2:
            d = Qm.shape[1]
            return np.zeros((N, K, d), dtype=float)
        raise ValueError(f"Q must be 1D or 2D, got shape {Qm.shape}")

    Qa_out = alloc(clean.Qa)
    Qv_out = alloc(clean.Qv)
    Qp_out = alloc(clean.Qp)

    # Deterministic, fold-specific RNG stream. Also separate modality streams for auditability.
    # (This is NOT “using TEST information” — it’s just seeding; the permutation itself is TEST-only by design.)
    root_seed = (int(base_seed) * 1000003 + int(seed) * 1009 + int(fold) * 9176) & 0xFFFFFFFF

    for k in range(max(K, 1)):
        rng_a = np.random.default_rng((root_seed + 11_000_000 + k) & 0xFFFFFFFF)
        rng_v = np.random.default_rng((root_seed + 22_000_000 + k) & 0xFFFFFFFF)
        rng_p = np.random.default_rng((root_seed + 33_000_000 + k) & 0xFFFFFFFF)

        Qa_k = _permute_within_indices(np.asarray(clean.Qa, dtype=float), idx_a, rng_a)
        Qv_k = _permute_within_indices(np.asarray(clean.Qv, dtype=float), idx_v, rng_v)
        Qp_k = _permute_within_indices(np.asarray(clean.Qp, dtype=float), idx_p, rng_p)

        # Re-enforce contract invariants post-permutation (should hold automatically).
        assert_q_missing_is_zero(union, Qa_k, Qv_k, Qp_k)
        assert_no_nan_in_present_q(union, Qa_k, Qv_k, Qp_k)

        if K <= 1:
            Qa_out = Qa_k
            Qv_out = Qv_k
            Qp_out = Qp_k
        else:
            if Qa_k.ndim == 1:
                Qa_out[:, k] = Qa_k
                Qv_out[:, k] = Qv_k
                Qp_out[:, k] = Qp_k
            else:
                Qa_out[:, k, :] = Qa_k
                Qv_out[:, k, :] = Qv_k
                Qp_out[:, k, :] = Qp_k

    meta = {
        "kind": "Broken-Q",
        "seed": int(seed),
        "fold": int(fold),
        "K": int(K),
        "base_seed": int(base_seed),
        "root_seed": int(root_seed),
        "permute_sets": {
            "a_n_test_present": int(idx_a.size),
            "v_n_test_present": int(idx_v.size),
            "p_n_test_present": int(idx_p.size),
        },
        "source_q_path": str(clean.q_path),
    }
    return Qa_out, Qv_out, Qp_out, meta


# ----------------------------
# I/O helpers
# ----------------------------

def save_broken_q_npz(
    out_path: Path,
    *,
    union: UnionData,
    Qa: np.ndarray,
    Qv: np.ndarray,
    Qp: np.ndarray,
    meta: Dict,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Alignment verification: include ids_str AND union_ids_hash.
    uhash = union_ids_hash(union.ids_str)

    # Must be loadable by q_contract.load_fold_q():
    # keys are Qa/Qv/Qp (the loader tolerates multiple synonyms; we keep canonical).
    np.savez_compressed(
        out_path,
        Qa=np.asarray(Qa, dtype=float),
        Qv=np.asarray(Qv, dtype=float),
        Qp=np.asarray(Qp, dtype=float),
        ids_str=np.asarray(union.ids_str, dtype=object),
        union_ids_hash=uhash,
        meta=meta,
    )


def out_file_for(out_root: Path, seed: int, fold: int) -> Path:
    # Deterministic file name; q_contract.find_fold_q_file can discover this via rglob patterns.
    return out_root / f"seed_{seed}" / f"union_quality_BROKENQ_avp_seed{seed}_fold{fold}.npz"


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

            # Load CLEAN fold-safe Q (must already satisfy the contract).
            clean_q = load_fold_q(
                q_root, seed=seed, fold=fold, union=union, require_ids_match=True, allow_hash_match=True
            )

            Qa_b, Qv_b, Qp_b, meta = build_broken_q(
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
            save_broken_q_npz(out_path, union=union, Qa=Qa_b, Qv=Qv_b, Qp=Qp_b, meta=meta)

            print(f"[OK] wrote {out_path}  (K={args.K})")

    print("[DONE]")


if __name__ == "__main__":
    main()

