#!/usr/bin/env python3
# FINAL_EXPERIMENTS/FINAL/v5_3/precompute_qsyn_decisional_corruption.py
#
# Experiment 2 artifact precompute (paper-consistent):
# - Build synthetic quality Qsyn that marks decisional corruption on FULL-only TEST.
# - Precompute leakage-safe Broken-Qsyn via TEST-only, present-only permutation (K perms).
# - Output is UNION-aligned arrays with ids_str + union_ids_hash so experiments can LOAD via load_fold_q().
#
# This is the ONLY place Qsyn_broken is constructed. Evaluation scripts must LOAD it.
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
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from qfd._shared.q_contract import (
    UnionData,
    FoldSplit,
    load_union,
    load_fold_split,
    make_train_test_masks,
    eval_mask_full_only,
    union_ids_hash,
    assert_q_missing_is_zero,
    assert_no_nan_in_present_q,
)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _permute_within_indices(Q: np.ndarray, idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.asarray(Q, dtype=float).copy()
    if idx.size <= 1:
        return out
    perm = rng.permutation(idx.size)
    if out.ndim == 1:
        out[idx] = out[idx][perm]
    elif out.ndim == 2:
        out[idx, :] = out[idx, :][perm, :]
    else:
        raise ValueError(f"Q must be 1D or 2D, got {out.shape}")
    return out


def _alloc_broken(Qm: np.ndarray, K: int) -> np.ndarray:
    Qm = np.asarray(Qm, dtype=float)
    N = Qm.shape[0]
    if K <= 1:
        return np.zeros_like(Qm, dtype=float)
    if Qm.ndim == 1:
        return np.zeros((N, K), dtype=float)
    if Qm.ndim == 2:
        d = Qm.shape[1]
        return np.zeros((N, K, d), dtype=float)
    raise ValueError(f"Q must be 1D or 2D, got {Qm.shape}")


def build_qsyn_clean(
    *,
    union: UnionData,
    test_mask: np.ndarray,
    k: float,
    base_seed: int,
    seed: int,
    fold: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Construct Qsyn in UNION space.

    - Corruption mask c sampled ONLY on eval_mask = FULL-only TEST.
    - Qa = 1 - c on eval_mask, else 0.
    - Qv = 1 on eval_mask, else 0.
    - Qp = 1 on eval_mask, else 0.

    Contract:
    - Missing rows (M==0) forced to 0 exactly.
    """
    N = len(union.ids_str)
    test_mask = np.asarray(test_mask, dtype=bool)
    if test_mask.shape != (N,):
        raise ValueError(f"test_mask must be (N,), got {test_mask.shape}")

    eval_mask = eval_mask_full_only(union, test_mask)
    idx = np.where(eval_mask)[0]
    if idx.size == 0:
        raise RuntimeError(f"Empty FULL-only TEST for seed={seed} fold={fold}")

    if not (0.0 < k < 1.0):
        raise ValueError(f"k must be in (0,1). Got {k}")

    # deterministic fold RNG stream
    rng_seed = (int(base_seed) * 1000003 + int(seed) * 1009 + int(fold) * 9176) & 0xFFFFFFFF
    rng = np.random.default_rng(rng_seed)

    c = (rng.random(idx.size) < float(k)).astype(np.uint8)  # 1 = corrupted

    Qa = np.zeros(N, dtype=float)
    Qv = np.zeros(N, dtype=float)
    Qp = np.zeros(N, dtype=float)

    Qa[idx] = 1.0 - c.astype(float)

    # IMPORTANT: keep denom>0 invariant under TEST-only permutations.
    # Make v/p always "available+trusted" whenever present, so only Qa carries the signal.
    Qv[union.Mv == 1] = 1.0
    Qp[union.Mp == 1] = 1.0

    # enforce contract on missing rows globally
    Qa[union.Ma == 0] = 0.0
    Qv[union.Mv == 0] = 0.0
    Qp[union.Mp == 0] = 0.0

    assert_q_missing_is_zero(union, Qa, Qv, Qp)
    assert_no_nan_in_present_q(union, Qa, Qv, Qp)

    meta = {
        "kind": "Qsyn_clean_decisional_corruption",
        "seed": int(seed),
        "fold": int(fold),
        "k": float(k),
        "rng_seed": int(rng_seed),
        "n_eval": int(idx.size),
        "n_corrupted_eval": int(c.sum()),
        "eval_subset": "FULL-only TEST",
        "definition": {
            "Qa": "1 - corruption_mask on eval subset; 0 elsewhere",
            "Qv": "1 on PRESENT(v) rows; 0 where missing",
            "Qp": "1 on PRESENT(p) rows; 0 where missing",
        },
    }
    return Qa, Qv, Qp, meta


def build_qsyn_broken(
    *,
    union: UnionData,
    test_mask: np.ndarray,
    clean_Qa: np.ndarray,
    clean_Qv: np.ndarray,
    clean_Qp: np.ndarray,
    K: int,
    base_seed: int,
    seed: int,
    fold: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Build Broken-Qsyn (K permutations), leakage-safe:
    - permute each modality Q within TEST ∩ PRESENT(modality)
    - missing rows remain exactly 0
    """
    N = len(union.ids_str)
    test_mask = np.asarray(test_mask, dtype=bool)
    if test_mask.shape != (N,):
        raise ValueError(f"test_mask must be (N,), got {test_mask.shape}")

    idx_a = np.where(test_mask & (union.Ma == 1))[0]
    idx_v = np.where(test_mask & (union.Mv == 1))[0]
    idx_p = np.where(test_mask & (union.Mp == 1))[0]

    assert_q_missing_is_zero(union, clean_Qa, clean_Qv, clean_Qp)
    assert_no_nan_in_present_q(union, clean_Qa, clean_Qv, clean_Qp)

    Qa_out = _alloc_broken(clean_Qa, K)
    Qv_out = _alloc_broken(clean_Qv, K)
    Qp_out = _alloc_broken(clean_Qp, K)

    root_seed = (int(base_seed) * 1000003 + int(seed) * 1009 + int(fold) * 9176) & 0xFFFFFFFF

    for k in range(max(K, 1)):
        rng_a = np.random.default_rng((root_seed + 11_000_000 + k) & 0xFFFFFFFF)
        rng_v = np.random.default_rng((root_seed + 22_000_000 + k) & 0xFFFFFFFF)
        rng_p = np.random.default_rng((root_seed + 33_000_000 + k) & 0xFFFFFFFF)

        Qa_k = _permute_within_indices(clean_Qa, idx_a, rng_a)
        Qv_k = _permute_within_indices(clean_Qv, idx_v, rng_v)
        Qp_k = _permute_within_indices(clean_Qp, idx_p, rng_p)

        # re-enforce contract
        Qa_k[union.Ma == 0] = 0.0
        Qv_k[union.Mv == 0] = 0.0
        Qp_k[union.Mp == 0] = 0.0
        assert_q_missing_is_zero(union, Qa_k, Qv_k, Qp_k)
        assert_no_nan_in_present_q(union, Qa_k, Qv_k, Qp_k)

        if K <= 1:
            Qa_out, Qv_out, Qp_out = Qa_k, Qv_k, Qp_k
        else:
            Qa_out[:, k] = Qa_k
            Qv_out[:, k] = Qv_k
            Qp_out[:, k] = Qp_k

    meta = {
        "kind": "Qsyn_broken_decisional_corruption",
        "seed": int(seed),
        "fold": int(fold),
        "K": int(K),
        "base_seed": int(base_seed),
        "permute_sets": {
            "a_n_test_present": int(idx_a.size),
            "v_n_test_present": int(idx_v.size),
            "p_n_test_present": int(idx_p.size),
        },
        "permute_rule": "permute within TEST ∩ PRESENT(modality)",
    }
    return Qa_out, Qv_out, Qp_out, meta


def _save_q_npz(path: Path, *, union: UnionData, Qa: np.ndarray, Qv: np.ndarray, Qp: np.ndarray, meta: Dict) -> None:
    _ensure_dir(path.parent)
    np.savez_compressed(
        path,
        Qa=np.asarray(Qa, dtype=float),
        Qv=np.asarray(Qv, dtype=float),
        Qp=np.asarray(Qp, dtype=float),
        ids_str=np.asarray(union.ids_str, dtype=object),
        union_ids_hash=union_ids_hash(union.ids_str),
        meta=meta,
    )


def _clean_path(root: Path, seed: int, fold: int) -> Path:
    # discoverable by q_contract.find_fold_q_file(): seed_{seed}/fold_{fold}.npz
    return root / f"seed_{seed}" / f"fold_{fold}.npz"


def _broken_path(root: Path, seed: int, fold: int) -> Path:
    # also discoverable by find_fold_q_file()
    return root / f"seed_{seed}" / f"fold_{fold}.npz"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--union_npz", type=str, required=True)
    ap.add_argument("--splits_dir", type=str, required=True)
    ap.add_argument("--out_qsyn_clean_root", type=str, required=True)
    ap.add_argument("--out_qsyn_broken_root", type=str, required=True)

    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])

    ap.add_argument("--k", type=float, required=True, help="corruption probability on FULL-only TEST")
    ap.add_argument("--L", type=float, required=True, help="logged for audit; corruption magnitude applied in eval script")
    ap.add_argument("--K", type=int, default=200)
    ap.add_argument("--base_seed", type=int, default=12345)
    ap.add_argument("--require_full_coverage", action="store_true")

    args = ap.parse_args()

    union = load_union(args.union_npz)
    clean_root = Path(args.out_qsyn_clean_root)
    broken_root = Path(args.out_qsyn_broken_root)

    for seed in args.seeds:
        for fold in args.folds:
            split = load_fold_split(args.splits_dir, seed=seed, fold=fold)
            train_mask, test_mask = make_train_test_masks(
                union, split, require_full_coverage=bool(args.require_full_coverage)
            )

            Qa, Qv, Qp, meta_c = build_qsyn_clean(
                union=union,
                test_mask=test_mask,
                k=float(args.k),
                base_seed=int(args.base_seed),
                seed=int(seed),
                fold=int(fold),
            )
            meta_c = dict(meta_c)
            meta_c.update(
                {
                    "union_npz": str(Path(args.union_npz).resolve()),
                    "splits_dir": str(Path(args.splits_dir).resolve()),
                    "k": float(args.k),
                    "L": float(args.L),
                    "K_broken": int(args.K),
                    "n_union": int(len(union.ids_str)),
                    "n_train": int(train_mask.sum()),
                    "n_test": int(test_mask.sum()),
                }
            )

            Qa_b, Qv_b, Qp_b, meta_b = build_qsyn_broken(
                union=union,
                test_mask=test_mask,
                clean_Qa=Qa,
                clean_Qv=Qv,
                clean_Qp=Qp,
                K=int(args.K),
                base_seed=int(args.base_seed),
                seed=int(seed),
                fold=int(fold),
            )
            meta_b = dict(meta_b)
            meta_b.update(
                {
                    "union_npz": str(Path(args.union_npz).resolve()),
                    "splits_dir": str(Path(args.splits_dir).resolve()),
                    "k": float(args.k),
                    "L": float(args.L),
                    "K": int(args.K),
                    "n_union": int(len(union.ids_str)),
                    "n_train": int(train_mask.sum()),
                    "n_test": int(test_mask.sum()),
                }
            )

            p_clean = _clean_path(clean_root, seed=int(seed), fold=int(fold))
            p_brok = _broken_path(broken_root, seed=int(seed), fold=int(fold))

            _save_q_npz(p_clean, union=union, Qa=Qa, Qv=Qv, Qp=Qp, meta=meta_c)
            _save_q_npz(p_brok, union=union, Qa=Qa_b, Qv=Qv_b, Qp=Qp_b, meta=meta_b)

            print(f"[OK] seed={seed} fold={fold} wrote Qsyn clean={p_clean} broken={p_brok}")

    print("[DONE]")


if __name__ == "__main__":
    main()