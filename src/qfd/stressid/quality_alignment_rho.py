#!/usr/bin/env python3
"""
Compute Spearman alignment between fold-scaled quality Q and unimodal error on FULL-only TEST.

Outputs per-modality rho(Q, error) and rho(Q, correctness) aggregated across 25 runs.

Run twice:
  --unimodal_preds_root .../unimodal_preds/lr
  --unimodal_preds_root .../unimodal_preds/hgb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from qfd._shared.q_contract import (
    UnionData,
    FoldSplit,
    FoldQ,
    load_union,
    load_fold_split,
    make_train_test_masks,
    eval_mask_full_only,
    load_fold_q,
    assert_q_missing_is_zero,
    assert_no_nan_in_present_q,
    assert_probs_nan_where_missing,
    preds_from_probs,
)

# -----------------------------
# Spearman (robust, no scipy dependency)
# -----------------------------

def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks for ties, 1..n."""
    a = np.asarray(a)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)

    # handle ties
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    return ranks


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape[0] != y.shape[0]:
        raise ValueError("spearman_rho: x and y must have same length")
    if x.shape[0] < 3:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx * rx).sum()) * np.sqrt((ry * ry).sum()))
    if denom == 0.0:
        return float("nan")
    return float((rx * ry).sum() / denom)


# -----------------------------
# Unimodal probs loader (matches your 03B patterns)
# -----------------------------

def _find_unimodal_preds_file(root: Path, seed: int, fold: int) -> Path:
    cand = [
        root / f"seed_{seed}" / f"union_unimodal_preds_seed{seed}_fold{fold}.npz",
        root / f"seed_{seed}" / f"fold_{fold}.npz",
        root / f"seed_{seed}" / f"fold{fold}.npz",
        root / f"union_unimodal_preds_seed{seed}_fold{fold}.npz",
        root / f"seed_{seed}_fold_{fold}.npz",
    ]
    for p in cand:
        if p.exists():
            return p
    hits = list(root.rglob(f"*seed{seed}*fold{fold}*.npz"))
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise FileNotFoundError(
            f"Ambiguous unimodal preds for seed={seed}, fold={fold} under {root}:\n"
            + "\n".join(map(str, hits[:50]))
        )
    raise FileNotFoundError(f"Missing unimodal preds for seed={seed}, fold={fold} under {root}")


def _load_unimodal_probs(npz_path: Path, *, expect_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(npz_path, allow_pickle=True)
    keys = set(z.files)

    def get_any(cands: List[str]) -> np.ndarray:
        for k in cands:
            if k in keys:
                return z[k]
        raise KeyError(f"{npz_path} missing keys {cands}. Found {sorted(keys)}")

    pa = get_any(["p_a", "pa", "prob_a", "probs_a", "p_audio", "audio_p", "pA"])
    pv = get_any(["p_v", "pv", "prob_v", "probs_v", "p_video", "video_p", "pV"])
    pp = get_any(["p_p", "pp", "prob_p", "probs_p", "p_phys", "phys_p", "pP"])

    for name, arr in [("p_a", pa), ("p_v", pv), ("p_p", pp)]:
        if arr.ndim != 1 or arr.shape[0] != expect_len:
            raise ValueError(f"{name} shape {arr.shape} != ({expect_len},) in {npz_path}")
        if np.isinf(arr).any():
            raise ValueError(f"{name} contains inf in {npz_path}")

    pa = np.clip(pa.astype(float), 0.0, 1.0)
    pv = np.clip(pv.astype(float), 0.0, 1.0)
    pp = np.clip(pp.astype(float), 0.0, 1.0)
    return pa, pv, pp


def _mean_std(xs: List[float]) -> Dict[str, float]:
    arr = np.array(xs, dtype=float)
    return {"mean": float(np.nanmean(arr)), "std": float(np.nanstd(arr, ddof=0))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, required=True, help="Label for this run (e.g., lr or hgb).")
    ap.add_argument("--union_npz", type=str, required=True)
    ap.add_argument("--splits_dir", type=str, required=True)
    ap.add_argument("--unimodal_preds_root", type=str, required=True)
    ap.add_argument("--q_clean_root", type=str, required=True)
    ap.add_argument("--out_json", type=str, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--thresh", type=float, default=0.5)
    args = ap.parse_args()

    union: UnionData = load_union(args.union_npz)
    N = union.ids.shape[0]

    unimodal_root = Path(args.unimodal_preds_root)
    q_clean_root = Path(args.q_clean_root)

    # Collect per-run rhos
    rhos_err = {"audio": [], "video": [], "phys": []}
    rhos_cor = {"audio": [], "video": [], "phys": []}

    for seed in args.seeds:
        for fold in args.folds:
            split: FoldSplit = load_fold_split(args.splits_dir, seed, fold)
            train_mask, test_mask = make_train_test_masks(union, split)
            eval_mask = eval_mask_full_only(union, test_mask)

            if int(eval_mask.sum()) == 0:
                raise RuntimeError(f"seed={seed} fold={fold}: FULL-only TEST empty.")

            unimodal_path = _find_unimodal_preds_file(unimodal_root, seed, fold)
            pa, pv, pp = _load_unimodal_probs(unimodal_path, expect_len=N)

            # Contract: NaN where missing (FULL-only eval implies none missing, but still validate globally)
            assert_probs_nan_where_missing(union, pa, pv, pp)

            q_clean: FoldQ = load_fold_q(
                q_clean_root,
                seed,
                fold,
                union=union,
                require_ids_match=True,
            )

            assert_q_missing_is_zero(union, q_clean.Qa, q_clean.Qv, q_clean.Qp)
            assert_no_nan_in_present_q(union, q_clean.Qa, q_clean.Qv, q_clean.Qp)

            y = union.y.astype(int)
            y_t = y[eval_mask]

            # unimodal correctness (binary label via same threshold)
            ya = preds_from_probs(pa[eval_mask], thresh=args.thresh)
            yv = preds_from_probs(pv[eval_mask], thresh=args.thresh)
            yp = preds_from_probs(pp[eval_mask], thresh=args.thresh)

            cor_a = (ya == y_t).astype(float)
            cor_v = (yv == y_t).astype(float)
            cor_p = (yp == y_t).astype(float)

            err_a = 1.0 - cor_a
            err_v = 1.0 - cor_v
            err_p = 1.0 - cor_p

            Qa = q_clean.Qa[eval_mask].astype(float)
            Qv = q_clean.Qv[eval_mask].astype(float)
            Qp = q_clean.Qp[eval_mask].astype(float)

            # Spearman rho between Q and error (paper phrase: quality--error alignment)
            rhos_err["audio"].append(spearman_rho(Qa, err_a))
            rhos_err["video"].append(spearman_rho(Qv, err_v))
            rhos_err["phys"].append(spearman_rho(Qp, err_p))

            # Also store rho(Q, correctness) for interpretability (should be opposite sign)
            rhos_cor["audio"].append(spearman_rho(Qa, cor_a))
            rhos_cor["video"].append(spearman_rho(Qv, cor_v))
            rhos_cor["phys"].append(spearman_rho(Qp, cor_p))

            print(
                f"[OK] {args.tag} seed={seed} fold={fold} | "
                f"rho(Q,error): a={rhos_err['audio'][-1]:.3f} v={rhos_err['video'][-1]:.3f} p={rhos_err['phys'][-1]:.3f}"
            )

    summary = {
        "meta": {
            "tag": args.tag,
            "union_npz": args.union_npz,
            "splits_dir": args.splits_dir,
            "unimodal_preds_root": str(unimodal_root),
            "q_clean_root": str(q_clean_root),
            "seeds": args.seeds,
            "folds": args.folds,
            "thresh": args.thresh,
            "eval_subset": "FULL-only within TEST",
            "rho_definition": "Spearman rank correlation computed per (seed,fold) on FULL-only TEST, then aggregated",
        },
        "rho_q_error": {m: _mean_std(v) for m, v in rhos_err.items()},
        "rho_q_correctness": {m: _mean_std(v) for m, v in rhos_cor.items()},
        "raw": {
            "rho_q_error": rhos_err,
            "rho_q_correctness": rhos_cor,
        },
    }

    # ---- print paper-ready summary (mean ± std over 25 folds) ----
    print("\n=== PAPER-READY (mean ± std over 25 folds; Spearman rho(Q, correctness)) ===")
    for m in ["audio", "video", "phys"]:
        mu = summary["rho_q_correctness"][m]["mean"]
        sd = summary["rho_q_correctness"][m]["std"]
        print(f"rho_{m}: {mu:.3f} ± {sd:.3f}")

    print("\n=== PAPER-READY (mean ± std over 25 folds; Spearman rho(Q, error)) ===")
    for m in ["audio", "video", "phys"]:
        mu = summary["rho_q_error"][m]["mean"]
        sd = summary["rho_q_error"][m]["std"]
        print(f"rho_{m}: {mu:.3f} ± {sd:.3f}")

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"[DONE] wrote {out_json}")



if __name__ == "__main__":
    main()



