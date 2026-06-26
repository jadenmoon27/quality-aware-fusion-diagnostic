#!/usr/bin/env python3
"""
compute_mosei_delta_rho_fullonly.py

Compute MOSEI structural stats on FULL-only TEST per (seed,fold), aligned to UNION row order.

Outputs:
  - per-fold CSV with Δ stats + rank-corr (Spearman + Kendall) for Q vs correctness
  - JSON with aggregated mean±std across folds (and optional per-fold records)

Assumptions (match q_contract_mosei.py you pasted):
  UNION keys: ids, y (or y2), E_l, E_a, E_v, M_l, M_a, M_v
  FULL-only mask: (M_l==1 & M_a==1 & M_v==1) within TEST
  Q files: per seed×fold NPZ with Ql/Qa/Qv + ids for order verification
  Unimodal preds: per seed×fold NPZ with p_l/p_a/p_v (UNION-length), aligned to UNION row order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

from qfd._shared.q_contract_mosei import (
    load_union,
    load_fold_split,
    make_train_test_masks,
    eval_mask_full_only,
    load_fold_q,
)


# -------------------------
# Helpers
# -------------------------

def _true_class_conf(p_pos: np.ndarray, y: np.ndarray) -> np.ndarray:
    """p(y)=p if y=1 else 1-p (binary)."""
    return np.where(y == 1, p_pos, 1.0 - p_pos)


def _pred_label(p_pos: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    return (p_pos >= thresh).astype(int)


def _as_scalar_q(Q: np.ndarray) -> np.ndarray:
    """Accept Q shape (N,) or (N,d); return scalar (N,) via mean over dims if needed."""
    Q = np.asarray(Q)
    if Q.ndim == 1:
        return Q.astype(float)
    if Q.ndim == 2:
        return np.mean(Q.astype(float), axis=1)
    raise ValueError(f"Q must be 1D or 2D, got shape {Q.shape}")


def _load_preds_file(preds_root: Path, seed: int, fold: int) -> Path:
    """
    Find a unimodal preds NPZ for this (seed,fold).
    We support a few common layouts:
      preds_root/seed_{seed}/fold_{fold}.npz
      preds_root/seed_{seed}/fold{fold}.npz
      preds_root/seed_{seed}/fold_{fold}/preds.npz
      preds_root/seed_{seed}_fold_{fold}.npz
    Otherwise rglob for *seed{seed}*fold{fold}*.npz.
    """
    root = Path(preds_root)

    cand = [
        root / f"seed_{seed}" / f"fold_{fold}.npz",
        root / f"seed_{seed}" / f"fold{fold}.npz",
        root / f"seed_{seed}" / f"fold_{fold}" / "preds.npz",
        root / f"seed_{seed}" / f"fold_{fold}" / "unimodal_preds.npz",
        root / f"seed_{seed}_fold_{fold}.npz",
        root / f"seed{seed}_fold{fold}.npz",
    ]
    for p in cand:
        if p.exists():
            return p

    hits = list(root.rglob(f"*seed{seed}*fold{fold}*.npz"))
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise FileNotFoundError(
            f"Ambiguous preds file for seed={seed}, fold={fold} under {root}. "
            f"Matches:\n" + "\n".join(map(str, hits[:40]))
        )
    raise FileNotFoundError(
        f"Could not find preds file for seed={seed}, fold={fold} under {root}. "
        f"Edit _load_preds_file() patterns to match your directory structure."
    )


def _load_unimodal_preds_union(
    preds_root: Path,
    seed: int,
    fold: int,
    union_ids_str: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Load p_l/p_a/p_v arrays of shape (N,) aligned to UNION.
    If ids exist in preds npz, verify exact order match to union_ids_str.
    """
    p_path = _load_preds_file(preds_root, seed, fold)
    z = np.load(p_path, allow_pickle=True)
    files = set(z.files)

    def req(key: str) -> None:
        if key not in files:
            raise KeyError(f"Preds file {p_path} missing key '{key}'. Found keys: {sorted(files)}")

    # Accept a couple name variants
    def get_any(keys) -> np.ndarray:
        for k in keys:
            if k in files:
                return np.asarray(z[k], dtype=float)
        raise KeyError(f"Preds file {p_path} missing keys {keys}. Found keys: {sorted(files)}")

    p_l = get_any(["p_l", "p_lang", "p_text", "pl"])
    p_a = get_any(["p_a", "p_audio", "pa"])
    p_v = get_any(["p_v", "p_video", "pv"])

    n = len(union_ids_str)
    for name, arr in [("p_l", p_l), ("p_a", p_a), ("p_v", p_v)]:
        if arr.shape != (n,):
            raise ValueError(f"{name} shape {arr.shape} != (N,) with N={n} in {p_path}")

    # Optional alignment verification
    ids_key = None
    for k in ["ids", "ids_str", "union_ids"]:
        if k in files:
            ids_key = k
            break
    if ids_key is not None:
        p_ids = np.asarray([str(x) for x in z[ids_key]], dtype=object)
        if p_ids.shape[0] != n:
            raise ValueError(f"Preds ids length {p_ids.shape[0]} != N={n} in {p_path}")
        if not np.array_equal(p_ids, union_ids_str):
            bad = np.where(p_ids != union_ids_str)[0][:10]
            sample = [(int(i), union_ids_str[i], p_ids[i]) for i in bad]
            raise ValueError(
                f"Preds ids do not match UNION ids order in {p_path}. "
                f"Mismatches (row, union_id, pred_id) sample: {sample}"
            )

    return p_l, p_a, p_v, str(p_path)


def _corr_rank(q: np.ndarray, correct: np.ndarray) -> Tuple[float, float]:
    """
    Return (spearman_r, kendall_tau) on valid rows.
    correct is {0,1}.
    """
    q = np.asarray(q, dtype=float)
    c = np.asarray(correct, dtype=float)

    ok = np.isfinite(q) & np.isfinite(c)
    if np.sum(ok) < 3:
        return float("nan"), float("nan")

    sp = spearmanr(q[ok], c[ok]).correlation
    kt = kendalltau(q[ok], c[ok]).correlation
    return float(sp) if sp is not None else float("nan"), float(kt) if kt is not None else float("nan")


def _delta_stats(y: np.ndarray, p_l: np.ndarray, p_a: np.ndarray, p_v: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Compute Δ distribution stats and disagreement rates on mask.
    Δ_i = top1(true-class conf) - top2(true-class conf) across {L,A,V}.
    """
    y = np.asarray(y).astype(int)
    mask = np.asarray(mask, dtype=bool)

    pl = np.asarray(p_l, dtype=float)
    pa = np.asarray(p_a, dtype=float)
    pv = np.asarray(p_v, dtype=float)

    ok = mask & np.isfinite(pl) & np.isfinite(pa) & np.isfinite(pv)
    n = int(np.sum(ok))
    if n == 0:
        return {
            "n_eval": 0,
            "delta_median": float("nan"),
            "delta_mean": float("nan"),
            "frac_delta_lt_005": float("nan"),
            "frac_delta_lt_010": float("nan"),
            "frac_any_pred_disagree": float("nan"),
            "frac_any_correctness_disagree": float("nan"),
        }

    tc_l = _true_class_conf(pl[ok], y[ok])
    tc_a = _true_class_conf(pa[ok], y[ok])
    tc_v = _true_class_conf(pv[ok], y[ok])

    C = np.stack([tc_l, tc_a, tc_v], axis=1)  # (n,3)
    Csort = np.sort(C, axis=1)               # ascending
    top1 = Csort[:, -1]
    top2 = Csort[:, -2]
    delta = top1 - top2

    # prediction disagreement across modalities
    yl = _pred_label(pl[ok])
    ya = _pred_label(pa[ok])
    yv = _pred_label(pv[ok])
    pred_disagree = (yl != ya) | (yl != yv) | (ya != yv)

    # correctness disagreement across modalities (at least one correct, at least one incorrect)
    cl = (yl == y[ok])
    ca = (ya == y[ok])
    cv = (yv == y[ok])
    any_correct = cl | ca | cv
    any_wrong = (~cl) | (~ca) | (~cv)
    corr_disagree = any_correct & any_wrong

    return {
        "n_eval": n,
        "delta_median": float(np.median(delta)),
        "delta_mean": float(np.mean(delta)),
        "frac_delta_lt_005": float(np.mean(delta < 0.05)),
        "frac_delta_lt_010": float(np.mean(delta < 0.10)),
        "frac_any_pred_disagree": float(np.mean(pred_disagree)),
        "frac_any_correctness_disagree": float(np.mean(corr_disagree)),
    }


def _alignment_stats(
    union_y: np.ndarray,
    union_Ml: np.ndarray,
    union_Ma: np.ndarray,
    union_Mv: np.ndarray,
    p_l: np.ndarray,
    p_a: np.ndarray,
    p_v: np.ndarray,
    Ql: np.ndarray,
    Qa: np.ndarray,
    Qv: np.ndarray,
    eval_mask: np.ndarray,
    thresh: float = 0.5,
) -> Dict[str, float]:
    """
    Compute Spearman and Kendall alignment between Q and unimodal correctness on eval_mask.
    Uses present-only rows per modality.
    """
    y = np.asarray(union_y).astype(int)
    eval_mask = np.asarray(eval_mask, dtype=bool)

    pl = np.asarray(p_l, dtype=float)
    pa = np.asarray(p_a, dtype=float)
    pv = np.asarray(p_v, dtype=float)

    Ql = _as_scalar_q(Ql)
    Qa = _as_scalar_q(Qa)
    Qv = _as_scalar_q(Qv)

    out: Dict[str, float] = {}

    def one(mod: str, M: np.ndarray, p: np.ndarray, Q: np.ndarray) -> None:
        use = eval_mask & (M == 1) & np.isfinite(p) & np.isfinite(Q)
        n = int(np.sum(use))
        if n < 3:
            out[f"n_{mod}"] = n
            out[f"rhoS_{mod}"] = float("nan")
            out[f"tauK_{mod}"] = float("nan")
            return
        yhat = _pred_label(p[use], thresh=thresh)
        correct = (yhat == y[use]).astype(int)
        sp, kt = _corr_rank(Q[use], correct)
        out[f"n_{mod}"] = n
        out[f"rhoS_{mod}"] = sp
        out[f"tauK_{mod}"] = kt

    one("L", union_Ml, pl, Ql)
    one("A", union_Ma, pa, Qa)
    one("V", union_Mv, pv, Qv)

    return out


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--union_npz", type=str, required=True)
    ap.add_argument("--splits_dir", type=str, required=True)
    ap.add_argument("--unimodal_preds_root", type=str, required=True)
    ap.add_argument("--quality_root", type=str, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--folds", type=int, nargs="+", required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--out_json", type=str, required=True)
    ap.add_argument("--thresh", type=float, default=0.5, help="threshold for correctness indicator from p_pos")
    args = ap.parse_args()

    union = load_union(args.union_npz)
    preds_root = Path(args.unimodal_preds_root)
    q_root = Path(args.quality_root)

    rows = []
    for seed in args.seeds:
        for fold in args.folds:
            split = load_fold_split(args.splits_dir, seed=seed, fold=fold)
            train_mask, test_mask = make_train_test_masks(union, split)
            eval_mask = eval_mask_full_only(union, test_mask)  # FULL-only TEST

            # Load unimodal probs aligned to UNION
            p_l, p_a, p_v, preds_path = _load_unimodal_preds_union(
                preds_root, seed=seed, fold=fold, union_ids_str=union.ids_str
            )

            # Load fold-scaled Q aligned to UNION (with ids verification)
            fq = load_fold_q(q_root, seed=seed, fold=fold, union=union)

            # Δ stats on FULL-only TEST (needs all three p finite)
            dstat = _delta_stats(union.y, p_l, p_a, p_v, eval_mask)

            # ρ stats: Q vs correctness on FULL-only TEST (per modality present)
            astat = _alignment_stats(
                union_y=union.y,
                union_Ml=union.Ml,
                union_Ma=union.Ma,
                union_Mv=union.Mv,
                p_l=p_l,
                p_a=p_a,
                p_v=p_v,
                Ql=fq.Ql,
                Qa=fq.Qa,
                Qv=fq.Qv,
                eval_mask=eval_mask,
                thresh=args.thresh,
            )

            rows.append({
                "seed": seed,
                "fold": fold,
                "N_union": int(len(union.ids_str)),
                "n_fullonly_test": int(np.sum(eval_mask)),
                "preds_path": preds_path,
                "q_path": fq.q_path,
                **dstat,
                **astat,
            })

    df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_csv, index=False)

    # Aggregate mean±std over folds (all rows)
    def mean_std(col: str) -> Dict[str, float]:
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        return {"mean": float(np.mean(x)) if x.size else float("nan"),
                "std": float(np.std(x)) if x.size else float("nan"),
                "n": int(x.size)}

    summary = {
        "args": vars(args),
        "n_rows": int(len(df)),
        "delta": {
            "delta_median": mean_std("delta_median"),
            "delta_mean": mean_std("delta_mean"),
            "frac_delta_lt_005": mean_std("frac_delta_lt_005"),
            "frac_delta_lt_010": mean_std("frac_delta_lt_010"),
            "frac_any_pred_disagree": mean_std("frac_any_pred_disagree"),
            "frac_any_correctness_disagree": mean_std("frac_any_correctness_disagree"),
        },
        "alignment": {
            "rhoS_L": mean_std("rhoS_L"),
            "rhoS_A": mean_std("rhoS_A"),
            "rhoS_V": mean_std("rhoS_V"),
            "tauK_L": mean_std("tauK_L"),
            "tauK_A": mean_std("tauK_A"),
            "tauK_V": mean_std("tauK_V"),
        },
        "sizes": {
            "n_fullonly_test": mean_std("n_fullonly_test"),
            "n_eval": mean_std("n_eval"),
            "n_L": mean_std("n_L"),
            "n_A": mean_std("n_A"),
            "n_V": mean_std("n_V"),
        },
        "per_fold": rows,  # keep for audits; delete if you want a tiny JSON
    }

    out_json.write_text(json.dumps(summary, indent=2))
    print("[DONE] wrote:")
    print(f"  {out_csv}")
    print(f"  {out_json}")


if __name__ == "__main__":
    main()
