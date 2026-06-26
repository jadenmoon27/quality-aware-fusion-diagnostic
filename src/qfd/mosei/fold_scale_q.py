#!/usr/bin/env python3
"""
fold_scale_mosei_q.py — StressID-compatible fold-scaled Q for MOSEI (25 files)

Inputs:
  --union_npz  : MOSEI UNION (.npz) with ids, M_l, M_a, M_v
  --rawq_npz   : MOSEI RAWQ (.npz) with ids, Q_l_raw, Q_a_raw, Q_v_raw
  --splits_dir : splits_mosei/seed_{seed}/train_ids_fold{k}.npy
Outputs:
  out_root/seed_{seed}/fold_{k}.npz containing:
    ids (canonical UNION ids)
    Q_l, Q_a, Q_v  (float32, shape (N,2), in [0,1])
    meta (json)

Scaling rule per modality m and dim d:
  Fit on: TRAIN ∩ (M_m==1) ∩ finite(Q_m_raw[:,d])
  Let lo = q05, hi = q95 of RAWQ on fit set
  Clip: x' = clip(x, lo, hi)
  Map:  (x' - lo) / (hi - lo)   (if hi==lo -> all zeros)
  Force: if M_m==0 or non-finite RAWQ -> 0

This matches StressID discipline (leakage-safe, fold-specific).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def load_ids(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=True).astype(str).reshape(-1)


def require(z, k: str):
    if k not in z.files:
        raise KeyError(f"Missing key '{k}'. Found: {sorted(z.files)}")
    return z[k]


def quantile_fit(x: np.ndarray, q_lo: float, q_hi: float) -> Tuple[float, float]:
    lo = float(np.quantile(x, q_lo))
    hi = float(np.quantile(x, q_hi))
    return lo, hi


def scale_dim(raw: np.ndarray, fit_mask: np.ndarray, present_mask: np.ndarray, q_lo: float, q_hi: float) -> Tuple[np.ndarray, Dict]:
    """
    raw: (N,) float
    fit_mask: (N,) bool for TRAIN ∩ PRESENT ∩ FINITE
    present_mask: (N,) bool for PRESENT (M==1)
    Returns:
      scaled: (N,) float32 in [0,1]
      info: dict with fit counts and quantiles
    """
    scaled = np.zeros_like(raw, dtype=np.float32)

    fit_x = raw[fit_mask]
    info = {
        "n_fit": int(fit_x.size),
        "q_lo": float(q_lo),
        "q_hi": float(q_hi),
        "lo": None,
        "hi": None,
        "degenerate": None,
    }

    if fit_x.size == 0:
        # Nothing to fit on; keep zeros everywhere (StressID-compatible safe fallback)
        info["lo"] = float("nan")
        info["hi"] = float("nan")
        info["degenerate"] = True
        return scaled, info

    lo, hi = quantile_fit(fit_x, q_lo, q_hi)
    info["lo"] = lo
    info["hi"] = hi

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        # Degenerate scaler; map everything to 0 (safe)
        info["degenerate"] = True
        return scaled, info

    info["degenerate"] = False

    # Apply scaling ONLY where present AND finite
    ok = present_mask & np.isfinite(raw)
    x = raw.copy()
    x = np.clip(x, lo, hi)
    scaled[ok] = ((x[ok] - lo) / (hi - lo)).astype(np.float32)

    # Hard clip numerical noise
    scaled = np.clip(scaled, 0.0, 1.0).astype(np.float32)
    return scaled, info


def scale_modality(Q_raw: np.ndarray, M: np.ndarray, train_mask: np.ndarray, q_lo: float, q_hi: float) -> Tuple[np.ndarray, Dict]:
    """
    Q_raw: (N,2)
    M: (N,) int8
    train_mask: (N,) bool
    Returns Q_scaled: (N,2) float32 in [0,1]
    """
    M = M.astype(np.int8).reshape(-1)
    present = (M == 1)

    out = np.zeros_like(Q_raw, dtype=np.float32)
    info = {"dim0": None, "dim1": None}

    for d in [0, 1]:
        raw_d = Q_raw[:, d].astype(float)
        fit_mask = train_mask & present & np.isfinite(raw_d)

        scaled_d, inf = scale_dim(raw_d, fit_mask, present, q_lo, q_hi)
        out[:, d] = scaled_d
        info[f"dim{d}"] = inf

    # StressID rule: missing modalities forced to 0 (already zeros)
    out[~present, :] = 0.0
    out = np.clip(out, 0.0, 1.0).astype(np.float32)
    return out, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union_npz", type=str, required=True)
    ap.add_argument("--rawq_npz", type=str, required=True)
    ap.add_argument("--splits_dir", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--q_lo", type=float, default=0.05)
    ap.add_argument("--q_hi", type=float, default=0.95)
    args = ap.parse_args()

    u = np.load(args.union_npz, allow_pickle=True)
    r = np.load(args.rawq_npz, allow_pickle=True)

    # Required keys
    u_ids = require(u, "ids").astype(str)
    r_ids = require(r, "ids").astype(str)
    if len(u_ids) != len(r_ids) or np.any(u_ids != r_ids):
        raise AssertionError("RAWQ ids must match UNION ids exactly (set + order).")

    N = len(u_ids)

    M_l = require(u, "M_l").astype(np.int8).reshape(-1)
    M_a = require(u, "M_a").astype(np.int8).reshape(-1)
    M_v = require(u, "M_v").astype(np.int8).reshape(-1)

    Q_l_raw = require(r, "Q_l_raw").astype(np.float32)
    Q_a_raw = require(r, "Q_a_raw").astype(np.float32)
    Q_v_raw = require(r, "Q_v_raw").astype(np.float32)

    if Q_l_raw.shape != (N, 2) or Q_a_raw.shape != (N, 2) or Q_v_raw.shape != (N, 2):
        raise ValueError("RAWQ shapes must be (N,2) for each modality.")

    # id -> row index
    id2idx = {sid: i for i, sid in enumerate(u_ids.tolist())}

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        seed_dir = Path(args.splits_dir) / f"seed_{seed}"
        if not seed_dir.exists():
            raise FileNotFoundError(f"Missing seed dir: {seed_dir}")

        out_seed_dir = out_root / f"seed_{seed}"
        out_seed_dir.mkdir(parents=True, exist_ok=True)

        for fold in args.folds:
            train_ids_path = seed_dir / f"train_ids_fold{fold}.npy"
            if not train_ids_path.exists():
                raise FileNotFoundError(f"Missing {train_ids_path}")

            train_ids = load_ids(train_ids_path)
            train_idx = np.array([id2idx[s] for s in train_ids], dtype=np.int64)

            train_mask = np.zeros((N,), dtype=bool)
            train_mask[train_idx] = True

            Q_l, info_l = scale_modality(Q_l_raw, M_l, train_mask, args.q_lo, args.q_hi)
            Q_a, info_a = scale_modality(Q_a_raw, M_a, train_mask, args.q_lo, args.q_hi)
            Q_v, info_v = scale_modality(Q_v_raw, M_v, train_mask, args.q_lo, args.q_hi)

            meta = {
                "dataset": "MOSEI",
                "scaling": {
                    "fit_set": "TRAIN ∩ PRESENT ∩ FINITE",
                    "q_lo": args.q_lo,
                    "q_hi": args.q_hi,
                    "missing_or_nonfinite_policy": "forced to 0",
                },
                "seed": int(seed),
                "fold": int(fold),
                "fit_info": {"L": info_l, "A": info_a, "V": info_v},
            }

            out_path = out_seed_dir / f"fold_{fold}.npz"
            np.savez_compressed(
                out_path,
                ids=u_ids.astype(object),
                Q_l=Q_l,
                Q_a=Q_a,
                Q_v=Q_v,
                meta=np.array([json.dumps(meta)], dtype=object),
            )

            # Minimal sanity: ranges
            for name, Q in [("L", Q_l), ("A", Q_a), ("V", Q_v)]:
                if not (np.min(Q) >= -1e-6 and np.max(Q) <= 1.0 + 1e-6):
                    raise AssertionError(f"Scaled Q_{name} not within [0,1] (min={Q.min()}, max={Q.max()})")

            print(f"[OK] seed={seed} fold={fold} wrote {out_path}")

    print("[OK] All fold-scaled Q files generated.")


if __name__ == "__main__":
    main()



