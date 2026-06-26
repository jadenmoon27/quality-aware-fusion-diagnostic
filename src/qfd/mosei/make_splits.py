#!/usr/bin/env python3
"""
make_splits_mosei.py  — StressID-compatible MOSEI splits

Creates 5 seeds × 5 folds = 25 splits using:
  - Stratification: y (binary)
  - Grouping: video_id (parsed from UNION ids like "video_id[j]")

Outputs (StressID form):
  splits_mosei/
    seed_11/
      train_ids_fold0.npy
      test_ids_fold0.npy
      ...
      fold_report.json
    seed_22/
    ...
Plus a top-level summary:
  splits_mosei/splits_summary.json

Hard constraints enforced:
  - Group-disjoint: no video_id appears in both train and test within a fold
  - Deterministic per seed
  - Uses UNION row order only to read ids/y; split membership is by ids

Requires:
  pip install scikit-learn numpy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def parse_video_id(seg_id: str) -> str:
    # Expect: "video_id[j]" (from your UNION builder)
    return seg_id.split("[", 1)[0]


def safe_rate(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.mean(x))


def fold_stats(
    y: np.ndarray,
    ids: np.ndarray,
    video_ids: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    extra_masks: Dict[str, np.ndarray] | None = None,
) -> Dict:
    tr_ids = ids[train_idx]
    te_ids = ids[test_idx]

    tr_vid = video_ids[train_idx]
    te_vid = video_ids[test_idx]

    # Hard group-disjoint check
    inter = set(tr_vid.tolist()).intersection(set(te_vid.tolist()))
    if len(inter) != 0:
        sample = list(sorted(inter))[:5]
        raise AssertionError(f"Group leakage: {len(inter)} overlapping video_ids (sample={sample})")

    out = {
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_train_groups": int(len(set(tr_vid.tolist()))),
        "n_test_groups": int(len(set(te_vid.tolist()))),
        "pos_rate_train": float(y[train_idx].mean()),
        "pos_rate_test": float(y[test_idx].mean()),
        "ids_train_preview": tr_ids[:3].tolist(),
        "ids_test_preview": te_ids[:3].tolist(),
    }

    if extra_masks is not None:
        for k, m in extra_masks.items():
            m = m.astype(np.int8).reshape(-1)
            out[f"{k}_rate_train"] = safe_rate(m[train_idx])
            out[f"{k}_rate_test"] = safe_rate(m[test_idx])

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union_npz", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="splits_mosei")
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--require_keys", action="store_true", help="Fail unless UNION contains required keys.")
    args = ap.parse_args()

    union = np.load(args.union_npz, allow_pickle=True)

    required = ["ids", "y"]
    if args.require_keys:
        missing = [k for k in required if k not in union.files]
        if missing:
            raise KeyError(f"UNION missing keys {missing}. Found: {sorted(union.files)}")

    ids = union["ids"].astype(str)
    y = union["y"].astype(int).reshape(-1)

    if ids.shape[0] != y.shape[0]:
        raise ValueError("UNION ids and y length mismatch.")

    # Video grouping: parsed from ids (canonical for MOSEI)
    video_ids = np.array([parse_video_id(s) for s in ids], dtype=object)

    # Optional: carry presence masks into fold reports (not used for splitting)
    extra_masks = {}
    for k in ["M_l", "M_a", "M_v"]:
        if k in union.files:
            extra_masks[k] = union[k].astype(np.int8).reshape(-1)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "union_npz": args.union_npz,
        "N": int(len(ids)),
        "pos_rate": float(y.mean()),
        "n_unique_groups": int(len(set(video_ids.tolist()))),
        "n_folds": int(args.n_folds),
        "seeds": args.seeds,
        "notes": [
            "Groups are video_id parsed from UNION ids 'video_id[j]'.",
            "StratifiedGroupKFold used: stratify on y, group-disjoint on video_id.",
            "Stored split membership as train/test id lists; never resplit downstream.",
        ],
        "per_seed": {},
    }

    # Generate splits
    for seed in args.seeds:
        seed_dir = out_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        cv = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=seed)

        fold_reports: List[Dict] = []
        all_test_ids = []

        for k, (train_idx, test_idx) in enumerate(cv.split(np.zeros_like(y), y, groups=video_ids)):
            train_ids = ids[train_idx]
            test_ids = ids[test_idx]

            np.save(seed_dir / f"train_ids_fold{k}.npy", train_ids.astype(object))
            np.save(seed_dir / f"test_ids_fold{k}.npy", test_ids.astype(object))

            rep = fold_stats(
                y=y,
                ids=ids,
                video_ids=video_ids,
                train_idx=train_idx,
                test_idx=test_idx,
                extra_masks=extra_masks if extra_masks else None,
            )
            rep["fold"] = int(k)
            fold_reports.append(rep)
            all_test_ids.append(set(test_ids.tolist()))

        # Optional sanity: test folds should be disjoint in ids (typical K-fold property)
        # Not strictly required but should hold.
        overlap_any = False
        for i in range(len(all_test_ids)):
            for j in range(i + 1, len(all_test_ids)):
                if len(all_test_ids[i].intersection(all_test_ids[j])) != 0:
                    overlap_any = True
        if overlap_any:
            raise AssertionError(f"Seed {seed}: test folds overlap in ids (unexpected).")

        fold_report_path = seed_dir / "fold_report.json"
        with open(fold_report_path, "w") as f:
            json.dump(
                {
                    "seed": seed,
                    "n_folds": args.n_folds,
                    "N": int(len(ids)),
                    "pos_rate": float(y.mean()),
                    "n_unique_groups": int(len(set(video_ids.tolist()))),
                    "folds": fold_reports,
                },
                f,
                indent=2,
            )

        summary["per_seed"][str(seed)] = {
            "fold_report": str(fold_report_path),
            "folds": [
                {
                    "fold": fr["fold"],
                    "pos_rate_train": fr["pos_rate_train"],
                    "pos_rate_test": fr["pos_rate_test"],
                    "n_train_groups": fr["n_train_groups"],
                    "n_test_groups": fr["n_test_groups"],
                }
                for fr in fold_reports
            ],
        }

        print(f"[OK] seed={seed}: wrote {args.n_folds} folds to {seed_dir}")

    with open(out_root / "splits_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] wrote {out_root / 'splits_summary.json'}")
    print("Done.")


if __name__ == "__main__":
    main()


