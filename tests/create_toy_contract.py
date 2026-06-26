#!/usr/bin/env python3
"""Create a tiny UNION/splits/Q contract for CI-style smoke tests.

This data is synthetic and only checks pipeline mechanics: row alignment, masks,
fold-safe Q loading, unimodal posterior dumping, and Broken-Q permutation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def make(out: Path, seed: int = 11, fold: int = 0, n: int = 72) -> None:
    rng = np.random.default_rng(123)
    out.mkdir(parents=True, exist_ok=True)
    ids = np.asarray([f"toy_{i:03d}" for i in range(n)], dtype=object)
    y = np.asarray([0, 1] * (n // 2), dtype=int)
    rng.shuffle(y)

    # Binary signal plus modality-specific noise.
    z = (2 * y - 1).astype(float)[:, None]
    Ea = np.hstack([z + 0.8 * rng.normal(size=(n, 1)), rng.normal(size=(n, 5))]).astype("float32")
    Ev = np.hstack([0.4 * z + rng.normal(size=(n, 1)), rng.normal(size=(n, 4))]).astype("float32")
    Ep = np.hstack([0.7 * z + rng.normal(size=(n, 1)), rng.normal(size=(n, 3))]).astype("float32")

    Ma = np.ones(n, dtype="uint8")
    Mv = np.ones(n, dtype="uint8")
    Mp = np.ones(n, dtype="uint8")
    # Some natural missingness outside the fully observed slice.
    Ma[::17] = 0
    Mv[5::19] = 0
    for E, M in [(Ea, Ma), (Ev, Mv), (Ep, Mp)]:
        E[M == 0] = 0

    union_dir = out / "union"
    union_dir.mkdir(exist_ok=True)
    np.savez(union_dir / "toy_union.npz", ids=ids, y2=y, E_a=Ea, E_v=Ev, E_p=Ep, M_a=Ma, M_v=Mv, M_p=Mp)

    # One complete split.
    splits = out / "splits" / f"seed_{seed}"
    splits.mkdir(parents=True, exist_ok=True)
    idx = np.arange(n)
    # balanced-ish deterministic split
    test = idx[idx % 5 == 0]
    train = idx[idx % 5 != 0]
    np.save(splits / f"train_ids_fold{fold}.npy", ids[train])
    np.save(splits / f"test_ids_fold{fold}.npy", ids[test])

    # Q is fold-scaled-like and zero when missing. Shape (N,2).
    qroot = out / "quality" / f"seed_{seed}"
    qroot.mkdir(parents=True, exist_ok=True)
    Qa = np.clip(rng.uniform(0.2, 1.0, size=(n, 2)), 0, 1).astype("float32")
    Qv = np.clip(rng.uniform(0.2, 1.0, size=(n, 2)), 0, 1).astype("float32")
    Qp = np.clip(rng.uniform(0.2, 1.0, size=(n, 2)), 0, 1).astype("float32")
    Qa[Ma == 0] = 0
    Qv[Mv == 0] = 0
    Qp[Mp == 0] = 0
    np.savez(qroot / f"fold_{fold}.npz", ids=ids, Qa=Qa, Qv=Qv, Qp=Qp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    make(Path(args.out))
