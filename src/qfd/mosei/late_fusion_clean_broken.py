#!/usr/bin/env python3
"""
mosei_step5_late_fusion_clean_vs_broken.py

MOSEI STEP 5 — Late fusion clean vs broken (StressID-style replication)

Report (FULL-only TEST):
NoQ         = arithmetic mean of unimodal probs (no training)
cleanQ      = quality-conditioned late fusion (train on TRAIN_FULL, eval on TEST_FULL)
brokenQ     = load from precomputed quality_fold_broken (leakage-safe, test-only permuted)
oracle-best = best unimodal per sample by log-loss (diagnostic ceiling)

Outputs:
- CSV: per seed/fold metrics
- JSON: mean±std table + mean|clean-broken| gap + flip rate % (one per fuser family)

Expected layouts:
  splits_dir/seed_{seed}/train_ids_fold{fold}.npy
  splits_dir/seed_{seed}/test_ids_fold{fold}.npy
  unimodal_root/seed_{seed}/fold_{fold}.npz    (ids + p_l/p_a/p_v)
  q_clean_root/seed_{seed}/fold_{fold}.npz     (ids + Ql/Qa/Qv shape (N,2) or (N,))
  q_broken_root/seed_{seed}/fold_{fold}.npz    (ids + Ql/Qa/Qv shape (N,2) or (N,))

Run example:
python python -m qfd.mosei.late_fusion_clean_broken \
  --union_npz output/final_experiments/mosei/union/mosei_union.npz \
  --splits_dir output/final_experiments/mosei/splits_mosei \
  --unimodal_root output/final_experiments/mosei/unimodal_preds/lr \
  --q_clean_root output/final_experiments/mosei/quality_fold \
  --q_broken_root output/final_experiments/mosei/quality_fold_broken \
  --out_csv output/final_experiments/mosei/reports/5/late_fusion_clean_vs_broken.csv \
  --out_json output/final_experiments/mosei/reports/5/late_fusion_clean_vs_broken.json \
  --seeds 11 22 33 44 55 --folds 0 1 2 3 4 \
  --thresh 0.5 \
  --fusers lr hgb
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


# -----------------------------
# Basic metrics
# -----------------------------

def preds_from_probs(p: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    return (p >= thresh).astype(int)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(int).reshape(-1)
    y_pred = y_pred.astype(int).reshape(-1)
    pos = (y_true == 1)
    neg = (y_true == 0)
    tpr = float((y_pred[pos] == 1).mean()) if pos.any() else 0.0
    tnr = float((y_pred[neg] == 0).mean()) if neg.any() else 0.0
    return 0.5 * (tpr + tnr)


def flip_rate(y1: np.ndarray, y2: np.ndarray) -> float:
    y1 = y1.astype(int).reshape(-1)
    y2 = y2.astype(int).reshape(-1)
    if y1.shape != y2.shape:
        raise ValueError("flip_rate: shape mismatch")
    return float((y1 != y2).mean())


def _mean_std(xs: List[float]) -> Dict[str, float]:
    a = np.asarray(xs, dtype=float)
    return {"mean": float(np.mean(a)), "std": float(np.std(a, ddof=0)), "n": int(a.size)}


# -----------------------------
# Loaders (MOSEI strict)
# -----------------------------

def load_union(union_npz: Path) -> Dict[str, np.ndarray]:
    z = np.load(union_npz, allow_pickle=True)
    req = ["ids", "y", "M_l", "M_a", "M_v"]
    for k in req:
        if k not in z.files:
            raise KeyError(f"{union_npz}: missing {k}. Found {sorted(z.files)}")

    ids = z["ids"].astype(str).reshape(-1)
    y = z["y"].astype(int).reshape(-1)
    Ml = z["M_l"].astype(np.int8).reshape(-1)
    Ma = z["M_a"].astype(np.int8).reshape(-1)
    Mv = z["M_v"].astype(np.int8).reshape(-1)

    if len(np.unique(ids)) != len(ids):
        raise AssertionError("UNION ids are not unique")

    for name, M in [("M_l", Ml), ("M_a", Ma), ("M_v", Mv)]:
        if not np.all((M == 0) | (M == 1)):
            raise AssertionError(f"{name} must be binary 0/1")

    return {"ids": ids, "y": y, "M_l": Ml, "M_a": Ma, "M_v": Mv}


def load_fold_split(splits_dir: Path, seed: int, fold: int) -> Dict[str, np.ndarray]:
    sd = splits_dir / f"seed_{seed}"
    trp = sd / f"train_ids_fold{fold}.npy"
    tep = sd / f"test_ids_fold{fold}.npy"
    if not trp.exists() or not tep.exists():
        raise FileNotFoundError(f"missing split files under {sd} for fold={fold}")
    tr = np.load(trp, allow_pickle=True).astype(str).reshape(-1)
    te = np.load(tep, allow_pickle=True).astype(str).reshape(-1)
    if len(np.unique(tr)) != len(tr):
        raise AssertionError(f"seed={seed} fold={fold}: duplicate ids in train split")
    if len(np.unique(te)) != len(te):
        raise AssertionError(f"seed={seed} fold={fold}: duplicate ids in test split")
    return {"train_ids": tr, "test_ids": te}


def make_train_test_masks(union_ids: np.ndarray, split: Dict[str, np.ndarray], *, seed: int, fold: int) -> Tuple[np.ndarray, np.ndarray]:
    id2i = {sid: i for i, sid in enumerate(union_ids.tolist())}
    train_mask = np.zeros((len(union_ids),), dtype=bool)
    test_mask = np.zeros((len(union_ids),), dtype=bool)

    missing = []
    for s in split["train_ids"]:
        j = id2i.get(s, None)
        if j is None:
            missing.append(s)
        else:
            train_mask[j] = True
    for s in split["test_ids"]:
        j = id2i.get(s, None)
        if j is None:
            missing.append(s)
        else:
            test_mask[j] = True

    if missing:
        raise AssertionError(f"seed={seed} fold={fold}: {len(missing)} split ids not in UNION. examples={missing[:10]}")

    if np.any(train_mask & test_mask):
        raise AssertionError(f"seed={seed} fold={fold}: train/test masks overlap")
    return train_mask, test_mask


def eval_mask_full_only(Ml: np.ndarray, Ma: np.ndarray, Mv: np.ndarray, test_mask: np.ndarray) -> np.ndarray:
    return test_mask & (Ml == 1) & (Ma == 1) & (Mv == 1)


def _pick_npz(z: np.lib.npyio.NpzFile, cands: List[str], path: Path) -> np.ndarray:
    for k in cands:
        if k in z.files:
            return z[k]
    raise KeyError(f"{path}: missing any of {cands}. Found {sorted(z.files)}")


def load_unimodal_probs(unimodal_root: Path, seed: int, fold: int, union_ids: np.ndarray) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    p = unimodal_root / f"seed_{seed}" / f"fold_{fold}.npz"
    if not p.exists():
        raise FileNotFoundError(p)
    z = np.load(p, allow_pickle=True)

    ids = _pick_npz(z, ["ids", "id", "union_ids"], p).astype(str).reshape(-1)
    if ids.shape != union_ids.shape or np.any(ids != union_ids):
        raise AssertionError(f"{p}: ids not aligned to UNION")

    pl = _pick_npz(z, ["p_l", "pl", "p_text", "p_lang", "p_L"], p).astype(float).reshape(-1)
    pa = _pick_npz(z, ["p_a", "pa", "p_audio", "p_A"], p).astype(float).reshape(-1)
    pv = _pick_npz(z, ["p_v", "pv", "p_video", "p_V"], p).astype(float).reshape(-1)

    def _clip_keep_nan(x: np.ndarray) -> np.ndarray:
        out = x.astype(float)
        fin = np.isfinite(out)
        out[fin] = np.clip(out[fin], 0.0, 1.0)
        return out

    return str(p), _clip_keep_nan(pl), _clip_keep_nan(pa), _clip_keep_nan(pv)


def load_fold_q(q_root: Path, seed: int, fold: int, union_ids: np.ndarray) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    p = q_root / f"seed_{seed}" / f"fold_{fold}.npz"
    if not p.exists():
        raise FileNotFoundError(p)
    z = np.load(p, allow_pickle=True)

    ids = _pick_npz(z, ["ids", "id", "union_ids"], p).astype(str).reshape(-1)
    if ids.shape != union_ids.shape or np.any(ids != union_ids):
        raise AssertionError(f"{p}: ids not aligned to UNION")

    # IMPORTANT: your broken-Q writer saves Ql/Qa/Qv (no underscore by default).
    Ql = _pick_npz(z, ["Ql", "Q_l", "Q_text", "Q_lang", "Q_L"], p)
    Qa = _pick_npz(z, ["Qa", "Q_a", "Q_audio", "Q_A"], p)
    Qv = _pick_npz(z, ["Qv", "Q_v", "Q_video", "Q_V"], p)

    Ql = np.asarray(Ql, dtype=float)
    Qa = np.asarray(Qa, dtype=float)
    Qv = np.asarray(Qv, dtype=float)

    def to_scalar(Q: np.ndarray) -> np.ndarray:
        if Q.ndim == 1:
            return Q.astype(float)
        if Q.ndim == 2:
            return Q.astype(float).mean(axis=1)
        raise ValueError(f"Q ndim {Q.ndim} unsupported")

    return str(p), to_scalar(Ql), to_scalar(Qa), to_scalar(Qv)


# -----------------------------
# StressID-style contracts
# -----------------------------

def assert_probs_nan_where_missing(M: np.ndarray, p: np.ndarray, name: str) -> None:
    if np.any((M == 0) & np.isfinite(p)):
        bad = int(((M == 0) & np.isfinite(p)).sum())
        raise AssertionError(f"{name}: {bad} finite probs where M==0 (should be NaN)")
    if np.any((M == 1) & (~np.isfinite(p))):
        bad = int(((M == 1) & (~np.isfinite(p))).sum())
        raise AssertionError(f"{name}: {bad} non-finite probs where M==1 (should be finite)")


def assert_q_missing_is_zero(M: np.ndarray, Q: np.ndarray, name: str) -> None:
    if np.any((M == 0) & (Q != 0.0)):
        bad = int(((M == 0) & (Q != 0.0)).sum())
        raise AssertionError(f"{name}: {bad} non-zero Q where M==0")


def assert_q_present_finite_in_01(M: np.ndarray, Q: np.ndarray, name: str) -> None:
    if np.any((M == 1) & (~np.isfinite(Q))):
        bad = int(((M == 1) & (~np.isfinite(Q))).sum())
        raise AssertionError(f"{name}: {bad} non-finite Q where M==1")
    if np.any((M == 1) & ((Q < 0.0) | (Q > 1.0))):
        bad = int(((M == 1) & ((Q < 0.0) | (Q > 1.0))).sum())
        raise AssertionError(f"{name}: {bad} present Q outside [0,1]")


# -----------------------------
# Late fusion features/models
# -----------------------------

def _make_X_noq(pl: np.ndarray, pa: np.ndarray, pv: np.ndarray) -> np.ndarray:
    return np.stack([pl, pa, pv], axis=1)  # (N,3)


def _make_X_q(pl: np.ndarray, pa: np.ndarray, pv: np.ndarray, Ql: np.ndarray, Qa: np.ndarray, Qv: np.ndarray) -> np.ndarray:
    Xp = np.stack([pl, pa, pv], axis=1)
    Xq = np.stack([Ql, Qa, Qv], axis=1)
    return np.concatenate([Xp, Xq], axis=1)  # (N,6)


def _fit_lr(X: np.ndarray, y: np.ndarray, seed: int) -> LogisticRegression:
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
        C=1.0,
    )
    clf.fit(X, y)
    return clf


def _fit_hgb(X: np.ndarray, y: np.ndarray, seed: int) -> HistGradientBoostingClassifier:
    clf = HistGradientBoostingClassifier(
        max_depth=3,
        learning_rate=0.05,
        max_iter=300,
        random_state=seed,
    )
    clf.fit(X, y)
    return clf


def _predict_proba_pos(model, X: np.ndarray) -> np.ndarray:
    p = model.predict_proba(X)
    return p[:, 1].astype(float)


# -----------------------------
# Oracle-best (per-sample best unimodal by log-loss)
# -----------------------------

def oracle_best_prob(y_true: np.ndarray, probs_3: np.ndarray) -> np.ndarray:
    eps = 1e-8
    y_true = y_true.astype(int).reshape(-1)
    p_true = np.where(y_true[:, None] == 1, probs_3, 1.0 - probs_3)  # (n,3)
    loss = -np.log(np.clip(p_true, eps, 1.0))                        # (n,3)
    best_m = loss.argmin(axis=1)                                     # (n,)
    return probs_3[np.arange(best_m.size), best_m].astype(float)     # (n,)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--union_npz", type=str, required=True)
    ap.add_argument("--splits_dir", type=str, required=True)
    ap.add_argument("--unimodal_root", type=str, required=True)
    ap.add_argument("--q_clean_root", type=str, required=True)
    ap.add_argument("--q_broken_root", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--out_json", type=str, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--fusers", type=str, nargs="+", default=["lr"], choices=["lr", "hgb"])
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    union = load_union(Path(args.union_npz))
    ids = union["ids"]
    y = union["y"]
    Ml, Ma, Mv = union["M_l"], union["M_a"], union["M_v"]

    splits_dir = Path(args.splits_dir)
    unimodal_root = Path(args.unimodal_root)
    q_clean_root = Path(args.q_clean_root)
    q_broken_root = Path(args.q_broken_root)

    rows: List[Dict] = []

    # per-family accumulators
    acc: Dict[str, List[float]] = {}
    abs_gap: Dict[str, List[float]] = {}
    flips: Dict[str, List[float]] = {}

    for fuser in args.fusers:
        acc[f"{fuser}_noq"] = []
        acc[f"{fuser}_cleanq"] = []
        acc[f"{fuser}_brokenq"] = []
        acc[f"{fuser}_oracle_best"] = []
        abs_gap[f"{fuser}_abs_clean_minus_broken"] = []
        flips[f"{fuser}_flip_clean_vs_broken"] = []

    # shared sanity checks
    acc["noq_mean"] = []
    acc["oracle_best"] = []

    for seed in args.seeds:
        for fold in args.folds:
            split = load_fold_split(splits_dir, seed, fold)
            train_mask, test_mask = make_train_test_masks(ids, split, seed=seed, fold=fold)

            eval_mask = eval_mask_full_only(Ml, Ma, Mv, test_mask)                   # FULL-only TEST
            train_full = train_mask & (Ml == 1) & (Ma == 1) & (Mv == 1)              # FULL-only TRAIN

            n_eval = int(eval_mask.sum())
            n_train = int(train_full.sum())
            if n_eval == 0:
                raise RuntimeError(f"seed={seed} fold={fold}: FULL-only TEST empty")
            if n_train == 0:
                raise RuntimeError(f"seed={seed} fold={fold}: FULL-only TRAIN empty")
            if len(np.unique(y[train_full])) < 2:
                raise RuntimeError(f"seed={seed} fold={fold}: FULL-only TRAIN has single class")

            unimodal_path, pl, pa, pv = load_unimodal_probs(unimodal_root, seed, fold, ids)

            q_clean_path, Ql, Qa, Qv = load_fold_q(q_clean_root, seed, fold, ids)
            q_broken_path, Ql_b, Qa_b, Qv_b = load_fold_q(q_broken_root, seed, fold, ids)

            # Contracts on probs
            assert_probs_nan_where_missing(Ml, pl, "p_l")
            assert_probs_nan_where_missing(Ma, pa, "p_a")
            assert_probs_nan_where_missing(Mv, pv, "p_v")

            # Contracts on Q (clean + broken)
            for tag, (QlX, QaX, QvX) in [
                ("clean", (Ql, Qa, Qv)),
                ("broken", (Ql_b, Qa_b, Qv_b)),
            ]:
                assert_q_missing_is_zero(Ml, QlX, f"Q_l({tag})")
                assert_q_missing_is_zero(Ma, QaX, f"Q_a({tag})")
                assert_q_missing_is_zero(Mv, QvX, f"Q_v({tag})")
                assert_q_present_finite_in_01(Ml, QlX, f"Q_l({tag})")
                assert_q_present_finite_in_01(Ma, QaX, f"Q_a({tag})")
                assert_q_present_finite_in_01(Mv, QvX, f"Q_v({tag})")

            # FULL-only implies probs are finite on eval
            for name, arr in [("p_l", pl), ("p_a", pa), ("p_v", pv)]:
                bad = eval_mask & (~np.isfinite(arr))
                if bad.any():
                    ex = np.flatnonzero(bad)[:10].tolist()
                    raise AssertionError(f"seed={seed} fold={fold}: {name} non-finite on FULL-only TEST. ex={ex}")

            y_true = y[eval_mask].astype(int)

            # NoQ baseline (arithmetic mean)
            probs_eval = np.stack([pl[eval_mask], pa[eval_mask], pv[eval_mask]], axis=1)
            p_noq_mean = probs_eval.mean(axis=1)
            noq_acc = balanced_accuracy(y_true, preds_from_probs(p_noq_mean, thresh=args.thresh))

            # oracle-best (diagnostic ceiling)
            p_oracle_best = oracle_best_prob(y_true, probs_eval)
            oracle_best_acc = balanced_accuracy(y_true, preds_from_probs(p_oracle_best, thresh=args.thresh))

            acc["noq_mean"].append(noq_acc)
            acc["oracle_best"].append(oracle_best_acc)

            if args.dry_run:
                print(f"[DRY] seed={seed} fold={fold} train_full={n_train} eval_full_test={n_eval}")
                print(f"      unimodal  ={unimodal_path}")
                print(f"      cleanQ    ={q_clean_path}")
                print(f"      brokenQ   ={q_broken_path}")
                continue

            # Features (UNION-space)
            X_clean = _make_X_q(pl, pa, pv, Ql, Qa, Qv)
            X_broken = _make_X_q(pl, pa, pv, Ql_b, Qa_b, Qv_b)

            for fuser in args.fusers:
                rng_seed = seed * 100 + fold

                if fuser == "lr":
                    model = _fit_lr(X_clean[train_full], y[train_full].astype(int), rng_seed)
                else:
                    model = _fit_hgb(X_clean[train_full], y[train_full].astype(int), rng_seed)

                # cleanQ / brokenQ (same trained model, different Q at test time)
                p_clean = _predict_proba_pos(model, X_clean[eval_mask])
                p_brok = _predict_proba_pos(model, X_broken[eval_mask])

                y_clean = preds_from_probs(p_clean, thresh=args.thresh)
                y_brok = preds_from_probs(p_brok, thresh=args.thresh)

                clean_acc = balanced_accuracy(y_true, y_clean)
                brok_acc = balanced_accuracy(y_true, y_brok)
                fr = flip_rate(y_clean, y_brok)
                gap = abs(clean_acc - brok_acc)

                # For table: fuser NoQ matches arithmetic mean baseline
                acc[f"{fuser}_noq"].append(noq_acc)
                acc[f"{fuser}_cleanq"].append(clean_acc)
                acc[f"{fuser}_brokenq"].append(brok_acc)
                acc[f"{fuser}_oracle_best"].append(oracle_best_acc)
                abs_gap[f"{fuser}_abs_clean_minus_broken"].append(gap)
                flips[f"{fuser}_flip_clean_vs_broken"].append(fr)

                rows.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "fuser": fuser,
                        "n_train_full": n_train,
                        "n_eval_full_test": n_eval,
                        "unimodal_path": unimodal_path,
                        "q_clean_path": q_clean_path,
                        "q_broken_path": q_broken_path,
                        "noq_mean_acc": noq_acc,
                        "cleanq_acc": clean_acc,
                        "brokenq_acc": brok_acc,
                        "oracle_best_acc": oracle_best_acc,
                        "abs_clean_minus_broken": gap,
                        "flip_clean_vs_broken": fr,
                    }
                )

                print(
                    f"[OK] fuser={fuser} seed={seed} fold={fold} | "
                    f"noQ/clean/broken/oracle={noq_acc:.4f}/{clean_acc:.4f}/{brok_acc:.4f}/{oracle_best_acc:.4f} | "
                    f"|clean-broken|={gap:.4f} flip={fr:.3f}"
                )

    if args.dry_run:
        print("[DONE] dry_run complete.")
        return

    # Write CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = sorted({k for r in rows for k in r.keys()})
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Summary table
    table: Dict[str, Dict[str, Dict[str, float]]] = {}
    for fuser in args.fusers:
        table[fuser] = {
            "NoQ": _mean_std(acc[f"{fuser}_noq"]),
            "cleanQ": _mean_std(acc[f"{fuser}_cleanq"]),
            "brokenQ": _mean_std(acc[f"{fuser}_brokenq"]),
            "oracle-best": _mean_std(acc[f"{fuser}_oracle_best"]),
        }

    summary = {
        "meta": {
            "union_npz": args.union_npz,
            "splits_dir": args.splits_dir,
            "unimodal_root": str(unimodal_root),
            "q_clean_root": str(q_clean_root),
            "q_broken_root": str(q_broken_root),
            "seeds": args.seeds,
            "folds": args.folds,
            "thresh": float(args.thresh),
            "eval_subset": "FULL-only within TEST (Ml==Ma==Mv==1)",
            "train_subset": "FULL-only within TRAIN (Ml==Ma==Mv==1)",
            "NoQ_definition": "arithmetic mean of unimodal probs",
            "brokenQ_definition": "loaded from precomputed q_broken_root (test-only permuted among present rows)",
            "oracle_best_definition": "best unimodal expert per-sample by log-loss (diagnostic ceiling)",
        },
        "table_balanced_accuracy": table,
        "gap_mean_abs_clean_minus_broken": {k: _mean_std(v) for k, v in abs_gap.items()},
        "flip_rate_percent_clean_vs_broken": {k: _mean_std([100.0 * x for x in v]) for k, v in flips.items()},
        "sanity_noq_mean_all_runs": _mean_std(acc["noq_mean"]),
        "sanity_oracle_best_all_runs": _mean_std(acc["oracle_best"]),
        "n_rows": len(rows),
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"[DONE] wrote CSV:  {out_csv}")
    print(f"[DONE] wrote JSON: {out_json}")


if __name__ == "__main__":
    main()



