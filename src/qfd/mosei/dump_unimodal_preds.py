#!/usr/bin/env python3
"""
dump_mosei_unimodal_preds.py — StressID-compatible unimodal posterior dump for MOSEI

Contract:
- Train each modality expert ONLY on TRAIN ∩ PRESENT(modality).
- Predict probabilities on:
    TRAIN ∩ PRESENT(modality)
    TEST  ∩ PRESENT(modality)
- Save length-N arrays aligned to UNION row order:
    p_l, p_a, p_v in [0,1], NaN only where modality missing.
    yhat_* defined where prob defined, else -1.

Outputs (per seed/fold):
  out_root/<model>/seed_<seed>/fold_<fold>.npz
with:
  ids, y, train_mask, test_mask, full_only_test_mask,
  p_l, p_a, p_v, yhat_l, yhat_a, yhat_v, meta(json)

FULL-only (MOSEI): Ml==1 & Ma==1 & Mv==1 within TEST.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedShuffleSplit


# -----------------------------
# Metrics
# -----------------------------
def preds_from_probs(p: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    return (p >= thresh).astype(int)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    # handle edge cases safely
    pos = (y_true == 1)
    neg = (y_true == 0)
    tpr = (y_pred[pos] == 1).mean() if pos.any() else 0.0
    tnr = (y_pred[neg] == 0).mean() if neg.any() else 0.0
    return float(0.5 * (tpr + tnr))


def balanced_accuracy_from_probs(y_true: np.ndarray, p_pos: np.ndarray, thresh: float = 0.5) -> float:
    return balanced_accuracy(y_true, preds_from_probs(p_pos, thresh=thresh))


def _eval_metrics_present_only(y: np.ndarray, p_pos: np.ndarray, present_mask: np.ndarray, thresh: float) -> Dict[str, float]:
    idx = np.where(present_mask)[0]
    if idx.size == 0:
        return {"balacc": float("nan"), "n": 0}
    return {
        "balacc": float(balanced_accuracy_from_probs(y[idx], p_pos[idx], thresh=thresh)),
        "n": int(idx.size),
    }


# -----------------------------
# IO helpers
# -----------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_npz(path: Path, **kwargs) -> None:
    _ensure_dir(path.parent)
    np.savez_compressed(path, **kwargs)


def load_union(union_npz: str):
    z = np.load(union_npz, allow_pickle=True)
    required = ["ids", "y", "E_l", "E_a", "E_v", "M_l", "M_a", "M_v"]
    missing = [k for k in required if k not in z.files]
    if missing:
        raise KeyError(f"UNION missing keys {missing}. Found: {sorted(z.files)}")

    class Union:  # lightweight container
        pass

    u = Union()
    u.ids = z["ids"].astype(str)
    u.ids_str = u.ids.astype(object)
    u.y = z["y"].astype(int).reshape(-1)

    u.El = z["E_l"].astype(np.float32)
    u.Ea = z["E_a"].astype(np.float32)
    u.Ev = z["E_v"].astype(np.float32)

    u.Ml = z["M_l"].astype(np.int8).reshape(-1)
    u.Ma = z["M_a"].astype(np.int8).reshape(-1)
    u.Mv = z["M_v"].astype(np.int8).reshape(-1)

    return u


def load_fold_split(splits_dir: str, seed: int, fold: int) -> Dict[str, np.ndarray]:
    sd = Path(splits_dir) / f"seed_{seed}"
    tr = np.load(sd / f"train_ids_fold{fold}.npy", allow_pickle=True).astype(str)
    te = np.load(sd / f"test_ids_fold{fold}.npy", allow_pickle=True).astype(str)
    return {"train_ids": tr, "test_ids": te}


def make_train_test_masks(union, split: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    id2i = {sid: i for i, sid in enumerate(union.ids.tolist())}
    train_mask = np.zeros((len(union.ids),), dtype=bool)
    test_mask = np.zeros((len(union.ids),), dtype=bool)

    for s in split["train_ids"]:
        train_mask[id2i[s]] = True
    for s in split["test_ids"]:
        test_mask[id2i[s]] = True

    if np.any(train_mask & test_mask):
        raise AssertionError("train/test masks overlap.")
    return train_mask, test_mask


def eval_mask_full_only(union, test_mask: np.ndarray) -> np.ndarray:
    return test_mask & (union.Ml == 1) & (union.Ma == 1) & (union.Mv == 1)


def _mask_counts(union, train_mask, test_mask, eval_full_mask) -> Dict[str, int]:
    return {
        "N_union": int(len(union.ids)),
        "N_train": int(train_mask.sum()),
        "N_test": int(test_mask.sum()),
        "N_full_only_test": int(eval_full_mask.sum()),
        "N_train_present_l": int((train_mask & (union.Ml == 1)).sum()),
        "N_train_present_a": int((train_mask & (union.Ma == 1)).sum()),
        "N_train_present_v": int((train_mask & (union.Mv == 1)).sum()),
        "N_test_present_l": int((test_mask & (union.Ml == 1)).sum()),
        "N_test_present_a": int((test_mask & (union.Ma == 1)).sum()),
        "N_test_present_v": int((test_mask & (union.Mv == 1)).sum()),
    }


# -----------------------------
# Models
# -----------------------------
def _make_lr(rng_seed: int) -> Pipeline:
    clf = LogisticRegression(
        solver="liblinear",
        penalty="l2",
        C=1.0,
        max_iter=2000,
        class_weight=None,  # use sample_weight instead (consistent across models)
        random_state=rng_seed,
    )
    return Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)), ("clf", clf)])


def _make_hgb(rng_seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=3,
        max_iter=300,
        l2_regularization=0.0,
        random_state=rng_seed,
    )


def _fit_model(model: str, X_train: np.ndarray, y_train: np.ndarray, rng_seed: int, sample_weight: np.ndarray | None):
    if model == "lr":
        m = _make_lr(rng_seed)
        m.fit(X_train, y_train, clf__sample_weight=sample_weight)
        return m
    if model == "hgb":
        m = _make_hgb(rng_seed)
        m.fit(X_train, y_train, sample_weight=sample_weight)
        return m
    raise ValueError(model)


# -----------------------------
# Main
# -----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--union_npz", type=str, required=True)
    ap.add_argument("--splits_dir", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--model", type=str, choices=["lr", "hgb"], default="lr")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--folds", type=int, nargs="+", required=True)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--write_summary_json", action="store_true")
    ap.add_argument("--calibrate", type=str, choices=["none", "sigmoid", "isotonic"], default="none")
    ap.add_argument("--calib_frac", type=float, default=0.2)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    union = load_union(args.union_npz)
    N = len(union.ids)

    out_root = Path(args.out_root) / args.model
    _ensure_dir(out_root)

    run_rows: List[Dict] = []

    for seed in args.seeds:
        for fold in args.folds:
            split = load_fold_split(args.splits_dir, seed=seed, fold=fold)
            train_mask, test_mask = make_train_test_masks(union, split)
            eval_full_mask = eval_mask_full_only(union, test_mask)

            rng_seed = seed * 100 + fold

            out_path = out_root / f"seed_{seed}" / f"fold_{fold}.npz"
            if out_path.exists() and not args.overwrite:
                run_rows.append({"seed": seed, "fold": fold, "status": "SKIP_EXISTS", "out": str(out_path)})
                continue

            # Allocate UNION-aligned arrays
            p_l = np.full(N, np.nan, dtype=float)
            p_a = np.full(N, np.nan, dtype=float)
            p_v = np.full(N, np.nan, dtype=float)

            yhat_l = np.full(N, -1, dtype=int)
            yhat_a = np.full(N, -1, dtype=int)
            yhat_v = np.full(N, -1, dtype=int)

            def train_and_predict_both(mod: str):
                if mod == "l":
                    X, M = union.El, union.Ml
                elif mod == "a":
                    X, M = union.Ea, union.Ma
                elif mod == "v":
                    X, M = union.Ev, union.Mv
                else:
                    raise ValueError(mod)

                tr = train_mask & (M == 1)
                te = test_mask & (M == 1)

                if tr.sum() == 0:
                    raise RuntimeError(f"seed={seed} fold={fold} modality={mod}: TRAIN∩PRESENT empty.")
                if te.sum() == 0:
                    raise RuntimeError(f"seed={seed} fold={fold} modality={mod}: TEST∩PRESENT empty.")

                X_tr = X[tr]
                y_tr = union.y[tr].astype(int)

                sw = compute_sample_weight(class_weight="balanced", y=y_tr)

                # Base model always exists (fixes your bug)
                base = None
                model_obj = None

                if args.calibrate != "none":
                    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.calib_frac, random_state=rng_seed)
                    idx_all = np.arange(len(y_tr))
                    sub_idx, cal_idx = next(sss.split(idx_all, y_tr))

                    X_sub, y_sub, sw_sub = X_tr[sub_idx], y_tr[sub_idx], sw[sub_idx]
                    X_cal, y_cal = X_tr[cal_idx], y_tr[cal_idx]

                    base = _fit_model(args.model, X_sub, y_sub, rng_seed=rng_seed, sample_weight=sw_sub)

                    # Calibrate on held-out calibration subset (still within TRAIN fold)
                    Calib = CalibratedClassifierCV
                    model_obj = Calib(estimator=base, method=args.calibrate, cv="prefit")
                    model_obj.fit(X_cal, y_cal)
                else:
                    base = _fit_model(args.model, X_tr, y_tr, rng_seed=rng_seed, sample_weight=sw)
                    model_obj = base

                # Predict on TRAIN ∩ PRESENT
                p_tr = model_obj.predict_proba(X_tr)[:, 1].astype(float)
                yhat_tr = preds_from_probs(p_tr, thresh=args.thresh)

                # Predict on TEST ∩ PRESENT
                X_te = X[te]
                p_te = model_obj.predict_proba(X_te)[:, 1].astype(float)
                yhat_te = preds_from_probs(p_te, thresh=args.thresh)

                return tr, p_tr, yhat_tr, te, p_te, yhat_te

            # Language / Audio / Visual
            tr_l, pl_tr, yl_tr, te_l, pl_te, yl_te = train_and_predict_both("l")
            tr_a, pa_tr, ya_tr, te_a, pa_te, ya_te = train_and_predict_both("a")
            tr_v, pv_tr, yv_tr, te_v, pv_te, yv_te = train_and_predict_both("v")

            # Fill
            p_l[tr_l] = np.clip(pl_tr, 0.0, 1.0)
            p_l[te_l] = np.clip(pl_te, 0.0, 1.0)
            yhat_l[tr_l] = yl_tr
            yhat_l[te_l] = yl_te

            p_a[tr_a] = np.clip(pa_tr, 0.0, 1.0)
            p_a[te_a] = np.clip(pa_te, 0.0, 1.0)
            yhat_a[tr_a] = ya_tr
            yhat_a[te_a] = ya_te

            p_v[tr_v] = np.clip(pv_tr, 0.0, 1.0)
            p_v[te_v] = np.clip(pv_te, 0.0, 1.0)
            yhat_v[tr_v] = yv_tr
            yhat_v[te_v] = yv_te

            # Metrics (report FULL-only test = L&A&V present)
            metrics = {
                "L_train_present": _eval_metrics_present_only(union.y, p_l, tr_l, thresh=args.thresh),
                "A_train_present": _eval_metrics_present_only(union.y, p_a, tr_a, thresh=args.thresh),
                "V_train_present": _eval_metrics_present_only(union.y, p_v, tr_v, thresh=args.thresh),

                "L_test_present": _eval_metrics_present_only(union.y, p_l, te_l, thresh=args.thresh),
                "A_test_present": _eval_metrics_present_only(union.y, p_a, te_a, thresh=args.thresh),
                "V_test_present": _eval_metrics_present_only(union.y, p_v, te_v, thresh=args.thresh),

                "L_full_only_test": _eval_metrics_present_only(union.y, p_l, eval_full_mask, thresh=args.thresh),
                "A_full_only_test": _eval_metrics_present_only(union.y, p_a, eval_full_mask, thresh=args.thresh),
                "V_full_only_test": _eval_metrics_present_only(union.y, p_v, eval_full_mask, thresh=args.thresh),
            }

            meta = {
                "script": "dump_mosei_unimodal_preds.py",
                "union_npz": str(Path(args.union_npz)),
                "splits_dir": str(Path(args.splits_dir)),
                "seed": int(seed),
                "fold": int(fold),
                "rng_seed": int(rng_seed),
                "model": args.model,
                "thresh": float(args.thresh),
                "counts": _mask_counts(union, train_mask, test_mask, eval_full_mask),
                "metrics": metrics,
                "calibrate": args.calibrate,
                "calib_frac": float(args.calib_frac),
                "note": "p_* filled for BOTH train∩present and test∩present. NaN only where modality missing.",
            }

            _save_npz(
                out_path,
                ids=union.ids_str,
                y=union.y.astype(int),
                train_mask=train_mask.astype(np.uint8),
                test_mask=test_mask.astype(np.uint8),
                full_only_test_mask=eval_full_mask.astype(np.uint8),
                p_l=p_l, p_a=p_a, p_v=p_v,
                yhat_l=yhat_l, yhat_a=yhat_a, yhat_v=yhat_v,
                meta=np.array([json.dumps(meta)], dtype=object),
            )

            run_rows.append({"seed": seed, "fold": fold, "status": "WROTE", "out": str(out_path)})

            print(
                f"[OK] seed={seed} fold={fold} wrote {out_path} | "
                f"L_full={metrics['L_full_only_test']['balacc']:.4f} "
                f"A_full={metrics['A_full_only_test']['balacc']:.4f} "
                f"V_full={metrics['V_full_only_test']['balacc']:.4f}"
            )

    if args.write_summary_json:
        summary_path = out_root / "unimodal_preds_summary.json"
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "args": vars(args),
                    "n_wrote": sum(1 for r in run_rows if r["status"] == "WROTE"),
                    "n_skipped": sum(1 for r in run_rows if r["status"] == "SKIP_EXISTS"),
                    "rows": run_rows,
                },
                f,
                indent=2,
            )
        print(f"[DONE] wrote summary: {summary_path}")


if __name__ == "__main__":
    main()


