#!/usr/bin/env python3
"""
paper/unimodal/01_dump_unimodal_preds.py

Defensible unimodal prediction dump for paper experiments.

Contract (matches q_contract.py):
- Load canonical UNION in fixed row order.
- Load subject-safe split ids; build UNION-aligned train/test masks.
- For each modality m ∈ {a,v,p}:
    * Train expert ONLY on TRAIN ∩ PRESENT(m).
    * Predict probabilities on BOTH:
        - TRAIN ∩ PRESENT(m)
        - TEST  ∩ PRESENT(m)
    * Write UNION-length arrays:
        p_m ∈ [0,1] where defined; NaN where modality missing (M_m==0).
        yhat_m ∈ {0,1} where defined; -1 otherwise.
- Save per (seed,fold,model) file:
    out_root/{lr|hgb}/seed_{seed}/fold_{fold}.npz
- Report unimodal Balanced Accuracy on:
    TRAIN∩PRESENT, TEST∩PRESENT, and FULL-only TEST.

Calibration:
- Optional, strictly train-only heldout split from TRAIN∩PRESENT(m).
- If calibrate=none: no calibrator applied (still probabilistic outputs).
- If calibrate=sigmoid|isotonic: use CalibratedClassifierCV on heldout calibration subset.
  No TEST leakage.

Notes:
- This script is for dumping unimodal probs used by downstream fusion.
- Permutation testing / BrokenQ handling is NOT here (by design).

python -m qfd.stressid.dump_unimodal_preds \
  --union_npz /path/to/project/output/fusion_table_noQ/new_fusion_table_union_y2_avp_v1_noQ.npz \
  --splits_dir /path/to/project/splits \
  --out_root /path/to/project/paper_output/unimodal_preds \
  --model lr \
  --seeds 11 22 33 44 55 \
  --folds 0 1 2 3 4 \
  --thresh 0.5 \
  --write_summary_json

python -m qfd.stressid.dump_unimodal_preds \
  --union_npz /path/to/project/output/fusion_table_noQ/new_fusion_table_union_y2_avp_v1_noQ.npz \
  --splits_dir /path/to/project/splits \
  --out_root /path/to/project/paper_output/unimodal_preds \
  --model hgb \
  --seeds 11 22 33 44 55 \
  --folds 0 1 2 3 4 \
  --thresh 0.5 \
  --write_summary_json
"""

from __future__ import annotations

import argparse
import json
import inspect
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

from qfd._shared.q_contract import (
    load_union,
    load_fold_split,
    make_train_test_masks,
    eval_mask_full_only,
    balanced_accuracy_from_probs,
    preds_from_probs,
    assert_probs_nan_where_missing,
)

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedShuffleSplit

try:
    # sklearn >= 1.6
    from sklearn.frozen import FrozenEstimator  # type: ignore
except Exception:
    FrozenEstimator = None


# -----------------------------
# Models
# -----------------------------

def _make_lr(rng_seed: int, C: float) -> Pipeline:
    clf = LogisticRegression(
        solver="liblinear",
        penalty="l2",
        C=float(C),
        max_iter=2000,
        class_weight=None,  # use explicit sample_weight instead (single source of truth)
        random_state=int(rng_seed),
    )
    return Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", clf),
        ]
    )


def _make_hgb(
    rng_seed: int,
    learning_rate: float,
    max_depth: int,
    max_iter: int,
    l2_regularization: float,
) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=float(learning_rate),
        max_depth=int(max_depth),
        max_iter=int(max_iter),
        l2_regularization=float(l2_regularization),
        random_state=int(rng_seed),
    )


def _fit_base_model(
    model: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    rng_seed: int,
    sample_weight: Optional[np.ndarray],
    *,
    lr_C: float,
    hgb_learning_rate: float,
    hgb_max_depth: int,
    hgb_max_iter: int,
    hgb_l2_regularization: float,
):
    if model == "lr":
        m = _make_lr(rng_seed, C=lr_C)
        m.fit(X_train, y_train, clf__sample_weight=sample_weight)
        return m
    if model == "hgb":
        m = _make_hgb(
            rng_seed,
            learning_rate=hgb_learning_rate,
            max_depth=hgb_max_depth,
            max_iter=hgb_max_iter,
            l2_regularization=hgb_l2_regularization,
        )
        m.fit(X_train, y_train, sample_weight=sample_weight)
        return m
    raise ValueError(f"Unknown --model {model}. Use lr or hgb.")


def _fit_with_optional_calibration(
    *,
    model: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    rng_seed: int,
    sample_weight: np.ndarray,
    calibrate: str,
    calib_frac: float,
    lr_C: float,
    hgb_learning_rate: float,
    hgb_max_depth: int,
    hgb_max_iter: int,
    hgb_l2_regularization: float,
):
    """
    Returns an object with predict_proba.

    calibrate=none:
      - train base model on all (TRAIN∩PRESENT)
    calibrate=sigmoid|isotonic:
      - split TRAIN∩PRESENT into (subtrain, calib)
      - train base on subtrain
      - fit calibrator on calib (train-only, no leakage)
    """
    if calibrate == "none":
        base = _fit_base_model(
            model,
            X_tr,
            y_tr,
            rng_seed=rng_seed,
            sample_weight=sample_weight,
            lr_C=lr_C,
            hgb_learning_rate=hgb_learning_rate,
            hgb_max_depth=hgb_max_depth,
            hgb_max_iter=hgb_max_iter,
            hgb_l2_regularization=hgb_l2_regularization,
        )
        return base

    # train-only calibration split
    calib_frac = float(calib_frac)
    if not (0.0 < calib_frac < 0.5):
        raise ValueError("--calib_frac must be in (0,0.5) for stable calibration.")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=calib_frac, random_state=int(rng_seed))
    idx_all = np.arange(len(y_tr))
    sub_idx, cal_idx = next(sss.split(idx_all, y_tr))

    X_sub, y_sub, sw_sub = X_tr[sub_idx], y_tr[sub_idx], sample_weight[sub_idx]
    X_cal, y_cal = X_tr[cal_idx], y_tr[cal_idx]

    base = _fit_base_model(
        model,
        X_sub,
        y_sub,
        rng_seed=rng_seed,
        sample_weight=sw_sub,
        lr_C=lr_C,
        hgb_learning_rate=hgb_learning_rate,
        hgb_max_depth=hgb_max_depth,
        hgb_max_iter=hgb_max_iter,
        hgb_l2_regularization=hgb_l2_regularization,
    )

    Calib = CalibratedClassifierCV
    sig = inspect.signature(Calib).parameters
    has_estimator_kw = "estimator" in sig

    if FrozenEstimator is not None:
        # sklearn >= 1.6 preferred path: cv=None + FrozenEstimator
        kwargs = {"method": calibrate, "cv": None}
        if has_estimator_kw:
            kwargs["estimator"] = FrozenEstimator(base)
        else:
            kwargs["base_estimator"] = FrozenEstimator(base)  # pragma: no cover
        cal = Calib(**kwargs)
        cal.fit(X_cal, y_cal)
        return cal

    # older sklearn: cv="prefit"
    kwargs = {"method": calibrate, "cv": "prefit"}
    if has_estimator_kw:
        kwargs["estimator"] = base
    else:
        kwargs["base_estimator"] = base  # pragma: no cover
    cal = Calib(**kwargs)
    cal.fit(X_cal, y_cal)
    return cal


# -----------------------------
# Helpers
# -----------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_npz(path: Path, **kwargs) -> None:
    _ensure_dir(path.parent)
    np.savez_compressed(path, **kwargs)


def _mask_counts(union, train_mask, test_mask, eval_full_mask) -> Dict[str, int]:
    return {
        "N_union": int(len(union.ids_str)),
        "N_train": int(train_mask.sum()),
        "N_test": int(test_mask.sum()),
        "N_full_only_test": int(eval_full_mask.sum()),
        "N_train_present_a": int((train_mask & (union.Ma == 1)).sum()),
        "N_train_present_v": int((train_mask & (union.Mv == 1)).sum()),
        "N_train_present_p": int((train_mask & (union.Mp == 1)).sum()),
        "N_test_present_a": int((test_mask & (union.Ma == 1)).sum()),
        "N_test_present_v": int((test_mask & (union.Mv == 1)).sum()),
        "N_test_present_p": int((test_mask & (union.Mp == 1)).sum()),
    }


def _eval_metrics_present_only(
    y: np.ndarray,
    p_pos: np.ndarray,
    present_mask: np.ndarray,
    *,
    thresh: float,
) -> Dict[str, float]:
    idx = np.where(present_mask)[0]
    if idx.size == 0:
        return {"balacc": float("nan"), "n": 0}
    return {
        "balacc": float(balanced_accuracy_from_probs(y[idx], p_pos[idx], thresh=thresh)),
        "n": int(idx.size),
    }


def _assert_probs_in_range_where_finite(p: np.ndarray, name: str) -> None:
    p = np.asarray(p, dtype=float)
    m = np.isfinite(p)
    if not np.any(m):
        return
    lo = np.min(p[m])
    hi = np.max(p[m])
    if lo < -1e-6 or hi > 1.0 + 1e-6:
        raise ValueError(f"{name}: probs out of [0,1] range: min={lo}, max={hi}")


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--union_npz", type=str, required=True)
    ap.add_argument("--splits_dir", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)

    ap.add_argument("--model", type=str, choices=["lr", "hgb"], required=True)

    # LR hyperparam
    ap.add_argument("--lr_C", type=float, default=1.0)

    # HGB hyperparams
    ap.add_argument("--hgb_learning_rate", type=float, default=0.05)
    ap.add_argument("--hgb_max_depth", type=int, default=3)
    ap.add_argument("--hgb_max_iter", type=int, default=300)
    ap.add_argument("--hgb_l2_regularization", type=float, default=0.0)

    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--folds", type=int, nargs="+", required=True)
    ap.add_argument("--thresh", type=float, default=0.5)

    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--write_summary_json", action="store_true")

    ap.add_argument("--calibrate", type=str, choices=["none", "sigmoid", "isotonic"], default="none")
    ap.add_argument("--calib_frac", type=float, default=0.2)

    # Split coverage is strict by default (matches paper contract for UNION coverage)
    ap.add_argument("--allow_partial_split_coverage", action="store_true")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    union = load_union(args.union_npz)
    N = len(union.ids_str)

    out_root = Path(args.out_root) / args.model
    _ensure_dir(out_root)

    run_rows: List[Dict] = []

    for seed in args.seeds:
        for fold in args.folds:
            split = load_fold_split(args.splits_dir, seed=seed, fold=fold)
            train_mask, test_mask = make_train_test_masks(
                union,
                split,
                require_full_coverage=not args.allow_partial_split_coverage,
            )
            eval_full_mask = eval_mask_full_only(union, test_mask)

            rng_seed = int(seed) * 100 + int(fold)

            out_path = out_root / f"seed_{seed}" / f"fold_{fold}.npz"
            if out_path.exists() and not args.overwrite:
                run_rows.append({"seed": seed, "fold": fold, "status": "SKIP_EXISTS", "out": str(out_path)})
                continue

            # Allocate full-length arrays (UNION-aligned)
            p_a = np.full(N, np.nan, dtype=float)
            p_v = np.full(N, np.nan, dtype=float)
            p_p = np.full(N, np.nan, dtype=float)

            yhat_a = np.full(N, -1, dtype=int)
            yhat_v = np.full(N, -1, dtype=int)
            yhat_p = np.full(N, -1, dtype=int)

            def _select_XM(mod: str) -> Tuple[np.ndarray, np.ndarray]:
                if mod == "a":
                    return union.Ea, union.Ma
                if mod == "v":
                    return union.Ev, union.Mv
                if mod == "p":
                    return union.Ep, union.Mp
                raise ValueError(mod)

            def train_and_predict_both(mod: str):
                X, M = _select_XM(mod)

                tr = train_mask & (M == 1)
                te = test_mask & (M == 1)

                if tr.sum() == 0:
                    raise RuntimeError(f"seed={seed} fold={fold} modality={mod}: TRAIN∩PRESENT is empty.")
                if te.sum() == 0:
                    raise RuntimeError(f"seed={seed} fold={fold} modality={mod}: TEST∩PRESENT is empty.")

                X_tr = X[tr]
                y_tr = union.y[tr].astype(int)

                # single source of truth for class balancing
                sw = compute_sample_weight(class_weight="balanced", y=y_tr).astype(float)

                model_obj = _fit_with_optional_calibration(
                    model=args.model,
                    X_tr=X_tr,
                    y_tr=y_tr,
                    rng_seed=rng_seed,
                    sample_weight=sw,
                    calibrate=args.calibrate,
                    calib_frac=args.calib_frac,
                    lr_C=args.lr_C,
                    hgb_learning_rate=args.hgb_learning_rate,
                    hgb_max_depth=args.hgb_max_depth,
                    hgb_max_iter=args.hgb_max_iter,
                    hgb_l2_regularization=args.hgb_l2_regularization,
                )

                # Predict on TRAIN∩PRESENT
                p_tr = model_obj.predict_proba(X_tr)[:, 1].astype(float)
                yhat_tr = preds_from_probs(p_tr, thresh=args.thresh)

                # Predict on TEST∩PRESENT
                X_te = X[te]
                p_te = model_obj.predict_proba(X_te)[:, 1].astype(float)
                yhat_te = preds_from_probs(p_te, thresh=args.thresh)

                return tr, p_tr, yhat_tr, te, p_te, yhat_te

            # Train + predict per modality
            tr_a, pa_tr, ya_tr, te_a, pa_te, ya_te = train_and_predict_both("a")
            tr_v, pv_tr, yv_tr, te_v, pv_te, yv_te = train_and_predict_both("v")
            tr_p, pp_tr, yp_tr, te_p, pp_te, yp_te = train_and_predict_both("p")

            # Fill UNION arrays at BOTH train and test indices
            p_a[tr_a] = np.clip(pa_tr, 0.0, 1.0)
            p_a[te_a] = np.clip(pa_te, 0.0, 1.0)
            yhat_a[tr_a] = ya_tr
            yhat_a[te_a] = ya_te

            p_v[tr_v] = np.clip(pv_tr, 0.0, 1.0)
            p_v[te_v] = np.clip(pv_te, 0.0, 1.0)
            yhat_v[tr_v] = yv_tr
            yhat_v[te_v] = yv_te

            p_p[tr_p] = np.clip(pp_tr, 0.0, 1.0)
            p_p[te_p] = np.clip(pp_te, 0.0, 1.0)
            yhat_p[tr_p] = yp_tr
            yhat_p[te_p] = yp_te

            # Contract sanity: missing modalities must remain NaN
            assert_probs_nan_where_missing(union, p_a, p_v, p_p)
            _assert_probs_in_range_where_finite(p_a, "p_a")
            _assert_probs_in_range_where_finite(p_v, "p_v")
            _assert_probs_in_range_where_finite(p_p, "p_p")

            # Metrics: TRAIN∩PRESENT, TEST∩PRESENT, FULL-only TEST
            metrics = {
                "A_train_present": _eval_metrics_present_only(union.y, p_a, tr_a, thresh=args.thresh),
                "V_train_present": _eval_metrics_present_only(union.y, p_v, tr_v, thresh=args.thresh),
                "P_train_present": _eval_metrics_present_only(union.y, p_p, tr_p, thresh=args.thresh),

                "A_test_present": _eval_metrics_present_only(union.y, p_a, te_a, thresh=args.thresh),
                "V_test_present": _eval_metrics_present_only(union.y, p_v, te_v, thresh=args.thresh),
                "P_test_present": _eval_metrics_present_only(union.y, p_p, te_p, thresh=args.thresh),

                "A_full_only_test": _eval_metrics_present_only(union.y, p_a, eval_full_mask, thresh=args.thresh),
                "V_full_only_test": _eval_metrics_present_only(union.y, p_v, eval_full_mask, thresh=args.thresh),
                "P_full_only_test": _eval_metrics_present_only(union.y, p_p, eval_full_mask, thresh=args.thresh),
            }

            meta = {
                "script": "paper/unimodal/01_dump_unimodal_preds.py",
                "union_npz": str(Path(args.union_npz)),
                "splits_dir": str(Path(args.splits_dir)),
                "seed": int(seed),
                "fold": int(fold),
                "rng_seed": int(rng_seed),
                "model": str(args.model),
                "thresh": float(args.thresh),
                "counts": _mask_counts(union, train_mask, test_mask, eval_full_mask),
                "metrics": metrics,
                "calibrate": str(args.calibrate),
                "calib_frac": float(args.calib_frac),
                "hyperparams": {
                    "lr_C": float(args.lr_C),
                    "hgb_learning_rate": float(args.hgb_learning_rate),
                    "hgb_max_depth": int(args.hgb_max_depth),
                    "hgb_max_iter": int(args.hgb_max_iter),
                    "hgb_l2_regularization": float(args.hgb_l2_regularization),
                },
                "note": "p_* filled for BOTH train∩present and test∩present; NaN where modality missing (M==0).",
            }

            _save_npz(
                out_path,
                ids=union.ids_str,
                y=union.y.astype(int),
                train_mask=train_mask.astype(np.uint8),
                test_mask=test_mask.astype(np.uint8),
                full_only_test_mask=eval_full_mask.astype(np.uint8),
                p_a=p_a,
                p_v=p_v,
                p_p=p_p,
                yhat_a=yhat_a,
                yhat_v=yhat_v,
                yhat_p=yhat_p,
                meta=meta,
            )

            run_rows.append({"seed": seed, "fold": fold, "status": "WROTE", "out": str(out_path), "meta": meta})

            print(
                f"[OK] model={args.model} seed={seed} fold={fold} wrote {out_path} | "
                f"A_full={metrics['A_full_only_test']['balacc']:.4f} "
                f"V_full={metrics['V_full_only_test']['balacc']:.4f} "
                f"P_full={metrics['P_full_only_test']['balacc']:.4f}"
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



