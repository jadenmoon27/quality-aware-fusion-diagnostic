#!/usr/bin/env python3
"""
build_mosei_union.py

StressID-grade UNION builder for CMU-MOSEI.

Contract (one row = one opinion segment):
  ids[i]      : unique segment id (video_id + segment index)
  groups[i]   : grouping key for subject-safe / video-safe splits (video_id)
  y[i]        : binary label (1 if sentiment>0 else 0) derived from MOSEI sentiment
  E_l/E_a/E_v : pooled embeddings/features for Language/Audio/Visual
  M_l/M_a/M_v : presence masks (1 iff usable pooled vector exists)
  A_l/A_a/A_v : availability masks (1 iff present and at least one fully-finite overlapping frame exists)
  seg_intervals[i] (optional) : [t0,t1] for audits
  Y_raw[i] (optional)         : raw label feature vector for audits (includes sentiment + emotions)

Rigor requirements (mirrors StressID discipline):
  - Canonical segmentation: align to "Opinion Segment Labels"
  - UNION row order is fixed and saved; never reorder downstream
  - Explicit separation: evidence E, availability M/A, later quality Q
  - Missing policy: if no usable frames => E=0 vector, M=0, A=0
  - Strong diagnostics and invariants
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


# -----------------------------
# Canonical MOSEI segmentation
# -----------------------------
LABEL_KEY = "Opinion Segment Labels"  # canonical unit in MOSEI SDK
SENTIMENT_COL = 0                    # sentiment column in label feature vector
BINARIZE_RULE = "y=1 if sentiment>0 else 0"


# -----------------------------
# Data loading (local .csd)
# -----------------------------
def load_local_dataset(data_path: str) -> "mmdatasdk.mmdataset":
    """
    Load local MOSEI .csd files and expose a canonical label key LABEL_KEY.

    IMPORTANT:
      We map LABEL_KEY -> CMU_MOSEI_Labels.csd so ds.align(LABEL_KEY) is valid.
      This makes your segmentation boundary explicit and stable.
    """
    try:
        from mmsdk import mmdatasdk
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "qfd.mosei.build_union requires the optional CMU Multimodal SDK "
            "dependency. Install the MOSEI extra, for example: "
            "python -m pip install '.[mosei]'"
        ) from exc

    p = Path(data_path)
    local = {
        "glove_vectors": str(p / "CMU_MOSEI_TimestampedWordVectors.csd"),
        "COVAREP": str(p / "CMU_MOSEI_COVAREP.csd"),
        "FACET 4.2": str(p / "CMU_MOSEI_VisualFacet42.csd"),
        LABEL_KEY: str(p / "CMU_MOSEI_Labels.csd"),
        # OpenFace_2 intentionally excluded (not needed for L/A/V)
    }
    for k, fp in local.items():
        if not Path(fp).exists():
            raise FileNotFoundError(f"Missing file for key '{k}': {fp}")
    return mmdatasdk.mmdataset(local, str(p))


# -----------------------------
# Interval overlap + pooling
# -----------------------------
def overlap_mask(frame_intervals: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """
    frame_intervals: (T,2) [start,end]
    Overlap iff frame_end > t0 and frame_start < t1.
    """
    a0 = frame_intervals[:, 0]
    a1 = frame_intervals[:, 1]
    return (a1 > t0) & (a0 < t1)


def pooled_segment(
    frame_intervals: np.ndarray,
    frame_features: np.ndarray,
    seg_t0: float,
    seg_t1: float,
) -> Tuple[Optional[np.ndarray], int, int]:
    """
    Mean pool fully-finite frames overlapping [seg_t0, seg_t1].

    Returns:
      vec_or_none,
      has_overlap (1 iff any overlapping frames exist),
      has_finite  (1 iff at least one fully-finite overlapping frame exists)
    """
    m = overlap_mask(frame_intervals, seg_t0, seg_t1)
    if not np.any(m):
        return None, 0, 0

    X = frame_features[m]
    finite_rows = np.isfinite(X).all(axis=1)
    X = X[finite_rows]
    if X.shape[0] == 0:
        return None, 1, 0

    return X.mean(axis=0).astype(np.float32), 1, 1


def infer_dim(mod_data: Dict[str, dict], name: str) -> int:
    """
    Robustly infer feature dimension from first usable entry.
    """
    for vid, item in mod_data.items():
        X = np.asarray(item.get("features", None))
        if X is None:
            continue
        if X.ndim == 2 and X.shape[1] > 0:
            return int(X.shape[1])
    raise RuntimeError(f"Could not infer feature dim for '{name}' (no usable features).")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="mosei_data/")
    ap.add_argument("--out_npz", type=str, default="output/mosei_union/mosei_union.npz")
    ap.add_argument(
        "--keep_intervals",
        action="store_true",
        help="Store seg_intervals in UNION for auditing (recommended).",
    )
    ap.add_argument(
        "--keep_label_features",
        action="store_true",
        help="Store raw label feature vectors per segment as Y_raw (recommended for audits).",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Fail hard if alignment or expected keys are missing.",
    )
    args = ap.parse_args()

    out_path = Path(args.out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_local_dataset(args.data_path)

    # Key presence check
    needed = [LABEL_KEY, "glove_vectors", "COVAREP", "FACET 4.2"]
    missing = [k for k in needed if k not in ds.keys()]
    if missing:
        msg = f"Missing required dataset keys: {missing}. Available: {list(ds.keys())}"
        if args.strict:
            raise KeyError(msg)
        print(f"[WARN] {msg}")

    # Canonical alignment: everything to Opinion Segment Labels
    try:
        ds.align(LABEL_KEY)
    except Exception as e:
        msg = f"ds.align('{LABEL_KEY}') failed: {e}"
        if args.strict:
            raise RuntimeError(msg)
        print(f"[WARN] {msg}")

    # Aligned data dicts
    lab = ds[LABEL_KEY].data
    gv = ds["glove_vectors"].data
    cov = ds["COVAREP"].data
    facet = ds["FACET 4.2"].data

    # Feature dims
    DL = infer_dim(gv, "glove_vectors")     # expected 300
    DA = infer_dim(cov, "COVAREP")          # expected 74
    DV = infer_dim(facet, "FACET 4.2")      # expected 35

    # UNION outputs
    ids, groups, y = [], [], []
    seg_intervals = []

    E_l, E_a, E_v = [], [], []
    M_l, M_a, M_v = [], [], []
    A_l, A_a, A_v = [], [], []

    Y_raw = []  # optional audits

    # Diagnostics (StressID-grade: separate causes)
    diag = {
        "missing_videos_with_labels": {"glove": 0, "covarep": 0, "facet": 0},
        "segments": {
            "total": 0,
            "label_nonfinite": 0,
            "label_shape_bad": 0,
            "label_dim_mismatch": 0,
        },
        "pooling": {
            "L": {"no_overlap": 0, "overlap_all_nonfinite": 0, "dim_mismatch": 0},
            "A": {"no_overlap": 0, "overlap_all_nonfinite": 0, "dim_mismatch": 0},
            "V": {"no_overlap": 0, "overlap_all_nonfinite": 0, "dim_mismatch": 0},
        },
    }

    # Canonical video order: labels define universe
    label_vids = sorted(lab.keys())

    for vid in label_vids:
        L = lab[vid]
        itv_lab = np.asarray(L.get("intervals", None), dtype=float)
        feat_lab = np.asarray(L.get("features", None), dtype=float)

        if itv_lab.ndim != 2 or feat_lab.ndim != 2 or itv_lab.shape[0] != feat_lab.shape[0]:
            diag["segments"]["label_shape_bad"] += 1
            if args.strict:
                raise RuntimeError(
                    f"Bad label shapes for vid={vid}: intervals={itv_lab.shape}, features={feat_lab.shape}"
                )
            # skip video if labels are malformed
            continue

        Dy = int(feat_lab.shape[1])
        if Dy <= SENTIMENT_COL:
            raise RuntimeError(
                f"Label feature dim Dy={Dy} does not include SENTIMENT_COL={SENTIMENT_COL} (vid={vid})."
            )

        sentiments = feat_lab[:, SENTIMENT_COL]

        # modality availability at video-level
        has_g = vid in gv
        has_a = vid in cov
        has_v = vid in facet
        if not has_g:
            diag["missing_videos_with_labels"]["glove"] += 1
        if not has_a:
            diag["missing_videos_with_labels"]["covarep"] += 1
        if not has_v:
            diag["missing_videos_with_labels"]["facet"] += 1

        if has_g:
            itv_g = np.asarray(gv[vid]["intervals"], dtype=float)
            X_g = np.asarray(gv[vid]["features"], dtype=float)
        if has_a:
            itv_a = np.asarray(cov[vid]["intervals"], dtype=float)
            X_a = np.asarray(cov[vid]["features"], dtype=float)
        if has_v:
            itv_v = np.asarray(facet[vid]["intervals"], dtype=float)
            X_v = np.asarray(facet[vid]["features"], dtype=float)

        # iterate segments
        for j in range(itv_lab.shape[0]):
            diag["segments"]["total"] += 1
            t0, t1 = float(itv_lab[j, 0]), float(itv_lab[j, 1])

            seg_id = f"{vid}[{j}]"
            ids.append(seg_id)
            groups.append(vid)
            seg_intervals.append([t0, t1])

            s = float(sentiments[j])
            if not np.isfinite(s):
                diag["segments"]["label_nonfinite"] += 1
                s = 0.0  # deterministic fallback; rare, but makes pipeline total
            y.append(1 if s > 0 else 0)

            if args.keep_label_features:
                yy = feat_lab[j].astype(np.float32)
                if yy.shape[0] != Dy:
                    diag["segments"]["label_dim_mismatch"] += 1
                Y_raw.append(yy)

            # ---- L ----
            if has_g:
                vec, has_ov, has_fin = pooled_segment(itv_g, X_g, t0, t1)
            else:
                vec, has_ov, has_fin = None, 0, 0

            if has_ov == 0:
                diag["pooling"]["L"]["no_overlap"] += 1
            elif has_fin == 0:
                diag["pooling"]["L"]["overlap_all_nonfinite"] += 1

            if vec is not None and vec.shape[0] == DL:
                E_l.append(vec); M_l.append(1); A_l.append(1)
            else:
                if vec is not None and vec.shape[0] != DL:
                    diag["pooling"]["L"]["dim_mismatch"] += 1
                E_l.append(np.zeros(DL, dtype=np.float32)); M_l.append(0); A_l.append(0)

            # ---- A ----
            if has_a:
                vec, has_ov, has_fin = pooled_segment(itv_a, X_a, t0, t1)
            else:
                vec, has_ov, has_fin = None, 0, 0

            if has_ov == 0:
                diag["pooling"]["A"]["no_overlap"] += 1
            elif has_fin == 0:
                diag["pooling"]["A"]["overlap_all_nonfinite"] += 1

            if vec is not None and vec.shape[0] == DA:
                E_a.append(vec); M_a.append(1); A_a.append(1)
            else:
                if vec is not None and vec.shape[0] != DA:
                    diag["pooling"]["A"]["dim_mismatch"] += 1
                E_a.append(np.zeros(DA, dtype=np.float32)); M_a.append(0); A_a.append(0)

            # ---- V ----
            if has_v:
                vec, has_ov, has_fin = pooled_segment(itv_v, X_v, t0, t1)
            else:
                vec, has_ov, has_fin = None, 0, 0

            if has_ov == 0:
                diag["pooling"]["V"]["no_overlap"] += 1
            elif has_fin == 0:
                diag["pooling"]["V"]["overlap_all_nonfinite"] += 1

            if vec is not None and vec.shape[0] == DV:
                E_v.append(vec); M_v.append(1); A_v.append(1)
            else:
                if vec is not None and vec.shape[0] != DV:
                    diag["pooling"]["V"]["dim_mismatch"] += 1
                E_v.append(np.zeros(DV, dtype=np.float32)); M_v.append(0); A_v.append(0)

    # Stack arrays
    ids = np.array(ids, dtype=object)
    groups = np.array(groups, dtype=object)
    y = np.array(y, dtype=np.int64)

    E_l = np.stack(E_l, axis=0).astype(np.float32)
    E_a = np.stack(E_a, axis=0).astype(np.float32)
    E_v = np.stack(E_v, axis=0).astype(np.float32)

    M_l = np.array(M_l, dtype=np.int8)
    M_a = np.array(M_a, dtype=np.int8)
    M_v = np.array(M_v, dtype=np.int8)

    A_l = np.array(A_l, dtype=np.int8)
    A_a = np.array(A_a, dtype=np.int8)
    A_v = np.array(A_v, dtype=np.int8)

    if args.keep_label_features:
        if len(Y_raw) == 0:
            raise RuntimeError("keep_label_features enabled but Y_raw is empty.")
        Dy0 = Y_raw[0].shape[0]
        for rr in Y_raw:
            if rr.shape[0] != Dy0:
                raise RuntimeError("Inconsistent label feature dims across segments; cannot stack Y_raw.")
        Y_raw = np.stack(Y_raw, axis=0).astype(np.float32)

    # Invariants (hard)
    if len(set(ids.tolist())) != len(ids):
        raise AssertionError("UNION ids not unique.")

    # M implies finite evidence; A implies M
    for name, E, M, A in [
        ("L", E_l, M_l, A_l),
        ("A", E_a, M_a, A_a),
        ("V", E_v, M_v, A_v),
    ]:
        if not np.isfinite(E[M == 1]).all():
            raise AssertionError(f"Non-finite values in E_{name} where M_{name}==1")
        if not ((A == 0) | (M == 1)).all():
            raise AssertionError(f"A_{name}==1 found where M_{name}==0")

    # FULL-only definitions (for later reports)
    full_M = (M_l == 1) & (M_a == 1) & (M_v == 1)
    full_A = (A_l == 1) & (A_a == 1) & (A_v == 1)

    meta = {
        "dataset": "CMU-MOSEI (local .csd via CMU Multimodal SDK)",
        "row_unit": "opinion segment (aligned)",
        "label_key": LABEL_KEY,
        "sentiment_col": SENTIMENT_COL,
        "binarize_rule": BINARIZE_RULE,
        "modalities": {"L": "glove_vectors", "A": "COVAREP", "V": "FACET 4.2"},
        "dims": {"L": int(DL), "A": int(DA), "V": int(DV)},
        "pooling": "mean over fully-finite overlapping frames",
        "overlap_rule": "frame_end>t0 and frame_start<t1",
        "missing_policy": "no valid finite overlapping frames => E=0, M=0, A=0",
        "availability_policy": "A=1 iff M=1 and at least one fully-finite overlapping frame exists",
        "stats": {
            "N": int(len(ids)),
            "pos_rate": float(y.mean()),
            "presence_rate": {"L": float(M_l.mean()), "A": float(M_a.mean()), "V": float(M_v.mean())},
            "availability_rate": {"L": float(A_l.mean()), "A": float(A_a.mean()), "V": float(A_v.mean())},
            "FULL_only_rate_M": float(full_M.mean()),
            "FULL_only_rate_A": float(full_A.mean()),
        },
        "diagnostics": diag,
    }

    save_kwargs = dict(
        ids=ids,
        groups=groups,
        y=y,
        E_l=E_l,
        E_a=E_a,
        E_v=E_v,
        M_l=M_l,
        M_a=M_a,
        M_v=M_v,
        A_l=A_l,
        A_a=A_a,
        A_v=A_v,
        meta=np.array([json.dumps(meta)], dtype=object),
    )
    if args.keep_intervals:
        save_kwargs["seg_intervals"] = np.asarray(seg_intervals, dtype=np.float32)
    if args.keep_label_features:
        save_kwargs["Y_raw"] = Y_raw

    np.savez(out_path, **save_kwargs)

    print(f"[OK] wrote {out_path}")
    print(f"N={len(ids)} pos_rate={y.mean():.4f}")
    print(f"presence:  L={M_l.mean():.4f} A={M_a.mean():.4f} V={M_v.mean():.4f}")
    print(f"avail:     L={A_l.mean():.4f} A={A_a.mean():.4f} V={A_v.mean():.4f}")
    print(f"FULL-only (M): {full_M.mean():.4f}")
    print(f"FULL-only (A): {full_A.mean():.4f}")
    print("missing videos among labeled:", diag["missing_videos_with_labels"])
    print("[OK] invariants passed")


if __name__ == "__main__":
    main()
