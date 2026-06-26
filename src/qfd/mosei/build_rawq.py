#!/usr/bin/env python3
"""
build_mosei_rawq.py — StressID-grade MOSEI RAWQ builder (UNION-aligned)

RAWQ computed ONCE globally, aligned 1:1 to UNION row order.

Outputs:
  ids, video_ids, seg_intervals, M_*, Q_*_raw (NaN when missing/unusable), frame/token counts, meta.

Hard rigor constraints:
- UNION row order is canonical.
- RAWQ never computed per fold.
- Missing modality rows (M==0) => RAWQ must remain NaN (never 0).
- If M==1 but we cannot retrieve stream frames/tokens, RAWQ remains NaN and is counted.
- Coverage checks prevent silent all-NaN failure.

RAWQ definitions (2D per modality):

Audio (COVAREP 74D):
  Qa0 = usable_frame_fraction  (fraction of frames with all dims finite)
  Qa1 = stability_quality      (1/(1+median_d MAD_t(x_t,d)))

Visual (Facet42 35D):
  Qv0 = stability_quality      (1/(1+median_d MAD_t(x_t,d)))
  Qv1 = smoothness_quality     (1/(1+median_t ||x_{t+1}-x_t||_2))

Language (TimestampedWordVectors 300D):
  Ql0 = token_density          (n_tokens / duration_sec)
  Ql1 = coherence              (median cosine similarity between adjacent token vectors)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


# ----------------------------
# Robust statistics
# ----------------------------
def robust_mad(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def median_cosine_adjacent(v: np.ndarray, eps: float = 1e-12) -> float:
    if v.shape[0] < 2:
        return np.nan
    a = v[:-1]
    b = v[1:]
    na = np.linalg.norm(a, axis=1) + eps
    nb = np.linalg.norm(b, axis=1) + eps
    cos = np.sum(a * b, axis=1) / (na * nb)
    cos = np.clip(cos, -1.0, 1.0)
    return float(np.median(cos))


def overlap_indices(intervals: np.ndarray, start: float, end: float) -> np.ndarray:
    if intervals.size == 0:
        return np.array([], dtype=np.int64)
    t0 = intervals[:, 0]
    t1 = intervals[:, 1]
    return np.where((t1 > start) & (t0 < end))[0]


def usable_frame_fraction(feat: np.ndarray) -> float:
    if feat.size == 0:
        return np.nan
    good = np.all(np.isfinite(feat), axis=1)
    return float(np.mean(good))


def stability_quality(feat: np.ndarray, min_frames: int = 4) -> float:
    if feat.size == 0:
        return np.nan
    good = np.all(np.isfinite(feat), axis=1)
    x = feat[good]
    if x.shape[0] < min_frames:
        return np.nan
    mads = np.empty((x.shape[1],), dtype=float)
    for d in range(x.shape[1]):
        mads[d] = robust_mad(x[:, d])
    disp = float(np.median(mads))
    return float(1.0 / (1.0 + disp))


def smoothness_quality(feat: np.ndarray, min_frames: int = 4) -> float:
    if feat.size == 0:
        return np.nan
    good = np.all(np.isfinite(feat), axis=1)
    x = feat[good]
    if x.shape[0] < min_frames:
        return np.nan
    d = x[1:] - x[:-1]
    mag = np.linalg.norm(d, axis=1)
    speed = float(np.median(mag))
    return float(1.0 / (1.0 + speed))


def safe_duration(start: float, end: float) -> float:
    return max(float(end) - float(start), 1e-6)


# ----------------------------
# CSD loading (SDK)
# ----------------------------
def load_csd(csd_path: Path):
    try:
        from mmsdk import mmdatasdk  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: CMU Multimodal SDK.\n"
            "Install in this env:\n"
            "  pip install cmu-multimodal-sdk\n"
            "or:\n"
            "  pip install git+https://github.com/A2Zadeh/CMU-MultimodalSDK.git\n"
        ) from e

    if not csd_path.exists():
        raise FileNotFoundError(f"Missing CSD: {csd_path}")
    return mmdatasdk.computational_sequence(str(csd_path))


def get_stream(seq, vid: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if vid not in seq.data:
        return None
    entry = seq.data[vid]
    intervals = np.asarray(entry["intervals"], dtype=float)
    features = np.asarray(entry["features"], dtype=float)
    return intervals, features


def parse_video_id_from_union_id(seg_id: str) -> str:
    # expects "video_id[j]" from UNION builder
    # robust: take substring before first '['
    p = seg_id.split("[", 1)[0]
    return p


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union_npz", type=str, required=True)
    ap.add_argument("--csd_root", type=str, required=True)
    ap.add_argument("--out_npz", type=str, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--progress_every", type=int, default=5000)
    ap.add_argument("--min_cover_frac", type=float, default=0.90,
                    help="Require at least this fraction of M==1 rows to have finite RAWQ (per modality).")
    ap.add_argument("--allow_low_coverage", action="store_true",
                    help="Do not hard-fail if coverage is low; still print diagnostics.")
    args = ap.parse_args()

    union_path = Path(args.union_npz)
    csd_root = Path(args.csd_root)
    out_path = Path(args.out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Load UNION (locked keys; no ambiguity)
    z = np.load(union_path, allow_pickle=True)
    required = ["ids", "groups", "seg_intervals", "M_l", "M_a", "M_v"]
    missing = [k for k in required if k not in z.files]
    if missing:
        raise KeyError(f"UNION missing keys {missing}. Found keys: {sorted(z.files)}")

    ids = z["ids"].astype(str)
    groups = z["groups"].astype(str)
    seg_intervals = np.asarray(z["seg_intervals"], dtype=float)

    Ml = np.asarray(z["M_l"], dtype=np.int8).reshape(-1)
    Ma = np.asarray(z["M_a"], dtype=np.int8).reshape(-1)
    Mv = np.asarray(z["M_v"], dtype=np.int8).reshape(-1)

    N = len(ids)
    if not (len(groups) == N and seg_intervals.shape == (N, 2) and len(Ml) == N and len(Ma) == N and len(Mv) == N):
        raise ValueError("UNION array length/shape mismatch.")

    N_use = min(N, args.limit) if args.limit and args.limit > 0 else N

    # ---- Load CSD streams
    lang_seq = load_csd(csd_root / "CMU_MOSEI_TimestampedWordVectors.csd")
    aud_seq  = load_csd(csd_root / "CMU_MOSEI_COVAREP.csd")
    vis_seq  = load_csd(csd_root / "CMU_MOSEI_VisualFacet42.csd")

    # Determine whether UNION groups are video ids or not
    # If most groups exist in stream keys, treat as video ids; otherwise parse from ids.
    stream_keys = set(lang_seq.data.keys())
    grp_hit = np.mean([g in stream_keys for g in groups[: min(5000, N)]])
    use_groups_as_vid = grp_hit > 0.95

    # Compute per-row video_id for stream lookup
    if use_groups_as_vid:
        video_ids = groups.copy()
    else:
        video_ids = np.array([parse_video_id_from_union_id(s) for s in ids], dtype=object)

    # ---- Allocate RAWQ (NaN by default)
    Ql_raw = np.full((N, 2), np.nan, dtype=np.float32)
    Qa_raw = np.full((N, 2), np.nan, dtype=np.float32)
    Qv_raw = np.full((N, 2), np.nan, dtype=np.float32)

    # Frame/token counts (diagnostics; -1 means not attempted)
    n_tok = np.full((N,), -1, dtype=np.int32)
    n_af  = np.full((N,), -1, dtype=np.int32)
    n_vf  = np.full((N,), -1, dtype=np.int32)

    # Cache streams per video
    cache_lang: Dict[str, Optional[Tuple[np.ndarray, np.ndarray]]] = {}
    cache_aud: Dict[str, Optional[Tuple[np.ndarray, np.ndarray]]] = {}
    cache_vis: Dict[str, Optional[Tuple[np.ndarray, np.ndarray]]] = {}

    miss_stream_counts = {"L": 0, "A": 0, "V": 0}
    no_overlap_counts = {"L": 0, "A": 0, "V": 0}

    for i in range(N_use):
        vid = str(video_ids[i])
        start, end = float(seg_intervals[i, 0]), float(seg_intervals[i, 1])
        dur = safe_duration(start, end)

        # --- Language
        if Ml[i] == 1:
            if vid not in cache_lang:
                cache_lang[vid] = get_stream(lang_seq, vid)
            stream = cache_lang[vid]
            if stream is None:
                miss_stream_counts["L"] += 1
            else:
                intervals, feats = stream
                idx = overlap_indices(intervals, start, end)
                if idx.size == 0:
                    n_tok[i] = 0
                    no_overlap_counts["L"] += 1
                else:
                    F = feats[idx]
                    good = np.all(np.isfinite(F), axis=1)
                    F = F[good]
                    K = int(F.shape[0])
                    n_tok[i] = K
                    if K > 0:
                        Ql_raw[i, 0] = np.float32(K / dur)
                        Ql_raw[i, 1] = np.float32(median_cosine_adjacent(F))

        # --- Audio
        if Ma[i] == 1:
            if vid not in cache_aud:
                cache_aud[vid] = get_stream(aud_seq, vid)
            stream = cache_aud[vid]
            if stream is None:
                miss_stream_counts["A"] += 1
            else:
                intervals, feats = stream
                idx = overlap_indices(intervals, start, end)
                if idx.size == 0:
                    n_af[i] = 0
                    no_overlap_counts["A"] += 1
                else:
                    F = feats[idx]
                    n_af[i] = int(F.shape[0])
                    Qa_raw[i, 0] = np.float32(usable_frame_fraction(F))
                    Qa_raw[i, 1] = np.float32(stability_quality(F))

        # --- Visual
        if Mv[i] == 1:
            if vid not in cache_vis:
                cache_vis[vid] = get_stream(vis_seq, vid)
            stream = cache_vis[vid]
            if stream is None:
                miss_stream_counts["V"] += 1
            else:
                intervals, feats = stream
                idx = overlap_indices(intervals, start, end)
                if idx.size == 0:
                    n_vf[i] = 0
                    no_overlap_counts["V"] += 1
                else:
                    F = feats[idx]
                    n_vf[i] = int(F.shape[0])
                    Qv_raw[i, 0] = np.float32(stability_quality(F))
                    Qv_raw[i, 1] = np.float32(smoothness_quality(F))

        if args.progress_every and (i + 1) % args.progress_every == 0:
            print(f"[RAWQ] processed {i+1}/{N_use}")

    # ---- Hard invariant: M==0 => RAWQ must be NaN
    def assert_missing_nan(M: np.ndarray, Q: np.ndarray, name: str):
        miss = (M == 0)
        if np.any(~np.isnan(Q[miss])):
            bad = int(np.sum(~np.isnan(Q[miss])))
            raise AssertionError(f"{name}: found {bad} non-NaN RAWQ entries where M==0.")

    assert_missing_nan(Ml, Ql_raw, "Q_l_raw")
    assert_missing_nan(Ma, Qa_raw, "Q_a_raw")
    assert_missing_nan(Mv, Qv_raw, "Q_v_raw")

    # ---- Coverage checks to prevent silent failure
    def cover_frac(M: np.ndarray, Q: np.ndarray) -> float:
        idx = (M == 1)
        if idx.sum() == 0:
            return 0.0
        return float(np.mean(np.all(np.isfinite(Q[idx]), axis=1)))

    cov_L = cover_frac(Ml[:N_use], Ql_raw[:N_use])
    cov_A = cover_frac(Ma[:N_use], Qa_raw[:N_use])
    cov_V = cover_frac(Mv[:N_use], Qv_raw[:N_use])

    if not args.allow_low_coverage:
        if cov_L < args.min_cover_frac:
            raise AssertionError(f"LOW COVERAGE: L cov={cov_L:.4f} (<{args.min_cover_frac}). Likely wrong video_id mapping.")
        if cov_A < args.min_cover_frac:
            raise AssertionError(f"LOW COVERAGE: A cov={cov_A:.4f} (<{args.min_cover_frac}). Likely wrong video_id mapping.")
        if cov_V < args.min_cover_frac:
            raise AssertionError(f"LOW COVERAGE: V cov={cov_V:.4f} (<{args.min_cover_frac}). Likely wrong video_id mapping.")

    meta = {
        "rawq_version": "mosei_rawq_v1_locked",
        "union_path": str(union_path),
        "csd_root": str(csd_root),
        "video_id_source": "groups" if use_groups_as_vid else "parsed_from_ids",
        "definitions": {
            "audio": ["usable_frame_fraction", "stability_quality(1/(1+median MAD))"],
            "visual": ["stability_quality(1/(1+median MAD))", "smoothness_quality(1/(1+median ||Δx||))"],
            "language": ["token_density(n_tokens/duration)", "coherence(median adjacent cosine)"],
        },
        "missing_policy": "M==0 => RAWQ NaN; fold-scaling later maps missing/non-finite to 0",
        "diagnostics": {
            "coverage_bothdims_finite": {"L": cov_L, "A": cov_A, "V": cov_V},
            "missing_stream_counts": miss_stream_counts,
            "no_overlap_counts": no_overlap_counts,
            "group_hit_rate_in_stream_keys": float(grp_hit),
        },
    }

    np.savez_compressed(
        out_path,
        ids=ids,
        video_ids=video_ids.astype(object),
        seg_intervals=seg_intervals.astype(np.float32),
        M_l=Ml, M_a=Ma, M_v=Mv,
        Q_l_raw=Ql_raw,
        Q_a_raw=Qa_raw,
        Q_v_raw=Qv_raw,
        n_tokens=n_tok,
        n_audio_frames=n_af,
        n_visual_frames=n_vf,
        meta=np.array([json.dumps(meta)], dtype=object),
    )

    print("=== MOSEI RAWQ BUILT (LOCKED) ===")
    print(f"Saved: {out_path}")
    print(f"N rows used: {N_use} / {N}")
    print(f"video_id_source: {'groups' if use_groups_as_vid else 'parsed_from_ids'} (group_hit_rate={grp_hit:.4f})")
    print("Coverage (both dims finite among M==1):")
    print(f"  L: {cov_L:.4f}")
    print(f"  A: {cov_A:.4f}")
    print(f"  V: {cov_V:.4f}")
    print("Missing streams:", miss_stream_counts)
    print("No-overlap counts:", no_overlap_counts)


if __name__ == "__main__":
    main()



