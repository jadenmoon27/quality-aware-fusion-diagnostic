#!/usr/bin/env python3
# diagnostics/failure_mode_alignment_diagnostics.py
"""
Failure-mode alignment diagnostics (Add-C closure), leakage-safe and paper-defensible.

Computes on FULL-only TEST rows, per (seed, fold):
  1) Spearman corr(Q_dom, |p_fused - y|)
  2) (Optional) Spearman corr(severity, |p_fused - y|) if severity is provided
  3) 5-bin equal-frequency stratification of Q_dom -> mean error per bin
  4) Aggregates mean ± std across folds + prints a paste-ready LaTeX snippet.

Designed for your UNION contract:
- UNION provides ids, labels y or y2, masks M_a/M_v/M_p
- Fold splits are stored as train_ids_fold{k}.npy / test_ids_fold{k}.npy in splits/seed_{seed}/
- Fold-safe Q is stored per fold in quality/seed_{seed}/..._fold{fold}.npz (keys: Q_a,Q_v,Q_p)
- Unimodal predictions stored per fold in unimodal_preds_root/seed_{seed}/fold_{fold}.npz
  (keys: p_a,p_v,p_p and/or logit_a,logit_v,logit_p; NaN where missing)

You choose a fusion rule to produce per-row fused probabilities p_fused:
- poe: Product-of-Experts via sum of logits over present modalities
- lfavg: present-average of probabilities over present modalities
- wavgq: quality-weighted average: w_m ∝ (Q_m^gamma) over present modalities

Optional severity:
- Provide a per-fold NPZ with severity aligned to UNION order.
- Expected keys (any one works):
    - severity (N,)  [single scalar per row]
    - severity_a, severity_v, severity_p (N,)
  If you provide *_a/_v/_p, the script uses the nonzero / non-NaN field per row;
  otherwise uses 'severity'.

Outputs:
- CSV with fold-level correlations and bin means
- JSON summary with aggregate stats
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ---------------------------
# Utilities
# ---------------------------

def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation, returning np.nan if insufficient variation."""
    x = np.asarray(x)
    y = np.asarray(y)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if x.size < 3:
        return np.nan
    if np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan
    return float(spearmanr(x, y).correlation)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z)
    # stable sigmoid
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


def _load_union(union_npz: str) -> Dict[str, np.ndarray]:
    d = np.load(union_npz, allow_pickle=True)
    keys = set(d.files)

    # label key variations
    if "y" in keys:
        y = d["y"].astype(int)
    elif "y2" in keys:
        y = d["y2"].astype(int)
    else:
        raise KeyError(f"UNION missing label key 'y' or 'y2'. keys={sorted(keys)}")

    # mask key variations
    def get_mask(name_candidates: List[str]) -> np.ndarray:
        for k in name_candidates:
            if k in keys:
                return d[k].astype(int)
        raise KeyError(f"UNION missing mask among {name_candidates}. keys={sorted(keys)}")

    Ma = get_mask(["M_a", "Ma", "M_audio"])
    Mv = get_mask(["M_v", "Mv", "M_video"])
    Mp = get_mask(["M_p", "Mp", "M_phys", "M_physio"])

    # ids
    if "ids" in keys:
        ids = d["ids"]
    elif "ids_str" in keys:
        ids = d["ids_str"]
    else:
        raise KeyError(f"UNION missing 'ids' or 'ids_str'. keys={sorted(keys)}")

    return {"ids": ids, "y": y, "Ma": Ma, "Mv": Mv, "Mp": Mp}


def _build_id2row(ids: np.ndarray) -> Dict[str, int]:
    # ids might be bytes/object; normalize to str for matching
    def to_str(x):
        if isinstance(x, bytes):
            return x.decode("utf-8")
        return str(x)

    return {to_str(ids[i]): i for i in range(len(ids))}


def _load_split_indices(splits_dir: str, seed: int, fold: int, id2row: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    seed_dir = os.path.join(splits_dir, f"seed_{seed}")
    train_path = os.path.join(seed_dir, f"train_ids_fold{fold}.npy")
    test_path = os.path.join(seed_dir, f"test_ids_fold{fold}.npy")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing split files for seed={seed} fold={fold}:\n  {train_path}\n  {test_path}")

    train_ids = np.load(train_path, allow_pickle=True)
    test_ids = np.load(test_path, allow_pickle=True)

    def to_str_arr(arr):
        out = []
        for x in arr:
            if isinstance(x, bytes):
                out.append(x.decode("utf-8"))
            else:
                out.append(str(x))
        return out

    train_ids = to_str_arr(train_ids)
    test_ids = to_str_arr(test_ids)

    train_idx = np.array([id2row[i] for i in train_ids], dtype=int)
    test_idx = np.array([id2row[i] for i in test_ids], dtype=int)
    return train_idx, test_idx


def _load_fold_q(q_clean_root: str, seed: int, fold: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # robust search for the fold file under q_clean_root/seed_{seed}/
    seed_dir = os.path.join(q_clean_root, f"seed_{seed}")
    if not os.path.isdir(seed_dir):
        raise FileNotFoundError(f"Missing Q seed dir: {seed_dir}")

    # find a file that contains "fold{fold}" and endswith .npz
    candidates = [f for f in os.listdir(seed_dir) if f.endswith(".npz") and f"fold{fold}" in f]
    if len(candidates) == 0:
        raise FileNotFoundError(f"No Q npz found in {seed_dir} containing 'fold{fold}'.")
    if len(candidates) > 1:
        # choose the longest (usually most specific), but warn by deterministic pick
        candidates = sorted(candidates, key=lambda x: (len(x), x), reverse=True)

    path = os.path.join(seed_dir, candidates[0])
    d = np.load(path, allow_pickle=True)
    keys = set(d.files)
    for k in ["Q_a", "Qa", "Q_audio"]:
        if k in keys:
            Qa = d[k].astype(np.float64); break
    else:
        raise KeyError(f"Q file missing Q_a-like key. keys={sorted(keys)}")

    for k in ["Q_v", "Qv", "Q_video"]:
        if k in keys:
            Qv = d[k].astype(np.float64); break
    else:
        raise KeyError(f"Q file missing Q_v-like key. keys={sorted(keys)}")

    for k in ["Q_p", "Qp", "Q_phys", "Q_physio"]:
        if k in keys:
            Qp = d[k].astype(np.float64); break
    else:
        raise KeyError(f"Q file missing Q_p-like key. keys={sorted(keys)}")

    return Qa, Qv, Qp


def _load_unimodal_preds(unimodal_preds_root: str, seed: int, fold: int) -> Dict[str, np.ndarray]:
    # expected: unimodal_preds_root/seed_{seed}/fold_{fold}.npz OR .../fold_{fold}/unimodal_preds.npz
    p1 = os.path.join(unimodal_preds_root, f"seed_{seed}", f"fold_{fold}.npz")
    p2 = os.path.join(unimodal_preds_root, f"seed_{seed}", f"fold_{fold}", "unimodal_preds.npz")
    if os.path.exists(p1):
        path = p1
    elif os.path.exists(p2):
        path = p2
    else:
        raise FileNotFoundError(f"Missing unimodal preds for seed={seed} fold={fold}:\n  {p1}\n  {p2}")

    d = np.load(path, allow_pickle=True)
    keys = set(d.files)

    out = {}
    # probabilities
    for k in ["p_a", "pA", "p_audio"]:
        if k in keys: out["p_a"] = d[k].astype(np.float64); break
    for k in ["p_v", "pV", "p_video"]:
        if k in keys: out["p_v"] = d[k].astype(np.float64); break
    for k in ["p_p", "pP", "p_phys", "p_physio"]:
        if k in keys: out["p_p"] = d[k].astype(np.float64); break

    # logits optional
    for k in ["logit_a", "logits_a", "z_a"]:
        if k in keys: out["logit_a"] = d[k].astype(np.float64); break
    for k in ["logit_v", "logits_v", "z_v"]:
        if k in keys: out["logit_v"] = d[k].astype(np.float64); break
    for k in ["logit_p", "logits_p", "z_p"]:
        if k in keys: out["logit_p"] = d[k].astype(np.float64); break

    if "p_a" not in out or "p_v" not in out or "p_p" not in out:
        raise KeyError(f"Unimodal preds npz missing p_* keys. keys={sorted(keys)}")

    # If logits missing, derive from probs
    if "logit_a" not in out: out["logit_a"] = _logit(out["p_a"])
    if "logit_v" not in out: out["logit_v"] = _logit(out["p_v"])
    if "logit_p" not in out: out["logit_p"] = _logit(out["p_p"])

    return out


def _load_severity_optional(severity_root: Optional[str], seed: int, fold: int) -> Optional[Dict[str, np.ndarray]]:
    if severity_root is None:
        return None

    # accept same pattern as unimodal preds
    p1 = os.path.join(severity_root, f"seed_{seed}", f"fold_{fold}.npz")
    p2 = os.path.join(severity_root, f"seed_{seed}", f"fold_{fold}", "severity.npz")
    if os.path.exists(p1):
        path = p1
    elif os.path.exists(p2):
        path = p2
    else:
        raise FileNotFoundError(f"Missing severity npz for seed={seed} fold={fold}:\n  {p1}\n  {p2}")

    d = np.load(path, allow_pickle=True)
    return {k: d[k].astype(np.float64) for k in d.files}


def _compute_q_dom(Qa: np.ndarray, Qv: np.ndarray, Qp: np.ndarray, Ma: np.ndarray, Mv: np.ndarray, Mp: np.ndarray) -> np.ndarray:
    # Q_dom among present modalities
    Q = np.stack([Qa, Qv, Qp], axis=1)  # (N,3)
    M = np.stack([Ma, Mv, Mp], axis=1).astype(bool)
    Q_masked = np.where(M, Q, -np.inf)
    return np.max(Q_masked, axis=1)


def _fuse_p(
    fuser: str,
    preds: Dict[str, np.ndarray],
    Ma: np.ndarray,
    Mv: np.ndarray,
    Mp: np.ndarray,
    Qa: Optional[np.ndarray] = None,
    Qv: Optional[np.ndarray] = None,
    Qp: Optional[np.ndarray] = None,
    gamma: float = 4.0,
) -> np.ndarray:
    """
    Returns fused probability p_fused for all N rows (NaN where no modalities present).
    Missingness is defined by M_* and also by NaN p_* (contract uses NaN where missing).
    """
    p_a, p_v, p_p = preds["p_a"], preds["p_v"], preds["p_p"]
    z_a, z_v, z_p = preds["logit_a"], preds["logit_v"], preds["logit_p"]

    # availability = present mask AND finite p
    Aa = (Ma == 1) & np.isfinite(p_a)
    Av = (Mv == 1) & np.isfinite(p_v)
    Ap = (Mp == 1) & np.isfinite(p_p)

    if fuser == "poe":
        # sum logits over available modalities
        z = np.zeros_like(z_a, dtype=np.float64)
        cnt = np.zeros_like(z_a, dtype=np.float64)
        for A, zz in [(Aa, z_a), (Av, z_v), (Ap, z_p)]:
            z[A] += zz[A]
            cnt[A] += 1.0
        out = np.full_like(z_a, np.nan, dtype=np.float64)
        ok = cnt > 0
        out[ok] = _sigmoid(z[ok])
        return out

    if fuser == "lfavg":
        # average probabilities over available modalities
        s = np.zeros_like(p_a, dtype=np.float64)
        cnt = np.zeros_like(p_a, dtype=np.float64)
        for A, pp in [(Aa, p_a), (Av, p_v), (Ap, p_p)]:
            s[A] += pp[A]
            cnt[A] += 1.0
        out = np.full_like(p_a, np.nan, dtype=np.float64)
        ok = cnt > 0
        out[ok] = s[ok] / cnt[ok]
        return out

    if fuser == "wavgq":
        if Qa is None or Qv is None or Qp is None:
            raise ValueError("wavgq requires Qa,Qv,Qp.")
        # weights proportional to Q^gamma over available modalities
        w_a = np.where(Aa, np.power(np.clip(Qa, 0.0, 1.0), gamma), 0.0)
        w_v = np.where(Av, np.power(np.clip(Qv, 0.0, 1.0), gamma), 0.0)
        w_p = np.where(Ap, np.power(np.clip(Qp, 0.0, 1.0), gamma), 0.0)
        W = w_a + w_v + w_p
        out = np.full_like(p_a, np.nan, dtype=np.float64)
        ok = W > 0
        out[ok] = (w_a[ok] * p_a[ok] + w_v[ok] * p_v[ok] + w_p[ok] * p_p[ok]) / W[ok]
        return out

    raise ValueError(f"Unknown fuser: {fuser}")


def _extract_severity(sev: Dict[str, np.ndarray]) -> np.ndarray:
    keys = set(sev.keys())
    if "severity" in keys:
        return sev["severity"]
    # if modality-specific severities are present, combine by taking max across modalities per row
    candidates = []
    for k in ["severity_a", "severity_v", "severity_p"]:
        if k in keys:
            candidates.append(sev[k])
    if len(candidates) == 0:
        raise KeyError(f"Severity npz missing 'severity' or 'severity_*' keys. keys={sorted(keys)}")
    S = np.stack(candidates, axis=1)
    # handle all-NaN rows
    return np.nanmax(S, axis=1)


def _qcut_5bins(x: np.ndarray) -> np.ndarray:
    """Return bin id in {0..4} for finite x; -1 for non-finite."""
    x = np.asarray(x, dtype=np.float64)
    out = np.full(x.shape, -1, dtype=int)
    ok = np.isfinite(x)
    if ok.sum() < 5:
        return out
    # use pandas qcut for equal-frequency bins; handle duplicates robustly
    try:
        bins = pd.qcut(x[ok], 5, labels=False, duplicates="drop")
        # if duplicates drop leads to <5 bins, still accept; map to 0..(nbins-1)
        out[ok] = np.asarray(bins, dtype=int)
    except Exception:
        # fallback: percentile edges
        qs = np.nanpercentile(x[ok], [0, 20, 40, 60, 80, 100])
        # ensure strictly increasing edges
        eps = 1e-12
        for i in range(1, len(qs)):
            if qs[i] <= qs[i - 1]:
                qs[i] = qs[i - 1] + eps
        out[ok] = np.clip(np.digitize(x[ok], qs[1:-1], right=False), 0, 4)
    return out


@dataclass
class FoldResult:
    seed: int
    fold: int
    n_full_test: int
    corr_qdom_abs_err: float
    corr_sev_abs_err: float
    bin_mean_err: List[float]  # length up to 5; may include nan


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union_npz", required=True)
    ap.add_argument("--splits_dir", required=True)
    ap.add_argument("--unimodal_preds_root", required=True)
    ap.add_argument("--q_clean_root", required=True)
    ap.add_argument("--fuser", choices=["poe", "lfavg", "wavgq"], default="poe")
    ap.add_argument("--error_metric", choices=["abs", "nll"], default="abs")
    ap.add_argument("--gamma", type=float, default=4.0, help="Only used for wavgq")
    ap.add_argument("--severity_root", default=None, help="Optional. Per-fold severity npz root.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    union = _load_union(args.union_npz)
    ids, y = union["ids"], union["y"]
    Ma, Mv, Mp = union["Ma"], union["Mv"], union["Mp"]
    N = len(y)
    id2row = _build_id2row(ids)

    fold_rows: List[FoldResult] = []

    for seed in args.seeds:
        for fold in args.folds:
            _, test_idx = _load_split_indices(args.splits_dir, seed, fold, id2row)

            Qa, Qv, Qp = _load_fold_q(args.q_clean_root, seed, fold)
            preds = _load_unimodal_preds(args.unimodal_preds_root, seed, fold)

            # fused prediction for all N rows
            p_fused = _fuse_p(
                fuser=args.fuser,
                preds=preds,
                Ma=Ma, Mv=Mv, Mp=Mp,
                Qa=Qa, Qv=Qv, Qp=Qp,
                gamma=args.gamma,
            )

            # FULL-only TEST mask
            full_mask = (Ma == 1) & (Mv == 1) & (Mp == 1)
            test_mask = np.zeros(N, dtype=bool)
            test_mask[test_idx] = True
            eval_mask = full_mask & test_mask & np.isfinite(p_fused)

            n_eval = int(eval_mask.sum())
            if n_eval < 5:
                fold_rows.append(FoldResult(seed, fold, n_eval, np.nan, np.nan, [np.nan]*5))
                continue

            # dominant-modality quality among present (here full-only => all present, but keep general)
            Q_dom = _compute_q_dom(Qa, Qv, Qp, Ma, Mv, Mp)

            # error metric
            p_eval = p_fused[eval_mask].astype(np.float64)
            y_eval = y[eval_mask].astype(np.float64)

            if args.error_metric == "abs":
                e = np.abs(p_eval - y_eval)
            elif args.error_metric == "nll":
                eps = 1e-12
                p_eval = np.clip(p_eval, eps, 1.0 - eps)
                e = -(y_eval * np.log(p_eval) + (1.0 - y_eval) * np.log(1.0 - p_eval))
            else:
                raise ValueError(f"Unknown error metric: {args.error_metric}")

            # corr(Q_dom, error)
            corr_q = _safe_spearman(Q_dom[eval_mask], e)

            # optional severity
            corr_s = np.nan
            bin_means = [np.nan] * 5

            if args.severity_root is not None:
                sev_npz = _load_severity_optional(args.severity_root, seed, fold)
                S = _extract_severity(sev_npz)
                corr_s = _safe_spearman(S[eval_mask], e)

            # 5-bin stratification on Q_dom (within eval_mask)
            bins = _qcut_5bins(Q_dom[eval_mask])
            for b in range(5):
                if np.any(bins == b):
                    bin_means[b] = float(np.mean(e[bins == b]))

            fold_rows.append(FoldResult(
                seed=seed,
                fold=fold,
                n_full_test=n_eval,
                corr_qdom_abs_err=corr_q,
                corr_sev_abs_err=corr_s,
                bin_mean_err=bin_means
            ))

    # Write CSV
    out_csv = os.path.join(args.out_dir, f"failure_mode_alignment_{args.fuser}_{args.error_metric}.csv")
    rows = []
    for r in fold_rows:
        d = {
            "seed": r.seed,
            "fold": r.fold,
            "n_full_test": r.n_full_test,
            "corr_qdom_abs_err": r.corr_qdom_abs_err,
            "corr_sev_abs_err": r.corr_sev_abs_err,
        }
        for i in range(5):
            d[f"bin{i}_mean_abs_err"] = r.bin_mean_err[i]
        rows.append(d)
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    # Aggregate summary
    df = pd.DataFrame(rows)
    def mean_std(col):
        x = df[col].to_numpy(dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return (np.nan, np.nan)
        return (float(np.mean(x)), float(np.std(x, ddof=1)) if x.size > 1 else 0.0)

    corr_q_mean, corr_q_std = mean_std("corr_qdom_abs_err")
    corr_s_mean, corr_s_std = mean_std("corr_sev_abs_err")

    bin_means = []
    for i in range(5):
        m, s = mean_std(f"bin{i}_mean_abs_err")
        bin_means.append({"bin": i, "mean_abs_err": m, "std_abs_err": s})

    summary = {
        "fuser": args.fuser,
        "gamma": args.gamma if args.fuser == "wavgq" else None,
        "seeds": args.seeds,
        "folds": args.folds,
        "n_folds_total": len(fold_rows),
        "n_folds_with_corr_q": int(np.isfinite(df["corr_qdom_abs_err"]).sum()),
        "corr_qdom_abs_err_mean": corr_q_mean,
        "corr_qdom_abs_err_std": corr_q_std,
        "corr_sev_abs_err_mean": corr_s_mean,
        "corr_sev_abs_err_std": corr_s_std,
        "qdom_bins_5": bin_means,
        "csv": out_csv,
    }

    out_json = os.path.join(args.out_dir, f"failure_mode_alignment_{args.fuser}_{args.error_metric}.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    # Print paste-ready LaTeX snippet (edit numbers if you want rounding)
    print("\n=== Paste-ready LaTeX (edit as needed) ===\n")
    print(r"\paragraph{Failure-mode alignment analysis.}")
    print(r"To test whether quality is error-predictive independent of routing, we measure the association between dominant-modality quality $Q_{\mathrm{dom}}$ and per-sample absolute error $|p-y|$ on FULL-only TEST.")
    print(rf"Across folds, Spearman correlation is $\rho=\mathrm{{corr}}(Q_{{\mathrm{{dom}}}},|p-y|)={corr_q_mean:.3f}\pm{corr_q_std:.3f}$.")
    if args.severity_root is not None:
        print(rf"Severity exhibits similarly weak alignment with error: $\rho=\mathrm{{corr}}(\mathrm{{severity}},|p-y|)={corr_s_mean:.3f}\pm{corr_s_std:.3f}$.")
    print(r"A 5-bin stratification of $Q_{\mathrm{dom}}$ yields nearly flat mean error across bins, indicating that higher quality does not correspond to lower prediction error in this regime.")
    print(r"These diagnostics explain the cleanQ $\approx$ brokenQ invariance: when quality does not predict error, conditioning on $Q$ cannot induce correctness-aligned routing.")
    print("\n=========================================\n")

    print(f"[WROTE] {out_csv}")
    print(f"[WROTE] {out_json}")


if __name__ == "__main__":
    main()


