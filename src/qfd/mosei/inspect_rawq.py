#!/usr/bin/env python3
"""
inspect_mosei_rawq.py

StressID-grade inspection for MOSEI RAWQ NPZ.

Checks:
1) Schema + shape consistency vs UNION
2) ids alignment
3) Missing-policy invariant: M==0 => RAWQ is NaN (all dims)
4) Coverage on M==1: fraction with both dims finite
5) Basic distribution sanity (non-degenerate variance, quantiles)
6) Cross-check: if UNION present but RAWQ missing -> counts reported (expected small)
7) Optional: correlation between the two RAWQ dims per modality (diagnostic only)
"""

import argparse
import json
import numpy as np


def _load_npz(path: str):
    return np.load(path, allow_pickle=True)


def _require(z, k):
    if k not in z.files:
        raise KeyError(f"Missing key '{k}'. Found keys: {sorted(z.files)}")
    return z[k]


def frac_finite_both(M: np.ndarray, Q: np.ndarray) -> float:
    idx = (M == 1)
    if idx.sum() == 0:
        return 0.0
    return float(np.mean(np.all(np.isfinite(Q[idx]), axis=1)))


def frac_any_finite(M: np.ndarray, Q: np.ndarray) -> float:
    idx = (M == 1)
    if idx.sum() == 0:
        return 0.0
    return float(np.mean(np.any(np.isfinite(Q[idx]), axis=1)))


def missing_policy_ok(M: np.ndarray, Q: np.ndarray) -> bool:
    miss = (M == 0)
    return bool(np.all(np.isnan(Q[miss])))


def summarize_Q(name: str, M: np.ndarray, Q: np.ndarray):
    idx = (M == 1) & np.all(np.isfinite(Q), axis=1)
    n = int(idx.sum())
    print(f"\n--- {name} ---")
    print("M==1 count:", int((M == 1).sum()))
    print("both-dims finite count:", n)
    if n == 0:
        print("[WARN] no finite rows to summarize.")
        return

    q0 = Q[idx, 0].astype(float)
    q1 = Q[idx, 1].astype(float)

    def stats(x):
        return {
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "min": float(np.min(x)),
            "p01": float(np.quantile(x, 0.01)),
            "p05": float(np.quantile(x, 0.05)),
            "p50": float(np.quantile(x, 0.50)),
            "p95": float(np.quantile(x, 0.95)),
            "p99": float(np.quantile(x, 0.99)),
            "max": float(np.max(x)),
        }

    s0 = stats(q0)
    s1 = stats(q1)

    print("q0 stats:", s0)
    print("q1 stats:", s1)

    # Non-degeneracy check (variance > 0)
    if np.std(q0) == 0.0:
        print("[FAIL] q0 appears degenerate (std=0).")
    if np.std(q1) == 0.0:
        print("[FAIL] q1 appears degenerate (std=0).")

    # Correlation (diagnostic only)
    c = np.corrcoef(q0, q1)[0, 1]
    print("corr(q0,q1):", float(c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union_npz", type=str, required=True)
    ap.add_argument("--rawq_npz", type=str, required=True)
    ap.add_argument("--max_print", type=int, default=5, help="how many example mismatches to print")
    args = ap.parse_args()

    u = _load_npz(args.union_npz)
    r = _load_npz(args.rawq_npz)

    print("===== LOAD =====")
    u_ids = _require(u, "ids").astype(str)
    r_ids = _require(r, "ids").astype(str)

    N = len(u_ids)
    print("UNION N:", N)
    print("RAWQ N :", len(r_ids))

    if len(r_ids) != N:
        raise AssertionError("RAWQ length != UNION length (must be 1:1).")

    # Alignment check
    mism = np.where(u_ids != r_ids)[0]
    if len(mism) > 0:
        print(f"[FAIL] ids mismatch count: {len(mism)}")
        for j in mism[: args.max_print]:
            print(" idx", j, "union:", u_ids[j], "rawq:", r_ids[j])
        raise AssertionError("RAWQ ids order does not match UNION ids order.")
    print("[OK] ids alignment")

    # Keys
    print("\n===== SCHEMA =====")
    needed = ["M_l", "M_a", "M_v", "Q_l_raw", "Q_a_raw", "Q_v_raw", "seg_intervals"]
    for k in needed:
        _require(r, k)
        print("OK:", k)

    # Shapes
    Ml = _require(r, "M_l").astype(np.int8).reshape(-1)
    Ma = _require(r, "M_a").astype(np.int8).reshape(-1)
    Mv = _require(r, "M_v").astype(np.int8).reshape(-1)

    Ql = _require(r, "Q_l_raw").astype(np.float32)
    Qa = _require(r, "Q_a_raw").astype(np.float32)
    Qv = _require(r, "Q_v_raw").astype(np.float32)

    assert Ml.shape == (N,)
    assert Ma.shape == (N,)
    assert Mv.shape == (N,)
    assert Ql.shape == (N, 2)
    assert Qa.shape == (N, 2)
    assert Qv.shape == (N, 2)

    print("\n===== M CONSISTENCY (RAWQ vs UNION) =====")
    u_Ml = _require(u, "M_l").astype(np.int8).reshape(-1)
    u_Ma = _require(u, "M_a").astype(np.int8).reshape(-1)
    u_Mv = _require(u, "M_v").astype(np.int8).reshape(-1)

    for name, Mu, Mr in [("L", u_Ml, Ml), ("A", u_Ma, Ma), ("V", u_Mv, Mv)]:
        diff = int(np.sum(Mu != Mr))
        print(f"{name} M diff count:", diff)
        if diff != 0:
            raise AssertionError(f"M_{name} mismatch between UNION and RAWQ (should be identical).")

    print("[OK] M masks identical to UNION")

    print("\n===== MISSING-POLICY INVARIANT =====")
    for name, M, Q in [("L", Ml, Ql), ("A", Ma, Qa), ("V", Mv, Qv)]:
        ok = missing_policy_ok(M, Q)
        print(f"{name}: M==0 => Q is all-NaN:", ok)
        if not ok:
            bad = int(np.sum(~np.isnan(Q[M == 0])))
            raise AssertionError(f"{name}: found {bad} non-NaN RAWQ entries where M==0")

    print("\n===== COVERAGE (AMONG M==1) =====")
    for name, M, Q in [("L", Ml, Ql), ("A", Ma, Qa), ("V", Mv, Qv)]:
        both = frac_finite_both(M, Q)
        anyf = frac_any_finite(M, Q)
        print(f"{name}: both-dims finite={both:.4f} | any-dim finite={anyf:.4f}")

    print("\n===== DISTRIBUTION SANITY =====")
    summarize_Q("LANG", Ml, Ql)
    summarize_Q("AUDIO", Ma, Qa)
    summarize_Q("VISUAL", Mv, Qv)

    print("\n===== META =====")
    if "meta" in r.files:
        try:
            meta = json.loads(r["meta"][0])
            print("rawq_version:", meta.get("rawq_version"))
            print("video_id_source:", meta.get("video_id_source"))
            print("coverage_bothdims_finite:", meta.get("diagnostics", {}).get("coverage_bothdims_finite"))
        except Exception:
            print("[WARN] Could not parse meta JSON.")
    else:
        print("[WARN] No meta field found in RAWQ NPZ.")

    print("\nALL RAWQ CHECKS PASSED.")


if __name__ == "__main__":
    main()



