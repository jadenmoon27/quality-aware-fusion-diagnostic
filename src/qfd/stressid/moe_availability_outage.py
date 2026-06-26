#!/usr/bin/env python3
# FINAL_EXPERIMENTS/FINAL/v5_2/04_moe_availability_outage_experiment.py
#
# Leakage-safe, post-training counterfactual outage experiment to test:
#   Does conditioning on availability M improve robustness under outages?
#
# Key design (fixes your earlier confound):
# - Start from FULL-only TEST for counterfactual outages (isolates outage from natural missingness).
# - Train routers on NATURAL-missingness TRAIN (M varies) using contract-aligned preds:
#     * unimodal probs are NaN where missing in UNION
#     * for router training ONLY, we fill missing expert probs with 0.5 (uninformative)
# - Compare two CAPACITY-MATCHED routers:
#     (A) E-only router (M-agnostic): bias-only softmax (fixed weights across instances)
#     (B) E+M router (availability-aware): softmax-linear on M (instance-varying weights)
#
# Both fuse in logit space:
#   l_i = sum_m w_{m,i} * logit(p_{m,i}); p_i = sigmoid(l_i)
#
# Counterfactual outage at inference (FULL-only TEST base):
#   - set dropped modality prob to 0.5 (logit 0)
#   - set M_m = 0
#   - (Q not used in this experiment; keep it out entirely)
#
# Outputs:
#   - JSON/CSV per-fold and aggregate
#   - LaTeX table
# python -m qfd.stressid.moe_availability_outage \
#   --dataset stressid \
#   --union_npz /path/to/project/output/fusion_table_noQ/new_fusion_table_union_y2_avp_v1_noQ.npz \
#   --splits_dir /path/to/project/splits \
#   --preds_root /path/to/project/paper_output/unimodal_preds/lr \
#   --seeds 11 22 33 44 55 \
#   --folds 0 1 2 3 4 \
#   --router_epochs 800 \
#   --router_lr 0.01 \
#   --router_l2 1e-4 \
#   --thresh 0.5 \
#   --out_dir /path/to/project/paper_output/reports/moe_avail_outage_stressid_lr \
#   --write_tex
# python -m qfd.stressid.moe_availability_outage \
#   --dataset mosei \
#   --union_npz /path/to/project/output/final_experiments/mosei/union/mosei_union.npz \
#   --splits_dir /path/to/project/output/final_experiments/mosei/splits_mosei \
#   --preds_root /path/to/project/output/final_experiments/mosei/unimodal_preds/lr \
#   --seeds 11 22 33 44 55 \
#   --folds 0 1 2 3 4 \
#   --router_epochs 800 \
#   --router_lr 0.01 \
#   --router_l2 1e-4 \
#   --thresh 0.5 \
#   --out_dir /path/to/project/paper_output/reports/moe_avail_outage_stressid_lr \
#   --write_tex
# python -m qfd.stressid.moe_availability_outage \
#   --family "MoE avail (M-only gate)" \
#   --union_npz /path/to/project/output/final_experiments/mosei/union/mosei_union.npz \
#   --splits_dir /path/to/project/output/final_experiments/mosei/splits_mosei \
#   --preds_root /path/to/project/output/final_experiments/mosei/unimodal_preds/lr \
#   --seeds 11 22 33 44 55 \
#   --folds 0 1 2 3 4 \
#   --thresh 0.5 \
#   --require_full_coverage \
#   --gate_train_on "train_all" \
#   --eonly_mode "bias_only" \
#   --router_epochs 800 \
#   --router_lr 0.01 \
#   --router_l2 1e-4 \
#   --outage_seed 12345 \
#   --outage_rates 0.1 0.3 0.5 0.7 \
#   --out_dir /path/to/project/output/final_experiments/mosei/reports/moe_avail_outage_lr \
#   --write_tex


from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from qfd._shared.q_contract import (
    load_union,
    load_fold_split,
    make_train_test_masks,
    eval_mask_full_only,
    full_only_mask,
    balanced_accuracy_from_probs,
    assert_probs_nan_where_missing,
)

# -----------------------------
# Utilities
# -----------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _mean_std(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    return float(np.mean(x)), float(np.std(x, ddof=0))

def _fmt_mean_std(m: float, s: float, digits: int = 3) -> str:
    return f"{m:.{digits}f}\\pm{s:.{digits}f}"

def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out

def _softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=1, keepdims=True)

def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)

def _resolve_pred_keys(z: np.lib.npyio.NpzFile, dataset: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    files = set(z.files)

    def pick(keys: List[str], required: bool = True) -> Optional[np.ndarray]:
        for k in keys:
            if k in files:
                return z[k]
        if required:
            raise KeyError(f"Missing required key among {keys}. Found keys={sorted(files)}")
        return None

    y = pick(["y", "y2", "label", "labels"]).astype(int)

    if dataset == "stressid":
        # audio/video/physio expected
        p0 = pick(["p_a", "p_audio", "pa"])
        p1 = pick(["p_v", "p_video", "pv"])
        p2 = pick(["p_p", "p_phys", "pp", "p_physio"])
    elif dataset == "mosei":
        # language/audio/video expected
        p0 = pick(["p_l", "p_lang", "pl"])
        p1 = pick(["p_a", "p_audio", "pa"])
        p2 = pick(["p_v", "p_video", "pv"])
    else:
        raise ValueError(f"Unknown dataset={dataset}")

    return np.asarray(y), np.asarray(p0, dtype=float), np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)

def _load_unimodal_preds_npz(preds_root: Path, seed: int, fold: int, dataset: str, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = preds_root / f"seed_{seed}" / f"fold_{fold}.npz"
    if not p.exists():
        raise FileNotFoundError(f"Missing unimodal preds file: {p}")
    z = np.load(p, allow_pickle=True)
    y, p0, p1, p2 = _resolve_pred_keys(z, dataset=dataset)
    if p0.shape != (n,) or p1.shape != (n,) or p2.shape != (n,):
        raise ValueError(f"Pred arrays must be (N,), got {p0.shape},{p1.shape},{p2.shape} (N={n})")
    return y, p0, p1, p2

def _latex_table_outage(caption: str, rows: List[Tuple[str, Tuple[float,float], Tuple[float,float]]], label: str) -> str:
    s = []
    s.append("\\begin{table}[t]\n\\centering\n")
    s.append(f"\\caption{{{caption}}}\n")
    s.append(f"\\label{{{label}}}\n")
    s.append("\\begin{tabular}{lcc}\n\\toprule\n")
    s.append("Condition & E-only (BA) & E+M (BA) \\\\\n\\midrule\n")
    for name, (mE, sE), (mM, sM) in rows:
        s.append(f"{name} & {_fmt_mean_std(mE, sE)} & {_fmt_mean_std(mM, sM)} \\\\\n")
    s.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    return "".join(s)

# -----------------------------
# Routers
# -----------------------------

class _Router:
    def __init__(self, W: np.ndarray, b: np.ndarray):
        self.W = np.asarray(W, dtype=float)  # (d,3)
        self.b = np.asarray(b, dtype=float)  # (3,)

    def weights(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return _softmax(X @ self.W + self.b[None, :])

def _train_router_adam(
    X: np.ndarray,     # (n,d)
    L: np.ndarray,     # (n,3) expert logits (filled; no NaNs)
    y: np.ndarray,     # (n,) {0,1}
    *,
    seed: int,
    epochs: int,
    lr: float,
    l2: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> _Router:
    X = np.asarray(X, dtype=float)
    L = np.asarray(L, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    n, d = X.shape
    rng = np.random.RandomState(int(seed))
    W = 0.01 * rng.randn(d, 3)
    b = np.zeros(3, dtype=float)

    mW = np.zeros_like(W); vW = np.zeros_like(W)
    mb = np.zeros_like(b); vb = np.zeros_like(b)
    t = 0

    for _ in range(int(epochs)):
        t += 1
        z = X @ W + b[None, :]     # (n,3)
        w = _softmax(z)            # (n,3)
        f = np.sum(w * L, axis=1)  # (n,)
        p = _sigmoid(f)            # (n,)

        g_f = (p - y)              # (n,)

        # df/dz_j = w_j * (L_j - f)
        g_z = (g_f[:, None]) * (w * (L - f[:, None]))  # (n,3)

        gW = (X.T @ g_z) / n + 2.0 * l2 * W
        gb = np.sum(g_z, axis=0) / n

        mW = beta1 * mW + (1 - beta1) * gW
        vW = beta2 * vW + (1 - beta2) * (gW * gW)
        mb = beta1 * mb + (1 - beta1) * gb
        vb = beta2 * vb + (1 - beta2) * (gb * gb)

        mW_hat = mW / (1 - beta1 ** t)
        vW_hat = vW / (1 - beta2 ** t)
        mb_hat = mb / (1 - beta1 ** t)
        vb_hat = vb / (1 - beta2 ** t)

        W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
        b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

    return _Router(W=W, b=b)

def _moe_fuse_probs(P: np.ndarray, X_gate: np.ndarray, router: _Router, eps_logit: float) -> np.ndarray:
    P = np.asarray(P, dtype=float)         # (n,3) probs (filled; no NaNs)
    X_gate = np.asarray(X_gate, dtype=float)
    L = _logit(P, eps=eps_logit)           # (n,3)
    w = router.weights(X_gate)             # (n,3)
    fused = np.sum(w * L, axis=1)          # (n,)
    return _sigmoid(fused)

# -----------------------------
# Counterfactual outages
# -----------------------------

def _apply_outage(
    P_full: np.ndarray,     # (n,3) probs (all finite)
    M_full: np.ndarray,     # (n,3) all ones
    condition: str,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (P_cf, M_cf) for FULL-only base rows.
    Missing modality => set p=0.5 (uninformative), M=0.
    """
    P = np.array(P_full, dtype=float, copy=True)
    M = np.array(M_full, dtype=float, copy=True)

    n = P.shape[0]
    assert P.shape == (n, 3) and M.shape == (n, 3)

    def drop(mod: int) -> None:
        P[:, mod] = 0.5
        M[:, mod] = 0.0

    def keep_only(mod: int) -> None:
        for j in range(3):
            if j != mod:
                drop(j)

    if condition == "clean":
        return P, M

    if condition == "drop_0":
        drop(0); return P, M
    if condition == "drop_1":
        drop(1); return P, M
    if condition == "drop_2":
        drop(2); return P, M

    if condition == "keep_0":
        keep_only(0); return P, M
    if condition == "keep_1":
        keep_only(1); return P, M
    if condition == "keep_2":
        keep_only(2); return P, M

    if condition.startswith("rand_r"):
        r = float(condition.split("rand_r", 1)[1])
        # independent drops per modality; ensure at least one remains
        for i in range(n):
            mask = (rng.rand(3) >= r).astype(float)  # 1=keep
            if mask.sum() == 0:
                mask[int(rng.randint(0, 3))] = 1.0
            for j in range(3):
                if mask[j] == 0.0:
                    P[i, j] = 0.5
                    M[i, j] = 0.0
        return P, M

    raise ValueError(f"Unknown condition={condition}")

# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", type=str, required=True, choices=["stressid", "mosei"])
    ap.add_argument("--union_npz", type=str, required=True)
    ap.add_argument("--splits_dir", type=str, required=True)
    ap.add_argument("--preds_root", type=str, required=True)

    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--folds", type=int, nargs="+", required=True)

    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--eps_logit", type=float, default=1e-6)

    ap.add_argument("--router_epochs", type=int, default=800)
    ap.add_argument("--router_lr", type=float, default=0.01)
    ap.add_argument("--router_l2", type=float, default=1e-4)

    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--write_tex", action="store_true")

    # Optional: enforce union split creation requires full coverage? usually off here (we want natural missingness in TRAIN)
    ap.add_argument("--require_full_coverage", action="store_true")

    return ap.parse_args()

def main() -> None:
    args = parse_args()

    union = load_union(args.union_npz)
    N = len(union.ids_str)

    # modality masks in UNION
    if args.dataset == "stressid":
        # order: audio, video, physio
        M_all = np.stack([union.Ma, union.Mv, union.Mp], axis=1).astype(float)
        names = ["a", "v", "p"]
    else:
        # order: language, audio, video
        M_all = np.stack([union.Ml, union.Ma, union.Mv], axis=1).astype(float)
        names = ["l", "a", "v"]

    preds_root = Path(args.preds_root)
    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)
    router_dir = out_dir / "routers"
    _ensure_dir(router_dir)

    # Conditions
    conds = [
        "clean",
        "drop_0", "keep_0",
        "drop_1", "keep_1",
        "drop_2", "keep_2",
        "rand_r0.1", "rand_r0.3", "rand_r0.5", "rand_r0.7",
    ]

    # storage: condition -> list over folds
    scores_E: Dict[str, List[float]] = {c: [] for c in conds}
    scores_EM: Dict[str, List[float]] = {c: [] for c in conds}

    fold_rows: List[Dict] = []

    for seed in args.seeds:
        for fold in args.folds:
            split = load_fold_split(args.splits_dir, seed=seed, fold=fold)
            train_mask, test_mask = make_train_test_masks(
                union, split, require_full_coverage=bool(args.require_full_coverage)
            )

            # FULL-only TEST base for counterfactual outages
            eval_full_test = eval_mask_full_only(union, test_mask)
            idx_eval = np.where(eval_full_test)[0]
            if idx_eval.size == 0:
                raise RuntimeError(f"seed={seed} fold={fold}: FULL-only TEST empty.")

            # Load unimodal preds (UNION-aligned)
            y_all, p0_all, p1_all, p2_all = _load_unimodal_preds_npz(preds_root, seed, fold, args.dataset, N)

            # Contract check: probs are NaN where missing in UNION.
            if args.dataset == "stressid":
                assert_probs_nan_where_missing(union, p0_all, p1_all, p2_all)
            else:
                # MOSEI contract in your canonical q_contract uses Ml/Ma/Mv and probs must be NaN where missing as well.
                assert_probs_nan_where_missing(union, p1_all, p2_all, p0_all)  # reuse helper shape; order irrelevant for NaN check

            P_all = np.stack([p0_all, p1_all, p2_all], axis=1)  # (N,3), NaNs where missing

            # Build router TRAIN data (natural missingness): fill missing expert probs with 0.5
            idx_tr = np.where(train_mask)[0]
            if idx_tr.size == 0:
                raise RuntimeError(f"seed={seed} fold={fold}: TRAIN empty.")

            P_tr = np.array(P_all[idx_tr], copy=True)
            M_tr = M_all[idx_tr]
            y_tr = y_all[idx_tr].astype(int)

            # Fill missing probs with 0.5 for training only
            missing_tr = ~np.isfinite(P_tr)
            P_tr[missing_tr] = 0.5

            L_tr = _logit(P_tr, eps=args.eps_logit)  # (ntr,3)

            # Router A: E-only (M-agnostic): bias-only softmax -> constant weights
            # X is a single constant feature. (W,b) still learn.
            X_tr_E = np.ones((idx_tr.size, 1), dtype=float)

            # Router B: E+M: condition on availability (3-d)
            X_tr_EM = M_tr.astype(float)

            # Train/load routers (deterministic)
            base_tag = f"{args.dataset}_seed{seed}_fold{fold}"
            path_E = router_dir / f"router_Eonly_{base_tag}.npz"
            path_EM = router_dir / f"router_EM_{base_tag}.npz"

            if path_E.exists():
                z = np.load(path_E, allow_pickle=True)
                router_E = _Router(W=z["W"], b=z["b"])
                status_E = "LOADED"
            else:
                router_E = _train_router_adam(
                    X=X_tr_E,
                    L=L_tr,
                    y=y_tr,
                    seed=int(seed) * 100 + int(fold) + 1,
                    epochs=int(args.router_epochs),
                    lr=float(args.router_lr),
                    l2=float(args.router_l2),
                )
                np.savez_compressed(path_E, W=router_E.W, b=router_E.b)
                status_E = "TRAINED_SAVED"

            if path_EM.exists():
                z = np.load(path_EM, allow_pickle=True)
                router_EM = _Router(W=z["W"], b=z["b"])
                status_EM = "LOADED"
            else:
                router_EM = _train_router_adam(
                    X=X_tr_EM,
                    L=L_tr,
                    y=y_tr,
                    seed=int(seed) * 100 + int(fold) + 2,
                    epochs=int(args.router_epochs),
                    lr=float(args.router_lr),
                    l2=float(args.router_l2),
                )
                np.savez_compressed(path_EM, W=router_EM.W, b=router_EM.b)
                status_EM = "TRAINED_SAVED"

            # Prepare FULL-only TEST base arrays (all finite, all M=1)
            y_ev = y_all[idx_eval].astype(int)
            P_ev = np.array(P_all[idx_eval], copy=True)
            if not np.isfinite(P_ev).all():
                raise ValueError(f"seed={seed} fold={fold}: FULL-only TEST should have finite probs; found non-finite.")
            M_ev = np.ones_like(P_ev, dtype=float)

            # Evaluate all conditions (counterfactual outages)
            rng = np.random.RandomState(int(seed) * 10_000 + int(fold))
            per_fold = {"seed": int(seed), "fold": int(fold), "router_E": status_E, "router_EM": status_EM, "conditions": {}}

            for c in conds:
                P_cf, M_cf = _apply_outage(P_ev, M_ev, c, rng=rng)

                # E-only: constant X
                X_E = np.ones((len(y_ev), 1), dtype=float)
                p_E = _moe_fuse_probs(P_cf, X_E, router_E, eps_logit=args.eps_logit)
                ba_E = balanced_accuracy_from_probs(y_ev, p_E, thresh=args.thresh)

                # E+M: X = M_cf
                X_EM = M_cf.astype(float)
                p_EM = _moe_fuse_probs(P_cf, X_EM, router_EM, eps_logit=args.eps_logit)
                ba_EM = balanced_accuracy_from_probs(y_ev, p_EM, thresh=args.thresh)

                scores_E[c].append(float(ba_E))
                scores_EM[c].append(float(ba_EM))
                per_fold["conditions"][c] = {"BA_Eonly": float(ba_E), "BA_EM": float(ba_EM)}

            fold_rows.append(per_fold)

            # print a compact clean line
            c0 = "clean"
            print(
                f"[OK] seed={seed} fold={fold} | routers: E={status_E}, EM={status_EM} | "
                f"clean BA: E-only={per_fold['conditions'][c0]['BA_Eonly']:.3f} E+M={per_fold['conditions'][c0]['BA_EM']:.3f}"
            )

    # Aggregate
    table_rows: List[Tuple[str, Tuple[float,float], Tuple[float,float]]] = []
    for c in conds:
        mE, sE = _mean_std(np.asarray(scores_E[c], dtype=float))
        mM, sM = _mean_std(np.asarray(scores_EM[c], dtype=float))
        # rename conditions to your preferred names
        if c == "drop_0": name = f"drop_{names[0]}"
        elif c == "drop_1": name = f"drop_{names[1]}"
        elif c == "drop_2": name = f"drop_{names[2]}"
        elif c == "keep_0": name = f"keep_{names[0]}"
        elif c == "keep_1": name = f"keep_{names[1]}"
        elif c == "keep_2": name = f"keep_{names[2]}"
        else: name = c
        table_rows.append((name, (mE, sE), (mM, sM)))

    caption = f"MoE availability robustness ({args.dataset.upper()}; natural-missingness router training; FULL-only TEST counterfactual outages)."
    tex = _latex_table_outage(caption, table_rows, label="tab:outage_M")

    # Write artifacts
    out_json = out_dir / f"moe_availability_outage_{args.dataset}.json"
    out_csv = out_dir / f"moe_availability_outage_{args.dataset}.csv"
    out_tex = out_dir / f"Table_outage_M_{args.dataset}.tex"

    with open(out_json, "w") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "args": vars(args),
                "conditions": conds,
                "aggregate": {
                    c: {
                        "Eonly_mean": _mean_std(np.asarray(scores_E[c]))[0],
                        "Eonly_std": _mean_std(np.asarray(scores_E[c]))[1],
                        "EM_mean": _mean_std(np.asarray(scores_EM[c]))[0],
                        "EM_std": _mean_std(np.asarray(scores_EM[c]))[1],
                    } for c in conds
                },
                "fold_rows": fold_rows,
            },
            f,
            indent=2,
        )

    # CSV (flatten)
    with open(out_csv, "w") as f:
        f.write("seed,fold,condition,BA_Eonly,BA_EM\n")
        for r in fold_rows:
            for c in conds:
                f.write(f"{r['seed']},{r['fold']},{c},{r['conditions'][c]['BA_Eonly']:.6f},{r['conditions'][c]['BA_EM']:.6f}\n")

    if args.write_tex:
        out_tex.write_text(tex)

    print("\n% ===============================")
    print("% LaTeX table (copy/paste)")
    print("% ===============================\n")
    print(tex)

    print(f"[DONE] wrote: {out_json}")
    print(f"[DONE] wrote: {out_csv}")
    if args.write_tex:
        print(f"[DONE] wrote: {out_tex}")

if __name__ == "__main__":
    main()