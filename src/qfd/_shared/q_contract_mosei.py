#!/usr/bin/env python3
# qfd/_shared/q_contract_mosei.py
#
# Single source of truth for MOSEI UNION / splits / Q variants / FULL-only masking / oracleQ.
# Intentionally strict: if a script violates the contract, it should fail loudly.
#
# Option1 fix: keep FoldQ strict (1D/2D), and add a separate FoldQBank + loader for K-permutation banks
# (e.g., brokenQ stored as (N,K) or (N,K,d)). Experiments select a permutation k to obtain a FoldQ.
#
# Defensible contract (paper):
# - UNION row order is canonical. Nothing is ever re-ordered at runtime.
# - Splits are subject-safe, stored as train/test id lists. Never resplit.
# - Availability (M) is binary; missing => M=0.
# - Quality (Q) is defined only when present; missing => Q==0 exactly (all dims).
# - Late-fusion/unimodal probabilities are defined only when present; missing => prob=NaN.
# - BrokenQ is precomputed externally; experiments only LOAD it.
# - OracleQ is derived from THIS RUN’s unimodal predictions on the eval subset, returned in UNION row space.
#
# Robustness upgrades (match StressID q_contract.py):
# - UNION loader tolerates common key variants (E_l/El, M_l/Ml, y/y2, etc).
# - Q alignment verification supports ids or union_ids_hash (sha256 over UNION ids_str).
# - Split coverage strictness configurable (default strict: train ∪ test covers UNION).
# - Q key resolution centralized; deterministic Q file discovery with stable sorting.
# - Explicit dtype/shape/finite checks; missing rows must be exactly zero (all dims).

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np


# ----------------------------
# Data containers
# ----------------------------

@dataclass(frozen=True)
class UnionData:
    ids: np.ndarray          # (N,) original dtype
    ids_str: np.ndarray      # (N,) dtype=object (strings), canonical ids in row order
    y: np.ndarray            # (N,) int {0,1}
    El: np.ndarray           # (N, dl) language/text embeddings
    Ea: np.ndarray           # (N, da) audio embeddings
    Ev: np.ndarray           # (N, dv) video embeddings
    Ml: np.ndarray           # (N,) uint8 {0,1}
    Ma: np.ndarray           # (N,) uint8 {0,1}
    Mv: np.ndarray           # (N,) uint8 {0,1}
    meta: Optional[object] = None


@dataclass(frozen=True)
class FoldSplit:
    seed: int
    fold: int
    train_ids: np.ndarray    # (n_train,) dtype=object (strings)
    test_ids: np.ndarray     # (n_test,) dtype=object (strings)


@dataclass(frozen=True)
class FoldQ:
    Ql: np.ndarray           # (N,) or (N,d)
    Qa: np.ndarray           # (N,) or (N,d)
    Qv: np.ndarray           # (N,) or (N,d)
    q_path: str              # audit: file path used


# Option1: Q bank container (e.g., brokenQ stored as K permutations)
@dataclass(frozen=True)
class FoldQBank:
    Ql: np.ndarray           # (N,K) or (N,K,d)
    Qa: np.ndarray           # (N,K) or (N,K,d)
    Qv: np.ndarray           # (N,K) or (N,K,d)
    q_path: str              # audit: file path used


# ----------------------------
# Canonical UNION loader
# ----------------------------

def load_union(union_npz: Union[str, Path]) -> UnionData:
    """Load canonical MOSEI UNION. Fails loudly if required keys are missing/inconsistent."""
    p = Path(union_npz)
    if not p.exists():
        raise FileNotFoundError(f"UNION not found: {p}")

    z = np.load(p, allow_pickle=True)
    files = set(z.files)

    ids = _get_union_any(z, files, ["ids", "id", "ID"], required=True, where=p)
    y = _get_union_any(z, files, ["y2", "y", "label", "labels"], required=True, where=p)

    # embeddings
    El = _get_union_any(z, files, ["E_l", "El", "Etext", "E_lang", "E_L"], required=True, where=p)
    Ea = _get_union_any(z, files, ["E_a", "Ea", "Eaudio", "E_A"], required=True, where=p)
    Ev = _get_union_any(z, files, ["E_v", "Ev", "Evideo", "E_V"], required=True, where=p)

    # masks
    Ml = _get_union_any(z, files, ["M_l", "Ml", "Mtext", "M_lang", "M_L"], required=True, where=p)
    Ma = _get_union_any(z, files, ["M_a", "Ma", "Maudio", "M_A"], required=True, where=p)
    Mv = _get_union_any(z, files, ["M_v", "Mv", "Mvideo", "M_V"], required=True, where=p)

    ids = np.asarray(ids)
    ids_str = np.asarray([str(x) for x in ids], dtype=object)

    # Canonical indexing uses ids_str; duplicates are fatal.
    if len(np.unique(ids_str)) != len(ids_str):
        uniq, counts = np.unique(ids_str, return_counts=True)
        dups = uniq[counts > 1]
        raise ValueError(f"UNION ids not unique after str(). Example duplicates: {dups[:10].tolist()}")

    y = np.asarray(y).astype(int)

    El = np.asarray(El)
    Ea = np.asarray(Ea)
    Ev = np.asarray(Ev)

    Ml = np.asarray(Ml).astype(np.uint8)
    Ma = np.asarray(Ma).astype(np.uint8)
    Mv = np.asarray(Mv).astype(np.uint8)

    meta = z["meta"].item() if "meta" in files else None

    _assert_union_shapes(ids_str, y, El, Ea, Ev, Ml, Ma, Mv)
    _assert_binary_mask(Ml, "M_l")
    _assert_binary_mask(Ma, "M_a")
    _assert_binary_mask(Mv, "M_v")
    _assert_binary_label(y, "y")

    return UnionData(
        ids=ids,
        ids_str=ids_str,
        y=y,
        El=El,
        Ea=Ea,
        Ev=Ev,
        Ml=Ml,
        Ma=Ma,
        Mv=Mv,
        meta=meta,
    )


def build_id2row(ids_str: np.ndarray) -> Dict[str, int]:
    """Map string id -> row index. ids_str must be unique."""
    return {str(v): i for i, v in enumerate(ids_str)}


def union_ids_hash(ids_str: np.ndarray) -> str:
    """
    Stable sha256 signature over UNION ids_str in row order.
    Use this in Q files when storing full ids is undesirable.
    """
    h = hashlib.sha256()
    for s in ids_str:
        h.update(str(s).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


# ----------------------------
# Splits
# ----------------------------

def load_fold_split(splits_dir: Union[str, Path], seed: int, fold: int) -> FoldSplit:
    """Load canonical split IDs as strings."""
    base = Path(splits_dir) / f"seed_{seed}"
    tr_path = base / f"train_ids_fold{fold}.npy"
    te_path = base / f"test_ids_fold{fold}.npy"
    if not tr_path.exists():
        raise FileNotFoundError(f"Missing split file: {tr_path}")
    if not te_path.exists():
        raise FileNotFoundError(f"Missing split file: {te_path}")

    train_ids = np.load(tr_path, allow_pickle=True)
    test_ids = np.load(te_path, allow_pickle=True)

    train_ids = np.asarray([str(x) for x in train_ids], dtype=object)
    test_ids = np.asarray([str(x) for x in test_ids], dtype=object)

    if len(np.unique(train_ids)) != len(train_ids):
        raise ValueError(f"Duplicate IDs in train split seed={seed} fold={fold}")
    if len(np.unique(test_ids)) != len(test_ids):
        raise ValueError(f"Duplicate IDs in test split seed={seed} fold={fold}")

    return FoldSplit(seed=seed, fold=fold, train_ids=train_ids, test_ids=test_ids)


def rows_from_ids(ids_subset: Iterable[str], id2row: Dict[str, int], *, name: str) -> np.ndarray:
    """Resolve ids -> row indices. Fails if any id is missing."""
    ids_subset = list(ids_subset)
    missing: List[str] = []
    rows: List[int] = []
    for sid in ids_subset:
        s = str(sid)
        r = id2row.get(s)
        if r is None:
            missing.append(s)
        else:
            rows.append(r)

    if missing:
        raise KeyError(f"{name}: {len(missing)} ids not found in UNION. Example: {missing[:20]}")

    return np.asarray(rows, dtype=int)


def make_train_test_masks(
    union: UnionData,
    split: FoldSplit,
    *,
    require_full_coverage: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (train_mask, test_mask) over UNION rows for given fold split.

    Invariants:
      - all split ids must exist in UNION
      - train/test disjoint

    Coverage:
      - if require_full_coverage=True (default), enforce train∪test covers UNION exactly
      - else allow splits to cover a subset
    """
    id2row = build_id2row(union.ids_str)

    train_rows = rows_from_ids(split.train_ids, id2row, name=f"train_ids (seed={split.seed}, fold={split.fold})")
    test_rows = rows_from_ids(split.test_ids, id2row, name=f"test_ids (seed={split.seed}, fold={split.fold})")

    if np.intersect1d(train_rows, test_rows).size != 0:
        raise ValueError(f"Split leakage: train/test overlap for seed={split.seed} fold={split.fold}")

    n = len(union.ids_str)
    train_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    train_mask[train_rows] = True
    test_mask[test_rows] = True

    if require_full_coverage:
        covered = train_mask | test_mask
        if not np.all(covered):
            missing_rows = np.where(~covered)[0]
            sample = union.ids_str[missing_rows[:20]].tolist()
            raise ValueError(
                f"Split does not cover all UNION rows for seed={split.seed} fold={split.fold}. "
                f"Missing rows={missing_rows.size}. Example ids: {sample}"
            )
        if train_mask.sum() + test_mask.sum() != n:
            raise ValueError(
                f"Split coverage mismatch for seed={split.seed} fold={split.fold}: "
                f"train+test={train_mask.sum()+test_mask.sum()} != N={n}"
            )

    return train_mask, test_mask


# ----------------------------
# FULL-only helpers
# ----------------------------

def full_only_mask(
    Ml: np.ndarray,
    Ma: np.ndarray,
    Mv: np.ndarray,
    base_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """FULL-only rows. If base_mask provided, intersect with it."""
    full = (Ml == 1) & (Ma == 1) & (Mv == 1)
    if base_mask is not None:
        full &= np.asarray(base_mask, dtype=bool)
    return full


def eval_mask_full_only(union: UnionData, test_mask: np.ndarray) -> np.ndarray:
    """FULL-only within test."""
    return full_only_mask(union.Ml, union.Ma, union.Mv, base_mask=test_mask)


def slice_mask(mask: np.ndarray, *more_masks: np.ndarray) -> np.ndarray:
    """AND multiple masks."""
    out = np.asarray(mask, dtype=bool).copy()
    for m in more_masks:
        out &= np.asarray(m, dtype=bool)
    return out


# ----------------------------
# Q loading (clean/broken) — contract checks inside
# ----------------------------

_QL_KEYS = ["Ql", "Q_l", "Qlang", "Q_text", "Q_l_scaled", "Q_l_fold", "Q_l_qscaled"]
_QA_KEYS = ["Qa", "Q_a", "Qaudio", "Q_a_scaled", "Q_a_fold", "Q_a_qscaled"]
_QV_KEYS = ["Qv", "Q_v", "Qvideo", "Q_v_scaled", "Q_v_fold", "Q_v_qscaled"]

_ID_KEYS = ["ids", "ids_str", "union_ids", "union_ids_str"]
_HASH_KEYS = ["union_ids_hash", "ids_hash", "union_hash"]


def find_fold_q_file(q_root: Union[str, Path], seed: int, fold: int) -> Path:
    """
    Deterministic discovery for per-(seed,fold) Q npz.

    Order:
      1) explicit canonical candidates
      2) glob candidates (stable sorted)
      3) rglob fallback (stable sorted)
    """
    root = Path(q_root)
    if not root.exists():
        raise FileNotFoundError(f"Q root not found: {root}")

    candidates: List[Path] = [
        root / f"seed_{seed}" / f"fold_{fold}.npz",
        root / f"seed_{seed}" / f"fold{fold}.npz",
        root / f"seed_{seed}" / f"q_fold{fold}.npz",
        root / f"seed_{seed}" / f"fold_{fold}" / "q.npz",
        root / f"seed_{seed}_fold_{fold}.npz",
        root / f"seed_{seed}" / f"union_quality_Q_lav_v1_seed{seed}_fold{fold}.npz",
        # bank-style files may share names; keep rglob fallback
    ]
    for p in candidates:
        if p.exists():
            return p

    glob_patterns = [
        f"seed_{seed}/union_quality_Q_*_seed{seed}_fold{fold}.npz",
        f"seed_{seed}/*seed{seed}*fold{fold}*.npz",
    ]
    hits: List[Path] = []
    for pat in glob_patterns:
        hits.extend(sorted(root.glob(pat)))

    if not hits:
        hits.extend(sorted(root.rglob(f"*seed{seed}*fold{fold}*.npz")))
        hits.extend(sorted(root.rglob(f"*seed_{seed}*fold{fold}*.npz")))

    hits = sorted({h.resolve() for h in hits}, key=lambda x: str(x))

    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise FileNotFoundError(
            f"Ambiguous Q file for seed={seed}, fold={fold} under {root}. "
            f"Edit find_fold_q_file() patterns. Candidates:\n" + "\n".join(map(str, hits[:40]))
        )
    raise FileNotFoundError(
        f"Could not find Q file for seed={seed}, fold={fold} under {root}. "
        f"Edit find_fold_q_file() patterns to match your naming."
    )


def load_fold_q(
    q_root: Union[str, Path],
    seed: int,
    fold: int,
    *,
    union: UnionData,
    require_ids_match: bool = True,
    allow_hash_match: bool = True,
) -> FoldQ:
    """
    Load a SINGLE Q realization (cleanQ or a single brokenQ), returning 1D/2D arrays.

    This function is intentionally strict: Q must be 1D or 2D.
    For K-permutation banks (e.g., brokenQ stored with K axis), use load_fold_q_bank().
    """
    q_path = find_fold_q_file(q_root, seed, fold)
    z = np.load(q_path, allow_pickle=True)
    files = set(z.files)

    Ql = _get_any_array(z, files, _QL_KEYS, q_path)
    Qa = _get_any_array(z, files, _QA_KEYS, q_path)
    Qv = _get_any_array(z, files, _QV_KEYS, q_path)

    N = len(union.ids_str)
    Ql = _as_q_array(Ql, "Ql", q_path, expect_len=N)
    Qa = _as_q_array(Qa, "Qa", q_path, expect_len=N)
    Qv = _as_q_array(Qv, "Qv", q_path, expect_len=N)

    _verify_q_alignment(union, z, files, q_path, require_ids_match=require_ids_match, allow_hash_match=allow_hash_match)

    # Contract checks
    assert_q_missing_is_zero(union, Ql, Qa, Qv)
    assert_no_nan_in_present_q(union, Ql, Qa, Qv)

    return FoldQ(Ql=Ql, Qa=Qa, Qv=Qv, q_path=str(q_path))


# ---------- Option1 additions: bank loader + selector ----------

def load_fold_q_bank(
    q_root: Union[str, Path],
    seed: int,
    fold: int,
    *,
    union: UnionData,
    require_ids_match: bool = True,
    allow_hash_match: bool = True,
) -> FoldQBank:
    """
    Load a K-permutation Q bank where arrays are (N,K) or (N,K,d).
    Intended for brokenQ banks.

    Contract (bank-level):
      - first dim must be N
      - Q must be numeric
      - Q must be 2D or 3D (bank axis is dim=1)
      - missing rows (M==0) must be exactly zero across ALL K (all dims)
      - alignment verified via ids or union_ids_hash
    """
    q_path = find_fold_q_file(q_root, seed, fold)
    z = np.load(q_path, allow_pickle=True)
    files = set(z.files)

    Ql = _get_any_array(z, files, _QL_KEYS, q_path)
    Qa = _get_any_array(z, files, _QA_KEYS, q_path)
    Qv = _get_any_array(z, files, _QV_KEYS, q_path)

    N = len(union.ids_str)
    Ql = _as_q_bank(Ql, "Ql", q_path, expect_len=N)
    Qa = _as_q_bank(Qa, "Qa", q_path, expect_len=N)
    Qv = _as_q_bank(Qv, "Qv", q_path, expect_len=N)

    # Bank axis K must match across modalities (strict)
    K_l = Ql.shape[1]
    K_a = Qa.shape[1]
    K_v = Qv.shape[1]
    if not (K_l == K_a == K_v):
        raise ValueError(
            f"BrokenQ bank K mismatch in {q_path}: "
            f"Ql.K={K_l}, Qa.K={K_a}, Qv.K={K_v}"
        )

    _verify_q_alignment(union, z, files, q_path, require_ids_match=require_ids_match, allow_hash_match=allow_hash_match)

    # Missing rows must be zero across all K
    _assert_missing_zero_bank(Ql, union.Ml, "Ql", q_path)
    _assert_missing_zero_bank(Qa, union.Ma, "Qa", q_path)
    _assert_missing_zero_bank(Qv, union.Mv, "Qv", q_path)

    # We do NOT enforce "present finite" across all K here (can be expensive / unnecessary).
    # Selection-time checks enforce FoldQ invariants.

    return FoldQBank(Ql=Ql, Qa=Qa, Qv=Qv, q_path=str(q_path))


def select_perm_from_bank(union: UnionData, bank: FoldQBank, k: int) -> FoldQ:
    """
    Select a single permutation k from a Q bank, returning a standard FoldQ (1D/2D).
    Enforces the standard contract on the selected slice.
    """
    if k < 0 or k >= bank.Ql.shape[1]:
        raise IndexError(f"k out of range: k={k}, K={bank.Ql.shape[1]} for {bank.q_path}")

    Ql = bank.Ql[:, k] if bank.Ql.ndim == 2 else bank.Ql[:, k, :]
    Qa = bank.Qa[:, k] if bank.Qa.ndim == 2 else bank.Qa[:, k, :]
    Qv = bank.Qv[:, k] if bank.Qv.ndim == 2 else bank.Qv[:, k, :]

    # Selected slice must satisfy the standard contract
    assert_q_missing_is_zero(union, Ql, Qa, Qv)
    assert_no_nan_in_present_q(union, Ql, Qa, Qv)

    return FoldQ(Ql=Ql, Qa=Qa, Qv=Qv, q_path=f"{bank.q_path}::k={k}")


def stack_q_for_three_experts(Ql: np.ndarray, Qa: np.ndarray, Qv: np.ndarray) -> np.ndarray:
    """
    Stack per-expert Q into (N,3) or (N,3,d) depending on Q dimensionality.
    Accepts Q shaped (N,) or (N,d).
    """
    Ql = np.asarray(Ql)
    Qa = np.asarray(Qa)
    Qv = np.asarray(Qv)
    if Ql.ndim not in (1, 2) or Qa.ndim != Ql.ndim or Qv.ndim != Ql.ndim:
        raise ValueError(f"Q ndim mismatch: {Ql.ndim}, {Qa.ndim}, {Qv.ndim}")
    return np.stack([Ql, Qa, Qv], axis=1)


# ----------------------------
# OracleQ (computed from THIS RUN)
# ----------------------------

def preds_from_probs(probs_pos: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    """Convert positive-class probabilities to 0/1 predictions."""
    return (np.asarray(probs_pos) >= float(thresh)).astype(int)


def compute_oracle_q_union(
    union: UnionData,
    eval_mask: np.ndarray,
    p_l: Optional[np.ndarray],
    p_a: Optional[np.ndarray],
    p_v: Optional[np.ndarray],
    *,
    thresh: float = 0.5,
) -> FoldQ:
    """
    Compute oracleQ in UNION row space from THIS RUN's expert predictions.

    Definition:
      Q*_m = 1[ŷ_m == y] on (eval_mask & present_m & finite(p_m)).
      All other rows are 0.

    Inputs p_* are UNION-length probabilities with NaN where modality missing/undefined.
    """
    n = len(union.ids_str)
    eval_mask = np.asarray(eval_mask, dtype=bool)
    if eval_mask.shape != (n,):
        raise ValueError(f"eval_mask must be shape (N,), got {eval_mask.shape}")

    Ql = np.zeros(n, dtype=float)
    Qa = np.zeros(n, dtype=float)
    Qv = np.zeros(n, dtype=float)

    y = union.y.astype(int)

    def fill(Qout: np.ndarray, M: np.ndarray, p: Optional[np.ndarray], name: str) -> None:
        if p is None:
            return
        p = np.asarray(p)
        if p.shape != (n,):
            raise ValueError(f"{name} probs must be shape (N,), got {p.shape}")
        use = (M == 1) & eval_mask & np.isfinite(p)
        if not np.any(use):
            return
        yhat = preds_from_probs(p[use], thresh=thresh)
        Qout[use] = (yhat == y[use]).astype(float)

    fill(Ql, union.Ml, p_l, "lang")
    fill(Qa, union.Ma, p_a, "audio")
    fill(Qv, union.Mv, p_v, "video")

    # Contract checks
    assert_q_missing_is_zero(union, Ql, Qa, Qv)
    assert_no_nan_in_present_q(union, Ql, Qa, Qv)

    return FoldQ(Ql=Ql, Qa=Qa, Qv=Qv, q_path="(oracleQ_from_preds)")


# ----------------------------
# Metrics (minimal, dependency-free)
# ----------------------------

def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Binary balanced accuracy."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    pos = y_true == 1
    neg = y_true == 0
    tpr = np.mean(y_pred[pos] == 1) if np.any(pos) else np.nan
    tnr = np.mean(y_pred[neg] == 0) if np.any(neg) else np.nan
    if np.isnan(tpr) or np.isnan(tnr):
        return float("nan")
    return float(0.5 * (tpr + tnr))


def balanced_accuracy_from_probs(y_true: np.ndarray, p_pos: np.ndarray, thresh: float = 0.5) -> float:
    return balanced_accuracy(np.asarray(y_true), preds_from_probs(np.asarray(p_pos), thresh=thresh))


def flip_rate(y_a: np.ndarray, y_b: np.ndarray) -> float:
    """Fraction of samples where predictions differ."""
    y_a = np.asarray(y_a)
    y_b = np.asarray(y_b)
    if y_a.shape != y_b.shape:
        raise ValueError(f"flip_rate requires same shapes, got {y_a.shape} vs {y_b.shape}")
    return float(np.mean(y_a != y_b))


# ----------------------------
# Contract assertions (fail loud)
# ----------------------------

def assert_q_missing_is_zero(union: UnionData, Ql: np.ndarray, Qa: np.ndarray, Qv: np.ndarray) -> None:
    """Enforce: missing modality (M==0) must have Q==0 (all dims)."""
    _assert_missing_zero(Ql, union.Ml, "Ql")
    _assert_missing_zero(Qa, union.Ma, "Qa")
    _assert_missing_zero(Qv, union.Mv, "Qv")


def assert_no_nan_in_present_q(union: UnionData, Ql: np.ndarray, Qa: np.ndarray, Qv: np.ndarray) -> None:
    """Enforce: for present rows, Q must be finite."""
    _assert_present_finite(Ql, union.Ml, "Ql")
    _assert_present_finite(Qa, union.Ma, "Qa")
    _assert_present_finite(Qv, union.Mv, "Qv")


def assert_probs_nan_where_missing(union: UnionData, p_l: np.ndarray, p_a: np.ndarray, p_v: np.ndarray) -> None:
    """Enforce: if M==0 then probs must be NaN/inf (undefined where missing)."""
    _assert_nan_where_missing(p_l, union.Ml, "p_l")
    _assert_nan_where_missing(p_a, union.Ma, "p_a")
    _assert_nan_where_missing(p_v, union.Mv, "p_v")


# ----------------------------
# Audit signatures
# ----------------------------

def q_signature(union: UnionData, q: FoldQ, mask: np.ndarray) -> Dict[str, float]:
    """
    Numeric signature for auditing that two scripts use the same subset and Q.
    mask typically = FULL-only test mask.
    Reports stats on:
      - mask
      - mask & present
    """
    n = len(union.ids_str)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (n,):
        raise ValueError(f"mask must be (N,), got {mask.shape}")

    def stats(arr: np.ndarray, present: np.ndarray) -> Dict[str, float]:
        arr = np.asarray(arr)
        if arr.ndim == 2:
            arr1 = np.mean(arr, axis=1)
        elif arr.ndim == 1:
            arr1 = arr
        else:
            raise ValueError("Q must be 1D or 2D")

        m_all = mask
        m_pres = mask & (present == 1)

        def s(m: np.ndarray) -> Dict[str, float]:
            idx = np.where(m)[0]
            if idx.size == 0:
                return {"n": 0, "mean": float("nan"), "std": float("nan"),
                        "min": float("nan"), "max": float("nan"), "frac0": float("nan")}
            x = arr1[idx]
            return {
                "n": int(idx.size),
                "mean": float(np.mean(x)),
                "std": float(np.std(x)),
                "min": float(np.min(x)),
                "max": float(np.max(x)),
                "frac0": float(np.mean(x == 0)),
            }

        out: Dict[str, float] = {}
        a = s(m_all)
        p = s(m_pres)
        out.update({f"all_{k}": v for k, v in a.items()})
        out.update({f"pres_{k}": v for k, v in p.items()})
        return out

    out: Dict[str, float] = {}
    out.update({f"Ql_{k}": v for k, v in stats(q.Ql, union.Ml).items()})
    out.update({f"Qa_{k}": v for k, v in stats(q.Qa, union.Ma).items()})
    out.update({f"Qv_{k}": v for k, v in stats(q.Qv, union.Mv).items()})
    return out


# ----------------------------
# Internals
# ----------------------------

def _get_union_any(z: np.lib.npyio.NpzFile, files: set, keys: Sequence[str], *, required: bool, where: Path):
    for k in keys:
        if k in files:
            return z[k]
    if required:
        raise KeyError(f"UNION {where} missing one of keys {list(keys)}. Found keys: {sorted(files)}")
    return None


def _get_any_array(z: np.lib.npyio.NpzFile, files: set, keys: List[str], q_path: Path) -> np.ndarray:
    for k in keys:
        if k in files:
            return z[k]
    raise KeyError(f"Q file {q_path} missing keys {keys}. Found keys: {sorted(files)}")


def _as_q_array(q: np.ndarray, name: str, q_path: Path, expect_len: int) -> np.ndarray:
    q = np.asarray(q)
    if q.shape[0] != expect_len:
        raise ValueError(f"{name} length {q.shape[0]} != expected {expect_len} in {q_path}")
    if q.ndim not in (1, 2):
        raise ValueError(f"{name} must be 1D or 2D. Got shape {q.shape} in {q_path}")
    if not np.issubdtype(q.dtype, np.number):
        raise TypeError(f"{name} must be numeric dtype. Got {q.dtype} in {q_path}")
    return q.astype(float, copy=False)


def _as_q_bank(q: np.ndarray, name: str, q_path: Path, expect_len: int) -> np.ndarray:
    q = np.asarray(q)
    if q.shape[0] != expect_len:
        raise ValueError(f"{name} bank length {q.shape[0]} != expected {expect_len} in {q_path}")
    if q.ndim not in (2, 3):
        raise ValueError(f"{name} bank must be 2D or 3D. Got shape {q.shape} in {q_path}")
    if not np.issubdtype(q.dtype, np.number):
        raise TypeError(f"{name} bank must be numeric dtype. Got {q.dtype} in {q_path}")
    return q.astype(float, copy=False)


def _maybe_load_ids(z: np.lib.npyio.NpzFile, files: set) -> Optional[np.ndarray]:
    for k in _ID_KEYS:
        if k in files:
            return np.asarray([str(x) for x in z[k]], dtype=object)
    return None


def _maybe_load_hash(z: np.lib.npyio.NpzFile, files: set) -> Optional[str]:
    for k in _HASH_KEYS:
        if k in files:
            v = z[k]
            if isinstance(v, np.ndarray) and v.shape == ():
                v = v.item()
            if isinstance(v, bytes):
                v = v.decode("utf-8")
            return str(v)

    # Some pipelines store it in meta
    if "meta" in files:
        try:
            meta = z["meta"].item()
            if isinstance(meta, dict):
                for k in _HASH_KEYS:
                    if k in meta:
                        return str(meta[k])
        except Exception:
            pass
    return None


def _verify_q_alignment(
    union: UnionData,
    z: np.lib.npyio.NpzFile,
    files: set,
    q_path: Path,
    *,
    require_ids_match: bool,
    allow_hash_match: bool,
) -> None:
    if not require_ids_match:
        return

    N = len(union.ids_str)
    q_ids = _maybe_load_ids(z, files)
    if q_ids is not None:
        if q_ids.shape[0] != N:
            raise ValueError(f"Q ids length {q_ids.shape[0]} != UNION length {N} in {q_path}")
        if not np.array_equal(q_ids, union.ids_str):
            bad = np.where(q_ids != union.ids_str)[0]
            sample = [(int(i), union.ids_str[i], q_ids[i]) for i in bad[:10]]
            raise ValueError(f"Q ids do not match UNION ids order in {q_path}. sample: {sample}")
        return

    if not allow_hash_match:
        raise KeyError(
            f"Q file {q_path} must contain ids for alignment verification. "
            f"Expected one of keys {_ID_KEYS}."
        )

    q_hash = _maybe_load_hash(z, files)
    if q_hash is None:
        raise KeyError(
            f"Q file {q_path} must contain ids OR union_ids_hash for alignment verification. "
            f"Expected ids key in {_ID_KEYS} or hash key in {_HASH_KEYS}. "
            f"Found keys: {sorted(files)}"
        )
    u_hash = union_ids_hash(union.ids_str)
    if str(q_hash) != str(u_hash):
        raise ValueError(f"Q file {q_path} union_ids_hash mismatch. Q has {q_hash}, UNION has {u_hash}.")


def _assert_union_shapes(ids_str, y, El, Ea, Ev, Ml, Ma, Mv) -> None:
    n = len(ids_str)
    for name, arr in [("y", y), ("M_l", Ml), ("M_a", Ma), ("M_v", Mv)]:
        if np.asarray(arr).shape[0] != n:
            raise ValueError(f"UNION {name}.shape[0]={np.asarray(arr).shape[0]} != ids length {n}")
    for name, emb in [("E_l", El), ("E_a", Ea), ("E_v", Ev)]:
        emb = np.asarray(emb)
        if emb.shape[0] != n:
            raise ValueError(f"UNION {name}.shape[0]={emb.shape[0]} != ids length {n}")
        if emb.ndim != 2:
            raise ValueError(f"UNION {name} must be 2D (N,d). Got shape {emb.shape}")


def _assert_binary_mask(M: np.ndarray, name: str) -> None:
    M = np.asarray(M)
    bad = ~np.isin(M, [0, 1])
    if np.any(bad):
        vals = np.unique(M[bad])
        raise ValueError(f"{name} must be binary in {{0,1}}. Found values: {vals[:10]}")


def _assert_binary_label(y: np.ndarray, name: str) -> None:
    y = np.asarray(y)
    bad = ~np.isin(y, [0, 1])
    if np.any(bad):
        vals = np.unique(y[bad])
        raise ValueError(f"{name} must be binary in {{0,1}}. Found values: {vals[:10]}")


def _assert_missing_zero(Q: np.ndarray, M: np.ndarray, name: str) -> None:
    Q = np.asarray(Q)
    M = np.asarray(M)
    idx = np.where(M == 0)[0]
    if idx.size == 0:
        return
    if Q.ndim == 1:
        if not np.all(Q[idx] == 0):
            bad = idx[np.where(Q[idx] != 0)[0][:10]]
            raise ValueError(f"{name}: expected Q==0 where M==0. Example bad rows: {bad.tolist()}")
    elif Q.ndim == 2:
        if not np.all(Q[idx, :] == 0):
            bad = idx[np.where(np.any(Q[idx, :] != 0, axis=1))[0][:10]]
            raise ValueError(f"{name}: expected Q==0 where M==0 (all dims). Example bad rows: {bad.tolist()}")
    else:
        raise ValueError(f"{name}: unsupported ndim={Q.ndim}. Expected 1 or 2.")


def _assert_missing_zero_bank(Q: np.ndarray, M: np.ndarray, name: str, q_path: Path) -> None:
    """
    Bank-level missingness check:
      - if M==0 then Q must be exactly 0 across ALL K (and dims).
    Q is (N,K) or (N,K,d).
    """
    Q = np.asarray(Q)
    M = np.asarray(M)
    idx = np.where(M == 0)[0]
    if idx.size == 0:
        return

    if Q.ndim == 2:
        bad_local = np.where(np.any(Q[idx, :] != 0, axis=1))[0]
    elif Q.ndim == 3:
        bad_local = np.where(np.any(Q[idx, :, :] != 0, axis=(1, 2)))[0]
    else:
        raise ValueError(f"{name} bank: unsupported ndim={Q.ndim} in {q_path}")

    if bad_local.size:
        bad_rows = idx[bad_local[:10]]
        raise ValueError(
            f"{name}: expected Q==0 where M==0 across all K (and dims). "
            f"Example bad rows: {bad_rows.tolist()} in {q_path}"
        )


def _assert_present_finite(Q: np.ndarray, M: np.ndarray, name: str) -> None:
    Q = np.asarray(Q)
    M = np.asarray(M)
    idx = np.where(M == 1)[0]
    if idx.size == 0:
        return
    if Q.ndim == 1:
        ok = np.isfinite(Q[idx]).all()
    elif Q.ndim == 2:
        ok = np.isfinite(Q[idx, :]).all()
    else:
        raise ValueError(f"{name}: unsupported ndim={Q.ndim}. Expected 1 or 2.")
    if not ok:
        if Q.ndim == 1:
            bad = idx[np.where(~np.isfinite(Q[idx]))[0][:10]]
        else:
            bad = idx[np.where(~np.isfinite(Q[idx, :]).all(axis=1))[0][:10]]
        raise ValueError(f"{name}: non-finite Q values where M==1. Example bad rows: {bad.tolist()}")


def _assert_nan_where_missing(p: np.ndarray, M: np.ndarray, name: str) -> None:
    p = np.asarray(p)
    M = np.asarray(M)
    if p.shape != M.shape:
        raise ValueError(f"{name}: shape {p.shape} must match mask shape {M.shape}")
    idx = np.where(M == 0)[0]
    if idx.size == 0:
        return
    # Undefined where missing: require non-finite (NaN or inf) to prevent accidental use.
    if not np.all(~np.isfinite(p[idx])):
        bad = idx[np.where(np.isfinite(p[idx]))[0][:10]]
        raise ValueError(
            f"{name}: expected NaN/inf where M==0, but found finite values. Example bad rows: {bad.tolist()}"
        )