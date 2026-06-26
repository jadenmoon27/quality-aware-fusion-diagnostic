# Data contract

The repository assumes a **UNION-first** workflow.

## StressID UNION

Required keys:

| Key | Shape | Meaning |
|---|---:|---|
| `ids` | `(N,)` | canonical row IDs; all other arrays follow this order |
| `y` or `y2` | `(N,)` | binary label |
| `E_a`, `E_v`, `E_p` | `(N, d_m)` | modality embeddings for audio, video, physiology |
| `M_a`, `M_v`, `M_p` | `(N,)` | binary availability masks |

Missing embeddings may be zero-filled for array consistency, but missingness is defined only by `M_m`.

## Fold splits

Each split is stored by IDs, not row indices:

```text
splits/seed_11/train_ids_fold0.npy
splits/seed_11/test_ids_fold0.npy
```

The contract loader checks for train/test overlap and, when requested, full coverage of UNION rows.

## Fold-scaled Q

For each seed/fold, Q files must contain `Qa`, `Qv`, `Qp` aligned to UNION order. Accepted shapes:

- clean Q: `(N,)` or `(N, d)`
- Broken-Q permutation bank: `(N, K)` or `(N, K, d)`

Hard constraints:

- present rows: finite Q
- missing rows: exactly zero Q
- include `ids` or `union_ids_hash` for alignment verification

## Unimodal posterior dump

The StressID dump script writes:

```text
unimodal_preds/lr/seed_11/fold_0.npz
```

Required arrays include `ids`, `y`, `train_mask`, `test_mask`, `p_a`, `p_v`, `p_p`. Posterior arrays are finite only where the modality is present and `NaN` where missing.

## CMU-MOSEI

MOSEI mirrors the same contract with modalities `[language, audio, visual]`.

Required UNION keys are `ids`, `y` or `y2`, `E_l/E_a/E_v`, and `M_l/M_a/M_v`.

Fold-safe Q files use `Ql`, `Qa`, and `Qv`. Clean Q is `(N,)` or `(N,d)`. Broken-Q banks produced by `qfd.mosei.precompute_brokenq` are one file per seed/fold with shape `(N,K)` or `(N,K,d)`; the permutation-test scripts also retain compatibility with an older `perm_###/seed_x/fold_y.npz` layout.
