from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from qfd._shared.q_contract import (
    FoldQ,
    FoldSplit,
    UnionData,
    full_only_mask,
    make_train_test_masks,
)
from qfd.stressid.precompute_brokenq import build_broken_q
from qfd._shared import q_contract_mosei as mosei_qc


def _stressid_union() -> UnionData:
    ids = np.asarray([f"row_{i}" for i in range(8)], dtype=object)
    y = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=int)
    Ea = np.zeros((8, 2), dtype=float)
    Ev = np.zeros((8, 2), dtype=float)
    Ep = np.zeros((8, 2), dtype=float)
    Ma = np.asarray([1, 1, 1, 0, 1, 1, 0, 1], dtype=np.uint8)
    Mv = np.asarray([1, 1, 0, 1, 1, 1, 1, 0], dtype=np.uint8)
    Mp = np.asarray([1, 0, 1, 1, 1, 0, 1, 1], dtype=np.uint8)
    return UnionData(ids=ids, ids_str=ids, y=y, Ea=Ea, Ev=Ev, Ep=Ep, Ma=Ma, Mv=Mv, Mp=Mp)


def _row_multiset(x: np.ndarray) -> list[tuple[float, ...]]:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    return sorted(tuple(row.tolist()) for row in x)


class BrokenQContractTests(unittest.TestCase):
    def test_stressid_brokenq_is_test_only_present_only_and_vector_safe(self) -> None:
        union = _stressid_union()
        split = FoldSplit(
            seed=11,
            fold=0,
            train_ids=union.ids_str[:4],
            test_ids=union.ids_str[4:],
        )
        train_mask, test_mask = make_train_test_masks(union, split)

        Qa = np.arange(16, dtype=float).reshape(8, 2)
        Qv = Qa + 100
        Qp = Qa + 200
        Qa[union.Ma == 0] = 0
        Qv[union.Mv == 0] = 0
        Qp[union.Mp == 0] = 0
        clean = FoldQ(Qa=Qa, Qv=Qv, Qp=Qp, q_path="synthetic")

        Qa_b, Qv_b, Qp_b, _ = build_broken_q(
            union=union,
            test_mask=test_mask,
            clean=clean,
            K=3,
            base_seed=123,
            seed=11,
            fold=0,
        )

        self.assertEqual(Qa_b.shape, (8, 3, 2))
        self.assertEqual(Qv_b.shape, (8, 3, 2))
        self.assertEqual(Qp_b.shape, (8, 3, 2))

        for Q_clean, Q_bank, M in [(Qa, Qa_b, union.Ma), (Qv, Qv_b, union.Mv), (Qp, Qp_b, union.Mp)]:
            eligible = test_mask & (M == 1)
            unchanged = ~eligible
            for k in range(Q_bank.shape[1]):
                np.testing.assert_array_equal(Q_bank[train_mask, k, :], Q_clean[train_mask])
                np.testing.assert_array_equal(Q_bank[M == 0, k, :], 0)
                np.testing.assert_array_equal(Q_bank[unchanged, k, :], Q_clean[unchanged])
                self.assertEqual(_row_multiset(Q_bank[eligible, k, :]), _row_multiset(Q_clean[eligible]))

        expected_full = (union.Ma == 1) & (union.Mv == 1) & (union.Mp == 1)
        np.testing.assert_array_equal(full_only_mask(union.Ma, union.Mv, union.Mp), expected_full)

    def test_train_test_overlap_is_rejected(self) -> None:
        union = _stressid_union()
        split = FoldSplit(
            seed=11,
            fold=0,
            train_ids=np.asarray(["row_0", "row_1"], dtype=object),
            test_ids=np.asarray(["row_1", "row_2"], dtype=object),
        )
        with self.assertRaises(ValueError):
            make_train_test_masks(union, split, require_full_coverage=False)

    def test_mosei_vector_bank_load_and_select(self) -> None:
        ids = np.asarray([f"seg_{i}" for i in range(5)], dtype=object)
        union = mosei_qc.UnionData(
            ids=ids,
            ids_str=ids,
            y=np.asarray([0, 1, 0, 1, 0], dtype=int),
            El=np.zeros((5, 2), dtype=float),
            Ea=np.zeros((5, 2), dtype=float),
            Ev=np.zeros((5, 2), dtype=float),
            Ml=np.asarray([1, 1, 0, 1, 1], dtype=np.uint8),
            Ma=np.asarray([1, 0, 1, 1, 1], dtype=np.uint8),
            Mv=np.asarray([1, 1, 1, 0, 1], dtype=np.uint8),
        )

        Ql = np.arange(30, dtype=float).reshape(5, 3, 2)
        Qa = Ql + 100
        Qv = Ql + 200
        Ql[union.Ml == 0] = 0
        Qa[union.Ma == 0] = 0
        Qv[union.Mv == 0] = 0

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "seed_11"
            out_dir.mkdir()
            np.savez(
                out_dir / "fold_0.npz",
                Ql=Ql,
                Qa=Qa,
                Qv=Qv,
                ids_str=ids,
                union_ids_hash=mosei_qc.union_ids_hash(ids),
            )

            bank = mosei_qc.load_fold_q_bank(root, seed=11, fold=0, union=union)
            selected = mosei_qc.select_perm_from_bank(union, bank, 2)

        self.assertEqual(bank.Ql.shape, (5, 3, 2))
        self.assertEqual(selected.Ql.shape, (5, 2))
        np.testing.assert_array_equal(selected.Ql, Ql[:, 2, :])
        np.testing.assert_array_equal(selected.Qa, Qa[:, 2, :])
        np.testing.assert_array_equal(selected.Qv, Qv[:, 2, :])


if __name__ == "__main__":
    unittest.main()
