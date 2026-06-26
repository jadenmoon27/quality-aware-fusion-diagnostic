#!/usr/bin/env bash
set -euo pipefail
: "${UNION:?Set UNION=/path/to/stressid_union.npz}"
: "${SPLITS:?Set SPLITS=/path/to/splits}"
: "${QUALITY:?Set QUALITY=/path/to/fold_scaled_quality}"
: "${OUT:?Set OUT=/path/to/output_dir}"
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
SEEDS=(11 22 33 44 55)
FOLDS=(0 1 2 3 4)
python -m qfd.stressid.dump_unimodal_preds --union_npz "$UNION" --splits_dir "$SPLITS" --out_root "$OUT/unimodal_preds" --model lr --seeds "${SEEDS[@]}" --folds "${FOLDS[@]}" --overwrite --write_summary_json
python -m qfd.stressid.precompute_brokenq --union_npz "$UNION" --splits_dir "$SPLITS" --q_clean_root "$QUALITY" --out_broken_root "$OUT/brokenQ_K200" --seeds "${SEEDS[@]}" --folds "${FOLDS[@]}" --K 200 --require_full_coverage
python -m qfd.stressid.compute_competitiveness_stats --union_npz "$UNION" --splits_dir "$SPLITS" --preds_root "$OUT/unimodal_preds/lr" --seeds "${SEEDS[@]}" --folds "${FOLDS[@]}" --out_csv "$OUT/reports/competitiveness_lr.csv" --out_json "$OUT/reports/competitiveness_lr.json"
python -m qfd.stressid.late_fusion_identifiability --family LR --union_npz "$UNION" --splits_dir "$SPLITS" --preds_root "$OUT/unimodal_preds/lr" --q_clean_root "$QUALITY" --q_broken_root "$OUT/brokenQ_K200" --seeds "${SEEDS[@]}" --folds "${FOLDS[@]}" --K 200 --require_full_coverage --out_dir "$OUT/reports/latefusion_lr" --write_tex
python -m qfd.stressid.moe_identifiability --family MoE --union_npz "$UNION" --splits_dir "$SPLITS" --preds_root "$OUT/unimodal_preds/lr" --q_clean_root "$QUALITY" --q_broken_root "$OUT/brokenQ_K200" --seeds "${SEEDS[@]}" --folds "${FOLDS[@]}" --K 200 --require_full_coverage --router_train_full_only_train --out_dir "$OUT/reports/moe_lr" --write_tex
