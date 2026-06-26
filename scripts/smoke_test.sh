#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
OUT="${1:-$ROOT/.smoke_out}"
rm -rf "$OUT"
python "$ROOT/tests/create_toy_contract.py" --out "$OUT/data"
python -m qfd.stressid.dump_unimodal_preds \
  --union_npz "$OUT/data/union/toy_union.npz" \
  --splits_dir "$OUT/data/splits" \
  --out_root "$OUT/unimodal_preds" \
  --model lr \
  --seeds 11 --folds 0 \
  --overwrite --write_summary_json
python -m qfd.stressid.precompute_brokenq \
  --union_npz "$OUT/data/union/toy_union.npz" \
  --splits_dir "$OUT/data/splits" \
  --q_clean_root "$OUT/data/quality" \
  --out_broken_root "$OUT/brokenQ_K3" \
  --seeds 11 --folds 0 --K 3 --require_full_coverage
python -m qfd.stressid.verify_brokenq_artifacts \
  --union_npz "$OUT/data/union/toy_union.npz" \
  --splits_dir "$OUT/data/splits" \
  --q_clean_root "$OUT/data/quality" \
  --q_broken_root "$OUT/brokenQ_K3" \
  --seed 11 --fold 0 --require_full_coverage
python -m qfd.stressid.late_fusion_identifiability \
  --family LR-toy \
  --union_npz "$OUT/data/union/toy_union.npz" \
  --splits_dir "$OUT/data/splits" \
  --preds_root "$OUT/unimodal_preds/lr" \
  --q_clean_root "$OUT/data/quality" \
  --q_broken_root "$OUT/brokenQ_K3" \
  --seeds 11 --folds 0 --K 3 --require_full_coverage \
  --out_dir "$OUT/reports/latefusion_lr"
echo "Smoke test completed: $OUT"
