# Quality-Aware Multimodal Fusion Diagnostic

This repository contains a cleaned, public-facing reproduction pipeline for the paper:

**When Does Quality-Aware Multimodal Fusion Matter? A Leakage-Safe Diagnostic for Decision-Level Dependence**

The code implements a diagnostic for whether a trained multimodal fusion rule actually depends on instance-aligned quality/reliability values at inference time. The key intervention is:

1. freeze unimodal experts and the fusion rule,
2. keep evidence `E` and availability masks `M` fixed,
3. compare matched quality values (`Clean-Q`) against test-time, present-only permutations (`Broken-Q`).

If quality is decision-relevant, Clean-Q should outperform Broken-Q. If predictions are invariant, the trained fusion rule is not relying on quality-instance alignment.

## Repository status

This repository is prepared from the uploaded experiment ZIPs and is organized for public release. It does **not** include restricted datasets, raw StressID/CMU-MOSEI files, extracted embeddings, subject split files, or paper result artifacts. Those should stay out of Git and be stored externally.

The main reproducibility boundary is prepared UNION, split, quality, and unimodal-prediction artifacts. The public StressID code does not claim to reproduce raw audio/video/physiology feature extraction from the restricted source data.

A synthetic smoke test is included and has been run successfully.

```bash
bash scripts/smoke_test.sh .smoke_out
```

## What is included

```text
src/qfd/_shared/          Strict data/Q contracts for StressID and CMU-MOSEI
src/qfd/stressid/         StressID unimodal, Broken-Q, late-fusion, MoE, controls
src/qfd/mosei/            CMU-MOSEI boundary-case builders and diagnostics
src/qfd/diagnostics/      Failure-mode alignment / Q-error diagnostics
scripts/                  Smoke test and reproduction shell entry points
tests/                    Synthetic toy contract generator for smoke testing
docs/                     Data contract, pipeline notes, code audit
```

## What you must provide for real reproduction

### StressID contract

The StressID pipeline starts from a prepared UNION `.npz` file. This is the correct public boundary because the raw preprocessing used Colab/HPC-specific extraction and because the dataset itself is externally governed.

Expected UNION keys:

```text
ids                shape (N,)
y or y2             shape (N,), binary labels
E_a, E_v, E_p       shape (N, d_m), audio/video/physio embeddings
M_a, M_v, M_p       shape (N,), binary availability masks
```

Expected split layout:

```text
splits/
  seed_11/
    train_ids_fold0.npy
    test_ids_fold0.npy
    ...
  seed_22/
  seed_33/
  seed_44/
  seed_55/
```

Expected fold-safe Q layout:

```text
quality/
  seed_11/
    fold_0.npz                       # or union_quality_Q_*_seed11_fold0.npz
    fold_1.npz
    ...
```

Each Q file must contain UNION-aligned `Qa`, `Qv`, `Qp` arrays, either shape `(N,)` or `(N, d)`. Missing rows must be exactly zero where the corresponding `M_m == 0`.

Expected unimodal prediction layout:

```text
unimodal_preds/
  lr/
    seed_11/
      fold_0.npz
      fold_1.npz
      ...
```

Each prediction file must contain UNION-aligned `ids`, `y`, `train_mask`, `test_mask`, and `p_a/p_v/p_p`. Posterior arrays are finite where the modality is present and `NaN` where the modality is missing.

### CMU-MOSEI contract

The MOSEI scripts can build the UNION and Q files from CMU Multimodal SDK artifacts, but raw data are not included. The MOSEI package mirrors the StressID contract using modalities `[language, audio, visual]` and keys `E_l/E_a/E_v`, `M_l/M_a/M_v`, `Ql/Qa/Qv`.

## Installation

```bash
git clone <your-repo-url>
cd quality-aware-fusion-diagnostic
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

For MOSEI raw-building scripts, install the CMU Multimodal SDK separately. The smoke test does not require MOSEI.

## Smoke test

Exact command:

```bash
bash scripts/smoke_test.sh .smoke_out
```

The smoke test creates a synthetic UNION table, splits, and fold-safe Q; trains LR unimodal experts; precomputes Broken-Q; verifies Broken-Q; and runs late-fusion identifiability. It is not intended to reproduce paper numbers.

## Minimal StressID reproduction pipeline

Set paths:

```bash
export UNION=path/to/new_fusion_table_union_y2_avp_v1_noQ.npz
export SPLITS=path/to/splits
export QUALITY=path/to/quality
export OUT=outputs/stressid
```

Exact pipeline command:

```bash
UNION="$UNION" SPLITS="$SPLITS" QUALITY="$QUALITY" OUT="$OUT" bash scripts/run_stressid_pipeline.sh
```

Train unimodal experts and dump UNION-aligned probabilities:

```bash
python -m qfd.stressid.dump_unimodal_preds \
  --union_npz "$UNION" \
  --splits_dir "$SPLITS" \
  --out_root "$OUT/unimodal_preds" \
  --model lr \
  --seeds 11 22 33 44 55 \
  --folds 0 1 2 3 4 \
  --overwrite --write_summary_json
```

Precompute leakage-safe Broken-Q banks:

```bash
python -m qfd.stressid.precompute_brokenq \
  --union_npz "$UNION" \
  --splits_dir "$SPLITS" \
  --q_clean_root "$QUALITY" \
  --out_broken_root "$OUT/brokenQ_K200" \
  --seeds 11 22 33 44 55 \
  --folds 0 1 2 3 4 \
  --K 200 \
  --require_full_coverage
```

Audit Broken-Q alignment:

```bash
python -m qfd.stressid.verify_brokenq_artifacts \
  --union_npz "$UNION" \
  --splits_dir "$SPLITS" \
  --q_clean_root "$QUALITY" \
  --q_broken_root "$OUT/brokenQ_K200" \
  --seed 11 --fold 0 \
  --require_full_coverage
```

Run late-fusion identifiability:

```bash
python -m qfd.stressid.late_fusion_identifiability \
  --family LR \
  --union_npz "$UNION" \
  --splits_dir "$SPLITS" \
  --preds_root "$OUT/unimodal_preds/lr" \
  --q_clean_root "$QUALITY" \
  --q_broken_root "$OUT/brokenQ_K200" \
  --seeds 11 22 33 44 55 \
  --folds 0 1 2 3 4 \
  --K 200 \
  --require_full_coverage \
  --out_dir "$OUT/reports/latefusion_lr" \
  --write_tex
```

Run MoE identifiability:

```bash
python -m qfd.stressid.moe_identifiability \
  --family "MoE" \
  --union_npz "$UNION" \
  --splits_dir "$SPLITS" \
  --preds_root "$OUT/unimodal_preds/lr" \
  --q_clean_root "$QUALITY" \
  --q_broken_root "$OUT/brokenQ_K200" \
  --seeds 11 22 33 44 55 \
  --folds 0 1 2 3 4 \
  --K 200 \
  --require_full_coverage \
  --router_train_full_only_train \
  --out_dir "$OUT/reports/moe_lr" \
  --write_tex
```

## Positive controls

The repository includes positive-control scripts for synthetic or correctness-aligned quality signals. These are diagnostic upper bounds, not deployable inference procedures:

```bash
python -m qfd.stressid.precompute_qsyn_decisional_corruption --help
python -m qfd.stressid.decisional_corruption_qsyn --help
python -m qfd.stressid.precompute_qcal_loss_predictor --help
python -m qfd.stressid.sufficiency_qcal_loss_router --help
```

## Git hygiene

Do not commit:

```text
*.npz
*.npy
outputs/
paper_output/
quality/
splits/
data/
```

These are ignored in `.gitignore`.

## Citation

If you use this repository, cite the repository metadata in `CITATION.cff` and the associated paper:

```text
When Does Quality-Aware Multimodal Fusion Matter? A Leakage-Safe Diagnostic for Decision-Level Dependence
INTERSPEECH 2026 diagnostic paper
```

## License

A license has intentionally not been finalized here. Before public release, choose a license with all coauthors/supervisors. MIT or BSD-3-Clause is typical for academic research code, but this should be a project decision.
