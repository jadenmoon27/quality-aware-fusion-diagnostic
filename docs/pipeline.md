# Reproduction pipeline

## StressID

1. Prepare UNION embeddings and masks.
2. Prepare subject-disjoint splits.
3. Prepare fold-safe Q files.
4. Dump unimodal posteriors with `qfd.stressid.dump_unimodal_preds`.
5. Precompute Broken-Q banks with `qfd.stressid.precompute_brokenq`.
6. Run structural diagnostics with `qfd.stressid.compute_competitiveness_stats` and `qfd.stressid.quality_alignment_rho`.
7. Run Clean-Q/Broken-Q diagnostics with:
   - `qfd.stressid.late_fusion_identifiability`
   - `qfd.stressid.moe_identifiability`
8. Run positive controls only as diagnostic upper bounds.

## CMU-MOSEI

1. Use `qfd.mosei.build_union` to build the UNION table from SDK artifacts.
2. Use `qfd.mosei.make_splits` to create video-disjoint splits.
3. Use `qfd.mosei.build_rawq` and `qfd.mosei.fold_scale_q` to create fold-safe Q.
4. Use `qfd.mosei.dump_unimodal_preds` to create posterior contracts.
5. Use `qfd.mosei.precompute_brokenq`, `qfd.mosei.late_fusion_permtest`, and `qfd.mosei.moe_identifiability_permtest` for the boundary-case diagnostic.

`qfd.mosei.precompute_brokenq` writes a K-permutation bank under the requested output root. The MOSEI permutation-test scripts load that bank directly and still support older `perm_###` directory layouts.

## HPC / Slurm usage pattern

The scripts are pure CLI modules. On an HPC cluster, use one Slurm job per stage and persist outputs between stages. The most important rule is to never regenerate splits or Q inside an experiment script.

Before touching real data, run `bash scripts/smoke_test.sh .smoke_out`. To run the standard StressID stages in order, use `UNION=... SPLITS=... QUALITY=... OUT=... bash scripts/run_stressid_pipeline.sh`.
