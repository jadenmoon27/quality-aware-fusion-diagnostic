# Code audit from uploaded ZIPs

## Source ZIPs inspected

- `FINAL.zip`: compact final scripts organized into `v5_1`, `v5_2`, `v5_3`.
- `FINAL_EXPERIMENTS.zip`: larger exploratory archive with StressID, CMU-MOSEI, diagnostics, sufficiency, and tooling folders.

## Public repo selection policy

Included:

- strict UNION/split/Q contract loaders,
- unimodal posterior dumpers,
- Broken-Q precomputation and auditors,
- late-fusion and MoE identifiability scripts,
- positive-control scripts supporting the paper's validation claim,
- CMU-MOSEI boundary-case scripts,
- failure-mode alignment diagnostics.

Excluded:

- `__MACOSX` metadata,
- `__pycache__` bytecode,
- duplicate `FINAL_EXPERIMENTS/FINAL/...` copies,
- exploratory scripts without a clear role in the final paper pipeline,
- generated outputs and private absolute paths,
- data files and embeddings.

## Mechanical checks completed

- `python -m compileall -q src` passes.
- `python -m compileall -q src tests` passes.
- `python -m unittest discover -s tests` passes.
- `bash scripts/smoke_test.sh .smoke_out` passes on synthetic data.
- Broken-Q 3D banks `(N,K,d)` are supported in the StressID contract and identifiability scripts.
- MOSEI permutation-test scripts load the public K-bank output from `qfd.mosei.precompute_brokenq` and retain compatibility with older `perm_###` layouts.
