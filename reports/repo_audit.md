# Repository audit (Phase 0)

**Date:** 2026-10-29 UTC  
**Host/cwd:** `foureyes` session, `/home/catid/pcalm`

## Initial state

The project directory existed but was empty (no files, subdirectories, or Git metadata). No nested `AGENTS.md` applies under the project; the user-provided project instructions and `/home/catid/.codex/AGENTS.md` apply. The repository is therefore a clean start, with no inherited implementation, tests, configuration, checkpoints, or run registry to audit.

Required directories were created:

- `configs/` — frozen model/method/run configurations
- `src/` — reference and optimized implementations
- `scripts/` — inventory, benchmark, pilot, and final-run entry points
- `tests/` — mathematical, Transformer, distributed, and stability gates
- `runs/` — registry, manifests, checkpoints (run-specific subdirectories)
- `artifacts/` — dataset/tokenizer caches and profiler outputs
- `reports/` — engineering gate, bibliography, plans, results, and figures

## Local software observations

- `/usr/bin/python3` is Python 3.12.3.
- The default Python environment currently has no importable `torch`, `transformers`, `datasets`, `tokenizers`, or `pytest` modules. Dependency installation/environment selection is a Phase 0 blocker and must be captured in the environment lock before any benchmark.
- `nvidia-smi` is available locally and reports four RTX PRO 6000 Blackwell Max-Q Workstation Edition GPUs. Full driver/CUDA/memory/topology details are being collected by the cluster inventory task.
- Git is available (`2.43.0`), but this directory was not initially a Git repository. Initialize it before recording the first implementation commit.

## Implementation implications

Because no prior code exists, the first implementation should establish a single method protocol (`init_state`, `relax_step`, `local_update`, `diagnostics`, `step_cost`) and a tiny deterministic Transformer/reference path before optimized kernels or distributed execution. Keep dependency installation, exact versions, and source/data identifiers in a committed environment lock. Do not claim a pilot or 1B-token result until the corresponding run manifest and gate records exist.

## Open Phase 0 actions

1. Initialize Git and commit the clean skeleton plus this audit.
2. Discover all authorized hosts and verify SSH, clock, source-tree distribution, rendezvous, GPU/driver/CUDA/PyTorch, storage, and interconnect details.
3. Choose/install a reproducible Python environment, then capture package and compiler locks.
4. Pin tokenizer/FineWeb identifiers and retrieve/check literature/code sources; write `reports/bibliography.md`.
5. Implement the reference model and mathematical gates before scaling or expensive runs.
