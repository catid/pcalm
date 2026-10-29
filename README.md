# PC-ALM and local-credit learning on FineWeb

This repository is a clean-start, phase-gated study of predictive-coding and related local-credit methods on a fixed causal Transformer. Phase 0 begins with cluster/environment discovery, dependency locking, literature/source verification, and engineering/correctness gates. No 1B-token campaign is authorized until those gates pass and the run registry records the approved configuration.

## Layout

- `configs/` frozen model, data, method, and run configurations
- `src/` reference and optimized implementations
- `scripts/` inventory, benchmark, pilot, and final-run commands
- `tests/` mathematical, model, distributed, and stability gates
- `runs/` run manifests, registry, and checkpoints
- `artifacts/` caches and profiler traces
- `reports/` engineering gate, bibliography, results, and figures

The initial repository audit is in [`reports/repo_audit.md`](reports/repo_audit.md). Exact reproduction commands will be added after Phase 0 selects and locks the runtime environment.
