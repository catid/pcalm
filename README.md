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

The initial repository audit is in [`reports/repo_audit.md`](reports/repo_audit.md), the cluster inventory is in [`reports/phase0_cluster.md`](reports/phase0_cluster.md), and the current gate is [`reports/engineering_gate.md`](reports/engineering_gate.md). The remote repository was empty at project start; the local Git history is the reproducible source of truth until a remote commit is available.

## Phase 0 commands

Use the pinned local environment on foureyes:

```bash
PY=/home/catid/venvs/dnn/bin/python
$PY scripts/data_manifest.py
$PY scripts/run_reference_tests.py
$PY scripts/phase0_inventory.py
for n in 1 2 4; do CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((n-1))) \
  /home/catid/venvs/dnn/bin/torchrun --standalone --nproc-per-node=$n scripts/bench_scaling.py; done
```

The 1–4 GPU benchmark is local to foureyes. Cross-node 8/12 GPU commands are intentionally omitted until every worker has the same pinned runtime and the source tree copied or checked out. Do not launch a 1B-token run while the engineering gate says `BLOCKED`.
