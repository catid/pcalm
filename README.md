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

The local and direct-rank cross-node all-reduce measurements are in `reports/scaling.csv`; the measured Ethernet rates make independent 4-GPU jobs the default. Do not launch a 1B-token run while the engineering gate says `BLOCKED`.

## Pilot and final-run commands

Create a manifest before either command. A bounded 1M-token BP pilot (diagnostic only) is:

```bash
PY=/home/catid/venvs/dnn/bin/python
$PY scripts/run_manifest.py --run-id pilot-bp-1m --phase 1 --tokens 1000000
$PY scripts/train.py --run-id pilot-bp-1m --method bp_adamw --source fineweb --tokens 1000000
```

The final command is intentionally guarded and will refuse the current blocked gate. After every required gate and method smoke has passed, the recorded command is:

```bash
$PY scripts/run_manifest.py --run-id final-bp-1b-seed0 --phase 4 --tokens 1000000000
$PY scripts/train.py --run-id final-bp-1b-seed0 --method bp_adamw --source fineweb \
  --tokens 1000000000 --allow-1b --checkpoint-every 10000000
```
