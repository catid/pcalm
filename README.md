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
$PY scripts/test_manifest.py
$PY scripts/phase0_inventory.py
CUDA_VISIBLE_DEVICES=0 /home/catid/venvs/dnn/bin/torchrun --standalone --nproc-per-node=1 scripts/ddp_equivalence.py --output /tmp/ddp1.json
CUDA_VISIBLE_DEVICES=0,1 /home/catid/venvs/dnn/bin/torchrun --standalone --nproc-per-node=2 scripts/ddp_equivalence.py --output /tmp/ddp2.json
for n in 1 2 4; do CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((n-1))) \
  /home/catid/venvs/dnn/bin/torchrun --standalone --nproc-per-node=$n scripts/bench_scaling.py; done
```

The local and direct-rank cross-node all-reduce measurements are in `reports/scaling.csv`; the measured Ethernet rates make independent 4-GPU jobs the default. Do not launch a 1B-token run while the engineering gate says `BLOCKED`.

The fixed-shape optimization gate can be run without a long training job:

```bash
# CPU harness smoke (does not claim CUDA-graph support)
PYTHONPATH=. /home/catid/venvs/dnn/bin/python scripts/bench_optimization_gate.py \
  --mode ddp --device cpu --tiny --seq-len 16 --batch-sizes 1,2 \
  --steps 2 --warmup 1 --output /tmp/ddp_gate.json

# Production-shape single-node batch benchmark
PYTHONPATH=. /home/catid/venvs/dnn/bin/python scripts/bench_optimization_gate.py \
  --mode ddp --device cuda --seq-len 128 --batch-sizes 1,2,4 \
  --steps 10 --warmup 3 --output reports/ddp_batch_gate.json

# CUDA graph capture/replay (fixed shape; run only when a GPU is available)
PYTHONPATH=. /home/catid/venvs/dnn/bin/python scripts/bench_optimization_gate.py \
  --mode graph --device cuda --seq-len 128 --batch-sizes 1 \
  --steps 10 --warmup 3 --output reports/cuda_graph_gate.json
```

For local multi-GPU DDP, launch the same `ddp` command with
`torchrun --standalone --nproc-per-node=N`; the output records world size,
rank hosts, peak memory, and the maximum median step time across ranks.

## Pilot and final-run commands

Create a manifest before either command. A bounded 1M-token BP pilot (diagnostic only) is:

```bash
PY=/home/catid/venvs/dnn/bin/python
$PY scripts/run_manifest.py --run-id pilot-bp-1m --method bp_adamw --phase 1 --tokens 1000000
$PY scripts/train.py --run-id pilot-bp-1m --method bp_adamw --source fineweb --validation-tokens 0 --tokens 1000000
```

For tuning runs, pass the resolved training settings and complete training
command when creating the manifest. For example:

```bash
$PY scripts/run_manifest.py --run-id tune-pc-10m --method pc_alm \
  --seed 20264100 --metadata-json '{"inner_steps":4,"eta_h":0.05,"alpha":0.01,"rho":1.0,"parameter_step":0.0003}' \
  --train-command 'scripts/train_local.py --run-id tune-pc-10m --method pc_alm --tokens 10000000 --inner-steps 4 --eta-h 0.05 --alpha 0.01 --rho 1.0 --parameter-step 0.0003'
```

`--train-command`, `--seed`, and `--training-metadata-json` are provenance
fields; they do not launch the training process.

The final command is intentionally guarded and will refuse the current blocked gate. After every required gate and method smoke has passed, the recorded command is:

```bash
$PY scripts/run_manifest.py --run-id final-bp-1b-seed0 --method bp_adamw --phase 4 --tokens 1000000000
$PY scripts/train.py --run-id final-bp-1b-seed0 --method bp_adamw --source fineweb \
  --validation-tokens 10000000 --tokens 1000000000 --allow-1b --checkpoint-every 10000000
```

For a read-only provenance audit of active processes and registry/results
consistency, run `$PY scripts/audit_registry.py`. It records a snapshot in
`reports/registry_audit.json` and never signals training jobs.
