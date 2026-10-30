# Engineering gate (Phase 0)

Date: 2026-10-29 UTC. This gate is a living record; a missing or failed item blocks the 1B-token campaign.

| Gate | Result | Evidence / limitation |
|---|---|---|
| Repository and plan | PASS | `reports/repo_audit.md`, `reports/plan.md`, local Git history beginning at `b3e963b` |
| Host/GPU/SSH/clock inventory | PASS for reachability | `reports/phase0_cluster.md`; all four hosts reachable, clocks synchronized, 12 GPUs identified |
| Persistent rendezvous TCP | PASS | `foureyes:29517` accepted clients from ripper/work/loki |
| Python/runtime lock | PARTIAL | `configs/environment.lock`; exact 5.8 GiB `/home/catid/venvs/dnn` was copied to ripper/work/loki and all now import `torch 2.12.0+cu130` (source-tree copies are under `/var/tmp/pcalm`), but package hashes and long-term image provisioning still need locking |
| Tokenizer/FineWeb pin | PASS metadata | `reports/data_manifest.json`; Mistral tokenizer vocab 32,000 and SHA256 recorded; FineWeb `sample-10BT` builder metadata recorded |
| Architecture/count | PASS | `configs/base.json`; 40,667,520 parameters with tied embeddings, 16 blocks, 384 width, six heads, 1,024 context |
| Reference mathematical/Transformer smoke | PASS | `scripts/run_reference_tests.py`, `reports/correctness.json`: parameter interval/tie, causal masking, next-token loss, tied-embedding gradient, linear KKT/finite difference, PC state diagnostics, and no-global-backward relaxation instrumentation |
| Local eager benchmark | PASS | `reports/phase0_inventory.json`: BF16/CUDA-capable model forward, 165,274 tokens/s at B=1,T=1024, peak allocated 985 MB (warm eager reference) |
| NCCL all-reduce 1/2/4 GPU | PASS, local only | `reports/scaling.csv`: 20 MB payload median 2,261 GB/s (world 1 baseline), 34.4 GB/s (world 2), 21.3 GB/s (world 4) on foureyes |
| NCCL cross-node 2/4/8/12 GPU | MEASURED, poor scaling | `reports/scaling.csv`, `reports/cross_node_attempt.md`: direct-rank Ethernet payload rates 11.1, 7.1, 6.3, 5.7 MB/s; use independent 4-GPU jobs by default |
| DDP single/2-GPU equivalence | PARTIAL PASS | `scripts/ddp_equivalence.py`: one fixed batch gives identical loss 10.4074726105 and parameter checksum 12852.84765625 at world 1 and local world 2; rank-shard and cross-node DDP equivalence remain pending |
| Dataloader/token counting | PASS (isolated worker) | `scripts/fineweb_worker.py`, `scripts/train.py`, `reports/data_smoke_worker.json`: the datasets/pyarrow reader is isolated in a subprocess that exits via `os._exit`; 2,046-token, resume-from-1,023, and 10,240-token live FineWeb BP smokes reached exact targets with exit 0. The worker now reserves 9,776 chunks (10,000,848 post-shift targets) by default before training; live validation evaluation remains pending. Original in-process reader failures remain recorded in `reports/loader_failure.md`. |
| Optimized/compiled/CUDA graph equivalence | PASS for fixed-shape compile path | `reports/optimized_equivalence.json`: CUDA eager versus `torch.compile` on identical weights/data, two optimizer steps; max loss error 4.77e-7 and max weight error 1.15e-5. CUDA graphs and fused kernels remain pending. |
| PC-ALM optimized relaxation | PASS reference equivalence and measured speedup | `reports/optimized_local_benchmark.json`: preallocated/in-place path matches the reference within 1.49e-8 hidden and 1.86e-9 dual absolute error and is 2.42x faster on the CUDA tiny benchmark. |
| Fixed-shape DDP/graph harness | CPU harness PASS; production CUDA evidence pending | `scripts/bench_optimization_gate.py` passes tiny fixed-shape CPU DDP and records structured CUDA-graph skip on CPU. Run frozen 40M CUDA DDP batch scaling and graph capture when a GPU slot frees; do not infer production speed from the tiny harness. |
| PC-ALM local method smoke | PASS for reference arm | `scripts/smoke_pcalm.py` passed on CUDA with detached latent/dual relaxation, nonzero local gradients, and global `Tensor.backward` instrumentation. |
| Ten-arm method registry | REGISTERED, not passed | `configs/method_registry.json`, `reports/method_registry_smoke.json`; PC-ALM, adapted ePC, and stable wrapper protocol smokes pass, while all required 10M-token method smokes and the remaining arms are pending |

## Measured bottleneck and next action

The immediate bottleneck is cross-node Ethernet communication, followed by local-method throughput and exact-budget FineWeb resume/validation checks. The runtime is now parity-tested and the source tree is copied to worker NVMe. Foureyes local all-reduce falls from 34.4 to 21.3 GB/s at two versus four ranks for the measured 20 MB bucket, while cross-node payload rates fall below 6 MB/s by world 8–12. Independent 4-GPU jobs therefore maximize useful progress without filler communication.

**1B-token status: BLOCKED.** No expensive 1B-token run has been launched.
