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
| Reference mathematical/Transformer smoke | PASS | `scripts/run_reference_tests.py`: parameter interval/tie, causal masking, next-token loss, PC state diagnostics |
| Local eager benchmark | PASS | `reports/phase0_inventory.json`: BF16/CUDA-capable model forward, 165,274 tokens/s at B=1,T=1024, peak allocated 985 MB (warm eager reference) |
| NCCL all-reduce 1/2/4 GPU | PASS, local only | `reports/scaling.csv`: 20 MB payload median 2,261 GB/s (world 1 baseline), 34.4 GB/s (world 2), 21.3 GB/s (world 4) on foureyes |
| NCCL cross-node 2/4/8/12 GPU | MEASURED, poor scaling | `reports/scaling.csv`, `reports/cross_node_attempt.md`: direct-rank Ethernet payload rates 11.1, 7.1, 6.3, 5.7 MB/s; use independent 4-GPU jobs by default |
| Dataloader/token counting | FAILED CLEANUP | `src/data_stream.py`, `scripts/smoke_data.py`: live FineWeb streaming and Mistral tokenization produced three deterministic 128-token chunks (381 post-shift targets), but even the data-only process exits 134 with `PyGILState_Release` during datasets/pyarrow finalization. The 2,046-token BP pilot has the same failure; exact budget/resume code is present, clean loader shutdown remains a blocker |
| Optimized/compiled/CUDA graph equivalence | NOT YET PASSED | Eager reference exists; no optimization is admitted before equivalence tests |
| Ten-arm method registry | REGISTERED, not passed | `configs/method_registry.json`; only PC-ALM reference scaffold exists; all arms require cards, tests, and 10M smoke |

## Measured bottleneck and next action

The immediate bottleneck is cross-node Ethernet communication, followed by exact-budget FineWeb resume/validation checks. The runtime is now parity-tested and the source tree is copied to worker NVMe. Foureyes local all-reduce falls from 34.4 to 21.3 GB/s at two versus four ranks for the measured 20 MB bucket, while cross-node payload rates fall below 6 MB/s by world 8–12. Independent 4-GPU jobs therefore maximize useful progress without filler communication.

**1B-token status: BLOCKED.** No expensive 1B-token run has been launched.
