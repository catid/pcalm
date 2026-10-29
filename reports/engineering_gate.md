# Engineering gate (Phase 0)

Date: 2026-10-29 UTC. This gate is a living record; a missing or failed item blocks the 1B-token campaign.

| Gate | Result | Evidence / limitation |
|---|---|---|
| Repository and plan | PASS | `reports/repo_audit.md`, `reports/plan.md`, local Git history beginning at `b3e963b` |
| Host/GPU/SSH/clock inventory | PASS for reachability | `reports/phase0_cluster.md`; all four hosts reachable, clocks synchronized, 12 GPUs identified |
| Persistent rendezvous TCP | PASS | `foureyes:29517` accepted clients from ripper/work/loki |
| Python/runtime lock | PARTIAL | `configs/environment.lock`; foureyes has `/home/catid/venvs/dnn` (`torch 2.12.0+cu130`), workers lack a matching environment; ripper has an unpinned source build and work has a development torch |
| Tokenizer/FineWeb pin | PASS metadata | `reports/data_manifest.json`; Mistral tokenizer vocab 32,000 and SHA256 recorded; FineWeb `sample-10BT` builder metadata recorded |
| Architecture/count | PASS | `configs/base.json`; 40,667,520 parameters with tied embeddings, 16 blocks, 384 width, six heads, 1,024 context |
| Reference mathematical/Transformer smoke | PASS | `scripts/run_reference_tests.py`: parameter interval/tie, causal masking, next-token loss, PC state diagnostics |
| Local eager benchmark | PASS | `reports/phase0_inventory.json`: BF16/CUDA-capable model forward, 39,294 tokens/s at B=1,T=128 (warm eager reference) |
| NCCL all-reduce 1/2/4 GPU | PASS, local only | `scripts/bench_scaling.py`: 20 MB median all-reduce 2,261 GB/s (world 1), 34.4 GB/s (world 2), 21.3 GB/s (world 4) on foureyes |
| NCCL cross-node 8/12 GPU | BLOCKED | Worker environments/source trees are not pinned; do not infer scaling from a failed launch |
| Dataloader/token counting | NOT YET PASSED | Streaming metadata is pinned, but deterministic sharding/token-count implementation and 10M validation isolation remain to be tested |
| Optimized/compiled/CUDA graph equivalence | NOT YET PASSED | Eager reference exists; no optimization is admitted before equivalence tests |
| Ten-arm method registry | REGISTERED, not passed | `configs/method_registry.json`; only PC-ALM reference scaffold exists; all arms require cards, tests, and 10M smoke |

## Measured bottleneck and next action

The immediate bottleneck is worker runtime parity, followed by deterministic FineWeb tokenization/sharding. The useful optimization is to provision one pinned environment (or a digest-pinned CUDA image) on all workers and copy the source tree, then rerun the actual model-shaped NCCL and DDP equivalence benchmark. Foureyes local all-reduce scaling already falls from 34.4 to 21.3 GB/s at two versus four ranks for the measured 20 MB bucket; 8- and 12-GPU choices must wait for cross-node measurements over Ethernet.

**1B-token status: BLOCKED.** No expensive 1B-token run has been launched.
