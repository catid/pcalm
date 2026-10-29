# PC-ALM and local-credit learning on FineWeb: execution plan

**Date:** 2026-10-29 UTC  
**Controller:** `foureyes`, `/home/catid/pcalm`  
**Repository:** `git@github.com:catid/pcalm.git` (the remote was empty at project start; local history begins with the audited skeleton)

This study is gated. The 1B-token campaign remains disabled until the environment, distributed scaling, reference implementation, mathematical checks, Transformer/data checks, and method registry are recorded as passing in `reports/engineering_gate.md` and the machine-readable gate record.

## Phase 0: inventory and engineering gate

1. Capture Git commit, Python/package/CUDA/driver versions, GPU inventory, memory, CPU/RAM/storage, topology, SSH reachability, clock skew, source-tree distribution, and rendezvous connectivity for `foureyes`, `ripper`, `work`, and `loki`.
2. Pin the model contract: a single pre-norm causal Transformer, tied embeddings, six 64-wide heads, 16 blocks, width/MLP selected in the 38–42M parameter interval, one tokenizer, BF16 production precision, and a declared context length.
3. Pin FineWeb identifiers, tokenizer identifiers, deterministic sharding/token counting, validation stream, and the run manifest format. No 1B run can start without these identifiers and hashes.
4. Implement the slow reference Transformer and one method interface (`init_state`, `relax_step`, `local_update`, `diagnostics`, `step_cost`), then measure eager forward, local-update, data-loader, and all-reduce throughput on a representative short workload.
5. Write a checked bibliography. Unavailable papers or repositories are recorded as unavailable; adaptations are never presented as faithful reproductions.

**Phase 0 exit:** every reachable host has an inventory record; the reference model runs deterministically; the distributed smoke test has no duplicate token ranges; all required mathematical and Transformer correctness tests pass; and the engineering report records measured bottlenecks and the chosen launcher/topology. A failed or unreachable host is a documented limitation, not silently omitted.

## Phase 1: correctness and pilots (1–10M tokens)

Use `work` and `loki` for analytical, finite-difference, eager/compiled, and method smoke tests. Use one 4-GPU production DDP smoke path on `foureyes` or `ripper`. Each arm gets a 10M-token smoke test before screening. These runs are diagnostic only and cannot be reported as 1B-token results.

## Phase 2: ten-arm screen (50–100M tokens)

Run the ten requested arms plus BP, checkpointed BP, standard PC, and iPC controls at one seed each, with identical architecture, tokenizer, sequence length, token checkpoints, validation stream, and accounting. Rebalance only from measured throughput and record the actual host/world-size assignment in `runs/registry.jsonl`.

## Phase 3: finalists (100–250M tokens)

Select by a predeclared combination of validation loss, tokens/s, GPU-hours, stability, and local-credit diagnostics. Run at least three seeds for the top five, including BP and standard PC. Preserve failed runs and reasons.

## Phase 4: final comparison (exactly 1B training tokens)

Run three seeds for the top three where compute permits and at least one complete run for every other finalist, always including BP, standard PC, and faithful PC-ALM. The final run is blocked until Phase 0–3 gates and the method registry are complete. Every checkpoint verifies the exact post-mask token count and can resume from its last valid checkpoint.

## Phase 5: targeted ablations and report

For leading methods vary one factor at a time (inference iterations, dual/parameter steps, precision, sequence length, batch, initialization, residual scaling, state/dual precision, CUDA graphs, and world size). Produce paired-token plots, confidence intervals over seeds, failed-run annotations, raw JSONL/CSV metrics, and the final recommendation by quality, time-to-quality, memory, strict-local-credit status, and follow-up value.

## Immediate bounded work

The first implementation pass prioritizes a correct, measurable reference over kernel speculation: model parameter audit, tiny deterministic data, local PC-ALM state/update equations, analytical tests, and one representative forward/relax/update benchmark. Only measured bottlenecks will receive compilation or fusion work. No filler GPU jobs are used while dependencies or correctness gates are unresolved.
