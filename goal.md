# Project goal: PC-ALM and local-credit learning on FineWeb

Lead a reproducible, paper-quality study of predictive-coding and related local-credit training methods using the `catid/pcalm` repository and the four-host cluster (`foureyes`, `ripper`, `work`, and `loki`). The final comparison must use one frozen approximately 40M-parameter causal Transformer, one pinned tokenizer and FineWeb stream, identical sequence length, precision, data order, evaluation protocol, checkpoint schedule, and exactly 1,000,000,000 training tokens. Report quality, throughput, wall-clock time, GPU-hours, memory, communication, stability, locality, and implementation tradeoffs. Failed and unstable methods remain visible results.

## Non-negotiable gates

- Inventory all hosts, runtimes, GPUs, topology, storage, SSH, clocks, and rendezvous connectivity from the real cluster; never invent hardware facts.
- Measure local and cross-node communication and step scaling before selecting DDP. Use independent jobs when Ethernet scaling is poor.
- Do not start a 1B-token run until the engineering gate, correctness tests, optimized/reference equivalence, and method-registry smokes pass. Keep the 1B campaign blocked while any required gate is unresolved.
- Record every run's commit, configuration, command line, environment lock, host/GPU inventory, dataset/tokenizer identifiers, rank mapping, NCCL/PyTorch settings, checkpoints, and exact post-mask token count in `runs/registry.jsonl`.
- Keep all runs resumable and preserve failed runs, loader failures, orchestration bugs, NaNs, divergence, and missing dependencies.

## Frozen model and data contract

Use one pre-norm RMSNorm causal residual Transformer with tied input/output embeddings, 16 blocks, width 384, six 64-dimensional heads, MLP width 1,536, RoPE, context 1,024 unless a measured memory/throughput gate supports 2,048, vocabulary 32,000, and BF16 production training. Declare and freeze the exact parameter count in the 38–42M interval. Use next-token cross-entropy with identical masking and normalization. Pin FineWeb (`HuggingFaceFW/fineweb`, `sample-10BT`) and the Mistral tokenizer, stream deterministically, reserve at least 10M held-out validation tokens, and verify exactly 1B post-shift training targets at checkpoints. Short budgets (1M, 10M, 50M, 100M, 250M) must be labeled pilots.

## Common method interface and comparison arms

Every method implements `init_state`, `relax_step`, `local_update`, `diagnostics`, and `step_cost`, with a slow reference path and an optimized path, a version identifier, and explicit `block_local` or `strict_local` locality. The required ten directions are:

1. faithful PC-ALM with primal hidden updates, dual updates, local parameter updates, output cross-entropy error, residual/dual diagnostics, equilibrium checks, and no global reverse graph;
2. error-based PC (ePC);
3. Meta-prediction-error PC (Meta-PCN), including normalized error statistics and a PC-ALM combination;
4. precision-weighted deep PC, with and without dual variables where practical;
5. stable parameterization/initialization wrapper with factorial ablations;
6. WF-Act-PC/local Jacobian-transpose feedback;
7. closed-form precision or hierarchical Gaussian-filter PC;
8. direct Kolen–Pollack feedback alignment/one-step PC;
9. stream-amortized or temporal carry-over PC with delayed-copy leakage checks;
10. PC-ALM plus activation-space control-variate probes, clearly labeled as a new hybrid.

Run ordinary AdamW BP, activation-checkpointed BP, standard PC-BP, iPC, and any precisely mapped Noise control as reference controls. Method cards must state literature source, equations, boundary information, local-autograd status, state/dual memory, failure modes, and reproduced versus adapted components. Standard causal attention is depth-block-local but not strict token-local.

## Correctness and performance evidence

Maintain analytical linear and nonlinear finite-difference checks, Transformer masking/shift/tied-embedding checks, checkpoint/resume and deterministic sharding checks, single-versus-multi-GPU equivalence, no-global-backward instrumentation, and NaN/Inf/dual/error stability checks. Profile before optimizing; preserve eager references; validate compilation, fused kernels, and CUDA graphs against FP32/production-precision references. Record local-update counts, inference iterations, residuals, dual/error norms, gradient/update cosine to BP, memory peaks, communication, and kernel timing.

## Staged campaign and deliverables

Complete Phase 0 inventory and engineering reports, then correctness and 1–10M pilots, a 50–100M screen, 100–250M finalist replication, and only then the gated 1B head-to-head. Use independent jobs and seeds to keep all eligible GPUs useful without filler work. Produce `reports/engineering_gate.md`, `reports/plan.md`, `reports/bibliography.md`, method cards, machine-readable gates and registry records, raw JSONL/CSV metrics, plots under `reports/figures/`, `reports/results.csv`, `reports/results.json`, and a paper-quality `reports/final_report.md` with quality/compute/memory tables, confidence intervals, failed-run annotations, limitations, and recommendations for quality, time-to-quality, memory, strict locality, and follow-up value. Keep `README.md` reproducible.

Current operating rule: continue measured, declared pilots and replication work on the cluster, but do not launch any 1B-token campaign until every required gate and the complete method registry are passed.
