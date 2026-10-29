# PC-ALM local-credit FineWeb study — current report

**Status:** Phase 0 executed; the 1B-token campaign is blocked. This report is an evidence snapshot, not a head-to-head result.

## Fixed contract

The frozen reference is a pre-norm RMSNorm causal Transformer with 16 blocks, width 384, six 64-dimensional heads, MLP width 1,536, RoPE, 1,024 context, tied input/output embeddings, and a 32,000-token Mistral tokenizer. It has 40,667,520 parameters. FineWeb is `HuggingFaceFW/fineweb`, `sample-10BT`; the tokenizer model SHA256 and dataset metadata are in `reports/data_manifest.json`. The intended training budget is exactly 1,000,000,000 post-shift targets with a 10M-token isolated validation stream.

## Environment and topology

All four hosts are reachable as `catid` over SSH: foureyes (4 GPUs), ripper (4), work (2), and loki (2). Every GPU is an RTX PRO 6000 Blackwell with 97,887 MiB and driver 615.71.10. The machines have Ethernet only, no NVLink or InfiniBand, and local GPU topology is PCIe `NODE`. The exact 5.8 GiB Python environment is provisioned on all hosts; each reports torch 2.12.0+cu130, CUDA 13.0, NCCL 2.29.7, and the same torch library hash.

Local foureyes all-reduce is 21.3 GB/s at world size 4 for a 20 MB payload. Direct-rank cross-node rates are 11.1 MB/s (world 2), 7.1 MB/s (world 4), 6.3 MB/s (world 8), and 5.7 MB/s (world 12). Independent 4-GPU jobs are therefore the default useful allocation; a single 8/12-GPU job is not justified by measured communication.

## Correctness and implementation evidence

The reference tests pass parameter counting/tied weights, causal masking, next-token loss normalization, tied-embedding gradients, PC state initialization/diagnostics, a linear PC-ALM KKT check, finite-difference local gradients, no-global-backward relaxation instrumentation, and deterministic rank chunk disjointness. A live FineWeb tokenizer smoke produced three 128-token chunks and 381 post-shift targets. The synthetic BP control completed 2,046 exact targets and wrote resumable checkpoints. A one-step fixed-batch single versus local 2-GPU DDP run has identical loss and parameter checksum.

The initial in-process FineWeb BP pilot consumed exactly 2,046 targets and wrote metrics/checkpoints, then aborted with `PyGILState_Release` during Python finalization in the streaming datasets/pyarrow stack; a retry reproduced it. Those failures remain recorded and excluded from quality claims. The reader is now isolated in `scripts/fineweb_worker.py`, which exits outside the training interpreter. The isolated 2,046-token pilot, resume-from-1,023 checkpoint, and 10,240-token live smoke all completed with exit 0 and exact counts. The training entrypoint now reserves 9,776 chunks (10,000,848 post-shift targets) by default before consuming training tokens; the required live validation evaluation and 10M-per-arm smoke remain pending.

## Method status

The ten-arm registry and method cards are committed. PC-ALM has an auditable block-local reference path; ePC has an explicitly labeled independent Transformer adaptation with EMA error states; and stable parameterization is implemented as a residual-scaling wrapper. Their deterministic CPU/GPU protocol smokes pass, while the required 10M-token method gates remain pending. Meta-PCN, precision weighting, WF-Act-PC, closed-form precision, direct feedback alignment, stream amortization, and the new control-variate hybrid have no implementation or screening results yet. The checked bibliography distinguishes published papers, contemporary preprints, official code commits, adaptations, and strict-locality limitations.

## Required next gates

1. Extend the isolated worker to reserve and evaluate the full 10M validation stream, then pass exact-budget, validation-isolation, and resume checks at 10M.
2. Add the tiny Transformer PC-ALM equilibrium/gradient-cosine diagnostic and strict local-autograd audit.
3. Implement each method card behind the common interface and run a 10M-token smoke per arm.
4. Complete the 10M ePC/PC-ALM method smokes and implement the remaining registry arms before any method ranking.

No validation loss, perplexity, time-to-1B, GPU-hour, or method ranking is claimed in this snapshot.

An ePC adaptation pilot ran on the full 40M model for 100,000 FineWeb targets on eight independent seeds across foureyes and ripper. It reached exact budgets with nonzero error/residual diagnostics at roughly 2.4k tokens/s, versus roughly 60–68k tokens/s for BP controls. This is an implementation-cost and stability observation only; it is below the 10M gate and used no held-out validation claim. One separate BP control was interrupted at 21.1M tokens after a duplicate GPU assignment was detected; it is retained and labeled as a failed run in the registry.

Meta-PCN and precision-weighted PC each completed corrected 100k-token pilots on work and loki with nonzero residual/error or precision diagnostics, at roughly 4.2–4.6k tokens/s. Their preceding 1M interface pilots are retained as failed zero-signal runs because the output-local teaching clamp was not yet wired into those adapters. A separate orchestration check caught and fixed a missing `CUDA_VISIBLE_DEVICES` binding that had initially placed two local pilots on GPU 0; all subsequent pilots verify per-GPU utilization.

Corrected one-million-token Meta-PCN and precision-weighted pilots reached exact budgets at roughly 14–17k tokens/s, with nonzero residuals and output-gradient diagnostics. A stable-parameterization ePC pair also reached 1M tokens at roughly 16.6–16.9k tokens/s. These are still tuning and implementation pilots; no held-out validation or ranking claim is made.

The faithful PC-ALM reference completed four 1M-token pilots at roughly 15.1–17.3k tokens/s. Primal residual and dual norms were finite and nonzero, with the output-local gradient norm recorded per step. This remains a short implementation pilot, not a final quality comparison.

A four-inference-step ePC ablation reached 1M tokens at 13.7–13.8k tokens/s; the matched stable-ePC variant reached 11.9–12.0k tokens/s. The two-step runs were roughly 14–17k tokens/s, so the extra relaxation work has a directly measured throughput cost. These curves are implementation diagnostics only.

The matched four-step PC-ALM ablation reached 1M tokens at 12.3–14.0k tokens/s, compared with 15.1–17.3k at two steps. Its primal residual and dual norms remained finite and nonzero across all four seeds.

Four-step Meta-PCN reached 1M tokens at about 12.7k tokens/s but developed large normalized error norms (about 11.5 average, with the deepest layer near 184) while remaining finite. Four-step precision-weighted PC reached about 12.0k tokens/s and raised its mean precision to about 345 with deepest-layer residual norms around 6.6. These are stability signals to investigate, not quality claims.

At eight inference steps, PC-ALM reached 1M tokens at 10.3–10.5k tokens/s, ePC at 8.6k, and stable-ePC at 8.5k. All remained finite; the stable-ePC deepest-layer error norm rose to about 0.42. These measurements make the sequential relaxation cost explicit and remain below the method gate.

## Phase 1 control pilots

Twelve independent single-GPU BP AdamW controls completed exactly 10M FineWeb training targets (four on each 4-GPU host and two on each 2-GPU host). Final training losses were 5.87–6.23. Mean throughput was 60.9k tokens/s on foureyes, 54.1k on ripper, 61.7k on work, and 63.4k on loki. These are control/pipeline measurements, not validation results or method rankings; raw metrics, manifests, and registry entries are retained per run.

## Phase 2 control pilots

The first 50M-token wave completed 12 independent BP AdamW FineWeb controls, four per 4-GPU host and two per 2-GPU host. Every run reached exactly 50,000,000 tokens with clean exit; final training losses were 4.79–4.97 and throughput was approximately 61–62k tokens/s on foureyes/ripper and 67k on work/loki. A second 50M seed wave completed on ripper, work, and loki and is registered with exact manifests and metrics; a third independent 50M replication wave is currently active across all 12 GPUs. These are additional BP controls for throughput and variance coverage, not method results.
