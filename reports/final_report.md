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

The ten-arm registry and method cards are committed. All ten arms now have runnable
protocol implementations and deterministic reference smokes. Exact 10M-token
screening runs completed for PC-ALM (training loss 8.920), ePC (8.957),
precision-weighted PC (10.493), WF-Act-PC (8.894), direct feedback alignment
(8.964), closed-form precision PC (10.441), stream-amortized PC (8.978), and
the new control-variate hybrid (10.398). A Meta-PCN run reached 10M with loss
8.942; stable ePC reached 10M with loss 10.164. These are training-batch losses,
not held-out validation results. The registry keeps all arms pending their
method-specific source, gradient-cosine, equivalence, leakage, or variance
sub-gates. The checked bibliography distinguishes published papers,
contemporary preprints, official code commits, adaptations, and strict-locality
limitations.

## Required next gates

1. Extend the isolated worker to reserve and evaluate the full 10M validation stream, then pass exact-budget, validation-isolation, and resume checks at 10M.
2. Add the tiny Transformer PC-ALM equilibrium/gradient-cosine diagnostic and strict local-autograd audit.
3. Implement each method card behind the common interface and run a 10M-token smoke per arm.
4. Complete the 10M ePC/PC-ALM method smokes and implement the remaining registry arms before any method ranking.

No 1B-token result, final validation ranking, or finalist selection is claimed in
this snapshot. The production 40M optimization gates are now measured: local
4-GPU DDP reaches 93.9--103.5k tokens/s at sequence 128 and the two-rank Loki
DDP path reaches 104.1k/225.1k tokens/s at sequence 1024 for per-rank batch
1/2. CUDA graphs pass fixed-shape eager equivalence. Replay speedup is
1.78--2.08x at sequence 128 and 1.06--1.31x at sequence 1024, with parameter
error 3.00e-4 below the 5.00e-4 gate tolerance. These measurements are recorded
in `reports/engineering_gate.md` and the optimization JSON artifacts.

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

The first 50M-token wave completed 12 independent BP AdamW FineWeb controls, four per 4-GPU host and two per 2-GPU host. Every run reached exactly 50,000,000 tokens with clean exit; final training losses were 4.79–4.97 and throughput was approximately 61–62k tokens/s on foureyes/ripper and 67k on work/loki. A second 50M seed wave completed on ripper, work, and loki and is registered with exact manifests and metrics; a fifth independent 50M replication wave (run IDs `phase2-bp50m-*-seed20263200` through `seed20263211`) is currently active across all 12 GPUs. The intervening third and fourth waves are retained in the registry. These are additional BP controls for throughput and variance coverage, not method results.

## Early tuning evidence

The 10M tuning runs provide bounded settings evidence without changing the main
contract. On Ripper, ePC with `beta=0.25` reached training loss 8.777 and with
`beta=0.75` 8.880; PC-ALM with four steps and `eta_h=0.05` reached 9.077. A
BP learning-rate pair on Loki reached 5.530 at `1e-4` and 6.045 at `6e-4`,
both at about 60.5k tokens/s. These are tuning-stream results and are not used
as validation claims.

The first gain ablations also completed at 1M tokens. WF-Act with gain 0.25
remained top-layer-only (loss 10.396), while direct feedback alignment with
gain 2.0 produced nonzero all-layer errors (loss 10.246, primal residual 3.64)
but remained a relaxed-feedback method. Additional 1M settings for 16-step
PC-ALM, 8-step ePC, precision weighting, Meta-PCN, carry-over, and the
control-variate probes remain in the registry as ablations or active jobs.
