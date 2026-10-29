# PC-ALM local-credit FineWeb study — current report

**Status:** Phase 0 executed; the 1B-token campaign is blocked. This report is an evidence snapshot, not a head-to-head result.

## Fixed contract

The frozen reference is a pre-norm RMSNorm causal Transformer with 16 blocks, width 384, six 64-dimensional heads, MLP width 1,536, RoPE, 1,024 context, tied input/output embeddings, and a 32,000-token Mistral tokenizer. It has 40,667,520 parameters. FineWeb is `HuggingFaceFW/fineweb`, `sample-10BT`; the tokenizer model SHA256 and dataset metadata are in `reports/data_manifest.json`. The intended training budget is exactly 1,000,000,000 post-shift targets with a 10M-token isolated validation stream.

## Environment and topology

All four hosts are reachable as `catid` over SSH: foureyes (4 GPUs), ripper (4), work (2), and loki (2). Every GPU is an RTX PRO 6000 Blackwell with 97,887 MiB and driver 615.71.10. The machines have Ethernet only, no NVLink or InfiniBand, and local GPU topology is PCIe `NODE`. The exact 5.8 GiB Python environment is provisioned on all hosts; each reports torch 2.12.0+cu130, CUDA 13.0, NCCL 2.29.7, and the same torch library hash.

Local foureyes all-reduce is 21.3 GB/s at world size 4 for a 20 MB payload. Direct-rank cross-node rates are 11.1 MB/s (world 2), 7.1 MB/s (world 4), 6.3 MB/s (world 8), and 5.7 MB/s (world 12). Independent 4-GPU jobs are therefore the default useful allocation; a single 8/12-GPU job is not justified by measured communication.

## Correctness and implementation evidence

The reference tests pass parameter counting/tied weights, causal masking, next-token loss normalization, tied-embedding gradients, PC state initialization/diagnostics, a linear PC-ALM KKT check, finite-difference local gradients, no-global-backward relaxation instrumentation, and deterministic rank chunk disjointness. A live FineWeb tokenizer smoke produced three 128-token chunks and 381 post-shift targets. The synthetic BP control completed 2,046 exact targets and wrote resumable checkpoints. A one-step fixed-batch single versus local 2-GPU DDP run has identical loss and parameter checksum.

The initial in-process FineWeb BP pilot consumed exactly 2,046 targets and wrote metrics/checkpoints, then aborted with `PyGILState_Release` during Python finalization in the streaming datasets/pyarrow stack; a retry reproduced it. Those failures remain recorded and excluded from quality claims. The reader is now isolated in `scripts/fineweb_worker.py`, which exits outside the training interpreter. The isolated 2,046-token pilot, resume-from-1,023 checkpoint, and 10,240-token live smoke all completed with exit 0 and exact counts. The required 10M-per-arm smoke remains pending.

## Method status

The ten-arm registry and method cards are committed, with PC-ALM as the only reference implementation scaffold. ePC, Meta-PCN, precision weighting, stable parameterization, WF-Act-PC, closed-form precision, direct feedback alignment, stream amortization, and the new control-variate hybrid have no screening results yet. The checked bibliography distinguishes published papers, contemporary preprints, official code commits, adaptations, and strict-locality limitations.

## Required next gates

1. Replace or isolate the FineWeb streaming reader and pass exact-budget, validation-isolation, and resume tests.
2. Add the tiny Transformer PC-ALM equilibrium/gradient-cosine checks and instrument global versus local autograd.
3. Implement each method card behind the common interface and run a 10M-token smoke per arm.
4. Re-run optimized/compiled equivalence and distributed single-versus-multi-GPU checks before any 50M screen.

No validation loss, perplexity, time-to-1B, GPU-hour, or method ranking is claimed in this snapshot.
