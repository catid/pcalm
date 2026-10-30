# Early local-method analysis

**Snapshot:** 2026-10-30 UTC. This is a read-only analysis of the completed 10M-token WF-Act-PC and direct feedback-alignment (DFA) pilots plus live local-method logs. It is an early gate diagnostic, not a finalist ranking or a 1B-token result.

## Completed 10M pilots

| Arm | Run | Tokens | Final train loss | Minimum loss | Mean loss, last 100 records | Tokens/s | Relax seconds/step | NaN/Inf records |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| WF-Act-PC | `phase2-wf-act-10m-work-g0-seed20264000` | 10,000,000 | 8.8936 | 7.6645 | 8.5786 +/- 0.1748 | 23,040.7 | 0.01695 | 0 |
| DFA | `phase2-dfa-10m-work-g1-seed20264002` | 10,000,000 | 8.9642 | 7.7832 | 8.6385 +/- 0.1708 | 24,299.0 | 0.01399 | 0 |

The 10M BP controls have final train loss around 6.0 and roughly 60--63k tokens/s on comparable single-GPU hosts. The local arms therefore trail the control by about 2.9 nats at this budget and run at about 36--40% of BP throughput. They should remain adaptation/gate results, not be promoted to quality finalists from this evidence alone.

Both pilots completed exactly 10,000,000 post-shift target tokens. Losses declined over the run and had no NaN/Inf records or obvious divergence. The last record is a partial-chunk boundary record, so its loss is noisy; use the fixed token checkpoint and the last-100 summary together.

### WF-Act-PC diagnostics

WF-Act has nearly zero residual/error in layers 0--14 and a final-layer error norm around `2.2e-3` for most late records (`5.7e-3` at the final partial record). This is consistent with forward initialization plus the current local-Jacobian scaffold: the hidden constraints are nearly exact and the output signal is not being propagated into earlier blocks. The reported `frozen_normalization_assumption=true` is therefore a material adaptation caveat, especially for RMSNorm and causal attention.

The final-layer local gradient norm is around `2e-7` (about `3e-6` at the final partial record), while earlier layers report zero. This is a useful failure/stability signal: the current feedback construction is effectively top-layer-only in this Transformer port. Recommended next test: instrument per-layer output-credit/update cosine against BP before tuning the feedback gain; then compare gains `{0.25, 1.0, 4.0}` at a 1M-token budget. Do not increase gain without a dual/error clipping check.

### DFA diagnostics

DFA has nonzero error in every layer. Late ordinary records show mean error norm about `1.31` and primal residual about `1.87`; the final partial record drops to `0.55` and `0.79`, which is a boundary artifact rather than convergence evidence. Layer errors are largest in the first few hidden blocks (roughly `1.5--1.6`) and decline gradually toward the output (about `1.2--1.3`). No explosion was observed, but the error signal is not equilibrated in two inference steps.

DFA is about 5.5% faster than WF-Act in this setup, but both are much slower than BP. Its local gradient norms are nonzero across all layers and decay with depth, so it is the better candidate for a gradient-cosine screen. Recommended next test: sweep feedback gain `{0.25, 1.0, 2.0}` and inference steps `{2, 4, 8}` on 1M tokens, recording layerwise cosine and peak memory. Keep the output feedback map explicitly labeled `relaxed_feedback`; it is not strict local credit.

## Active local-method logs at snapshot

These runs were still executing or had not yet produced a completed manifest when inspected. Counts are read from their local metrics files and are not treated as completed results.

| Arm | Host/GPU | Tokens observed | Tokens/s observed | Latest loss | Key diagnostic |
|---|---|---:|---:|---:|---|
| PC-ALM | foureyes/g0 | 5.35M | 19.0k | 8.857 | primal residual `2.14e-4`, dual `3.26e-6` |
| ePC | foureyes/g1 | 5.13M | 18.9k | 9.268 | error norm `1.43e-4`, final-layer only |
| Meta-PCN | foureyes/g2 | 4.87M | 18.0k | 9.348 | error norm `0.247`, final-layer `3.95`; cross-layer imbalance `16.0` |
| Precision-weighted PC | foureyes/g3 | 5.26M | 19.4k | 10.434 | precision `190.81` in every layer; residual `9.35e-4` |
| PC-ALM control variate | work/g0 | process active; metrics not copied at snapshot | -- | -- | inspect probe variance after completion |
| Stable ePC | work/g1 | process active; metrics not copied at snapshot | -- | -- | compare to clean ePC at paired token checkpoints |
| Closed-form precision PC | loki/g1 | process active; metrics not copied at snapshot | -- | -- | check strict no-autograd and precision bounds |

The active PC-ALM/ePC/Meta/precision logs show the same structural issue as the completed WF-Act run: forward initialization leaves layers 0--14 at nearly zero local residual, so the current two-step signal is concentrated at the final block. Meta-PCN additionally has a cross-layer imbalance of 16 by construction in these records, and precision weighting has converged to a uniform high scalar (`190.81`) instead of layer-specific precisions. These are tuning/debugging targets, not evidence that the underlying literature methods fail.

## Tuning and gate recommendations

1. Keep these five arms at the 10M gate only until a matched diagnostic batch reports layerwise update/gradient cosine to BP. Add the cosine and relative-error records before any 50--250M screen.
2. For PC-ALM/ePC/WF-Act, test a non-forward latent initializer (small deterministic Gaussian perturbation or declared carry) and `inner_steps={2,4,8}`. Preserve the forward-initialized result as the control so the effect is separable from the credit rule.
3. For Meta-PCN, test `meta_lr={0.1,0.25,0.5}` with variance floors and monitor cross-layer imbalance; reject settings that keep the ratio near 16 or produce non-finite errors.
4. For precision weighting, lower the initial/maximum precision or use a log-precision parameterization; record precision by layer. The current uniform `190.81` indicates the scalar adaptation is dominated by the same near-zero residual regime.
5. For DFA, perform the gain/inference sweep above and report the communication/feedback-map cost separately. Do not call it strict local-credit.
6. Use fixed token checkpoints and exclude the final partial chunk from convergence claims. Keep all failed/unstable settings in the registry.

## Provenance and comparability caveats

The completed worker manifests under `runs/phase2-*` were copied from `/var/tmp/pcalm` and report `commit: ERROR` because those worker trees are not Git checkouts. Their manifest method version is `reference-0.1`, while the metrics JSONL records the actual adapted versions (`adapted-local-jacobian-0.1` and `transpose-tied-feedback-0.1`). Preserve these runs, but repair provenance in future launches by checking out the source tree and recording the exact commit/configuration before starting. The `runs/registry.jsonl` entries identify the methods and exact 10M-token budget; this report does not overwrite them.
