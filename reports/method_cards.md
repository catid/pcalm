# Method cards (registry v0.1)

These cards define the interface and locality claims before implementation. They are adaptation plans until the corresponding source equation checks, unit tests, and 10M-token smoke run pass. Each arm exposes `init_state`, `relax_step`, `local_update`, `diagnostics`, and `step_cost`.

## 1. Faithful PC-ALM (`pc_alm`)

**Source/equations:** Seely & Gould, *Augmented Lagrangian Predictive Coding*, arXiv:2605.31022. For each block constraint `c_i=h_i-f_i(h_{i-1};W_i)`, optimize `ell(h_L,y)+sum(lambda_i*c_i)+(rho/2)sum||c_i||²`; update hidden states, duals, and local parameters. **Boundary:** adjacent block states and dual/error messages; causal attention remains token-global over prior positions inside a block, so locality is `block_local`, not strict token-local. **Autograd:** reference permits local detached-block `autograd.grad`; no global reverse graph. **State:** per-batch hidden states and BF16/FP32 duals; no persistence across batches. **Risks:** finite relaxation, dual explosion, attention/normalization Jacobian mismatch. **Status:** equations adapted to pre-norm Transformer; source smoke cell and linear equilibrium diagnostic still required.

## 2. Error-based PC (`epc`)

**Source/equations:** Goemaere et al., ICML 2026 PMLR 306, official code commit `95c5551`. Replace depth-propagated state signal with explicit error-state optimization and local block energy updates. Our independent Transformer adaptation stores one error tensor per block, `r_i=h[i+1]-f_i(h[i];W_i)`, then applies `e_i <- (1-beta)e_i + beta*r_i` and `h[i+1] <- h[i+1]-eta_h*e_i`; local weights use `0.5||r_i||^2` with detached block inputs/targets. **Boundary:** adjacent errors/states; causal attention token scope is unchanged. **Autograd:** public code calls `E.backward()` on error tensors and uses detached local energy, so label local-autograd rather than strict no-autograd. **State:** error tensors per batch; no dual unless the matched PC-ALM hybrid. **Risks:** error variance imbalance, nonlinear attention adaptation, optimizer sensitivity. **Status:** adapted/independent reproduction; the implementation is smoke-tested only and does not claim the published state/error diagnostic until the source cell is reproduced.

## 3. Meta-PCN (`meta_pcn`)

**Source/equations:** Ha et al., ICLR 2026, *Stable and Scalable Deep Predictive Coding Networks with Meta-Prediction Errors*. Our adapted controller computes `r_i=h[i+1]-f_i(h[i])`, normalizes it as `(r_i-mu_i)/sqrt(var_i+eps)`, and applies `e_i <- (1-meta_lr)e_i + meta_lr*normalized_r_i`, followed by `h[i+1] <- h[i+1]-eta_h*e_i`. **Boundary:** adjacent block errors plus meta statistics; block-local. **Autograd:** local only, instrumented. **State:** per-batch errors and running mean/variance. **Risks:** controller collapse, cross-layer imbalance, extra synchronization if statistics are global. **Status:** independent Transformer adaptation; tiny deterministic smoke passes, while source-cell and 10M-token gates remain pending.

## 4. Precision-weighted deep PC (`precision_weighted_pc`)

**Source/equations:** precision-weighted latent and weight updates from deep PC work; our adaptation keeps a positive scalar `p_i` per block, updates `p_i <- (1-gamma)p_i + gamma/(mean(r_i^2)+eps)`, and uses `h[i+1] <- h[i+1]-eta_h*p_i*r_i`; local weights minimize `0.5*p_i*||r_i||^2`. **Boundary:** adjacent block plus local precision; block-local. **Autograd:** local detached block allowed. **State:** latent states, dual-free precision (FP32/log-precision), optional auxiliary neurons. **Risks:** non-positive precision, ill-conditioning, extra memory. **Status:** independent Transformer adaptation; tiny deterministic smoke passes, while source-cell and 10M-token gates remain pending. Compare with and without dual variables at matched local updates.

## 5. Stable parameterization wrapper (`stable_parameterization`)

**Source/equations:** Innocenti, Achour & Bogacz, ICML 2026 PMLR 306, *On the Infinite Width and Depth Limits of Predictive Coding Networks*. The current wrapper applies `s=1/sqrt(L)` to both attention and MLP residual branches and multiplies block linear weights by `s` once at initialization; this is an explicit stable-depth adaptation, not a full reproduction of every theorem prescription. **Boundary:** unchanged from wrapped method. **Autograd/state:** unchanged; wrapper owns no credit state. **Risks:** changing optimization scale or effective architecture; isolate factorial ablation. **Status:** adaptation; no claim of a standalone learning rule. Smoke-tested with ePC; 10M gate remains pending.

## 6. WF-Act-PC (`wf_act_pc`)

**Source/equations:** Shen & Li, arXiv:2607.13380; code lead `pcax` commit `95e37f7`. The scaffold uses a detached local squared-error Jacobian-transpose action and records the frozen-normalization assumption; attention Q/K/V remains an unverified adaptation. **Boundary:** feedback matrix, local derivative, and residual path cross each layer; relaxed biological locality. **Autograd:** analytic/local derivative only where feasible. **State:** feedback maps and local errors; learned feedback adds parameters. **Risks:** RMSNorm/LayerNorm and attention Q/K/V corrections violate assumptions; feedback drift. **Status:** protocol smoke only; 10M/source-equation gates pending.

## 7. Closed-form precision PC (`closed_form_precision_pc`)

**Source/equations:** Baskakovs et al., arXiv:2605.20293. The scaffold applies the scalar Gaussian posterior `h_new=(h+p*f(h))/(1+p)` with positive clipped precision, replacing repeated relaxation. **Boundary:** local block sufficient statistics; block-local. **Autograd:** strict no-autograd in the update path. **State:** mean and positive precision per latent, FP32. **Risks:** nonlinear attention makes the closed form approximate; precision under/overflow. **Status:** protocol smoke only; tiny linear equivalence and 10M gates pending.

## 8. Direct feedback alignment (`direct_feedback_alignment`)

**Source/equations:** Casnici et al., arXiv:2602.15571. The scaffold projects the local output cross-entropy error through the tied readout transpose and applies one-step feedback to every block. **Boundary:** feedback map from output to each block; not strict local credit. **Autograd:** no global reverse pass in relaxation; local readout error only. **State:** low-rank/transpose-tied feedback and local errors; full vocabulary maps are prohibited by the 40M budget. **Risks:** feedback dimensionality, poor cosine, added communication/parameters. **Status:** protocol smoke only; adaptation and 10M gates pending.

## 9. Stream-amortized PC (`stream_amortized_pc`)

**Source/equations:** Pinchetti et al., arXiv:2601.20895. The scaffold carries detached hidden states across contiguous chunks with declared `carry_decay`, and exposes `reset_carry()` for document boundaries. **Boundary:** adjacent blocks plus temporal carry across examples; token-locality is not strict because attention sees prior context. **Autograd/state:** local updates; carry state is rank-local and reset at declared document boundaries. **Risks:** future-token leakage at chunk boundaries, stale state, data-order dependence. **Status:** protocol smoke only; delayed-copy leakage and 10M gates pending.

## 10. PC-ALM activation-space control variate (`pc_alm_control_variate`)

**Source/equations:** new hybrid. The scaffold uses PC-ALM residual/dual scalars as a predictor and fresh antithetic parameter probes for residual directional credit; it reports probe variance and predictor mean. **Boundary:** block-local predictor plus fresh local probes; strict token locality only for a separately windowed attention control. **Autograd:** strict arm forbids global and local autograd; estimator uses finite differences. **State:** PC hidden/dual state and probe RNG only. **Risks:** conditional-independence failure, variance/bias tradeoff, probe cost. **Status:** protocol smoke only; new proposal, not a literature reproduction, with 10M gate pending.

## Controls

`bp_adamw` and checkpointed BP use ordinary global reverse-mode autograd and establish attainable quality/speed; `standard_pc_bp` removes augmented-Lagrangian duals; `ipc` updates weights during inference as in DeepPCNs (official commit `99472b0`). The strict-locality label is never applied to these controls.
