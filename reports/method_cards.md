# Method cards (registry v0.1)

These cards define the interface and locality claims before implementation. They are adaptation plans until the corresponding source equation checks, unit tests, and 10M-token smoke run pass. Each arm exposes `init_state`, `relax_step`, `local_update`, `diagnostics`, and `step_cost`.

## 1. Faithful PC-ALM (`pc_alm`)

**Source/equations:** Seely & Gould, *Augmented Lagrangian Predictive Coding*, arXiv:2605.31022. For each block constraint `c_i=h_i-f_i(h_{i-1};W_i)`, optimize `ell(h_L,y)+sum(lambda_i*c_i)+(rho/2)sum||c_i||²`; update hidden states, duals, and local parameters. **Boundary:** adjacent block states and dual/error messages; causal attention remains token-global over prior positions inside a block, so locality is `block_local`, not strict token-local. **Autograd:** reference permits local detached-block `autograd.grad`; no global reverse graph. **State:** per-batch hidden states and BF16/FP32 duals; no persistence across batches. **Risks:** finite relaxation, dual explosion, attention/normalization Jacobian mismatch. **Status:** equations adapted to pre-norm Transformer; source smoke cell and linear equilibrium diagnostic still required.

## 2. Error-based PC (`epc`)

**Source/equations:** Goemaere et al., ICML 2026 PMLR 306, official code commit `95c5551`. Replace depth-propagated state signal with explicit error-state optimization and local block energy updates. **Boundary:** adjacent errors/states; causal attention token scope is unchanged. **Autograd:** public code calls `E.backward()` on error tensors and uses detached local energy, so label local-autograd rather than strict no-autograd. **State:** error tensors per batch; no dual unless the matched PC-ALM hybrid. **Risks:** error variance imbalance, nonlinear attention adaptation, optimizer sensitivity. **Status:** independent Transformer adaptation; reproduce public smallest state/error diagnostic first.

## 3. Meta-PCN (`meta_pcn`)

**Source/equations:** Ha et al., ICLR 2026, *Stable and Scalable Deep Predictive Coding Networks with Meta-Prediction Errors*. Add a meta prediction-error controller and normalization/variance stabilization to the local errors. **Boundary:** adjacent block errors plus meta statistics; block-local. **Autograd:** local only, instrumented. **State:** per-batch errors and running mean/variance. **Risks:** controller collapse, cross-layer imbalance, extra synchronization if statistics are global. **Status:** clean Meta-PCN and Meta-on-PC-ALM are adaptations; source code was not located in the audit.

## 4. Precision-weighted deep PC (`precision_weighted_pc`)

**Source/equations:** precision-weighted latent and weight updates from deep PC work; positive precision scales each local residual and optional auxiliary/residual neurons. **Boundary:** adjacent block plus local precision; block-local. **Autograd:** local detached block allowed. **State:** latent states, dual-free precision (FP32/log-precision), optional auxiliary neurons. **Risks:** non-positive precision, ill-conditioning, extra memory. **Status:** adaptation; compare with and without dual variables at matched local updates.

## 5. Stable parameterization wrapper (`stable_parameterization`)

**Source/equations:** Innocenti, Achour & Bogacz, ICML 2026 PMLR 306, *On the Infinite Width and Depth Limits of Predictive Coding Networks*. Apply explicit residual scaling, initialization, and normalization prescriptions as a wrapper to PC-ALM/ePC/Meta-PCN. **Boundary:** unchanged from wrapped method. **Autograd/state:** unchanged; wrapper owns no credit state. **Risks:** changing optimization scale or effective architecture; isolate factorial ablation. **Status:** adaptation; no claim of a standalone learning rule.

## 6. WF-Act-PC (`wf_act_pc`)

**Source/equations:** Shen & Li, arXiv:2607.13380; code lead `pcax` commit `95e37f7`. Use forward-weight feedback to form a local Jacobian-transpose action under frozen-normalization assumptions. **Boundary:** feedback matrix, local derivative, and residual path cross each layer; relaxed biological locality. **Autograd:** analytic/local derivative only where feasible. **State:** feedback maps and local errors; learned feedback adds parameters. **Risks:** RMSNorm/LayerNorm and attention Q/K/V corrections violate assumptions; feedback drift. **Status:** Transformer adaptation with explicit correction audit.

## 7. Closed-form precision PC (`closed_form_precision_pc`)

**Source/equations:** Baskakovs et al., arXiv:2605.20293. Treat latents as Gaussian means/precisions and replace repeated relaxation by a positive-precision posterior/Kalman-like update. **Boundary:** local block sufficient statistics; block-local. **Autograd:** intended strict no-autograd after tiny equivalence check. **State:** mean and positive precision per latent, FP32. **Risks:** nonlinear attention makes the closed form approximate; precision under/overflow. **Status:** tiny causal-model reproduction first, then batched adaptation.

## 8. Direct feedback alignment (`direct_feedback_alignment`)

**Source/equations:** Casnici et al., arXiv:2602.15571. One/few-step output-error feedback through fixed, learned, low-rank, or transpose-tied maps. **Boundary:** feedback map from output to each block; not strict local credit. **Autograd:** no global reverse pass; local readout error only. **State:** low-rank feedback parameters and local errors; full vocabulary maps are prohibited by the 40M budget. **Risks:** feedback dimensionality, poor cosine, added communication/parameters. **Status:** adaptation with parameter and memory accounting.

## 9. Stream-amortized PC (`stream_amortized_pc`)

**Source/equations:** Pinchetti et al., arXiv:2601.20895. Initialize hidden states by forward prediction, EMA/associative carry, and boundary-first scheduling over contiguous chunks. **Boundary:** adjacent blocks plus temporal carry across examples; token-locality is not strict because attention sees prior context. **Autograd/state:** local updates; carry state is rank-local and reset at declared document boundaries. **Risks:** future-token leakage at chunk boundaries, stale state, data-order dependence. **Status:** adaptation; delayed-copy leakage diagnostic is mandatory.

## 10. PC-ALM activation-space control variate (`pc_alm_control_variate`)

**Source/equations:** new hybrid. Use the PC-ALM dual/error signal as a predictor and estimate only residual credit with fresh random activation probes; compare plain finite differences, predictor, and residual estimator. **Boundary:** block-local predictor plus fresh local probes; strict token locality only for a separately windowed attention control. **Autograd:** strict arm forbids global and local autograd; reference estimator may use analytic finite differences. **State:** PC hidden/dual state and probe RNG only. **Risks:** conditional-independence failure, variance/bias tradeoff, probe cost. **Status:** new proposal; do not call a literature reproduction.

## Controls

`bp_adamw` and checkpointed BP use ordinary global reverse-mode autograd and establish attainable quality/speed; `standard_pc_bp` removes augmented-Lagrangian duals; `ipc` updates weights during inference as in DeepPCNs (official commit `99472b0`). The strict-locality label is never applied to these controls.
