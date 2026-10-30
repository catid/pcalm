"""Common local-credit method interface and a slow, auditable PC-ALM path.

The reference implementation deliberately keeps each block's graph short lived.
``relax_step`` updates detached latent states and duals; ``local_update`` builds
one block graph at a time and uses ``autograd.grad`` only on that block.  It
never calls ``Tensor.backward`` and does not retain a reverse-mode graph across
layers or inference iterations.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence
import time

import torch
import torch.nn.functional as F


@dataclass
class MethodState:
    """Per-minibatch state owned by a local-credit method.

    ``hidden`` has one entry for the token embedding and one per Transformer
    block.  ``dual`` has one multiplier per block constraint.  The tensors are
    rank-local and must not be persisted between minibatches unless an explicit
    temporal-carry method opts in.
    """

    hidden: List[torch.Tensor]
    dual: List[torch.Tensor]
    step: int = 0
    residual_norms: List[float] = field(default_factory=list)
    # ePC keeps explicit error states instead of depth-propagated error
    # messages.  PC-ALM leaves this field unset.
    errors: List[torch.Tensor] | None = None
    error_norms: List[float] = field(default_factory=list)
    error_mean: List[float] = field(default_factory=list)
    error_var: List[float] = field(default_factory=list)
    # Precision-weighted PC keeps one positive scalar (or a broadcastable
    # tensor in future variants) for each local block.
    precisions: List[torch.Tensor] | None = None
    precision_mean: List[float] = field(default_factory=list)
    # Optional method-owned work buffers.  Optimized implementations reuse
    # these tensors across relaxation sweeps; reference paths leave it None.
    work: List[torch.Tensor] | None = None


class PCALMReference:
    """Faithful slow PC-ALM block-local reference path.

    The constraint for block ``i`` is ``c_i = h[i+1] - f_i(h[i]; W_i)`` and
    the local augmented-Lagrangian term is
    ``0.5 * rho * ||h[i+1] - f_i(h[i]) + lambda_i/rho||^2``.
    Hidden primal updates are Jacobi/Gauss-Seidel style in depth order; duals
    use ``lambda <- lambda + alpha*c``.  Parameter updates use a detached
    input/target and one ``autograd.grad`` call per block, which is allowed in
    the block-local mode but does not form a global reverse graph.
    """

    name = "pc_alm"
    version = "reference-0.2"
    locality = "block_local"
    strict_no_autograd = False
    uses_local_autograd = True

    def __init__(
        self,
        model,
        inner_steps: int = 4,
        eta_h: float = 0.1,
        alpha: float = 0.01,
        rho: float = 1.0,
        parameter_step: float = 3e-4,
        clip_dual: float | None = None,
    ):
        if inner_steps < 1:
            raise ValueError("inner_steps must be positive")
        if eta_h <= 0 or alpha < 0 or rho <= 0 or parameter_step < 0:
            raise ValueError("invalid PC-ALM step sizes")
        self.model = model
        self.inner_steps = int(inner_steps)
        self.eta_h = float(eta_h)
        self.alpha = float(alpha)
        self.rho = float(rho)
        self.parameter_step = float(parameter_step)
        self.clip_dual = clip_dual

    def init_state(self, input_ids: torch.Tensor) -> MethodState:
        """Initialize local latent states from a detached forward pass."""
        with torch.no_grad():
            _, _, hidden = self.model(input_ids, labels=input_ids, return_hidden=True)
        # Clone so later primal updates cannot alias model activations.
        hs = [h.detach().clone() for h in hidden]
        dual = [torch.zeros_like(h) for h in hs[1:]]
        return MethodState(hs, dual)

    @torch.no_grad()
    def relax_step(self, state: MethodState, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        """Perform one local primal/dual relaxation sweep.

        ``input_ids`` is accepted for protocol uniformity and future output
        constraints; the block constraints only consume the state tensors.
        """
        del input_ids
        residuals: List[torch.Tensor] = []
        # Apply the output cross-entropy clamp through a short, output-local
        # graph.  Forward initialization makes all hidden constraints exact;
        # without this term there is no teaching signal to relax backward.
        if labels is not None:
            h_last = state.hidden[-1].detach().requires_grad_(True)
            with torch.enable_grad():
                logits = self.model.lm_head(self.model.norm(h_last))
                loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
                output_grad = torch.autograd.grad(loss, h_last, only_inputs=True)[0]
            state.hidden[-1] = (h_last.detach() - self.eta_h * output_grad.detach()).detach()
        # One sweep in depth order.  Every block sees a detached tensor and the
        # temporary prediction is discarded before the next block.
        for i, block in enumerate(self.model.blocks):
            h_prev = state.hidden[i]
            h_next = state.hidden[i + 1]
            pred = block(h_prev)
            residual = h_next - pred
            # Gradient of the ALM term with respect to h_next is rho*c + lambda.
            h_next_new = h_next - self.eta_h * (self.rho * residual + state.dual[i])
            state.hidden[i + 1] = h_next_new.detach()
            state.dual[i].add_(self.alpha * residual)
            if self.clip_dual is not None:
                state.dual[i].clamp_(-self.clip_dual, self.clip_dual)
            residuals.append(residual.detach())
        state.step += 1
        state.residual_norms = [float(r.float().norm().cpu()) for r in residuals]
        return residuals

    def local_update(
        self,
        state: MethodState,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        apply: bool = True,
    ) -> Dict[str, Any]:
        """Update block parameters with one detached local graph each.

        Returns per-layer gradient norms and the local ALM objective.  The
        default ``apply=True`` performs an in-place SGD step; callers that use
        an external optimizer can set ``apply=False`` and consume the returned
        diagnostics (or extend this method with a gradient sink).
        """
        del input_ids
        losses: List[float] = []
        grad_norms: List[float] = []
        for i, block in enumerate(self.model.blocks):
            # No graph from another block or previous iteration can survive.
            h_prev = state.hidden[i].detach()
            target = state.hidden[i + 1].detach()
            lam = state.dual[i].detach()
            pred = block(h_prev)
            shifted = target - pred + lam / self.rho
            local_loss = 0.5 * self.rho * shifted.float().pow(2).mean()
            params = tuple(p for p in block.parameters() if p.requires_grad)
            grads = torch.autograd.grad(local_loss, params, allow_unused=True)
            sq = torch.zeros((), device=local_loss.device, dtype=torch.float32)
            if apply and self.parameter_step:
                with torch.no_grad():
                    for p, g in zip(params, grads):
                        if g is None:
                            continue
                        gf = g.to(dtype=p.dtype)
                        p.add_(gf, alpha=-self.parameter_step)
                        sq = sq + g.float().pow(2).sum()
            else:
                for g in grads:
                    if g is not None:
                        sq = sq + g.float().pow(2).sum()
            losses.append(float(local_loss.detach().cpu()))
            grad_norms.append(float(sq.sqrt().detach().cpu()))
            # Explicitly drop local references before proceeding to the next
            # block, avoiding accidental graph retention.
            del pred, shifted, local_loss, grads
        output_grad_norm = 0.0
        if labels is not None:
            h_out = state.hidden[-1].detach()
            logits = self.model.lm_head(self.model.norm(h_out))
            output_loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
            out_param = (self.model.lm_head.weight,)
            out_grads = torch.autograd.grad(output_loss, out_param, allow_unused=True)
            if out_grads[0] is not None:
                output_grad_norm = float(out_grads[0].float().norm().detach().cpu())
                if apply and self.parameter_step:
                    with torch.no_grad():
                        self.model.lm_head.weight.add_(out_grads[0].to(dtype=self.model.lm_head.weight.dtype), alpha=-self.parameter_step)
        return {
            "local_loss": sum(losses) / max(1, len(losses)),
            "layer_grad_norm": grad_norms,
            "output_grad_norm": output_grad_norm,
            "updated_layers": len(losses) if apply and self.parameter_step else 0,
        }

    @torch.no_grad()
    def diagnostics(self, state: MethodState) -> Dict[str, Any]:
        residual = state.residual_norms
        if not residual:
            residual = [float((state.hidden[i + 1] - self.model.blocks[i](state.hidden[i])).float().norm().cpu()) for i in range(len(self.model.blocks))]
        dual = [float(d.float().norm().cpu()) for d in state.dual]
        return {
            "step": state.step,
            "primal_residual_norm": sum(residual) / max(1, len(residual)),
            "primal_residual_by_layer": residual,
            "dual_norm": sum(dual) / max(1, len(dual)),
            "dual_norm_by_layer": dual,
        }

    def step_cost(self, batch_tokens: int, elapsed_s: float, *, inference_steps: int | None = None) -> Dict[str, Any]:
        """Return a stable cost record used by the run registry."""
        return {
            "batch_tokens": int(batch_tokens),
            "elapsed_s": float(elapsed_s),
            "tokens_per_s": int(batch_tokens) / max(float(elapsed_s), 1e-9),
            "inner_steps": self.inner_steps if inference_steps is None else int(inference_steps),
            "local_update_count": len(self.model.blocks),
            "method": self.name,
            "version": self.version,
            "locality": self.locality,
            "strict_no_autograd": self.strict_no_autograd,
        }

    def run_inference(self, state: MethodState, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> Dict[str, Any]:
        """Convenience helper for fixed-count relaxation diagnostics."""
        t0 = time.perf_counter()
        for _ in range(self.inner_steps):
            self.relax_step(state, input_ids, labels)
        return self.step_cost(int(input_ids.numel()), time.perf_counter() - t0)


class PCALMOptimized(PCALMReference):
    """Preallocated PC-ALM relaxation path with the reference interface.

    The reference path allocates a new residual and hidden tensor for every
    block on every sweep.  This implementation allocates one residual buffer
    per block in ``init_state`` and updates latent tensors in place.  The
    arithmetic order is kept explicit (residual first, then primal and dual
    updates), and ``collect_diagnostics=False`` avoids synchronizing every
    layer's norm to the host during a steady-state benchmark.  Set it to true
    for equivalent per-sweep diagnostics.

    This optimization changes storage/launch behavior only; it does not add a
    global reverse-mode graph or change the block-local update equations.
    """

    name = "pc_alm_optimized"
    version = "optimized-preallocated-0.1"

    def __init__(self, *args, collect_diagnostics: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.collect_diagnostics = bool(collect_diagnostics)

    def init_state(self, input_ids: torch.Tensor) -> MethodState:
        state = super().init_state(input_ids)
        state.work = [torch.empty_like(h) for h in state.hidden[1:]]
        return state

    @torch.no_grad()
    def relax_step(self, state: MethodState, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        del input_ids
        if state.work is None or len(state.work) != len(self.model.blocks):
            raise ValueError("optimized PC-ALM state is missing preallocated work buffers")
        # Match the reference output-local objective exactly.  This short
        # graph is discarded before entering the block-local sweep.
        if labels is not None:
            h_last = state.hidden[-1].detach().requires_grad_(True)
            with torch.enable_grad():
                logits = self.model.lm_head(self.model.norm(h_last))
                loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
                output_grad = torch.autograd.grad(loss, h_last, only_inputs=True)[0]
            state.hidden[-1].copy_(h_last.detach() - self.eta_h * output_grad.detach())
        residuals = state.work
        for i, block in enumerate(self.model.blocks):
            h_prev = state.hidden[i]
            h_next = state.hidden[i + 1]
            pred = block(h_prev)
            residual = residuals[i]
            # out= avoids a fresh residual allocation.  Do not overwrite
            # h_next until the residual has been copied.
            torch.sub(h_next, pred, out=residual)
            h_next.add_(residual, alpha=-self.eta_h * self.rho)
            h_next.add_(state.dual[i], alpha=-self.eta_h)
            state.dual[i].add_(residual, alpha=self.alpha)
            if self.clip_dual is not None:
                state.dual[i].clamp_(-self.clip_dual, self.clip_dual)
            del pred
        state.step += 1
        if self.collect_diagnostics:
            state.residual_norms = [float(r.float().norm().cpu()) for r in residuals]
        else:
            state.residual_norms = []
        return residuals


class EPCReference:
    """Independent Transformer adaptation of error-based predictive coding.

    This is an *adapted reproduction*, not a claim that the public ePC code
    directly supports this architecture.  We keep an explicit error state
    ``e_i`` for every local block constraint instead of propagating one state
    error through all deeper blocks.  With ``r_i = h[i+1]-f_i(h[i];W_i)`` the
    reference update is

    ``e_i <- (1-beta)e_i + beta*r_i``
    ``h[i+1] <- h[i+1] - eta_h*e_i``
    ``W_i <- W_i - eta_w*d(0.5*||r_i||^2)/dW_i``.

    The target and input to each block are detached before its local graph is
    built.  ``autograd.grad`` is therefore local to one block; no global
    backward graph is retained.  Causal attention remains token-global within
    a block, so locality is ``block_local`` rather than strict token-local.
    """

    name = "epc"
    version = "independent-transformer-adaptation-0.1"
    locality = "block_local"
    strict_no_autograd = False
    uses_local_autograd = True

    def __init__(
        self,
        model,
        inner_steps: int = 4,
        eta_h: float = 0.1,
        beta: float = 0.5,
        parameter_step: float = 3e-4,
    ):
        if inner_steps < 1 or eta_h <= 0 or not (0 < beta <= 1) or parameter_step < 0:
            raise ValueError("invalid ePC step sizes")
        self.model = model
        self.inner_steps = int(inner_steps)
        self.eta_h = float(eta_h)
        self.beta = float(beta)
        self.parameter_step = float(parameter_step)

    def init_state(self, input_ids: torch.Tensor) -> MethodState:
        with torch.no_grad():
            _, _, hidden = self.model(input_ids, labels=input_ids, return_hidden=True)
        hs = [h.detach().clone() for h in hidden]
        # Explicit per-block errors are separate tensors, not aliases of h.
        errors = [torch.zeros_like(h) for h in hs[1:]]
        return MethodState(hs, [], errors=errors)

    @torch.no_grad()
    def relax_step(self, state: MethodState, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        del input_ids
        if state.errors is None or len(state.errors) != len(self.model.blocks):
            raise ValueError("ePC state is missing one error tensor per block")
        residuals: List[torch.Tensor] = []
        if labels is not None:
            h_last = state.hidden[-1].detach().requires_grad_(True)
            with torch.enable_grad():
                logits = self.model.lm_head(self.model.norm(h_last))
                loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
                output_grad = torch.autograd.grad(loss, h_last, only_inputs=True)[0]
            state.hidden[-1] = (h_last.detach() - self.eta_h * output_grad.detach()).detach()
        for i, block in enumerate(self.model.blocks):
            pred = block(state.hidden[i])
            residual = state.hidden[i + 1] - pred
            # EMA error-state optimization avoids repeatedly multiplying a
            # state signal by all preceding Jacobians.
            state.errors[i].lerp_(residual, self.beta)
            state.hidden[i + 1].add_(-self.eta_h * state.errors[i])
            residuals.append(residual.detach())
        state.step += 1
        state.residual_norms = [float(r.float().norm().cpu()) for r in residuals]
        state.error_norms = [float(e.float().norm().cpu()) for e in state.errors]
        state.error_mean = [float(e.float().mean().cpu()) for e in state.errors]
        state.error_var = [float(e.float().var(unbiased=False).cpu()) for e in state.errors]
        return residuals

    def local_update(
        self,
        state: MethodState,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        apply: bool = True,
    ) -> Dict[str, Any]:
        del input_ids
        losses: List[float] = []
        grad_norms: List[float] = []
        for i, block in enumerate(self.model.blocks):
            h_prev = state.hidden[i].detach()
            target = state.hidden[i + 1].detach()
            pred = block(h_prev)
            residual = target - pred
            # The ePC state is the explicit controller; the local weight
            # objective is the squared prediction error from this block.
            local_loss = 0.5 * residual.float().pow(2).mean()
            params = tuple(p for p in block.parameters() if p.requires_grad)
            grads = torch.autograd.grad(local_loss, params, allow_unused=True)
            sq = torch.zeros((), device=local_loss.device, dtype=torch.float32)
            for p, g in zip(params, grads):
                if g is None:
                    continue
                sq = sq + g.float().pow(2).sum()
                if apply and self.parameter_step:
                    with torch.no_grad():
                        p.add_(g.to(dtype=p.dtype), alpha=-self.parameter_step)
            losses.append(float(local_loss.detach().cpu()))
            grad_norms.append(float(sq.sqrt().detach().cpu()))
            del pred, residual, local_loss, grads
        output_grad_norm = 0.0
        if labels is not None:
            h_out = state.hidden[-1].detach()
            logits = self.model.lm_head(self.model.norm(h_out))
            output_loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
            out_grads = torch.autograd.grad(output_loss, (self.model.lm_head.weight,), allow_unused=True)
            if out_grads[0] is not None:
                output_grad_norm = float(out_grads[0].float().norm().detach().cpu())
                if apply and self.parameter_step:
                    with torch.no_grad():
                        self.model.lm_head.weight.add_(out_grads[0].to(dtype=self.model.lm_head.weight.dtype), alpha=-self.parameter_step)
        return {
            "local_loss": sum(losses) / max(1, len(losses)),
            "layer_grad_norm": grad_norms,
            "error_norm": sum(state.error_norms) / max(1, len(state.error_norms)),
            "output_grad_norm": output_grad_norm,
            "updated_layers": len(losses) if apply and self.parameter_step else 0,
        }

    @torch.no_grad()
    def diagnostics(self, state: MethodState) -> Dict[str, Any]:
        errors = state.errors or []
        if errors and not state.error_norms:
            state.error_norms = [float(e.float().norm().cpu()) for e in errors]
            state.error_mean = [float(e.float().mean().cpu()) for e in errors]
            state.error_var = [float(e.float().var(unbiased=False).cpu()) for e in errors]
        return {
            "step": state.step,
            "primal_residual_norm": sum(state.residual_norms) / max(1, len(state.residual_norms)),
            "error_norm": sum(state.error_norms) / max(1, len(state.error_norms)),
            "error_norm_by_layer": list(state.error_norms),
            "error_mean_by_layer": list(state.error_mean),
            "error_var_by_layer": list(state.error_var),
        }

    def step_cost(self, batch_tokens: int, elapsed_s: float, *, inference_steps: int | None = None) -> Dict[str, Any]:
        return {
            "batch_tokens": int(batch_tokens),
            "elapsed_s": float(elapsed_s),
            "tokens_per_s": int(batch_tokens) / max(float(elapsed_s), 1e-9),
            "inner_steps": self.inner_steps if inference_steps is None else int(inference_steps),
            "local_update_count": len(self.model.blocks),
            "method": self.name,
            "version": self.version,
            "locality": self.locality,
            "strict_no_autograd": self.strict_no_autograd,
        }

    def run_inference(self, state: MethodState, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> Dict[str, Any]:
        t0 = time.perf_counter()
        for _ in range(self.inner_steps):
            self.relax_step(state, input_ids, labels)
        return self.step_cost(int(input_ids.numel()), time.perf_counter() - t0)


class MetaPCNReference(EPCReference):
    """Adapted Meta-PCN controller with normalized prediction errors.

    This implementation isolates the meta controller from the ePC error rule.
    For each block, ``r_i=h[i+1]-f_i(h[i])`` is normalized with its current
    batch mean and variance, then the explicit error state is updated as
    ``e_i <- (1-meta_lr)e_i + meta_lr*(r_i-mu_i)/sqrt(var_i+eps)``.
    Hidden states receive ``-eta_h*e_i``.  Layer statistics are retained for
    diagnostics and the local weight objective remains the detached squared
    prediction error.  It is an independent Transformer adaptation; the
    published Meta-PCN controller and normalization details require a separate
    source-cell reproduction.
    """

    name = "meta_pcn"
    version = "meta-error-controller-0.1"
    locality = "block_local"
    strict_no_autograd = False
    uses_local_autograd = True

    def __init__(self, model, inner_steps: int = 4, eta_h: float = 0.1,
                 meta_lr: float = 0.5, parameter_step: float = 3e-4,
                 eps: float = 1e-5):
        super().__init__(model, inner_steps=inner_steps, eta_h=eta_h,
                         beta=meta_lr, parameter_step=parameter_step)
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.meta_lr = float(meta_lr)
        self.eps = float(eps)

    @torch.no_grad()
    def relax_step(self, state: MethodState, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        del input_ids
        if state.errors is None or len(state.errors) != len(self.model.blocks):
            raise ValueError("Meta-PCN state is missing one error tensor per block")
        residuals: List[torch.Tensor] = []
        if labels is not None:
            h_last = state.hidden[-1].detach().requires_grad_(True)
            with torch.enable_grad():
                logits = self.model.lm_head(self.model.norm(h_last))
                loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
                output_grad = torch.autograd.grad(loss, h_last, only_inputs=True)[0]
            state.hidden[-1] = (h_last.detach() - self.eta_h * output_grad.detach()).detach()
        means: List[float] = []
        variances: List[float] = []
        for i, block in enumerate(self.model.blocks):
            pred = block(state.hidden[i])
            residual = state.hidden[i + 1] - pred
            mu = residual.float().mean()
            var = residual.float().var(unbiased=False)
            normalized = (residual.float() - mu) * torch.rsqrt(var + self.eps)
            state.errors[i].lerp_(normalized.to(dtype=state.errors[i].dtype), self.meta_lr)
            state.hidden[i + 1].add_(-self.eta_h * state.errors[i])
            residuals.append(residual.detach())
            means.append(float(mu.cpu())); variances.append(float(var.cpu()))
        state.step += 1
        state.residual_norms = [float(r.float().norm().cpu()) for r in residuals]
        state.error_norms = [float(e.float().norm().cpu()) for e in state.errors]
        state.error_mean = [float(e.float().mean().cpu()) for e in state.errors]
        state.error_var = [float(e.float().var(unbiased=False).cpu()) for e in state.errors]
        # Keep raw residual moments in the same fields used by diagnostics;
        # layerwise error moments remain available above.
        state._meta_raw_mean = means
        state._meta_raw_var = variances
        return residuals

    @torch.no_grad()
    def diagnostics(self, state: MethodState) -> Dict[str, Any]:
        out = super().diagnostics(state)
        means = list(getattr(state, "_meta_raw_mean", state.error_mean))
        variances = list(getattr(state, "_meta_raw_var", state.error_var))
        norms = out["error_norm_by_layer"]
        avg = sum(norms) / max(1, len(norms))
        imbalance = max(norms) / max(avg, self.eps) if norms else 0.0
        out.update({"prediction_error_mean_by_layer": means,
                    "prediction_error_var_by_layer": variances,
                    "cross_layer_imbalance": imbalance,
                    "meta_lr": self.meta_lr})
        return out


class PrecisionWeightedPCReference:
    """Adapted precision-weighted deep-PC reference method.

    Each block keeps a positive scalar precision ``p_i``.  For residual
    ``r_i=h[i+1]-f_i(h[i])`` the reference update uses
    ``p_i <- (1-gamma)p_i + gamma/(mean(r_i^2)+eps)`` and
    ``h[i+1] <- h[i+1]-eta_h*p_i*r_i``.  Local weights minimize
    ``0.5*p_i*||r_i||^2`` on a detached block graph.  No dual variables are
    used, and precision remains bounded to expose instability rather than
    silently allowing overflow.  This is an adaptation, not a claim of exact
    reproduction of the cited deep-PC equations.
    """

    name = "precision_weighted_pc"
    version = "precision-weighted-0.1"
    locality = "block_local"
    strict_no_autograd = False
    uses_local_autograd = True

    def __init__(self, model, inner_steps: int = 4, eta_h: float = 0.1,
                 precision_ema: float = 0.1, parameter_step: float = 3e-4,
                 eps: float = 1e-5, precision_min: float = 1e-3,
                 precision_max: float = 1e3):
        if inner_steps < 1 or eta_h <= 0 or not (0 < precision_ema <= 1):
            raise ValueError("invalid precision-PC step sizes")
        if eps <= 0 or precision_min <= 0 or precision_max < precision_min:
            raise ValueError("invalid precision bounds")
        self.model = model; self.inner_steps = int(inner_steps)
        self.eta_h = float(eta_h); self.precision_ema = float(precision_ema)
        self.parameter_step = float(parameter_step); self.eps = float(eps)
        self.precision_min = float(precision_min); self.precision_max = float(precision_max)

    def init_state(self, input_ids: torch.Tensor) -> MethodState:
        with torch.no_grad():
            _, _, hidden = self.model(input_ids, labels=input_ids, return_hidden=True)
        hs = [h.detach().clone() for h in hidden]
        precisions = [torch.ones((), device=h.device, dtype=torch.float32) for h in hs[1:]]
        return MethodState(hs, [], errors=[torch.zeros_like(h) for h in hs[1:]], precisions=precisions)

    @torch.no_grad()
    def relax_step(self, state: MethodState, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        del input_ids
        if state.precisions is None or len(state.precisions) != len(self.model.blocks):
            raise ValueError("precision-PC state is missing one precision per block")
        residuals: List[torch.Tensor] = []
        if labels is not None:
            h_last = state.hidden[-1].detach().requires_grad_(True)
            with torch.enable_grad():
                logits = self.model.lm_head(self.model.norm(h_last))
                loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
                output_grad = torch.autograd.grad(loss, h_last, only_inputs=True)[0]
            state.hidden[-1] = (h_last.detach() - self.eta_h * output_grad.detach()).detach()
        for i, block in enumerate(self.model.blocks):
            pred = block(state.hidden[i])
            residual = state.hidden[i + 1] - pred
            variance = residual.float().pow(2).mean()
            target_precision = torch.clamp(variance.add(self.eps).reciprocal(), self.precision_min, self.precision_max)
            state.precisions[i].lerp_(target_precision, self.precision_ema)
            state.errors[i].copy_(residual)
            state.hidden[i + 1].add_(-self.eta_h * state.precisions[i].to(residual.dtype) * residual)
            residuals.append(residual.detach())
        state.step += 1
        state.residual_norms = [float(r.float().norm().cpu()) for r in residuals]
        state.error_norms = [float(e.float().norm().cpu()) for e in state.errors]
        state.error_mean = [float(e.float().mean().cpu()) for e in state.errors]
        state.error_var = [float(e.float().var(unbiased=False).cpu()) for e in state.errors]
        state.precision_mean = [float(p.cpu()) for p in state.precisions]
        return residuals

    def local_update(self, state: MethodState, input_ids: torch.Tensor,
                     labels: torch.Tensor | None = None, *, apply: bool = True) -> Dict[str, Any]:
        del input_ids, labels
        if state.precisions is None:
            raise ValueError("precision-PC state has no precisions")
        losses: List[float] = []; grad_norms: List[float] = []
        for i, block in enumerate(self.model.blocks):
            pred = block(state.hidden[i].detach())
            residual = state.hidden[i + 1].detach() - pred
            local_loss = 0.5 * state.precisions[i].detach() * residual.float().pow(2).mean()
            params = tuple(p for p in block.parameters() if p.requires_grad)
            grads = torch.autograd.grad(local_loss, params, allow_unused=True)
            sq = torch.zeros((), device=local_loss.device, dtype=torch.float32)
            for p, g in zip(params, grads):
                if g is None: continue
                sq = sq + g.float().pow(2).sum()
                if apply and self.parameter_step:
                    with torch.no_grad(): p.add_(g.to(dtype=p.dtype), alpha=-self.parameter_step)
            losses.append(float(local_loss.detach().cpu())); grad_norms.append(float(sq.sqrt().cpu()))
            del pred, residual, local_loss, grads
        return {"local_loss": sum(losses) / max(1, len(losses)),
                "layer_grad_norm": grad_norms,
                "precision_mean": sum(state.precision_mean) / max(1, len(state.precision_mean)),
                "updated_layers": len(losses) if apply and self.parameter_step else 0}

    @torch.no_grad()
    def diagnostics(self, state: MethodState) -> Dict[str, Any]:
        precisions = state.precisions or []
        if precisions and not state.precision_mean:
            state.precision_mean = [float(p.cpu()) for p in precisions]
        norms = state.error_norms
        avg = sum(norms) / max(1, len(norms))
        return {"step": state.step,
                "primal_residual_norm": sum(state.residual_norms) / max(1, len(state.residual_norms)),
                "error_norm_by_layer": list(norms),
                "precision_by_layer": list(state.precision_mean),
                "precision_mean": sum(state.precision_mean) / max(1, len(state.precision_mean)),
                "cross_layer_imbalance": max(norms) / max(avg, self.eps) if norms else 0.0}

    def step_cost(self, batch_tokens: int, elapsed_s: float, *, inference_steps: int | None = None) -> Dict[str, Any]:
        return {"batch_tokens": int(batch_tokens), "elapsed_s": float(elapsed_s),
                "tokens_per_s": int(batch_tokens) / max(float(elapsed_s), 1e-9),
                "inner_steps": self.inner_steps if inference_steps is None else int(inference_steps),
                "local_update_count": len(self.model.blocks), "method": self.name,
                "version": self.version, "locality": self.locality,
                "strict_no_autograd": self.strict_no_autograd}

    def run_inference(self, state: MethodState, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> Dict[str, Any]:
        t0 = time.perf_counter()
        for _ in range(self.inner_steps): self.relax_step(state, input_ids, labels)
        return self.step_cost(int(input_ids.numel()), time.perf_counter() - t0)


class StableParameterizationWrapper:
    """Apply the stable-depth initialization wrapper to a local-credit method.

    The wrapper scales each Transformer residual branch by ``1/sqrt(L)`` and
    optionally scales block linear weights at initialization by the same
    factor.  It delegates the common method interface to the wrapped method;
    no credit state is added and no claims are made for a standalone learner.
    """

    name = "stable_parameterization"
    version = "residual-scale-init-0.1"
    locality = "block_local"
    strict_no_autograd = False

    def __init__(self, base_method, residual_scale: float | None = None, init_scale: float | None = None):
        if not hasattr(base_method, "model") or not hasattr(base_method, "local_update"):
            raise TypeError("base_method must implement the local method interface")
        self.base_method = base_method
        self.model = base_method.model
        depth = max(1, len(self.model.blocks))
        self.residual_scale = float(residual_scale if residual_scale is not None else depth ** -0.5)
        self.init_scale = float(init_scale if init_scale is not None else depth ** -0.5)
        if self.residual_scale <= 0 or self.init_scale <= 0:
            raise ValueError("stable scales must be positive")
        for block in self.model.blocks:
            block.residual_scale = self.residual_scale
            # Idempotent marker prevents accidental double scaling on repeated
            # wrapper construction.
            if not getattr(block, "_stable_init_applied", False):
                with torch.no_grad():
                    for module in (block.attn.qkv, block.attn.proj, block.mlp.fc, block.mlp.proj):
                        module.weight.mul_(self.init_scale)
                block._stable_init_applied = True

    def __getattr__(self, name):
        # Delegate init_state, relax_step, local_update, diagnostics, and
        # step_cost while preserving this wrapper's metadata above.
        if name in {"base_method", "model", "residual_scale", "init_scale"}:
            raise AttributeError(name)
        return getattr(self.base_method, name)

    def diagnostics(self, state: MethodState) -> Dict[str, Any]:
        out = dict(self.base_method.diagnostics(state))
        out.update({"stable_residual_scale": self.residual_scale, "stable_init_scale": self.init_scale})
        return out

    def step_cost(self, batch_tokens: int, elapsed_s: float, *, inference_steps: int | None = None) -> Dict[str, Any]:
        out = dict(self.base_method.step_cost(batch_tokens, elapsed_s, inference_steps=inference_steps))
        out.update({"method": self.name, "version": self.version, "stable_residual_scale": self.residual_scale, "stable_init_scale": self.init_scale})
        return out


class WFActPCReference(EPCReference):
    """Minimal WF-Act-PC adapter using a local Jacobian-transpose action.

    The Transformer block graph is detached at each boundary and the local
    squared-error gradient is the available Jacobian-transpose action.  This
    keeps the forward-weight/frozen-normalization assumption explicit without
    pretending that attention Q/K/V derivatives have already been replaced by
    an analytic WF rule.  It is therefore an adapted scaffold, not a faithful
    reproduction of the vision implementation.
    """

    name = "wf_act_pc"
    version = "adapted-local-jacobian-0.1"
    locality = "block_local"

    def __init__(self, model, *args, feedback_gain: float = 1.0, **kwargs):
        super().__init__(model, *args, **kwargs)
        if feedback_gain <= 0:
            raise ValueError("feedback_gain must be positive")
        self.feedback_gain = float(feedback_gain)

    def local_update(self, state, input_ids, labels=None, *, apply=True):
        out = super().local_update(state, input_ids, labels, apply=apply)
        out.update({"feedback_rule": "local_jacobian_transpose", "feedback_gain": self.feedback_gain,
                    "frozen_normalization_assumption": True})
        return out

    @torch.no_grad()
    def diagnostics(self, state):
        out = super().diagnostics(state)
        out.update({"feedback_rule": "local_jacobian_transpose", "feedback_gain": self.feedback_gain,
                    "frozen_normalization_assumption": True})
        return out


class ClosedFormPrecisionPCReference(PrecisionWeightedPCReference):
    """One-step Gaussian posterior approximation for Transformer latents.

    Given predicted mean ``m=f_i(h_i)`` and current latent ``h``, use the
    scalar precision ``p_i`` to form ``h_new=(h+p_i*m)/(1+p_i)``.  Precision is
    adapted from the local residual variance and clipped positive.  This
    removes the repeated gradient relaxation loop but is only exact for the
    linear/Gaussian local block; nonlinear attention makes it an adaptation.
    """

    name = "closed_form_precision_pc"
    version = "adapted-gaussian-posterior-0.1"
    strict_no_autograd = True
    uses_local_autograd = False

    @torch.no_grad()
    def relax_step(self, state, input_ids, labels=None):
        del input_ids
        if state.precisions is None or len(state.precisions) != len(self.model.blocks):
            raise ValueError("closed-form state is missing one precision per block")
        if labels is not None:
            h_last = state.hidden[-1]
            logits = self.model.lm_head(self.model.norm(h_last)).float()
            err = torch.softmax(logits[:, :-1], -1)
            err.scatter_add_(-1, labels[:, 1:, None], -torch.ones_like(labels[:, 1:, None], dtype=err.dtype))
            feedback = err.matmul(self.model.lm_head.weight.detach().float())
            h_last[:, :-1].add_(-self.eta_h * feedback.to(dtype=h_last.dtype))
        residuals = []
        for i, block in enumerate(self.model.blocks):
            h_next = state.hidden[i + 1]
            pred = block(state.hidden[i])
            residual = h_next - pred
            var = residual.float().pow(2).mean()
            target_p = torch.clamp((var + self.eps).reciprocal(), self.precision_min, self.precision_max)
            state.precisions[i].lerp_(target_p, self.precision_ema)
            # Gaussian posterior mean with unit prior precision.
            weight = state.precisions[i] / (1.0 + state.precisions[i])
            h_next.lerp_(pred, weight.to(dtype=h_next.dtype))
            state.errors[i].copy_(residual)
            residuals.append(residual.detach())
        state.step += 1
        state.residual_norms = [float(r.float().norm().cpu()) for r in residuals]
        state.error_norms = [float(e.float().norm().cpu()) for e in state.errors]
        state.error_mean = [float(e.float().mean().cpu()) for e in state.errors]
        state.error_var = [float(e.float().var(unbiased=False).cpu()) for e in state.errors]
        state.precision_mean = [float(p.cpu()) for p in state.precisions]
        return residuals

    def local_update(self, state, input_ids, labels=None, *, apply=True):
        # The closed-form path has no local reverse graph.  Weight learning is
        # intentionally deferred to the matched local precision baseline until
        # an analytic block Jacobian is validated.
        return {"local_loss": 0.0, "layer_grad_norm": [0.0] * len(self.model.blocks),
                "precision_mean": sum(state.precision_mean) / max(1, len(state.precision_mean)),
                "updated_layers": 0, "analytic_weight_update": False}


class DirectFeedbackAlignmentReference(EPCReference):
    """One-step output-error feedback with transpose-tied low-cost maps."""

    name = "direct_feedback_alignment"
    version = "transpose-tied-feedback-0.1"
    locality = "relaxed_feedback"
    strict_no_autograd = False

    def __init__(self, model, *args, feedback_gain: float = 1.0, **kwargs):
        super().__init__(model, *args, **kwargs)
        if feedback_gain <= 0:
            raise ValueError("feedback_gain must be positive")
        self.feedback_gain = float(feedback_gain)

    @torch.no_grad()
    def relax_step(self, state, input_ids, labels=None):
        del input_ids
        if state.errors is None or len(state.errors) != len(self.model.blocks):
            raise ValueError("feedback-alignment state is missing error tensors")
        feedback = torch.zeros_like(state.hidden[-1])
        if labels is not None:
            logits = self.model.lm_head(self.model.norm(state.hidden[-1])).float()
            err = torch.softmax(logits[:, :-1], -1)
            err.scatter_add_(-1, labels[:, 1:, None], -torch.ones_like(labels[:, 1:, None], dtype=err.dtype))
            feedback[:, :-1].copy_(err.matmul(self.model.lm_head.weight.detach().float()).to(dtype=feedback.dtype))
        residuals = []
        for i, block in enumerate(self.model.blocks):
            pred = block(state.hidden[i])
            residual = state.hidden[i + 1] - pred
            state.errors[i].lerp_(residual, self.beta)
            state.hidden[i + 1].add_(residual, alpha=-self.eta_h)
            # The same low-cost output feedback reaches each layer in one
            # sweep; this is deliberately marked relaxed_feedback.
            state.hidden[i + 1].add_(feedback, alpha=-self.eta_h * self.feedback_gain)
            residuals.append(residual.detach())
        state.step += 1
        state.residual_norms = [float(r.float().norm().cpu()) for r in residuals]
        state.error_norms = [float(e.float().norm().cpu()) for e in state.errors]
        return residuals

    @torch.no_grad()
    def diagnostics(self, state):
        out = super().diagnostics(state)
        out.update({"feedback_rule": "transpose_tied_output", "feedback_gain": self.feedback_gain,
                    "feedback_latency": 1})
        return out


class StreamAmortizedPCReference(EPCReference):
    """ePC adapter with rank-local hidden-state carry between contiguous chunks."""

    name = "stream_amortized_pc"
    version = "carry-over-0.1"
    locality = "block_local"

    def __init__(self, model, *args, carry_decay: float = 1.0, **kwargs):
        super().__init__(model, *args, **kwargs)
        if not (0.0 <= carry_decay <= 1.0):
            raise ValueError("carry_decay must lie in [0,1]")
        self.carry_decay = float(carry_decay)
        self._carry_hidden = None

    def reset_carry(self):
        self._carry_hidden = None

    def init_state(self, input_ids):
        state = super().init_state(input_ids)
        if self._carry_hidden is not None and len(self._carry_hidden) == len(state.hidden):
            for i in range(1, len(state.hidden)):
                if self._carry_hidden[i].shape == state.hidden[i].shape:
                    state.hidden[i].lerp_(self._carry_hidden[i].to(state.hidden[i]), self.carry_decay)
            state.carry_used = True
        else:
            state.carry_used = False
        return state

    def local_update(self, state, input_ids, labels=None, *, apply=True):
        out = super().local_update(state, input_ids, labels, apply=apply)
        self._carry_hidden = [h.detach().clone() for h in state.hidden]
        out.update({"carry_decay": self.carry_decay, "carry_saved": True})
        return out

    @torch.no_grad()
    def diagnostics(self, state):
        out = super().diagnostics(state)
        out.update({"carry_decay": self.carry_decay, "carry_used": bool(getattr(state, "carry_used", False))})
        return out


class PCALMControlVariateReference(PCALMReference):
    """Forward-only PC-ALM predictor plus fresh antithetic probes.

    For each block, a random parameter direction ``u`` is probed at ``+eps``
    and ``-eps``.  The directional finite-difference credit is decomposed as
    ``d = predictor + residual`` where the predictor is the local dual/error
    scalar and only the residual is estimated after the probe.  This is a new
    hybrid adaptation; probes are parameter-space directions because a direct
    activation perturbation alone cannot update Transformer weights.
    """

    name = "pc_alm_control_variate"
    version = "activation-probe-control-variate-0.1"
    strict_no_autograd = True
    uses_local_autograd = False

    def __init__(self, model, *args, probe_count: int = 1, probe_eps: float = 1e-3,
                 use_predictor: bool = True, **kwargs):
        super().__init__(model, *args, **kwargs)
        if probe_count < 1 or probe_eps <= 0:
            raise ValueError("probe_count/probe_eps must be positive")
        self.probe_count = int(probe_count); self.probe_eps = float(probe_eps)
        self.use_predictor = bool(use_predictor)
        self._last_probe_variance = 0.0; self._last_predictor = 0.0

    @torch.no_grad()
    def relax_step(self, state, input_ids, labels=None):
        # The strict arm uses the PC-ALM latent/dual signal without an output
        # autograd graph; output supervision is represented by later probes.
        return super().relax_step(state, input_ids, None)

    @torch.no_grad()
    def local_update(self, state, input_ids, labels=None, *, apply=True):
        del input_ids, labels
        losses = []; grad_norms = []; residual_samples = []; predictor_samples = []
        for i, block in enumerate(self.model.blocks):
            h_prev = state.hidden[i].detach(); target = state.hidden[i + 1].detach()
            pred = block(h_prev); residual = target - pred
            base_loss = 0.5 * residual.float().pow(2).mean()
            # Dual/error predictor is a scalar control variate for the local
            # directional credit; center it to avoid changing the estimator's
            # expectation when conditional independence holds.
            predictor = float((residual.float() * state.dual[i].float()).mean().cpu())
            params = tuple(p for p in block.parameters() if p.requires_grad)
            accum = [torch.zeros_like(p) for p in params]
            for _ in range(self.probe_count):
                directions = [torch.randn_like(p) for p in params]
                for p, d in zip(params, directions): p.add_(self.probe_eps * d)
                lp = 0.5 * (target - block(h_prev)).float().pow(2).mean()
                for p, d in zip(params, directions): p.add_(-2.0 * self.probe_eps * d)
                lm = 0.5 * (target - block(h_prev)).float().pow(2).mean()
                for p, d in zip(params, directions): p.add_(self.probe_eps * d)
                directional = float(((lp - lm) / (2.0 * self.probe_eps)).cpu())
                residual_credit = directional - predictor if self.use_predictor else directional
                credit = (predictor + residual_credit) if self.use_predictor else residual_credit
                for a, d in zip(accum, directions): a.add_(d, alpha=credit / self.probe_count)
                residual_samples.append(residual_credit); predictor_samples.append(predictor)
            if apply and self.parameter_step:
                for p, g in zip(params, accum): p.add_(g, alpha=-self.parameter_step)
            losses.append(float(base_loss.cpu())); grad_norms.append(float(torch.stack([g.float().pow(2).sum() for g in accum]).sum().sqrt().cpu()))
        self._last_probe_variance = float(torch.tensor(residual_samples).var(unbiased=False).cpu()) if residual_samples else 0.0
        self._last_predictor = sum(predictor_samples) / max(1, len(predictor_samples))
        return {"local_loss": sum(losses) / max(1, len(losses)), "layer_grad_norm": grad_norms,
                "probe_variance": self._last_probe_variance, "predictor_mean": self._last_predictor,
                "updated_layers": len(losses) if apply and self.parameter_step else 0}

    @torch.no_grad()
    def diagnostics(self, state):
        out = super().diagnostics(state)
        out.update({"probe_count": self.probe_count, "probe_eps": self.probe_eps,
                    "probe_variance": self._last_probe_variance,
                    "predictor_mean": self._last_predictor})
        return out
