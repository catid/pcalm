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
        del input_ids, labels
        residuals: List[torch.Tensor] = []
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
        del input_ids, labels
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
        return {
            "local_loss": sum(losses) / max(1, len(losses)),
            "layer_grad_norm": grad_norms,
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
