"""Common method protocol and slow PC-ALM reference state path."""
from dataclasses import dataclass
import torch

@dataclass
class MethodState:
    hidden: list
    dual: list
    step: int = 0

class PCALMReference:
    name, version, locality, strict_no_autograd = "pc_alm", "reference-0.1", "block_local", False
    def __init__(self, model, inner_steps=4, eta_h=0.1, alpha=0.01, rho=1.0, parameter_step=3e-4):
        self.model, self.inner_steps, self.eta_h, self.alpha, self.rho, self.parameter_step = model, inner_steps, eta_h, alpha, rho, parameter_step
    def init_state(self, input_ids):
        with torch.no_grad(): _, _, hidden = self.model(input_ids, labels=input_ids, return_hidden=True)
        return MethodState([h.detach().clone().requires_grad_(True) for h in hidden], [torch.zeros_like(h) for h in hidden[1:]])
    def relax_step(self, state, input_ids, labels):
        residuals = []
        for i, block in enumerate(self.model.blocks):
            pred = block(state.hidden[i].detach()); residual = state.hidden[i + 1].detach() - pred.detach()
            state.hidden[i + 1] = (state.hidden[i + 1].detach() - self.eta_h * (self.rho * residual + state.dual[i])).requires_grad_(True)
            state.dual[i] = state.dual[i] + self.alpha * residual; residuals.append(residual)
        state.step += 1; return residuals
    def local_update(self, state, input_ids, labels):
        with torch.no_grad():
            x = self.model.token_embedding(input_ids)
            for block in self.model.blocks: x = block(x)
            logits = self.model.lm_head(self.model.norm(x)); logits_for_loss, labels_for_loss = logits[:, :-1], labels[:, 1:]
            loss = torch.nn.functional.cross_entropy(logits_for_loss.reshape(-1, logits.size(-1)), labels_for_loss.reshape(-1))
            err = torch.softmax(logits_for_loss.float(), -1); err.scatter_add_(-1, labels_for_loss[..., None], -torch.ones_like(labels_for_loss[..., None], dtype=err.dtype))
        return {"loss": float(loss), "output_error_norm": float(err.norm())}
    def diagnostics(self, state):
        vals = [d.detach().float().norm() for d in state.dual]
        return {"step": state.step, "dual_norm": float(torch.stack(vals).mean()) if vals else 0.0}
    def step_cost(self, batch_tokens, elapsed_s):
        return {"batch_tokens": batch_tokens, "elapsed_s": elapsed_s, "tokens_per_s": batch_tokens / max(elapsed_s, 1e-9), "inner_steps": self.inner_steps}
