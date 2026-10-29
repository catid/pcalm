#!/usr/bin/env python
"""Check eager/compiled Transformer equivalence on a fixed batch.

The gate intentionally uses a small model by default so it can run on a
developer machine.  ``--full`` switches to the frozen 40M architecture.  The
same weights, inputs, labels, and optimizer update are used for eager and
compiled paths; the script writes a JSON record that can be attached to the
engineering gate.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from src.model import CausalTransformer, GPTConfig


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float() - b.detach().float()).abs().max().cpu())


def _run(model: torch.nn.Module, ids: torch.Tensor, *, steps: int, lr: float):
    """Run a fixed eager/compiled path and return detached observables."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    model.train()
    losses = []
    t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss, _ = model(ids, ids)
        loss.backward()
        opt.step()
        losses.append(loss.detach().float().cpu())
    if torch.cuda.is_available() and ids.is_cuda:
        torch.cuda.synchronize(ids.device)
    return {
        "losses": torch.stack(losses),
        "weight": model.token_embedding.weight.detach().float().cpu().clone(),
        "elapsed_s": time.perf_counter() - t0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/optimized_equivalence.json")
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--full", action="store_true", help="use the frozen 40M model")
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()
    if args.steps < 1 or args.batch < 1 or args.seq_len < 2:
        raise SystemExit("steps, batch, and seq-len must be positive (seq-len >= 2)")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    cfg = GPTConfig() if args.full else GPTConfig(vocab_size=257, n_layers=2, d_model=64, n_heads=4, d_ff=256, max_seq_len=max(args.seq_len, 16))
    # Keep identical initial state and fixed data for both paths.
    template = CausalTransformer(cfg).to(device)
    ids = torch.randint(cfg.vocab_size, (args.batch, args.seq_len), device=device)
    state = {k: v.detach().clone() for k, v in template.state_dict().items()}
    eager = CausalTransformer(cfg).to(device)
    eager.load_state_dict(state)
    eager_result = _run(eager, ids, steps=args.steps, lr=3e-4)
    result = {"status": "pass", "device": str(device), "compiled": False,
              "seed": args.seed, "steps": args.steps, "batch": args.batch,
              "seq_len": args.seq_len, "full": args.full,
              "eager_elapsed_s": eager_result["elapsed_s"]}
    if not args.no_compile and hasattr(torch, "compile"):
        compiled_base = CausalTransformer(cfg).to(device)
        compiled_base.load_state_dict(state)
        # Reduce surprise recompiles for fixed-shape gate.  Fullgraph is not
        # required because SDPA/backend details vary by torch release.
        compiled = torch.compile(compiled_base, mode="default", dynamic=False)
        compiled_result = _run(compiled, ids, steps=args.steps, lr=3e-4)
        loss_err = _max_abs(eager_result["losses"], compiled_result["losses"])
        weight_err = _max_abs(eager_result["weight"], compiled_result["weight"])
        result.update({"compiled": True, "compiled_elapsed_s": compiled_result["elapsed_s"],
                       "max_loss_abs_err": loss_err, "max_weight_abs_err": weight_err})
        # BF16/SDPA kernels can differ by a few ulps; this is intentionally
        # tighter for FP32 CPU and relaxed for CUDA production kernels.
        tol = 5e-4 if device.type == "cuda" else 1e-6
        if loss_err > tol or weight_err > 5e-4:
            result["status"] = "fail"
    else:
        result["reason"] = "torch.compile unavailable or disabled"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
