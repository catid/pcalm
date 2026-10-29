#!/usr/bin/env python
"""Deterministic PC-ALM reference smoke; intended for one GPU or CPU."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.model import GPTConfig, CausalTransformer
from src.methods import PCALMReference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261029)
    args = ap.parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be positive")
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device(args.device)
    cfg = GPTConfig(vocab_size=97, n_layers=2, d_model=32, n_heads=4, d_ff=64, max_seq_len=16)
    model = CausalTransformer(cfg).to(device)
    ids = torch.randint(cfg.vocab_size, (1, cfg.max_seq_len), device=device)
    method = PCALMReference(model, inner_steps=args.steps, parameter_step=1e-4)
    state = method.init_state(ids)
    # Perturb one lifted latent state so the smoke exercises nonzero primal,
    # dual, and local parameter signals instead of only the forward fixed point.
    state.hidden[-1].add_(0.01 * torch.randn_like(state.hidden[-1]))
    for _ in range(args.steps):
        method.relax_step(state, ids, ids)
    # Instrument global backward: local_update is allowed to use autograd.grad,
    # but a global Tensor.backward call is a correctness failure.
    original = torch.Tensor.backward
    def forbidden(*_args, **_kwargs):
        raise AssertionError("global backward called by PC-ALM local update")
    torch.Tensor.backward = forbidden
    try:
        update = method.local_update(state, ids, ids)
    finally:
        torch.Tensor.backward = original
    diag = method.diagnostics(state)
    if not all(torch.isfinite(x).all() for x in state.hidden + state.dual):
        raise RuntimeError("non-finite PC-ALM state")
    out = {"device": str(device), "seed": args.seed, "steps": args.steps, "update": update, "diagnostics": diag, "status": "pass"}
    print(json.dumps(out, sort_keys=True))


if __name__ == "__main__":
    main()
