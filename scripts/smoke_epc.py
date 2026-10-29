#!/usr/bin/env python
"""Tiny deterministic smoke for the adapted ePC and stable wrapper.

This validates shape/finite-state/local-update behavior only; it is not a
10M-token method gate or a claim of faithful Transformer reproduction.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from src.model import GPTConfig, CausalTransformer
from src.methods import EPCReference, StableParameterizationWrapper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261029)
    args = ap.parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    cfg = GPTConfig(vocab_size=97, n_layers=2, d_model=32, n_heads=4, d_ff=64, max_seq_len=16)
    model = CausalTransformer(cfg).to(device)
    ids = torch.randint(cfg.vocab_size, (1, cfg.max_seq_len), device=device)
    method = StableParameterizationWrapper(EPCReference(model, inner_steps=args.steps, parameter_step=1e-4))
    state = method.init_state(ids)
    # Ensure the explicit error state is exercised away from the exact forward
    # fixed point while retaining deterministic inputs and initialization.
    state.hidden[-1].add_(0.01 * torch.randn_like(state.hidden[-1]))
    for _ in range(args.steps):
        method.relax_step(state, ids, ids)
    update = method.local_update(state, ids, ids)
    diagnostics = method.diagnostics(state)
    if state.errors is None or not all(torch.isfinite(x).all() for x in state.errors):
        raise RuntimeError("non-finite ePC error state")
    print(json.dumps({"device": str(device), "seed": args.seed, "steps": args.steps,
                      "update": update, "diagnostics": diagnostics, "status": "pass"}, sort_keys=True))


if __name__ == "__main__":
    main()
