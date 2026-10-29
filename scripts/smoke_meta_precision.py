#!/usr/bin/env python
"""Tiny deterministic smoke for Meta-PCN and precision-weighted PC adaptations."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.model import GPTConfig, CausalTransformer
from src.methods import MetaPCNReference, PrecisionWeightedPCReference


def run(cls, seed, device):
    torch.manual_seed(seed)
    cfg = GPTConfig(vocab_size=97, n_layers=2, d_model=32, n_heads=4, d_ff=64, max_seq_len=8)
    model = CausalTransformer(cfg).to(device)
    ids = torch.randint(cfg.vocab_size, (1, 8), device=device)
    method = cls(model, inner_steps=2, parameter_step=1e-4)
    state = method.init_state(ids)
    state.hidden[-1].add_(0.01 * torch.randn_like(state.hidden[-1]))
    for _ in range(2): method.relax_step(state, ids, ids)
    update = method.local_update(state, ids, ids)
    diagnostics = method.diagnostics(state)
    if not all(torch.isfinite(x).all() for x in state.hidden):
        raise RuntimeError(f"non-finite hidden state in {method.name}")
    if state.precisions is not None and not all(torch.isfinite(x).all() and (x > 0).all() for x in state.precisions):
        raise RuntimeError("invalid precision")
    return {"method": method.name, "update": update, "diagnostics": diagnostics}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=20261029); args = ap.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available(): raise SystemExit("CUDA unavailable")
    device = torch.device(args.device)
    out = [run(MetaPCNReference, args.seed, device), run(PrecisionWeightedPCReference, args.seed + 1, device)]
    print(json.dumps({"device": str(device), "results": out, "status": "pass"}, sort_keys=True))

if __name__ == "__main__": main()
