#!/usr/bin/env python
"""Short eager benchmark/equivalence check for PC-ALM relaxation paths.

The default tiny configuration runs quickly on one GPU or CPU and prints JSON
with warm timing and max state differences.  It is an engineering benchmark,
not a training result.
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.model import GPTConfig, CausalTransformer
from src.methods import PCALMReference, PCALMOptimized


def clone_state(state):
    # Keep a detached copy suitable for feeding a second method instance.
    from src.methods import MethodState
    return MethodState([h.detach().clone() for h in state.hidden], [d.detach().clone() for d in state.dual],
                       step=state.step, residual_norms=list(state.residual_norms))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20261030)
    ap.add_argument("--seq-len", type=int, default=32)
    args = ap.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if args.iters < 1 or args.warmup < 0:
        raise SystemExit("iters must be positive and warmup nonnegative")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    cfg = GPTConfig(vocab_size=97, n_layers=2, d_model=32, n_heads=4, d_ff=64, max_seq_len=args.seq_len)
    base = CausalTransformer(cfg).to(device)
    ref_model = CausalTransformer(cfg).to(device); ref_model.load_state_dict(base.state_dict())
    opt_model = CausalTransformer(cfg).to(device); opt_model.load_state_dict(base.state_dict())
    ids = torch.randint(cfg.vocab_size, (1, args.seq_len), device=device)
    labels = ids.clone()
    ref = PCALMReference(ref_model, inner_steps=1, eta_h=0.07, alpha=0.01, parameter_step=0.0)
    opt = PCALMOptimized(opt_model, inner_steps=1, eta_h=0.07, alpha=0.01, parameter_step=0.0,
                         collect_diagnostics=False)
    sr, so = ref.init_state(ids), opt.init_state(ids)
    # Create a nontrivial local residual while retaining identical state input.
    noise = 0.01 * torch.randn_like(sr.hidden[-1]); sr.hidden[-1].add_(noise); so.hidden[-1].add_(noise)
    for _ in range(args.warmup):
        ref.relax_step(sr, ids, labels); opt.relax_step(so, ids, labels)
    # Reset states after warmup to compare the same exact trajectory.
    sr, so = ref.init_state(ids), opt.init_state(ids)
    sr.hidden[-1].add_(noise); so.hidden[-1].add_(noise)
    if device.type == "cuda": torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(args.iters): ref.relax_step(sr, ids, labels)
    if device.type == "cuda": torch.cuda.synchronize(device)
    ref_s = time.perf_counter() - t0
    if device.type == "cuda": torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(args.iters): opt.relax_step(so, ids, labels)
    if device.type == "cuda": torch.cuda.synchronize(device)
    opt_s = time.perf_counter() - t0
    hidden_diff = max(float((a - b).abs().max().cpu()) for a, b in zip(sr.hidden, so.hidden))
    dual_diff = max(float((a - b).abs().max().cpu()) for a, b in zip(sr.dual, so.dual))
    if hidden_diff > 2e-5 or dual_diff > 2e-5:
        raise RuntimeError(f"optimized/reference mismatch hidden={hidden_diff} dual={dual_diff}")
    print(json.dumps({"status": "pass", "device": str(device), "iters": args.iters,
                      "reference_s": ref_s, "optimized_s": opt_s,
                      "reference_steps_per_s": args.iters / max(ref_s, 1e-12),
                      "optimized_steps_per_s": args.iters / max(opt_s, 1e-12),
                      "speedup": ref_s / max(opt_s, 1e-12),
                      "max_hidden_abs_diff": hidden_diff, "max_dual_abs_diff": dual_diff}, sort_keys=True))


if __name__ == "__main__": main()
