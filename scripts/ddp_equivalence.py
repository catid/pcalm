#!/usr/bin/env python
"""One-step DDP versus single-rank equivalence on the fixed model."""
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from src.model import GPTConfig, CausalTransformer

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output", required=True); args = ap.parse_args()
    dist.init_process_group("nccl"); rank = dist.get_rank(); local = int(os.environ.get("LOCAL_RANK", rank)); torch.cuda.set_device(local)
    torch.manual_seed(777); model = CausalTransformer(GPTConfig(vocab_size=32000)).cuda(); wrapped = DDP(model, device_ids=[local], gradient_as_bucket_view=True, static_graph=True)
    g = torch.Generator(device="cuda").manual_seed(1234); ids = torch.randint(32000, (1, 128), generator=g, device="cuda"); opt = torch.optim.AdamW(wrapped.parameters(), lr=3e-4)
    opt.zero_grad(set_to_none=True); loss, _ = wrapped(ids, ids); loss.backward(); opt.step(); torch.cuda.synchronize()
    checksum = float(sum(p.detach().float().sum() for p in model.parameters()))
    if rank == 0: Path(args.output).write_text(json.dumps({"world_size": dist.get_world_size(), "loss": float(loss.detach()), "checksum": checksum}, indent=2))
    dist.barrier(); dist.destroy_process_group()
if __name__ == "__main__": main()
