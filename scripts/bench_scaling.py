#!/usr/bin/env python
"""Short local NCCL/data-parallel scaling benchmark.

Run with torchrun --standalone --nproc-per-node=N scripts/bench_scaling.py.
The tensor size approximates the 40M-parameter model's BF16 gradient bucket.
"""
import json, os, time, socket
import torch
import torch.distributed as dist

def main():
    backend = os.environ.get("PCALM_BACKEND", "nccl")
    dist.init_process_group(backend)
    rank, world = dist.get_rank(), dist.get_world_size(); local = int(os.environ.get("LOCAL_RANK", rank))
    if backend == "nccl": torch.cuda.set_device(local)
    # 40M BF16 values, reduced in chunks to stay comfortably below VRAM.
    device = "cuda" if backend == "nccl" else "cpu"
    x = torch.ones(10_000_000, dtype=torch.bfloat16, device=device) * (rank + 1)
    for _ in range(3):
        dist.all_reduce(x)
        if device == "cuda": torch.cuda.synchronize()
    times = []
    for _ in range(10):
        x.fill_(rank + 1)
        if device == "cuda": torch.cuda.synchronize()
        start = time.perf_counter(); dist.all_reduce(x)
        if device == "cuda": torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    payload = {"host": socket.gethostname(), "rank": rank, "world_size": world, "local_rank": local, "median_s": sorted(times)[len(times)//2], "mean_s": sum(times)/len(times), "bytes": x.numel()*x.element_size(), "bandwidth_GBps": x.numel()*x.element_size()/sorted(times)[len(times)//2]/1e9}
    if rank == 0: print(json.dumps(payload))
    dist.destroy_process_group()

if __name__ == "__main__": main()
