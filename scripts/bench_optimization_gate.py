#!/usr/bin/env python
"""Measured fixed-shape optimization gate for the frozen Transformer.

Two short, independent checks live behind one command:

``--mode ddp``
    Runs fixed-shape AdamW steps for each requested per-rank batch size.  Run
    with ``torchrun`` to measure a DDP world size; with no launcher it uses a
    single rank.  The reported step time is the maximum median across ranks,
    so a slow rank cannot be hidden by rank-0 timing.

``--mode graph``
    Captures a fixed-shape forward/backward/AdamW step in a CUDA graph,
    compares its first update with an eager update from identical state, then
    measures replay.  The graph path is deliberately opt-in and reports a
    structured skip on CPU rather than pretending that graph support passed.

The default is the frozen 40M model.  ``--tiny`` is only for a fast local
smoke of the harness and must not be used as a production result.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.model import CausalTransformer, GPTConfig, count_parameters


def _config(args: argparse.Namespace) -> GPTConfig:
    if args.tiny:
        return GPTConfig(vocab_size=257, n_layers=2, d_model=64, n_heads=4,
                         d_ff=256, max_seq_len=max(args.seq_len, 16))
    return GPTConfig(max_seq_len=args.seq_len)


def _device(args: argparse.Namespace) -> torch.device:
    requested = args.device
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _autocast(device: torch.device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                          enabled=device.type == "cuda")


def _step(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
          ids: torch.Tensor, device: torch.device) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)
    with _autocast(device):
        loss, _ = model(ids, ids)
    loss.backward()
    optimizer.step()
    return loss.detach()


def _common(args: argparse.Namespace, cfg: GPTConfig, device: torch.device) -> dict[str, Any]:
    parts, nparams = count_parameters(CausalTransformer(cfg))
    return {
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": " ".join(sys.argv),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "tiny": bool(args.tiny),
        "model_parameters": int(nparams),
        "model_parameter_components": parts,
        "seq_len": int(args.seq_len),
        "seed": int(args.seed),
    }


def _init_dist(device: torch.device) -> tuple[int, int, int, str]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", str(rank)))
    backend = "nccl" if device.type == "cuda" else "gloo"
    if world > 1:
        dist.init_process_group(backend=backend)
    if device.type == "cuda":
        torch.cuda.set_device(local)
    return rank, world, local, backend


def _barrier(world: int) -> None:
    if world > 1:
        dist.barrier()


def _ddp(args: argparse.Namespace) -> dict[str, Any]:
    device = _device(args)
    cfg = _config(args)
    rank, world, local, backend = _init_dist(device)
    if device.type == "cuda":
        torch.cuda.set_device(local)
    torch.manual_seed(args.seed)
    base = _common(args, cfg, device)
    records: list[dict[str, Any]] = []
    for batch in args.batch_sizes:
        model = CausalTransformer(cfg).to(device)
        wrapped: torch.nn.Module = model
        if world > 1:
            wrapped = DDP(model, device_ids=[local] if device.type == "cuda" else None,
                          gradient_as_bucket_view=True, static_graph=True)
        optimizer = torch.optim.AdamW(wrapped.parameters(), lr=args.lr, weight_decay=0.0)
        generator = torch.Generator(device=device).manual_seed(args.seed + batch)
        ids = torch.randint(cfg.vocab_size, (batch, args.seq_len), generator=generator, device=device)
        for _ in range(args.warmup):
            _step(wrapped, optimizer, ids, device)
        _barrier(world)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
        times: list[float] = []
        for _ in range(args.steps):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _step(wrapped, optimizer, ids, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        local_median = sorted(times)[len(times) // 2]
        if world > 1:
            t = torch.tensor(local_median, dtype=torch.float64, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            step_s = float(t.cpu())
            hosts = [None] * world
            dist.all_gather_object(hosts, socket.gethostname())
        else:
            step_s = local_median
            hosts = [socket.gethostname()]
        peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        record = {
            "batch_per_rank": int(batch),
            "world_size": int(world),
            "rank": int(rank),
            "local_rank": int(local),
            "hosts": hosts,
            "backend": backend,
            "warmup_steps": int(args.warmup),
            "measure_steps": int(args.steps),
            "median_step_s_max_rank": step_s,
            "tokens_per_s_global": float(batch * args.seq_len * world / max(step_s, 1e-12)),
            "peak_memory_allocated_bytes_rank0": peak if rank == 0 else None,
            "same_fixed_input_each_step": True,
            "status": "pass" if all(torch.isfinite(x).all() for x in model.parameters()) else "fail",
        }
        records.append(record)
        del wrapped, model, optimizer, ids
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _barrier(world)
    result = dict(base, gate="ddp_fixed_shape_batch_scaling", world_size=world,
                  rank=rank, local_rank=local, backend=backend, records=records)
    if world > 1:
        _barrier(world)
        if rank == 0:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        dist.barrier()
        dist.destroy_process_group()
    else:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if rank == 0:
        print(json.dumps(result, sort_keys=True))
    return result


def _state_reset(optimizer: torch.optim.Optimizer) -> None:
    """Reset initialized AdamW state without reallocating graph-captured tensors."""
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                value.zero_()


def _graph(args: argparse.Namespace) -> dict[str, Any]:
    device = _device(args)
    cfg = _config(args)
    base = _common(args, cfg, device)
    result: dict[str, Any] = dict(base, gate="cuda_graph_capture_replay")
    if device.type != "cuda":
        result.update(status="skipped_no_cuda", reason="CUDA graphs require a CUDA device")
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, sort_keys=True))
        return result

    torch.manual_seed(args.seed)
    ids = torch.randint(cfg.vocab_size, (args.batch_sizes[0], args.seq_len), device=device)
    template = CausalTransformer(cfg).to(device)
    initial = {k: v.detach().clone() for k, v in template.state_dict().items()}
    eager = CausalTransformer(cfg).to(device)
    eager.load_state_dict(initial)
    graph_model = CausalTransformer(cfg).to(device)
    graph_model.load_state_dict(initial)
    eager_opt = torch.optim.AdamW(eager.parameters(), lr=args.lr, weight_decay=0.0)
    graph_opt = torch.optim.AdamW(graph_model.parameters(), lr=args.lr, weight_decay=0.0, capturable=True)
    # Initialize AdamW's CUDA state before capture, then return it to the exact
    # zero state.  Allocating state tensors during capture is a graph bug.
    for _ in range(max(1, args.warmup)):
        _step(graph_model, graph_opt, ids, device)
    # Warmup is only for allocator/kernel setup.  Restore the exact initial
    # parameters before capture so the eager and graph first updates compare
    # the same state; resetting AdamW moments alone would otherwise leave the
    # graph model several optimization steps ahead.
    graph_model.load_state_dict(initial)
    _state_reset(graph_opt)
    torch.cuda.synchronize()
    eager_loss = _step(eager, eager_opt, ids, device)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    # PyTorch requires graph capture on a non-default stream for this
    # execution path.  Keep all capture operations on one dedicated stream;
    # the explicit wait makes the stream observe the eager warmup/allocation
    # work without introducing an implicit default-stream capture.
    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(torch.cuda.current_stream(device))
    static_loss: torch.Tensor
    captured_loss: torch.Tensor
    try:
        with torch.cuda.stream(capture_stream):
            graph.capture_begin()
            # Keep gradient storage allocated and stable across replay.
            # Setting grads to None inside a captured graph can introduce
            # pointer changes or allocator work on replay.
            graph_opt.zero_grad(set_to_none=False)
            with _autocast(device):
                static_loss, _ = graph_model(ids, ids)
            # Keep a dedicated scalar copy of the forward result before
            # backward.  Some CUDA/SDPA backward kernels reuse the original
            # loss storage; reading ``static_loss`` after capture can then
            # report zero despite a correct parameter update.
            captured_loss = static_loss.detach().clone()
            static_loss.backward()
            graph_opt.step()
            graph.capture_end()
        capture_stream.synchronize()
        torch.cuda.synchronize()
        graph_loss = captured_loss.detach().clone()
        param_err = max(float((p.detach().float() - q.detach().float()).abs().max().cpu())
                        for p, q in zip(eager.parameters(), graph_model.parameters()))
        loss_err = float((eager_loss.float() - graph_loss.float()).abs().cpu())
        replay_times: list[float] = []
        for _ in range(args.steps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            graph.replay()
            torch.cuda.synchronize()
            replay_times.append(time.perf_counter() - t0)
        eager_times: list[float] = []
        for _ in range(args.steps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _step(eager, eager_opt, ids, device)
            torch.cuda.synchronize()
            eager_times.append(time.perf_counter() - t0)
        replay_median = sorted(replay_times)[len(replay_times) // 2]
        eager_median = sorted(eager_times)[len(eager_times) // 2]
        result.update({
            "status": "pass" if loss_err <= args.loss_tolerance and param_err <= args.parameter_tolerance else "fail",
            "batch": int(args.batch_sizes[0]),
            "warmup_steps": int(args.warmup),
            "measure_steps": int(args.steps),
            "eager_loss": float(eager_loss.cpu()),
            "graph_loss": float(graph_loss.cpu()),
            "max_loss_abs_error": loss_err,
            "max_parameter_abs_error": param_err,
            "loss_tolerance": args.loss_tolerance,
            "parameter_tolerance": args.parameter_tolerance,
            "eager_median_step_s": eager_median,
            "graph_replay_median_step_s": replay_median,
            "eager_tokens_per_s": args.batch_sizes[0] * args.seq_len / eager_median,
            "graph_tokens_per_s": args.batch_sizes[0] * args.seq_len / replay_median,
            "graph_speedup": eager_median / replay_median,
            "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "fixed_shape": True,
            "replay_finite": bool(torch.isfinite(static_loss).all().item()),
        })
    except Exception as exc:
        # Preserve the failed capture as evidence; callers can gate on status.
        result.update(status="fail", error_type=type(exc).__name__, error=str(exc),
                      fixed_shape=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("ddp", "graph"), required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch-sizes", default="1,2,4", help="comma-separated per-rank batches")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261030)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--loss-tolerance", type=float, default=5e-4)
    ap.add_argument("--parameter-tolerance", type=float, default=5e-4)
    args = ap.parse_args()
    try:
        args.batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    except ValueError as exc:
        raise SystemExit("--batch-sizes must be comma-separated positive integers") from exc
    if not args.batch_sizes or any(x < 1 for x in args.batch_sizes):
        raise SystemExit("--batch-sizes must contain positive integers")
    if args.steps < 1 or args.warmup < 0 or args.seq_len < 2:
        raise SystemExit("steps/warmup/seq-len are invalid")
    result = _ddp(args) if args.mode == "ddp" else _graph(args)
    if result.get("status") == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
