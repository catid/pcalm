#!/usr/bin/env python
"""Run a short deterministic pilot through the common local-method interface.

This runner is intentionally limited to screening budgets (<=10M tokens unless
``--allow-long-local`` is passed). It uses the isolated FineWeb worker used by
the BP control, exact post-shift target accounting, and JSONL metrics/checkpoints.
The current PC-ALM/ePC classes are reference/adaptation implementations; this
pilot is an interface and accounting smoke, not a 1B-token result.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from src.data_stream import StreamSpec, iter_token_chunks
from src.methods import (
    EPCReference, PCALMReference, StableParameterizationWrapper,
    MetaPCNReference, PrecisionWeightedPCReference, WFActPCReference,
    ClosedFormPrecisionPCReference, DirectFeedbackAlignmentReference,
    StreamAmortizedPCReference, PCALMControlVariateReference,
)
from src.model import CausalTransformer, GPTConfig


def _synthetic_chunks(seed: int, skip: int = 0):
    # Match FineWeb worker's 1024-token chunks and deterministic ordering.
    g = torch.Generator().manual_seed(seed)
    for i in range(skip):
        torch.randint(32000, (1024,), generator=g)
    while True:
        yield [int(x) for x in torch.randint(32000, (1024,), generator=g)]


def _fineweb_chunks(skip_chunks: int):
    worker = subprocess.Popen(
        [sys.executable, str(Path(__file__).with_name("fineweb_worker.py")), "--skip-chunks", str(skip_chunks)],
        stdout=subprocess.PIPE, text=True, bufsize=1,
    )
    def rows():
        try:
            for line in worker.stdout:
                yield json.loads(line)
        finally:
            if worker.poll() is None:
                worker.terminate()
                try:
                    worker.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    worker.kill(); worker.wait()
    return worker, rows()


def _build_method(name, model, args):
    if name == "pc_alm":
        return PCALMReference(model, inner_steps=args.inner_steps, eta_h=args.eta_h,
                              alpha=args.alpha, rho=args.rho,
                              parameter_step=args.parameter_step)
    if name == "epc":
        method = EPCReference(model, inner_steps=args.inner_steps, eta_h=args.eta_h,
                              beta=args.beta, parameter_step=args.parameter_step)
        if args.stable:
            method = StableParameterizationWrapper(method)
        return method
    if name == "meta_pcn":
        return MetaPCNReference(model, inner_steps=args.inner_steps, parameter_step=args.parameter_step)
    if name == "precision_weighted_pc":
        return PrecisionWeightedPCReference(model, inner_steps=args.inner_steps, parameter_step=args.parameter_step)
    if name == "wf_act_pc":
        return WFActPCReference(model, inner_steps=args.inner_steps, eta_h=args.eta_h,
                                beta=args.beta, parameter_step=args.parameter_step,
                                feedback_gain=args.feedback_gain)
    if name == "closed_form_precision_pc":
        return ClosedFormPrecisionPCReference(model, inner_steps=args.inner_steps, eta_h=args.eta_h,
                                              precision_ema=args.precision_ema,
                                              parameter_step=args.parameter_step)
    if name == "direct_feedback_alignment":
        return DirectFeedbackAlignmentReference(model, inner_steps=args.inner_steps, eta_h=args.eta_h,
                                                beta=args.beta, parameter_step=args.parameter_step,
                                                feedback_gain=args.feedback_gain)
    if name == "stream_amortized_pc":
        return StreamAmortizedPCReference(model, inner_steps=args.inner_steps, eta_h=args.eta_h,
                                          beta=args.beta, parameter_step=args.parameter_step,
                                          carry_decay=args.carry_decay)
    if name == "pc_alm_control_variate":
        return PCALMControlVariateReference(model, inner_steps=args.inner_steps, eta_h=args.eta_h,
                                            alpha=args.alpha, rho=args.rho,
                                            parameter_step=args.parameter_step,
                                            probe_count=args.probe_count, probe_eps=args.probe_eps)
    raise ValueError(f"unsupported local method: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--method", choices=("pc_alm", "epc", "meta_pcn", "precision_weighted_pc",
                                          "wf_act_pc", "closed_form_precision_pc",
                                          "direct_feedback_alignment", "stream_amortized_pc",
                                          "pc_alm_control_variate"), required=True)
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--source", choices=("fineweb", "synthetic"), default="fineweb")
    ap.add_argument("--validation-tokens", type=int, default=10_000_000)
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--inner-steps", type=int, default=2)
    ap.add_argument("--eta-h", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--precision-ema", type=float, default=0.1)
    ap.add_argument("--carry-decay", type=float, default=1.0)
    ap.add_argument("--probe-count", type=int, default=1)
    ap.add_argument("--probe-eps", type=float, default=1e-3)
    ap.add_argument("--feedback-gain", type=float, default=1.0)
    ap.add_argument("--parameter-step", type=float, default=3e-4)
    ap.add_argument("--checkpoint-every", type=int, default=1_000_000)
    ap.add_argument("--resume")
    ap.add_argument("--tiny", action="store_true", help="use a tiny model for deterministic smoke tests")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--stable", action="store_true", help="apply stable parameterization (ePC only)")
    ap.add_argument("--allow-long-local", action="store_true", help="allow budgets above the 10M screening limit")
    args = ap.parse_args()
    if args.tokens <= 0 or args.tokens > 1_000_000_000:
        raise SystemExit("tokens must be in (0, 1B]")
    if args.tokens > 10_000_000 and not args.allow_long_local:
        raise SystemExit("local pilot budget is gated at 10M; pass --allow-long-local for a declared longer run")
    if args.method != "epc" and args.stable:
        raise SystemExit("--stable currently requires --method epc")
    if args.validation_tokens < 0:
        raise SystemExit("validation-tokens must be nonnegative")
    if args.inner_steps < 1:
        raise SystemExit("inner-steps must be positive")

    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    if args.tiny:
        cfg = GPTConfig(vocab_size=32000, n_layers=2, d_model=32, n_heads=2, d_ff=64, max_seq_len=1024)
    else:
        cfg = GPTConfig()
    model = CausalTransformer(cfg).to(device)
    method = _build_method(args.method, model, args)

    outdir = Path("runs") / args.run_id
    outdir.mkdir(parents=True, exist_ok=True)
    metrics_path = outdir / "metrics.jsonl"
    start_tokens, step = 0, 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_tokens, step = int(ckpt["tokens"]), int(ckpt["step"])
        if start_tokens % 1023:
            raise SystemExit("resume checkpoint must end on a complete 1023-target chunk")

    skip_validation = math.ceil(args.validation_tokens / 1023)
    skip_train = start_tokens // 1023
    worker = None
    if args.source == "fineweb":
        worker, chunks = _fineweb_chunks(skip_validation + skip_train)
    else:
        chunks = _synthetic_chunks(args.seed, skip_validation + skip_train)

    seen = start_tokens
    started = time.perf_counter()
    model.train()
    with metrics_path.open("a") as mf:
        for chunk in chunks:
            if seen >= args.tokens:
                break
            target_here = min(len(chunk) - 1, args.tokens - seen)
            if target_here <= 0:
                continue
            ids = torch.tensor(chunk[:target_here + 1], dtype=torch.long, device=device)[None, :]
            # All state/dual/error tensors remain rank-local. Local methods do
            # not call a global backward pass in this runner.
            state = method.init_state(ids)
            relax_t0 = time.perf_counter()
            for _ in range(args.inner_steps):
                method.relax_step(state, ids, ids)
            relax_s = time.perf_counter() - relax_t0
            update = method.local_update(state, ids, ids)
            # Evaluation loss is recorded on the same batch for comparability;
            # it is not used to update parameters in this local runner.
            with torch.no_grad():
                loss, _ = model(ids, ids)
            seen += target_here
            step += 1
            elapsed = time.perf_counter() - started
            diag = method.diagnostics(state)
            rec = {
                "step": step, "tokens": seen, "seed": args.seed,
                "loss": float(loss.detach()), "elapsed_s": elapsed,
                "tokens_per_s": seen / max(elapsed, 1e-9),
                "relax_elapsed_s": relax_s, "method": args.method,
                "method_version": getattr(method, "version", "unknown"),
                "inner_steps": args.inner_steps, "update": update,
                "diagnostics": diag,
            }
            mf.write(json.dumps(rec) + "\n"); mf.flush()
            if seen % args.checkpoint_every < target_here or seen == args.tokens:
                torch.save({"model": model.state_dict(), "tokens": seen, "step": step,
                            "method": args.method, "seed": args.seed}, outdir / f"checkpoint-{seen}.pt")
    if worker is not None and worker.poll() is None:
        worker.terminate()
        try:
            worker.wait(timeout=5)
        except subprocess.TimeoutExpired:
            worker.kill(); worker.wait()
    if seen != args.tokens:
        raise RuntimeError(f"stream ended at {seen} tokens, target {args.tokens}")
    print(json.dumps({"run_id": args.run_id, "method": args.method, "tokens": seen,
                      "status": "completed", "metrics": str(metrics_path)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
