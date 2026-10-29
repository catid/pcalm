#!/usr/bin/env python
"""Small resumable BP control pilot with a hard 1B gate.

This is deliberately conservative: only the ordinary AdamW control is enabled
until each local-credit method has its own passed gate. It records exact target
counts and writes a checkpoint at token boundaries.
"""
import argparse, json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import subprocess, signal
import torch
from src.model import GPTConfig, CausalTransformer
from src.data_stream import StreamSpec, iter_token_chunks

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run-id", required=True); ap.add_argument("--tokens", type=int, required=True); ap.add_argument("--method", default="bp_adamw"); ap.add_argument("--source", choices=["fineweb", "synthetic"], default="fineweb"); ap.add_argument("--resume"); ap.add_argument("--checkpoint-every", type=int, default=1_000_000); ap.add_argument("--allow-1b", action="store_true"); args = ap.parse_args()
    if args.method != "bp_adamw": raise SystemExit("Only bp_adamw control is implemented in this Phase 0 pilot; method arms remain gated.")
    if args.tokens > 1_000_000_000: raise SystemExit("token budget exceeds exactly 1B contract")
    if args.tokens == 1_000_000_000:
        if not args.allow_1b: raise SystemExit("1B run requires --allow-1b and a separately passed engineering/method gate")
        gates = json.loads(Path("reports/gates.json").read_text())["gates"]
        required = ("repository_plan", "runtime_parity", "architecture_parameter_count", "reference_math_transformer_smoke", "streaming_data_smoke", "exact_budget_resume_validation", "optimized_equivalence", "method_registry_smokes")
        bad = [k for k in required if not str(gates.get(k, "")).startswith("pass")]
        if bad: raise SystemExit("1B gate blocked; unresolved gates: " + ", ".join(bad))
    device = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(20261029)
    model = CausalTransformer(GPTConfig(vocab_size=32000)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    start_tokens, step = 0, 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device); model.load_state_dict(ckpt["model"]); opt.load_state_dict(ckpt["optimizer"]); start_tokens, step = ckpt["tokens"], ckpt["step"]
    worker = None
    if args.source == "fineweb":
        worker = subprocess.Popen([sys.executable, str(Path(__file__).with_name("fineweb_worker.py"))], stdout=subprocess.PIPE, text=True, bufsize=1)
        def worker_chunks():
            for line in worker.stdout:
                yield json.loads(line)
        chunks = worker_chunks()
    else:
        g = torch.Generator().manual_seed(20261029); chunks = ([int(x) for x in torch.randint(32000, (1024,), generator=g)] for _ in range((args.tokens+1023)//1024))
    outdir = Path("runs") / args.run_id; outdir.mkdir(parents=True, exist_ok=True); metrics = outdir / "metrics.jsonl"; seen = start_tokens; started = time.perf_counter()
    model.train()
    with metrics.open("a") as mf:
        for chunk in chunks:
            if seen >= args.tokens: break
            target_here = min(len(chunk) - 1, args.tokens - seen)
            # Slice the final chunk so compute and accounting cover the same
            # post-shift targets; no extra token is silently trained.
            ids = torch.tensor(chunk[:target_here + 1], dtype=torch.long, device=device)[None, :]
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                loss, _ = model(ids, ids)
            loss.backward(); opt.step(); step += 1; contributed = target_here; seen += contributed
            rec = {"step": step, "tokens": seen, "loss": float(loss.detach()), "elapsed_s": time.perf_counter()-started, "tokens_per_s": seen/max(time.perf_counter()-started, 1e-9)}; mf.write(json.dumps(rec)+"\n"); mf.flush()
            if seen % args.checkpoint_every < contributed or seen == args.tokens:
                torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "tokens": seen, "step": step}, outdir / f"checkpoint-{seen}.pt")
    if seen != args.tokens: raise RuntimeError(f"stream ended at {seen} tokens, target {args.tokens}")
    result = {"run_id": args.run_id, "method": args.method, "tokens": seen, "status": "completed", "metrics": str(metrics)}
    print(json.dumps(result), flush=True)
    # datasets' streaming parquet reader can leave an aiohttp/pyarrow worker
    # during interpreter finalization on this environment. Close references and
    # let the process exit through the normal path; the failure is tracked if it
    # recurs rather than being counted as a training result.
    if worker is not None:
        worker.terminate()
        try: worker.wait(timeout=5)
        except subprocess.TimeoutExpired: worker.kill(); worker.wait()

if __name__ == "__main__": main()
