#!/usr/bin/env python
"""Capture local software, model count, and a representative eager benchmark."""
import json, os, platform, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.model import GPTConfig, CausalTransformer, count_parameters

def run(cmd):
    try: return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as e: return f"ERROR: {e}"

def main():
    torch.manual_seed(20261029)
    cfg = GPTConfig(vocab_size=32000); model = CausalTransformer(cfg)
    parts, total = count_parameters(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"; model = model.to(device)
    b, t = (1, min(cfg.max_seq_len, int(os.environ.get("PCALM_BENCH_SEQ", "128")))); ids = torch.randint(cfg.vocab_size, (b, t), device=device)
    labels = torch.roll(ids, -1, -1)
    for _ in range(2): model(ids, labels)
    if device == "cuda": torch.cuda.synchronize()
    start = time.perf_counter(); loss, _ = model(ids, labels)
    if device == "cuda": torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    out = {"host": platform.node(), "python": sys.version, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
           "cuda_available": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
           "config": cfg.__dict__, "parameter_parts": parts, "parameter_count": total, "parameter_interval_pass": 38_000_000 <= total <= 42_000_000,
           "benchmark": {"batch": b, "seq": t, "elapsed_s": elapsed, "tokens_per_s": b*t/elapsed, "loss": float(loss.detach())}}
    if device == "cuda": out["memory_peak_bytes"] = torch.cuda.max_memory_allocated()
    Path("reports").mkdir(exist_ok=True); Path("reports/phase0_inventory.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2)); print("nvidia-smi:\n" + run("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"))

if __name__ == "__main__": main()
