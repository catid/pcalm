#!/usr/bin/env python
"""Create a run manifest before any pilot or benchmark."""
import argparse, json, os, platform, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

def cmd(x):
    try: return subprocess.check_output(x, shell=True, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as e: return f"ERROR: {e}"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run-id", required=True); ap.add_argument("--config", default="configs/base.json"); ap.add_argument("--phase", default="0"); ap.add_argument("--tokens", type=int, default=0); args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]; outdir = root / "runs" / args.run_id; outdir.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        torch_info = {"version": torch.__version__, "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
    except Exception as e: torch_info = {"error": str(e)}
    manifest = {"run_id": args.run_id, "created_utc": datetime.now(timezone.utc).isoformat(), "phase": args.phase, "tokens_budget": args.tokens,
                "commit": cmd("git rev-parse HEAD"), "command": " ".join(sys.argv), "cwd": str(root), "host": platform.node(), "kernel": platform.release(),
                "config": json.loads((root / args.config).read_text()), "torch": torch_info,
                "gpu_inventory": cmd("nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader"),
                "environment_lock": (root / "configs/environment.lock").read_text(), "nccl_env": {k:v for k,v in os.environ.items() if k.startswith("NCCL_") or k.startswith("MASTER_")}}
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2)); print(json.dumps(manifest, indent=2))

if __name__ == "__main__": main()
