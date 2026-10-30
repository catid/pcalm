#!/usr/bin/env python
"""Create a run manifest before any pilot or benchmark."""
import argparse, json, os, platform, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

def cmd(x):
    try: return subprocess.check_output(x, shell=True, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as e: return f"ERROR: {e}"


def registry_method_defaults(root, method):
    """Return the canonical method metadata, or an empty mapping if unknown."""
    try:
        entries = json.loads((root / "configs/method_registry.json").read_text()).get("methods", [])
    except (OSError, json.JSONDecodeError):
        return {}
    for entry in entries:
        if entry.get("id") == method:
            return {k: entry[k] for k in ("version", "locality", "strict_no_autograd") if k in entry}
    return {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--config", default="configs/base.json")
    ap.add_argument("--phase", default="0")
    ap.add_argument("--tokens", type=int, default=0)
    # The base config is PC-ALM.  A manifest made for a BP control must state
    # that override explicitly; otherwise run_manifest followed by
    # ``train.py --method bp_adamw`` records contradictory provenance.
    ap.add_argument("--method", help="override config.method.name in the manifest")
    ap.add_argument("--method-version", help="override config.method.version")
    ap.add_argument("--locality", help="override config.method.locality")
    ap.add_argument("--strict-no-autograd", action=argparse.BooleanOptionalAction, default=None,
                    help="override config.method.strict_no_autograd")
    ap.add_argument("--metadata-json", help="JSON object merged into config.method")
    ap.add_argument("--training-metadata-json", help="JSON object merged into config.train")
    ap.add_argument("--train-command", help="complete training command recorded in the manifest")
    ap.add_argument("--seed", type=int, help="per-run training seed recorded in the manifest")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]; outdir = root / "runs" / args.run_id; outdir.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        torch_info = {"version": torch.__version__, "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
    except Exception as e: torch_info = {"error": str(e)}
    config = json.loads((root / args.config).read_text())
    method_cfg = config.setdefault("method", {})
    if args.method:
        method_cfg["name"] = args.method
        # Resolve canonical defaults from the registry before applying any
        # explicit command-line or JSON overrides.  Unknown/ad-hoc methods
        # retain the base config and remain visibly unversioned.
        defaults = registry_method_defaults(root, args.method)
        for key, value in defaults.items():
            if key not in method_cfg or method_cfg[key] in (None, "", "reference-0.1"):
                method_cfg[key] = value
        if not defaults and method_cfg.get("version") == "reference-0.1":
            method_cfg["version"] = "unregistered"
    if args.method_version:
        method_cfg["version"] = args.method_version
    if args.locality:
        method_cfg["locality"] = args.locality
    if args.strict_no_autograd is not None:
        method_cfg["strict_no_autograd"] = args.strict_no_autograd
    if args.metadata_json:
        extra = json.loads(args.metadata_json)
        if not isinstance(extra, dict):
            ap.error("--metadata-json must decode to a JSON object")
        method_cfg.update(extra)
    train_cfg = config.setdefault("train", {})
    if args.training_metadata_json:
        extra = json.loads(args.training_metadata_json)
        if not isinstance(extra, dict):
            ap.error("--training-metadata-json must decode to a JSON object")
        train_cfg.update(extra)
    manifest = {"run_id": args.run_id, "created_utc": datetime.now(timezone.utc).isoformat(), "phase": args.phase, "tokens_budget": args.tokens,
                "commit": cmd("git rev-parse HEAD"), "command": " ".join(sys.argv), "cwd": str(root), "host": platform.node(), "kernel": platform.release(),
                "config": config, "torch": torch_info,
                "gpu_inventory": cmd("nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader"),
                "environment_lock": (root / "configs/environment.lock").read_text(), "nccl_env": {k:v for k,v in os.environ.items() if k.startswith("NCCL_") or k.startswith("MASTER_")}}
    if args.train_command:
        manifest["train_command"] = args.train_command
    if args.seed is not None:
        manifest["run_seed"] = args.seed
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2)); print(json.dumps(manifest, indent=2))

if __name__ == "__main__": main()
