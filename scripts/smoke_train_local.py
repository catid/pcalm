#!/usr/bin/env python
"""Dependency-light synthetic exact-budget smoke for train_local.py."""
import json
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
run_id = f"smoke-local-synthetic-{int(time.time())}"
cmd = [sys.executable, str(root / "scripts" / "train_local.py"),
       "--run-id", run_id, "--method", "epc", "--source", "synthetic",
       "--tokens", "2046", "--validation-tokens", "0", "--seed", "17",
       "--tiny", "--device", "cpu", "--inner-steps", "2",
       "--checkpoint-every", "1023", "--stable"]
proc = subprocess.run(cmd, cwd=root, text=True, capture_output=True)
if proc.returncode:
    raise SystemExit(proc.stderr or proc.stdout)
result = json.loads(proc.stdout.strip().splitlines()[-1])
if result.get("status") != "completed" or result.get("tokens") != 2046:
    raise SystemExit(f"unexpected result: {result}")
metrics = root / "runs" / run_id / "metrics.jsonl"
rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
if len(rows) != 2 or rows[-1]["tokens"] != 2046:
    raise SystemExit(f"unexpected metrics: {rows}")
print(json.dumps({"run_id": run_id, "tokens": rows[-1]["tokens"], "steps": len(rows), "status": "pass"}, sort_keys=True))
