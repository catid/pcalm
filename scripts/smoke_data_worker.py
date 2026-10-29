#!/usr/bin/env python
"""Data-only smoke using the isolated FineWeb worker."""
import itertools, json, subprocess, sys
from pathlib import Path
worker = subprocess.Popen([sys.executable, str(Path(__file__).with_name("fineweb_worker.py"))], stdout=subprocess.PIPE, text=True, bufsize=1)
chunks = [json.loads(line) for line in itertools.islice(worker.stdout, 3)]
worker.terminate(); worker.wait(timeout=10)
out = {"chunks": len(chunks), "lengths": [len(c) for c in chunks], "loss_tokens": sum(len(c)-1 for c in chunks), "first_ids": chunks[0][:12] if chunks else []}
Path("reports/data_smoke_worker.json").write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=2))
