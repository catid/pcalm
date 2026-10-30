#!/usr/bin/env python3
"""Smoke-test manifest method metadata resolution without starting training."""

import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run(run_id, *extra):
    cmd = [PYTHON, str(ROOT / "scripts/run_manifest.py"), "--run-id", run_id,
           "--phase", "0", "--tokens", "1", *extra]
    subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    path = ROOT / "runs" / run_id / "manifest.json"
    try:
        return json.loads(path.read_text())
    finally:
        shutil.rmtree(path.parent)


def main():
    auto = run("manifest-smoke-auto-20261030", "--method", "epc")
    method = auto["config"]["method"]
    assert method["version"] == "independent-transformer-adaptation-0.1", method
    assert method["locality"] == "block_local", method
    assert method["strict_no_autograd"] is False, method

    explicit = run("manifest-smoke-explicit-20261030", "--method", "epc",
                   "--method-version", "custom-test-1", "--locality", "strict_local",
                   "--no-strict-no-autograd", "--metadata-json", '{"inner_steps":7}',
                   "--seed", "123", "--train-command", "train_local --tokens 1")
    method = explicit["config"]["method"]
    assert method["version"] == "custom-test-1", method
    assert method["locality"] == "strict_local", method
    assert method["strict_no_autograd"] is False, method
    assert method["inner_steps"] == 7, method
    assert explicit["run_seed"] == 123
    assert explicit["train_command"] == "train_local --tokens 1"
    print("manifest metadata smoke: PASS")


if __name__ == "__main__":
    main()
