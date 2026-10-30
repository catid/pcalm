#!/usr/bin/env python3
"""Audit run-registry/result consistency and live training processes.

This is intentionally read-only: it does not signal or modify training jobs.
It catches a common provenance error where ``run_manifest.py`` is invoked with
the default method config and a later ``train.py --method`` override.  The
process command is treated as the source of truth for a live run, and the
manifest mismatch is reported until the next run uses explicit manifest
overrides.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import socket
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HOSTS = ("foureyes", "ripper", "work", "loki")


def read_registry(path: Path):
    rows = []
    if not path.exists():
        return rows
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            row["_line"] = n
            rows.append(row)
        except json.JSONDecodeError as exc:
            rows.append({"_line": n, "_parse_error": str(exc)})
    return rows


def read_results(path: Path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def duplicate_ids(rows):
    ids = defaultdict(list)
    for row in rows:
        if row.get("run_id"):
            ids[row["run_id"]].append(row)
    return {k: v for k, v in ids.items() if len(v) > 1}


def command_output(cmd, *, timeout=10):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"ERROR: {exc}"


def parse_live_processes(host: str, local: bool):
    if local:
        text = command_output(["ps", "-eo", "pid=,etime=,cmd="])
    else:
        text = command_output(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host,
             "ps -eo pid=,etime=,cmd="], timeout=15
        )
    out = []
    for line in text.splitlines():
        if not re.search(r"scripts/train(?:_local)?\.py", line) or "--run-id" not in line:
            continue
        # The controller's process table also contains detached SSH launch
        # wrappers for remote jobs.  Those are audited on the destination
        # host, where their actual Python process and metric file live.
        if local and re.search(r"(?:^|\s)ssh\s", line):
            continue
        # The command includes wrappers and an optional CUDA_VISIBLE_DEVICES.
        rid = re.search(r"--run-id\s+([^\s'\"]+)", line)
        method = re.search(r"--method\s+([^\s'\"]+)", line)
        budget = re.search(r"--tokens\s+(\d+)", line)
        gpu = re.search(r"CUDA_VISIBLE_DEVICES[= ]([^\s;]+)", line)
        if not rid:
            continue
        out.append({
            "host": host,
            "run_id": rid.group(1),
            "method": method.group(1) if method else None,
            "tokens_budget": int(budget.group(1)) if budget else None,
            "gpu_binding": gpu.group(1) if gpu else None,
            "raw": line.strip(),
        })
    return out


def remote_last_metric(host: str, run_id: str):
    # Run IDs are generated from a restricted [A-Za-z0-9_.-] alphabet.  Keep
    # the explicit check because this value is interpolated into an SSH command.
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        return {"error": "invalid run id"}
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host,
           f"tail -1 /var/tmp/pcalm/runs/{run_id}/metrics.jsonl"]
    text = command_output(cmd, timeout=15).strip()
    if not text or text.startswith("ERROR:"):
        return {"error": text or "missing metrics"}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": text[-400:]}


def remote_manifest_method(host: str, run_id: str):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        return None
    text = command_output(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host,
         f"python3 -c 'import json; print(json.load(open(\"/var/tmp/pcalm/runs/{run_id}/manifest.json\"))[\"config\"][\"method\"].get(\"name\"))'"],
        timeout=15,
    ).strip()
    if text.startswith("ERROR:") or not text:
        return None
    return text.splitlines()[-1].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="reports/registry_audit.json")
    ap.add_argument("--hosts", default=",".join(HOSTS), help="comma-separated SSH hosts")
    ap.add_argument("--no-ssh", action="store_true", help="audit local files/processes only")
    args = ap.parse_args()

    registry = read_registry(ROOT / "runs/registry.jsonl")
    results = read_results(ROOT / "reports/results.csv")
    reg_ids = {r.get("run_id") for r in registry if r.get("run_id")}
    result_ids = {r.get("run_id") for r in results if r.get("run_id")}
    manifests = {}
    for p in sorted((ROOT / "runs").glob("*/manifest.json")):
        try:
            manifests[p.parent.name] = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            manifests[p.parent.name] = {"_error": str(exc)}

    active_processes = []
    hosts = [h.strip() for h in args.hosts.split(",") if h.strip()]
    if args.no_ssh:
        active_processes = parse_live_processes(socket.gethostname(), True)
    else:
        for host in hosts:
            active_processes.extend(parse_live_processes(host, host in {socket.gethostname(), "localhost", "foureyes"}))

    # Attach current progress for active process IDs.  Remote workers use the
    # NVMe source tree; foureyes jobs use the project checkout.
    for item in active_processes:
        host = item["host"]
        if host in {socket.gethostname(), "localhost", "foureyes"}:
            path = ROOT / "runs" / item["run_id"] / "metrics.jsonl"
            try:
                line = path.read_text().splitlines()[-1]
                item["last_metric"] = json.loads(line)
            except (OSError, IndexError, json.JSONDecodeError):
                item["last_metric"] = {"error": "missing local metric"}
        elif not args.no_ssh:
            item["last_metric"] = remote_last_metric(host, item["run_id"])

        manifest = manifests.get(item["run_id"])
        if manifest and isinstance(manifest.get("config"), dict):
            configured = manifest["config"].get("method", {}).get("name")
            item["manifest_method"] = configured
            item["manifest_method_mismatch"] = bool(item.get("method") and configured and configured != item["method"])
        elif not args.no_ssh and host not in {socket.gethostname(), "localhost", "foureyes"}:
            configured = remote_manifest_method(host, item["run_id"])
            item["manifest_method"] = configured
            item["manifest_method_mismatch"] = bool(item.get("method") and configured and configured != item["method"])

    # A run normally has a shell wrapper, an SSH process, the Python training
    # process, and its FineWeb worker.  Collapse those process rows so the
    # report counts active *runs*, while retaining a process count and raw
    # command lines for auditability.
    active = {}
    for item in active_processes:
        key = (item["host"], item["run_id"])
        if key not in active:
            active[key] = {k: v for k, v in item.items() if k != "raw"}
            active[key]["process_count"] = 0
            active[key]["raw_commands"] = []
        active[key]["process_count"] += 1
        active[key]["raw_commands"].append(item["raw"])
    active_runs = list(active.values())

    out = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "cwd": str(ROOT),
        "host": socket.gethostname(),
        "registry_rows": len(registry),
        "registry_unique_ids": len(reg_ids),
        "results_rows": len(results),
        "results_unique_ids": len(result_ids),
        "registry_duplicate_ids": {k: [r.get("_line") for r in v] for k, v in duplicate_ids(registry).items()},
        "results_duplicate_ids": {k: [i + 2 for i, r in enumerate(results) if r.get("run_id") == k] for k in duplicate_ids(results)},
        "registry_not_results": sorted(reg_ids - result_ids),
        "results_not_registry": sorted(result_ids - reg_ids),
        "manifest_not_registry": sorted(set(manifests) - reg_ids),
        "registry_missing_manifest": sorted(reg_ids - set(manifests)),
        "active": active_runs,
    }
    outpath = ROOT / args.output
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    print(json.dumps({
        "registry_rows": out["registry_rows"],
        "registry_unique_ids": out["registry_unique_ids"],
        "results_rows": out["results_rows"],
        "registry_duplicate_ids": out["registry_duplicate_ids"],
        "registry_not_results": len(out["registry_not_results"]),
        "active": len(active_runs),
        "manifest_method_mismatches": sum(bool(x.get("manifest_method_mismatch")) for x in active_runs),
        "output": str(outpath),
    }, indent=2))


if __name__ == "__main__":
    main()
