# Active-run and registry audit

**Audit time:** 2026-10-30 UTC  
**Command:** `venvs/dnn/bin/python scripts/audit_registry.py --output reports/registry_audit.json`  
**Scope:** read-only inspection; no active process was signalled or modified.

## Live fifth-wave activity

The fifth 50M-token BP replication wave is active as twelve independent
single-GPU jobs: four on `foureyes`, four on `ripper`, two on `work`, and two
on `loki`. At the audit, each GPU reported roughly 56–69% utilization and
2,156 MiB allocated. Live metric tails were advancing at approximately
22–26M targets per run. These are phase-2 controls, not 1B-token campaigns;
the 1B gate remains blocked.

The live jobs were launched with `train.py --method bp_adamw`, but all twelve
manifests were created from the default `configs/base.json` whose method name
is `pc_alm`. The audit therefore reports a manifest-method mismatch for all
twelve runs. The remote launch wrappers are audited on their destination
hosts; their source trees do not expose the controller's `runs/` manifests
locally, so remote manifests and metric tails are collected over SSH. Existing
runs are preserved and were not rewritten.

`scripts/run_manifest.py` now accepts explicit `--method`,
`--method-version`, `--locality`, `--strict-no-autograd`, and
`--metadata-json` overrides. New commands must pass `--method bp_adamw` for
BP controls (or the exact local method) so the manifest and training command
cannot silently disagree. The launch scripts should also retain the resolved
method metadata in their run registry entry.

## Registry/result consistency

At audit time:

- `runs/registry.jsonl`: 113 rows, 111 unique run IDs;
- `reports/results.csv`: 105 rows, 103 unique run IDs;
- two historical registry IDs are duplicated (the Loki 10M BP entries), and
  their duplicate rows are also present in `results.csv`;
- eight registry entries are intentionally absent from `results.csv`: phase-0
  infrastructure/reference records and the local worker resume checks;
- no result row lacks a registry row;
- six phase-0 registry IDs have no local `runs/<id>/manifest.json`, because
  those records predate manifest persistence;
- several debug and remote-run directories are not represented in the local
  registry, including the active remote jobs. This is a provenance gap to
  close after completion by importing their manifests/metrics, without
  altering the running jobs.

The machine-readable details, including duplicate line numbers, missing IDs,
active process rows, metric tails, and method mismatch flags, are in
[`registry_audit.json`](registry_audit.json). This audit intentionally does
not recast training-loss values as validation quality: the active jobs use
`--validation-tokens 0` and remain controls/pipeline measurements.

## Required follow-up

1. Let the active wave finish; import each remote manifest and final metric
   into the registry/results files with its remote commit/source-tree status.
2. Keep duplicate historical rows visible but annotate them as duplicate
   records in the final report rather than silently deduplicating evidence.
3. Before any future launch, run `run_manifest.py --method ...` and verify the
   resulting method metadata against the training command.
4. Keep the 1B campaign blocked while the method registry and required 10M
   method smokes remain pending.
