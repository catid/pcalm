# Local-method 10M run audit

**Snapshot:** 2026-10-30 UTC  
**Scope:** phase-2 local-method screening and BP tuning; process inspection was
read-only and no active job was stopped or rewritten.

## Live jobs

The audit found eight active jobs using all available GPUs on the four hosts:

| Host | Run(s) | Current target count at snapshot | Provenance check |
|---|---|---:|---|
| `foureyes` | `phase2-meta_pcn-10m-foureyes-g2-seed20264012` | ~9.94M / 10M | method matches |
| `ripper` | `phase2-tune-pc_alm-ripper-g0/g1-seed20264100/101`, `phase2-tune-epc-ripper-g2/g3-seed20264102/103` | ~1.4–3.8M / 10M | method matches |
| `work` | `phase2-epc-stable-10m-work-g1-seed20264005` | ~9.22M / 10M | method matches; stable flag is only in command/metadata |
| `loki` | `phase2-tune-bp-lr0p0001/lr0p0006-seed20264200/201` | ~9.5M / 10M | method matches |

The other four jobs from this batch completed during the audit window:
`phase2-pc_alm-10m-foureyes-g0-seed20264010`,
`phase2-epc-10m-foureyes-g1-seed20264011`,
`phase2-precision_weighted_pc-10m-foureyes-g3-seed20264013`, and
`phase2-control-variate-10m-work-g0-seed20264004`. Their manifests and
metrics are present locally and their registry/results rows were imported by
the controller. No held-out validation loss is claimed: these local runners
reserve the validation prefix but record the training-batch loss only.

At the same snapshot, the terminal `work` WF-Act-PC and direct-feedback
alignment runs were already registered locally. The terminal Loki
closed-form-precision and stream-amortized runs were copied from
`/var/tmp/pcalm` into local `runs/` with their manifests and complete JSONL
metrics, then registered in `runs/registry.jsonl`, `reports/results.csv`, and
`reports/results.json`.

## Provenance findings

All current active method names agree between the process command and method
field in the manifest. The older BP wave mismatch remains documented in
`reports/registry_audit.md`.

The imported 10M terminal manifests also retain the base method version
`reference-0.1`, while their metric records identify the resolved adapters
(`reference-0.2`, `independent-transformer-adaptation-0.1`,
`meta-error-controller-0.1`, `precision-weighted-0.1`,
`adapted-local-jacobian-0.1`, `transpose-tied-feedback-0.1`,
`adapted-gaussian-posterior-0.1`, `carry-over-0.1`, and
`activation-probe-control-variate-0.1`). This version mismatch is preserved
as an audit finding and is not silently corrected in historical manifests.

The active tuning commands expose a second provenance gap: their manifests'
`command` field records only the `run_manifest.py` invocation, while the
actual `train_local.py` command and tuning values are in the shell process.
For example, the two ripper PC-ALM runs vary `eta_h` and `inner_steps`, and
the two Loki BP runs use learning rates `1e-4` and `6e-4`; those values are not
represented in their manifest `config`. The stable ePC run similarly carries
`--stable` only in its training command. This does not invalidate the runs,
but they must be labeled as tuning/provenance-limited evidence until the
launch command is imported.

`run_manifest.py` now accepts `--train-command`, `--seed`,
`--metadata-json`, and `--training-metadata-json` so future manifests contain
the complete command and resolved hyperparameters. Existing active manifests
were deliberately left unchanged. The audit JSON in
`reports/registry_audit.json` contains the
live process list, metric tails, registry/result counts, and duplicate-ID
findings.

## Import policy

Only runs with no matching `train_local.py` process and an exact final
10,000,000-token metric were imported. Failed, partial, or active runs were
not rewritten or promoted. The source-tree manifests copied from workers
report `git rev-parse` errors because `/var/tmp/pcalm` is not a Git checkout;
that limitation is retained in each registry note rather than replaced with
the controller commit.
