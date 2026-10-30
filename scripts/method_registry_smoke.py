#!/usr/bin/env python
"""Audit the method registry and run the smallest PC-ALM protocol smoke.

This is a *registry* gate, not an implementation claim.  Every entry is
checked for the fields needed to make a method auditable.  Only methods with
an explicitly registered implementation are exercised.  Unknown or planned
arms remain ``pending`` in the output, so adding a name to the registry never
silently makes an unimplemented method pass.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.model import CausalTransformer, GPTConfig
from src.methods import (
    EPCReference,
    MetaPCNReference,
    PCALMReference,
    PrecisionWeightedPCReference,
    StableParameterizationWrapper,
    WFActPCReference,
    ClosedFormPrecisionPCReference,
    DirectFeedbackAlignmentReference,
    StreamAmortizedPCReference,
    PCALMControlVariateReference,
)


REQUIRED_ENTRY_FIELDS = {
    "id", "rank", "literature", "implementation", "version", "locality",
    "strict_no_autograd", "status",
}
ALLOWED_LOCALITY = {"block_local", "strict_local", "relaxed_feedback", "global_reverse"}
PROTOCOL = ("init_state", "relax_step", "local_update", "diagnostics", "step_cost")

# Keep this mapping deliberately explicit.  A registry entry with a plausible
# description is not an implementation until its factory is added here.  A
# factory is used instead of a class map because wrappers (such as stable
# parameterization) need a fully constructed base method.
def _pc_alm_factory(model):
    return PCALMReference(model, inner_steps=1, parameter_step=0.0)


def _epc_factory(model):
    return EPCReference(model, inner_steps=1, parameter_step=0.0)


def _meta_pcn_factory(model):
    return MetaPCNReference(model, inner_steps=1, parameter_step=0.0)


def _precision_weighted_factory(model):
    return PrecisionWeightedPCReference(model, inner_steps=1, parameter_step=0.0)


def _stable_factory(model):
    return StableParameterizationWrapper(_pc_alm_factory(model))


def _wf_act_factory(model):
    return WFActPCReference(model, inner_steps=1, parameter_step=0.0)


def _closed_form_factory(model):
    return ClosedFormPrecisionPCReference(model, inner_steps=1, parameter_step=0.0)


def _dfa_factory(model):
    return DirectFeedbackAlignmentReference(model, inner_steps=1, parameter_step=0.0)


def _stream_factory(model):
    return StreamAmortizedPCReference(model, inner_steps=1, parameter_step=0.0)


def _control_variate_factory(model):
    return PCALMControlVariateReference(model, inner_steps=1, parameter_step=0.0,
                                        probe_count=1, probe_eps=1e-3)


IMPLEMENTATIONS = {
    "pc_alm": _pc_alm_factory,
    "epc": _epc_factory,
    "meta_pcn": _meta_pcn_factory,
    "precision_weighted_pc": _precision_weighted_factory,
    "stable_parameterization": _stable_factory,
    "wf_act_pc": _wf_act_factory,
    "closed_form_precision_pc": _closed_form_factory,
    "direct_feedback_alignment": _dfa_factory,
    "stream_amortized_pc": _stream_factory,
    "pc_alm_control_variate": _control_variate_factory,
}


def _entry_check(entry: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_ENTRY_FIELDS - set(entry))
    if missing:
        errors.append("missing fields: " + ", ".join(missing))
    if entry.get("locality") not in ALLOWED_LOCALITY:
        errors.append(f"invalid locality {entry.get('locality')!r}")
    if not isinstance(entry.get("strict_no_autograd"), bool):
        errors.append("strict_no_autograd must be boolean")
    if not isinstance(entry.get("status"), str) or not entry.get("status"):
        errors.append("status must be a non-empty string")
    if not isinstance(entry.get("version"), str) or not entry.get("version"):
        errors.append("version must be a non-empty string")
    return errors


def _method_smoke(factory, device: torch.device, seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    cfg = GPTConfig(vocab_size=97, n_layers=2, d_model=32, n_heads=4, d_ff=64, max_seq_len=8)
    model = CausalTransformer(cfg).to(device)
    ids = torch.randint(cfg.vocab_size, (1, 8), device=device)
    method = factory(model)
    missing_protocol = [name for name in PROTOCOL if not callable(getattr(method, name, None))]
    if missing_protocol:
        raise RuntimeError("missing protocol methods: " + ", ".join(missing_protocol))
    state = method.init_state(ids)
    t0 = time.perf_counter()
    residuals = method.relax_step(state, ids, ids)
    update = method.local_update(state, ids, ids, apply=False)
    diag = method.diagnostics(state)
    cost = method.step_cost(int(ids.numel()), time.perf_counter() - t0)
    state_tensors = list(state.hidden) + list(state.dual) + list(state.errors or []) + list(state.precisions or [])
    finite = all(torch.isfinite(x).all().item() for x in state_tensors + residuals)
    if not finite:
        raise RuntimeError("non-finite state/residual in PC-ALM smoke")
    return {
        "protocol_smoke": "pass",
        "class": f"{type(method).__module__}.{type(method).__name__}",
        "version": method.version,
        "locality": method.locality,
        "strict_no_autograd": method.strict_no_autograd,
        "protocol": list(PROTOCOL),
        "state_step": state.step,
        "residual_count": len(residuals),
        "diagnostics": diag,
        "local_update": update,
        "step_cost": cost,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="configs/method_registry.json")
    ap.add_argument("--out", default="reports/method_registry_smoke.json")
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-smoke", action="store_true")
    args = ap.parse_args()

    path = Path(args.registry)
    payload = json.loads(path.read_text())
    entries = payload.get("methods")
    if not isinstance(entries, list) or not entries:
        raise SystemExit("registry methods must be a non-empty list")
    ids: set[str] = set()
    records: list[dict[str, Any]] = []
    schema_errors: list[str] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            schema_errors.append(f"entry {i} is not an object")
            continue
        eid = str(entry.get("id", f"entry-{i}"))
        errors = _entry_check(entry)
        if eid in ids:
            errors.append("duplicate id")
        ids.add(eid)
        record: dict[str, Any] = {"id": eid, "registry_status": entry.get("status"),
                                  "implementation": entry.get("implementation"),
                                  "version": entry.get("version"),
                                  "locality": entry.get("locality"),
                                  "schema_errors": errors}
        factory = IMPLEMENTATIONS.get(eid)
        if errors:
            record.update(status="fail", reason="registry schema error")
        elif factory is None:
            # This is an honest pending result, even when the entry says
            # "planned", "wrapper pending", or another implementation hint.
            record.update(status="pending", reason="no registered implementation")
        elif args.skip_smoke:
            record.update(status="pending", reason="smoke skipped")
        else:
            try:
                # Construct the method before checking metadata: wrappers
                # expose protocol methods through delegation and therefore do
                # not advertise all of them on the class object itself.
                torch.manual_seed(args.seed)
                cfg = GPTConfig(vocab_size=97, n_layers=2, d_model=32, n_heads=4, d_ff=64, max_seq_len=8)
                model = CausalTransformer(cfg).to(torch.device(args.device))
                method = factory(model)
                method_version = getattr(method, "version", "")
                method_locality = getattr(method, "locality", "")
                if method_version != entry["version"]:
                    raise RuntimeError(f"registry version {entry['version']!r} != class version {method_version!r}")
                if method_locality != entry["locality"]:
                    raise RuntimeError(f"registry locality {entry['locality']!r} != class locality {method_locality!r}")
                record.update(_method_smoke(factory, torch.device(args.device), args.seed))
                registry_status = str(entry.get("status", "")).lower()
                if "pending" in registry_status:
                    record.update(status="pending", reason="10M-token smoke pending")
                else:
                    record.update(status="pass")
            except Exception as exc:  # preserve failure evidence in JSON
                record.update(status="fail", reason=f"{type(exc).__name__}: {exc}")
        if record.get("status") == "fail":
            schema_errors.append(f"{eid}: {record.get('reason', 'failure')}")
        records.append(record)

    pending = sum(r.get("status") == "pending" for r in records)
    failed = sum(r.get("status") == "fail" for r in records)
    passed = sum(r.get("status") == "pass" for r in records)
    result = {
        "status": "pass" if not failed and not pending else ("fail" if failed else "pending"),
        "gate": "method_registry_smokes",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(args.device),
        "protocol": list(PROTOCOL),
        "counts": {"pass": passed, "pending": pending, "fail": failed},
        "methods": records,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if result["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
