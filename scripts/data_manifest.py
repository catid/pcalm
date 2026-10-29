#!/usr/bin/env python
"""Pin tokenizer files and FineWeb builder metadata without materializing data."""
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets import load_dataset_builder
from transformers import AutoTokenizer

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def main():
    name = "mistralai/Mistral-7B-v0.1"; tok = AutoTokenizer.from_pretrained(name)
    builder = load_dataset_builder("HuggingFaceFW/fineweb", name="sample-10BT")
    files = {}
    for k in ("vocab_file",):
        p = tok.init_kwargs.get(k)
        if p and Path(p).exists(): files[k] = {"path": p, "sha256": sha(p), "bytes": Path(p).stat().st_size}
    out = {"dataset": {"name": "HuggingFaceFW/fineweb", "config": "sample-10BT", "builder_version": str(builder.info.version), "features": str(builder.info.features)}, "tokenizer": {"name": name, "vocab_size": tok.vocab_size, "files": files}, "protocol": {"train_tokens": 1000000000, "validation_tokens": 10000000, "mask": "all positions after next-token shift", "order": "streaming dataset order; rank shards by contiguous deterministic example ranges"}}
    Path("reports/data_manifest.json").write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=2))

if __name__ == "__main__": main()
