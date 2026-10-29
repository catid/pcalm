#!/usr/bin/env python
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets import load_dataset
from transformers import AutoTokenizer
from src.data_stream import StreamSpec, iter_token_chunks, count_loss_tokens

def main():
    tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    spec = StreamSpec(chunk_tokens=128, validation_chunks=0)
    chunks = list(__import__("itertools").islice(iter_token_chunks(ds, tok, spec), 3))
    out = {"chunks": len(chunks), "lengths": [len(c) for c in chunks], "loss_tokens": count_loss_tokens(chunks), "first_ids": chunks[0][:12] if chunks else []}
    Path("reports/data_smoke.json").write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=2))
if __name__ == "__main__": main()
