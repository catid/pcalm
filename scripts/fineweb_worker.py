#!/usr/bin/env python
"""Emit deterministic FineWeb token chunks to stdout.

The worker owns datasets/pyarrow. The parent process can terminate it after an
exact token budget, avoiding the known extension finalization crash in the
training process.
"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets import load_dataset
from transformers import AutoTokenizer
from src.data_stream import StreamSpec, iter_token_chunks

def main():
    tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    for chunk in iter_token_chunks(ds, tok, StreamSpec(chunk_tokens=1024, validation_chunks=0)):
        try:
            sys.stdout.write(json.dumps(chunk) + "\n"); sys.stdout.flush()
        except BrokenPipeError:
            os._exit(0)
    os._exit(0)

if __name__ == "__main__": main()
