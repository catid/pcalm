"""FineWeb streaming iterator with explicit rank and token accounting."""
from dataclasses import dataclass
import itertools

@dataclass(frozen=True)
class StreamSpec:
    dataset: str = "HuggingFaceFW/fineweb"
    config: str = "sample-10BT"
    split: str = "train"
    chunk_tokens: int = 1024
    validation_chunks: int = 10000

def iter_token_chunks(dataset_iter, tokenizer, spec: StreamSpec):
    buf = []
    for row in dataset_iter:
        ids = tokenizer.encode(row["text"], add_special_tokens=False)
        buf.extend(ids)
        while len(buf) >= spec.chunk_tokens:
            yield buf[:spec.chunk_tokens]
            del buf[:spec.chunk_tokens]

def rank_token_chunks(dataset_iter, tokenizer, spec, rank, world_size):
    for i, chunk in enumerate(iter_token_chunks(dataset_iter, tokenizer, spec)):
        if i < spec.validation_chunks: continue
        if (i - spec.validation_chunks) % world_size == rank: yield chunk

def count_loss_tokens(chunks):
    # A chunk of length T contributes T-1 next-token targets. The final budget
    # is reached only by retaining exactly the partial final chunk metadata.
    return sum(max(len(c) - 1, 0) for c in chunks)
