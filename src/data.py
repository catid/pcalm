"""Deterministic token-stream helpers for FineWeb.

The stream is chunked before rank assignment. A chunk is either wholly assigned
to one rank or withheld for validation, so token ranges cannot overlap and the
post-shift loss count is explicit.
"""
from itertools import islice

def fixed_chunks(tokens, chunk_tokens):
    it = iter(tokens)
    while True:
        x = list(islice(it, chunk_tokens))
        if not x: return
        if len(x) == chunk_tokens: yield x

def rank_chunks(chunks, rank, world_size, validation_chunks=0):
    """Yield deterministic training chunks for rank; first chunks are validation."""
    for i, chunk in enumerate(chunks):
        if i < validation_chunks: continue
        if (i - validation_chunks) % world_size == rank: yield chunk

def next_token_batch(chunk):
    """Return input/label arrays and the exact post-shift token count."""
    if len(chunk) < 2: return chunk, [], 0
    return chunk[:-1], chunk[1:], len(chunk) - 1
