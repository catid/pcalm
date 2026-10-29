import math
import torch
from src.data import fixed_chunks, rank_chunks, next_token_batch
from src.data_stream import count_loss_tokens, StreamSpec, rank_token_chunks

def test_linear_alm_equilibrium():
    # min_h 1/2(h-y)^2 subject to h-wx=0, w fixed. KKT gives h=wx and
    # lambda=y-h for the Lagrangian convention used in the study.
    w, x, y = 2.0, 3.0, 7.0
    h_star, lam_star = w*x, y-w*x
    assert math.isclose(h_star, 6.0) and math.isclose(lam_star, 1.0)
    c = h_star - w*x
    assert abs(c) < 1e-12
    assert math.isclose((h_star-y) + lam_star, 0.0, abs_tol=1e-12)

def test_local_linear_finite_difference():
    w = torch.tensor(1.7, dtype=torch.float64, requires_grad=True); x = torch.tensor(2.3, dtype=torch.float64); y = torch.tensor(0.4, dtype=torch.float64)
    loss = 0.5 * (w*x-y)**2; (g,) = torch.autograd.grad(loss, (w,))
    eps = 1e-6
    fd = (0.5*(w.detach()+eps).mul(x).sub(y).pow(2) - 0.5*(w.detach()-eps).mul(x).sub(y).pow(2))/(2*eps)
    assert torch.allclose(g, fd, atol=1e-7, rtol=1e-7)

def test_rank_chunks_are_disjoint_and_counted():
    chunks = list(fixed_chunks(range(40), 10)); a = list(rank_chunks(chunks, 0, 2, validation_chunks=1)); b = list(rank_chunks(chunks, 1, 2, validation_chunks=1))
    assert a == [[10,11,12,13,14,15,16,17,18,19], [30,31,32,33,34,35,36,37,38,39]]
    assert b == [[20,21,22,23,24,25,26,27,28,29]]
    i, lab, n = next_token_batch(a[0]); assert i == list(range(10,19)) and lab == list(range(11,20)) and n == 9
    assert count_loss_tokens(a) == 18

def test_validation_chunk_isolation():
    # A synthetic tokenizer/stream demonstrates the protocol: validation owns
    # the first deterministic chunk range and no training rank receives it.
    class Tok:
        def encode(self, text, add_special_tokens=False):
            n, start = map(int, text.split(",")); return list(range(start, start+n))
    spec = StreamSpec(chunk_tokens=4, validation_chunks=1)
    rows = [{"text": "4,0"}, {"text": "4,4"}, {"text": "4,8"}, {"text": "4,12"}]
    all_chunks = list(__import__("src.data_stream", fromlist=["iter_token_chunks"]).iter_token_chunks(rows, Tok(), spec))
    train0 = list(rank_token_chunks(rows, Tok(), spec, 0, 2)); train1 = list(rank_token_chunks(rows, Tok(), spec, 1, 2))
    assert all_chunks[0] not in train0 and all_chunks[0] not in train1 and set(map(tuple, train0)).isdisjoint(set(map(tuple, train1)))
