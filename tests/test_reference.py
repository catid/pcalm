import torch
from src.model import GPTConfig, CausalTransformer, count_parameters
from src.methods import PCALMReference, EPCReference, StableParameterizationWrapper

def test_parameter_interval_and_tying():
    torch.manual_seed(0); m = CausalTransformer(GPTConfig())
    _, n = count_parameters(m)
    assert 38_000_000 <= n <= 42_000_000, n
    assert m.token_embedding.weight is m.lm_head.weight

def test_causal_shape_and_mask():
    torch.manual_seed(0); m = CausalTransformer(GPTConfig(vocab_size=97, n_layers=2, d_model=32, n_heads=4, d_ff=64, max_seq_len=8))
    x = torch.randint(97, (2, 8)); loss, logits = m(x, x)
    assert logits.shape == (2, 8, 97) and torch.isfinite(loss)
    x2 = x.clone(); x2[:, -1] = (x2[:, -1] + 1) % 97
    a = m(x, x)[1][:, :-1]; b = m(x2, x2)[1][:, :-1]
    assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)

def test_next_token_loss_contract():
    torch.manual_seed(1); cfg = GPTConfig(vocab_size=31, n_layers=1, d_model=16, n_heads=2, d_ff=32, max_seq_len=6)
    m = CausalTransformer(cfg); x = torch.randint(cfg.vocab_size, (1, 6)); loss, logits = m(x, x)
    expected = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, cfg.vocab_size), x[:, 1:].reshape(-1))
    assert torch.allclose(loss, expected)

def test_pcalm_state_and_diagnostics():
    torch.manual_seed(0); cfg = GPTConfig(vocab_size=97, n_layers=2, d_model=32, n_heads=4, d_ff=64, max_seq_len=8)
    m = CausalTransformer(cfg); ids = torch.randint(cfg.vocab_size, (1, 8)); method = PCALMReference(m)
    state = method.init_state(ids); assert len(state.hidden) == 3 and len(state.dual) == 2
    method.relax_step(state, ids, ids); d = method.diagnostics(state)
    assert d["step"] == 1 and all(torch.isfinite(x).all() for x in state.hidden)

def test_pcalm_relax_has_no_global_backward():
    torch.manual_seed(3); cfg = GPTConfig(vocab_size=53, n_layers=2, d_model=32, n_heads=4, d_ff=64, max_seq_len=8)
    m = CausalTransformer(cfg); ids = torch.randint(cfg.vocab_size, (1, 8)); method = PCALMReference(m); state = method.init_state(ids)
    original = torch.Tensor.backward
    def forbidden(*args, **kwargs): raise AssertionError("global backward called during PC relaxation")
    torch.Tensor.backward = forbidden
    try: method.relax_step(state, ids, ids)
    finally: torch.Tensor.backward = original
    assert state.step == 1

def test_tied_embedding_receives_output_gradient():
    torch.manual_seed(4); cfg = GPTConfig(vocab_size=53, n_layers=1, d_model=16, n_heads=2, d_ff=32, max_seq_len=8)
    m = CausalTransformer(cfg); ids = torch.randint(cfg.vocab_size, (1, 8)); loss, _ = m(ids, ids); loss.backward()
    assert m.token_embedding.weight.grad is not None and torch.isfinite(m.token_embedding.weight.grad).all()

def test_tiny_transformer_parameter_finite_difference():
    torch.manual_seed(5); cfg = GPTConfig(vocab_size=31, n_layers=1, d_model=16, n_heads=2, d_ff=32, max_seq_len=6)
    m = CausalTransformer(cfg).double(); ids = torch.randint(cfg.vocab_size, (1, 6)); loss, _ = m(ids, ids); p = m.blocks[0].mlp.fc.weight
    (g,) = torch.autograd.grad(loss, (p,)); idx = (0, 0); old = p.data[idx].item(); eps = 1e-4
    p.data[idx] = old + eps; lp = m(ids, ids)[0].item(); p.data[idx] = old - eps; lm = m(ids, ids)[0].item(); p.data[idx] = old
    assert abs(g[idx].item() - (lp-lm)/(2*eps)) < 1e-6

def test_epc_and_stable_wrapper_smoke():
    torch.manual_seed(6); cfg = GPTConfig(vocab_size=31, n_layers=2, d_model=16, n_heads=2, d_ff=32, max_seq_len=6)
    m = CausalTransformer(cfg); ids = torch.randint(cfg.vocab_size, (1, 6))
    method = StableParameterizationWrapper(EPCReference(m, inner_steps=2, parameter_step=1e-4))
    state = method.init_state(ids)
    state.hidden[-1].add_(0.01 * torch.randn_like(state.hidden[-1]))
    for _ in range(2): method.relax_step(state, ids, ids)
    update = method.local_update(state, ids, ids)
    diag = method.diagnostics(state)
    assert state.errors is not None and len(state.errors) == len(m.blocks)
    assert update["updated_layers"] == len(m.blocks)
    assert diag["step"] == 2 and all(torch.isfinite(x).all() for x in state.errors)
