"""Auditable pre-norm causal Transformer reference."""
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 32000
    n_layers: int = 16
    d_model: int = 384
    n_heads: int = 6
    d_ff: int = 1536
    max_seq_len: int = 1024
    dropout: float = 0.0
    tie_embeddings: bool = True

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.weight = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight

class Rotary(nn.Module):
    def __init__(self, head_dim, max_seq_len):
        super().__init__()
        inv = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        freqs = torch.outer(torch.arange(max_seq_len).float(), inv)
        self.register_buffer("cos", freqs.cos()[None, None], persistent=False)
        self.register_buffer("sin", freqs.sin()[None, None], persistent=False)
    def forward(self, q, k):
        t = q.shape[-2]; c, s = self.cos[..., :t, :], self.sin[..., :t, :]
        def rotate(x):
            xe, xo = x[..., ::2], x[..., 1::2]
            return torch.stack((xe * c - xo * s, xe * s + xo * c), dim=-1).flatten(-2)
        return rotate(q), rotate(k)

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.n_heads = cfg.n_heads; self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model); self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.rope = Rotary(self.head_dim, cfg.max_seq_len); self.dropout = cfg.dropout
    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim).unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2); q, k = self.rope(q, k)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        return self.proj(y.transpose(1, 2).contiguous().view(b, t, c))

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.fc = nn.Linear(cfg.d_model, cfg.d_ff); self.proj = nn.Linear(cfg.d_ff, cfg.d_model)
    def forward(self, x): return self.proj(F.gelu(self.fc(x), approximate="tanh"))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.norm1 = RMSNorm(cfg.d_model); self.attn = CausalSelfAttention(cfg); self.norm2 = RMSNorm(cfg.d_model); self.mlp = MLP(cfg)
    def forward(self, x): return x + self.mlp(self.norm2(x + self.attn(self.norm1(x))))

class CausalTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg
        if cfg.d_model % cfg.n_heads: raise ValueError("d_model must be divisible by n_heads")
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)]); self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings: self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init)
    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding): nn.init.normal_(m.weight, std=0.02)
    def forward(self, input_ids, labels=None, return_hidden=False):
        x = self.token_embedding(input_ids); hidden = [x] if return_hidden else None
        for block in self.blocks:
            x = block(x)
            if return_hidden: hidden.append(x)
        logits = self.lm_head(self.norm(x))
        if labels is None: return (logits, hidden) if return_hidden else logits
        # Labels use the natural same-length stream convention: token t predicts
        # token t+1. The final position has no target and is excluded.
        if labels.shape == input_ids.shape:
            logits_for_loss, labels_for_loss = logits[:, :-1], labels[:, 1:]
        else:
            logits_for_loss, labels_for_loss = logits, labels
        loss = F.cross_entropy(logits_for_loss.reshape(-1, logits.size(-1)), labels_for_loss.reshape(-1))
        return (loss, logits, hidden) if return_hidden else (loss, logits)

def count_parameters(model):
    parts = {}
    for name, p in model.named_parameters():
        root = name.split(".")[0]; parts[root] = parts.get(root, 0) + p.numel()
    return parts, sum(parts.values())
