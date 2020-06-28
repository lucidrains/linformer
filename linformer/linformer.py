import torch
from torch import nn
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return x + self.fn(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.LeakyReLU(inplace = True),
            nn.Linear(dim * mult, dim)
        )
    def forward(self, x):
        return self.net(x)

class LinformerSelfAttention(nn.Module):
    def __init__(self, dim, seq_len, k = 256, heads = 8, one_kv_head = False, share_kv = False):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'

        self.seq_len = seq_len
        self.k = k

        self.heads = heads
        self.dim_head = dim // heads

        self.to_q = nn.Linear(dim, dim, bias = False)

        kv_dim = self.dim_head if one_kv_head else dim
        self.to_k = nn.Linear(dim, kv_dim, bias = False)
        self.proj_k = nn.Parameter(torch.randn(seq_len, k))

        self.share_kv = share_kv
        if not share_kv:
            self.to_v = nn.Linear(dim, kv_dim, bias = False)
            self.proj_v = nn.Parameter(torch.randn(seq_len, k))

    def forward(self, x, context = None):
        b, n, d, d_h, h, k = *x.shape, self.dim_head, self.heads, self.k

        kv_len = n if context is None else context.shape[1] 
        assert kv_len == self.seq_len, f'the sequence length of the key / values must be {self.seq_len} - {kv_len} given'

        queries = self.to_q(x)

        proj_seq_len = lambda args: torch.einsum('bnd,nk->bkd', *args)

        kv_input = x if context is None else context

        keys = self.to_k(kv_input)
        values = self.to_v(kv_input) if not self.share_kv else keys

        kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)

        # project keys and values along the sequence length dimension to k
        
        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))

        # merge head into batch for queries and key / values

        queries = queries.reshape(b, n, h, -1).transpose(1, 2)

        merge_key_values = lambda t: t.reshape(b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
        keys, values = map(merge_key_values, (keys, values))

        # attention

        dots = torch.einsum('bhnd,bhkd->bhnk', queries, keys)
        attn = dots.softmax(dim=-1)
        attn = torch.einsum('bhnk,bhkd->bhnd', attn, values)

        # split heads

        attn = attn.transpose(1, 2).reshape(b, n, d)
        return attn

class Linformer(nn.Module):
    def __init__(self, dim, seq_len, depth, k = 256, heads = 8, one_kv_head = False, share_kv = False):
        super().__init__()
        layers = []
        for _ in range(depth):
            attn = LinformerSelfAttention(dim, seq_len, k = k, heads = heads, one_kv_head = one_kv_head, share_kv = share_kv)
            ff = FeedForward(dim)

            attn, ff = map(lambda fn: Residual(PreNorm(dim, fn)), (attn, ff))
            layers.extend([attn, ff])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class LinformerLM(nn.Module):
    def __init__(self, num_tokens, dim, seq_len, depth, k = 256, heads = 8, one_kv_head = False, share_kv = False):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.linformer = Linformer(dim, seq_len, depth, k = k, heads = heads, one_kv_head = one_kv_head, share_kv = share_kv)
        self.to_logits = nn.Linear(dim, num_tokens)

    def forward(self, x):
        x = self.token_emb(x)
        x = self.pos_emb(torch.arange(x.shape[1], device=x.device)) + x
        x = self.linformer(x)
        out = self.to_logits(x)
        return out
