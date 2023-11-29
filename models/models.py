"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# -----------------------------------------------------------------------------

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, dim, seq_len, heads):
        super().__init__()        
        self.c_attn = nn.Linear(dim, 3 * dim) # dim is embedding
        # output projection
        self.c_proj = nn.Linear(dim,dim)
        # regularization
        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(seq_len, seq_len))
                                     .view(1, 1, seq_len, seq_len))
        self.n_head = heads
        self.n_embd = dim

    def forward(self, x):
        B, T, C = x.size() # batch size=B, sequence length=T, embedding dimensionality (n_embd)=C

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2) # x=(B,T,dim) T is seq_len
        k = k.view(B, T, 8, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, 8, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, 8, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, dim, seq_len, heads):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, seq_len, heads)
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(dim, 4 * dim),
            c_proj  = nn.Linear(4 * dim, dim),
            act     = NewGELU(),
            dropout = nn.Dropout(0.1),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class GPT(nn.Module):
    """ GPT Language Model """
    
    
    def __init__(self, dim, num_unique_tokens=256, num_layers=6, heads=8, dim_head=None, max_seq_len=1024): # modified to match our way of creating model
        super().__init__()        
        self.block_size = max_seq_len
        self.max_seq_len = max_seq_len
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(num_unique_tokens, dim),
            wpe = nn.Embedding(max_seq_len, dim),
            drop = nn.Dropout(0.1),
            h = nn.ModuleList([Block(dim, max_seq_len, heads) for _ in range(heads)]), # added Block() modified to Block(dim,max_seq_len,heads)
            ln_f = nn.LayerNorm(dim),
            ))
        self.lm_head = nn.Linear(dim, num_unique_tokens, bias=False)       
        

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * 8))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)        

        return logits

    
