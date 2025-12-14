from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# shared network component
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = RMSNorm(dim)
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h*d) -> b h n d', h = self.heads), (q, k, v))

        # scaled dot-product attention
        sim = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = sim.softmax(dim = -1)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TransformerBlock(nn.Module):
    """
    Contains: Self-Attention -> Add & Norm -> MLP -> Add & Norm
    """
    def __init__(self, dim, heads = 8, dim_head = 64, mlp_mult = 4):
        super().__init__()
        self.attn = Attention(dim, heads = heads, dim_head = dim_head)
        self.ff = FeedForward(dim, mult = mlp_mult)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ff(x)
        return x

class TinyRecursiveModel(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        num_refinement_blocks = 3,   # N_sup (Outer Loop)
        num_latent_refinements = 6,  # n (Inner Loop)
        heads = 8,
        dim_head = 64
    ):
        super().__init__()
        
        # 1. inputs (x)
        self.input_embed = nn.Embedding(num_tokens, dim)
        
        # 2. state init (y and z)
        self.output_init_embed = nn.Parameter(torch.randn(dim) * 0.02) # Prediction (y)
        self.latent_init_embed = nn.Parameter(torch.randn(dim) * 0.02) # Latent (z)

        # 3. shared network (reused recursively)
        self.network = TransformerBlock(dim, heads=heads, dim_head=dim_head)

        # loop counters
        self.num_refinement_blocks = num_refinement_blocks
        self.num_latent_refinements = num_latent_refinements

        # 4. final prediction head (reverse embedding)
        self.to_pred = nn.Linear(dim, num_tokens, bias = False)

    @property
    def device(self):
        return next(self.parameters()).device

    def get_initial_states(self, batch_size, seq_len):
        """
        Helper to expand the learnable init vectors to the batch size.
        """
        # (dim) -> (b, n, dim)
        y = repeat(self.output_init_embed, 'd -> b n d', b = batch_size, n = seq_len)
        z = repeat(self.latent_init_embed, 'd -> b n d', b = batch_size, n = seq_len)
        return y, z

    def refine_step(self, x, y, z):
        """
        Performs one full pass of:
        1. Refine Latent z 
        2. Refine Prediction y
        """
        
        # updating z given x, y, z
        # inner loop runs 'n' times
        for _ in range(self.num_latent_refinements):
            # Input is sum of x, y, z
            combined = x + y + z
            
            # Update z
            z = self.network(combined)

        # updating y given y, z
        combined = y + z
        
        # update y
        y = self.network(combined)

        return y, z

    def forward(self, seq):
        """
        seq: (batch, seq_len) - input token IDs
        """
        b, n = seq.shape
        
        # 1.Embed Input
        x = self.input_embed(seq)

        # 2.Get Initial States
        # Prediction (y) and Latent (z)
        y, z = self.get_initial_states(b, n)

        # 3.Main Recursive Loop (applied N_sup times)
        for step in range(self.num_refinement_blocks):
            
            y, z = self.refine_step(x, y, z)

        # 4.Final Prediction
        # Reverse Embedding -> Cross Entropy
        logits = self.to_pred(y)
        
        return logits