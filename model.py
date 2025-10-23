import torch
from torch import nn
import torch.nn.functional as F

import math
from dataclasses import dataclass

from .utils import make_tuple


@dataclass
class ModulationOut:
    shift: torch.Tensor
    scale: torch.Tensor
    gate: torch.Tensor


class Modulation(nn.Module):
    def __init__(self, dim, double=False):
        super().__init__()
        self.make_double = double
        self.multiplier = 3 # scale, shift, gate
        self.proj = nn.Linear(dim, self.multiplier * dim)

    def forward(self, cond_vec):
        out = self.proj(
            F.silu(cond_vec)
        )[:, None].chunk(self.multiplier, dim=-1)

        return ModulationOut(*out[:3])


class QKNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qn = nn.RMSNorm(dim)
        self.kn = nn.RMSNorm(dim)
    
    def forward(self, q, k, v):
        q = self.qn(q)
        k = self.kn(k)
        return q.to(v), k.to(v)
    


class StreamBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, qk_scale=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = qk_scale or self.head_dim**-.5

        self.hidden_mlp = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, embed_dim * 3 + self.hidden_mlp)
        self.fc2 = nn.Linear(embed_dim + self.hidden_mlp, embed_dim)
        self.pre_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=1e-6)
        self.norm = QKNorm(self.head_dim)
        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(self.embed_dim)

    def forward(self, x, pos_embed, cond_vec):
        b, l, _ = x.shape
        mod = self.modulation(cond_vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(
            self.fc1(x_mod), [3 * self.embed_dim, self.hidden_mlp], dim=-1
        )

        qkv = qkv.view(b, l, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).contiguous() # (3, B, nh, L, hd)
        q, k = self.norm(q, k, v)

        attn = self.attention(q, k, v, pos_embed)
        out = self.fc2(torch.cat((attn, self.mlp_act(mlp)), dim=-1))
        return x + mod.gate * out


    def attention(self, q, k, v, pos_embed):
        q, k = self.apply_rope(q, k, pos_embed)
        x = F.scaled_dot_product_attention(q, k, v) # b, nh, l, hd
        b, nh, l, hd = x.shape
        
        return x.permute(0, 2, 1, 3).contiguous().view( # b, l, nh, hd
            b, l, nh*hd
        ) # b, l, d
        

    def apply_rope(self, q, k, rot_mats):
        q_shape, k_shape = q.shape, k.shape
        q = q.reshape(*q.shape[:-1], -1, 1, 2) # ... dim/2, 1, 2
        k = k.reshape(*k.shape[:-1], -1, 1, 2) # ... idm/2, 1, 2

        q = rot_mats[..., 0] * q[..., 0] + rot_mats[..., 1] * q[..., 1]
        k = rot_mats[..., 0] * k[..., 0] + rot_mats[..., 1] * k[..., 1]
        return q.reshape(q_shape), k.reshape(k_shape)


class NDRoPE(nn.Module):
    def __init__(self, dim, theta, axes_dim):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim
    
    def forward(self, ids):
        ndims = ids.shape[-1]
        ropes = torch.cat([
            self.rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(ndims)
        ], dim=-3)
        return ropes.unsqueeze(1) # attention broadcast


    def rope(self, pos, dim, theta):
        assert dim % 2 == 0, f"invalid dim ({dim}) passed to rope"
        scale = torch.arange(0, dim, 2, dtype=torch.float32) / dim
        omega = 1. / (theta ** scale)
        angles = torch.einsum("...n, d -> ...nd", pos, omega)
        rot_mats = torch.stack([
            torch.cos(angles), -torch.sin(angles), torch.sin(angles), torch.cos(angles)
        ], dim=-1) # pos, dim/2, 4
        return rot_mats.reshape(*rot_mats.shape[:-1], 2, 2)


class AdaLN(nn.Module):
    def __init__(self, embed_dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.proj_out = nn.Linear(embed_dim, out_dim)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim*2)
        )

    def forward(self, x, cond_vec):
        shift, scale = self.modulation(cond_vec).chunk(2, dim=1)
        x = (1 + scale[:, None]) * self.norm(x) + shift[:, None]
        return self.proj_out(x)


class DiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ph, self.pw = make_tuple(config.patch_size)
        frame_h, frame_w = make_tuple(config.frame_size)
        self.nh = frame_h // self.ph
        self.nw = frame_w // self.pw

        self.cond_dim = config.cond_dim
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        c = config.n_channels
        assert self.embed_dim % self.num_heads == 0, f"Invalid embed dim({self.embed_dim}), num heads({self.num_heads}) config"

        self.proj_embed = nn.Linear(
            self.ph * self.pw * c,
            self.embed_dim,
        )

        self.rope_dim = self.embed_dim // self.num_heads
        self.nd_rope = NDRoPE(self.rope_dim, 100_000, [self.rope_dim//2, self.rope_dim//4, self.rope_dim//4])

        self.blocks = nn.ModuleList(
            [StreamBlock(self.embed_dim, self.num_heads) for _ in range(config.depth)]
        )

        self.time_embed_proj = nn.Sequential(
            nn.Linear(self.cond_dim, self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        self.adaln = AdaLN(self.embed_dim, self.ph * self.pw * c)

    def forward(self, x, x_ids, timestep):
        b, t, c, h, w = x.shape
        x = x.view(b, t, c, self.nh, self.ph, self.nw, self.pw) # b, t, c, n, p, n, p
        x = x.permute(0, 1, 3, 5, 2, 4, 6) # b, t, n, n, c, p, p
        x = x.contiguous().view(
            b, 
            t * self.nh * self.nw, 
            c * self.ph * self.pw
        ) # b, N, D

        x = self.proj_embed(x)
        cond_vec = self.time_embed_proj(self.time_embedding(timestep, self.cond_dim))
        rope_embeds = self.nd_rope(x_ids)

        for block in self.blocks:
            x = block(x, rope_embeds, cond_vec)
        
        return self.adaln(x, cond_vec).reshape(b, t, c, h, w)


    def time_embedding(self, t, dim, max_period=10_000, time_factor=1000.):
        t *= time_factor
        assert dim % 2 == 0, f"invalid dim ({dim}) passed to sinusoidal pos embedding"
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, dim/2))
        args = t[:, None] * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    


if __name__ == "__main__":
    from types import SimpleNamespace
    from torch._subclasses.fake_tensor import FakeTensorMode

    config = SimpleNamespace(
        patch_size=4,
        frame_size=16,
        embed_dim=256,
        num_heads=4,
        depth=2,
        n_channels=1,
        cond_dim=256
    )

    B, T, C, H, W = 2, 4, 1, 16, 16
    ph = pw = config.patch_size
    nh, nw = H // ph, W // pw
    L = T * nh * nw

    x = torch.randn(B, T, C, H, W)
    tt, yy, xx = torch.meshgrid(
        torch.arange(T), torch.arange(nh), torch.arange(nw), indexing="ij"
    )
    x_ids = torch.stack([tt, yy, xx], dim=-1).reshape(1, L, 3).expand(B, -1, -1).to(torch.float32)
    timestep = torch.randint(0, 1000, (B,), dtype=torch.float32)

    fake_mode = FakeTensorMode()
    converter = fake_mode.fake_tensor_converter
    x, x_ids, timestep = [converter.from_real_tensor(fake_mode, d) for d in [x, x_ids, timestep]]
    
    with fake_mode:
        model = DiT(config)
        print(f'{sum(p.numel() for p in model.parameters()) * 1e-6:.3f}M params')
        recon_x = model(x, x_ids, timestep)
        print(recon_x)