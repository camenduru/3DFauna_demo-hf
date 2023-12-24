import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from typing import Union, List, Tuple
import os
import video3d.utils.misc as misc
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Transformer_layer(nn.Module):
    def __init__(self, dim_feat=384, dim=1024, hidden_dim=1024, heads=16):
        super().__init__()
        '''
        dim: the dim between each attention, mlp, also the input and output dim for the layer
        hidden_dim: the dim inside qkv
        dim_feat: condition feature dim
        '''
        dim_head = hidden_dim // heads
        self.heads = heads
        self.scale = dim_head ** -0.5  # 8

        self.norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(
            dim=dim,
            hidden_dim=(4 * dim),
            dropout=0.
        )

        # cross attention part
        self.to_cross_q = nn.Linear(dim, hidden_dim, bias=False)
        self.to_cross_kv = nn.Linear(dim_feat, hidden_dim*2, bias=False)
        self.cross_attend = nn.Softmax(dim=-1)

        # self attention part
        self.to_self_qkv = nn.Linear(dim, hidden_dim*3, bias=False)
        self.self_attend = nn.Softmax(dim=-1)

    def forward_cross_attn(self, x, feature):
        x = self.norm(x)

        q = self.to_cross_q(x)
        k, v = self.to_cross_kv(feature).chunk(2, dim=-1)
        qkv = (q, k, v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.cross_attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return out
    
    def forward_self_attn(self, x):
        x = self.norm(x)
        qkv = self.to_self_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.self_attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return out
    
    def forward(self, x, feature):
        '''
        x: [B, N, dim]
        feature: [B, N, dim_feat]
        '''
        cross_token = self.forward_cross_attn(x, feature)
        cross_token = cross_token + x

        self_token = self.forward_self_attn(cross_token)
        self_token = self_token + cross_token

        out = self.ffn(self_token)
        out = out + self_token

        return out
    

class Triplane_Transformer(nn.Module):
    def __init__(self, emb_dim=1024, emb_num=1024, num_layers=16,
                 triplane_dim=80, triplane_scale=7.):
        super().__init__()

        self.learnable_embedding = nn.Parameter(torch.randn(1, emb_num, emb_dim))
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(
                Transformer_layer(
                    dim_feat=384,
                    dim=emb_dim,
                    hidden_dim=emb_dim
                )
            )
        
        self.triplane_dim = triplane_dim
        self.triplane_scale = triplane_scale

        self.to_triplane = nn.ConvTranspose2d(
            in_channels=emb_dim,
            out_channels=3 * triplane_dim,
            kernel_size=4,
            padding=1,
            stride=2
        )

        self.norm = nn.LayerNorm(emb_dim)
    
    def sample_feat(self, feat_maps, pts):
        '''
        feat_maps: [B, 3, C, H, W]
        pts: [B, K, 3]
        '''
        pts = pts / (self.triplane_scale / 2)

        pts_xy = pts[..., [0,1]]
        pts_yz = pts[..., [1,2]]
        pts_xz = pts[..., [0,2]]

        feat_xy = feat_maps[:, 0, :, :, :]
        feat_yz = feat_maps[:, 1, :, :, :]
        feat_xz = feat_maps[:, 2, :, :, :]

        sampled_feat_xy = F.grid_sample(
            feat_xy, pts_xy.unsqueeze(1), mode='bilinear', align_corners=True
        )
        sampled_feat_yz = F.grid_sample(
            feat_yz, pts_yz.unsqueeze(1), mode='bilinear', align_corners=True
        )
        sampled_feat_xz = F.grid_sample(
            feat_xz, pts_xz.unsqueeze(1), mode='bilinear', align_corners=True
        )

        sampled_feat = torch.cat([sampled_feat_xy, sampled_feat_yz, sampled_feat_xz], dim=1).squeeze(-2)  # [B, F, K]
        sampled_feat = sampled_feat.permute(0, 2, 1)
        return sampled_feat
        
    def forward(self, feature, pts):
        '''
        feature: [B, N, dim_feat]
        '''
        batch_size = feature.shape[0]
        embedding = self.learnable_embedding.repeat(batch_size, 1, 1)

        x = embedding
        for layer in self.layers:
            x = layer(x, feature)
        x = self.norm(x)
        # x: [B, 32x32, 1024]
        batch_size, pwph, feat_dim = x.shape
        ph = int(pwph ** 0.5)
        pw = int(pwph ** 0.5)
        triplane_feat = x.reshape(batch_size, ph, pw, feat_dim).permute(0, 3, 1, 2)
        triplane_feat = self.to_triplane(triplane_feat)  # [B, C, 64, 64]

        triplane_feat = triplane_feat.reshape(triplane_feat.shape[0], 3, self.triplane_dim, triplane_feat.shape[-2], triplane_feat.shape[-1])

        pts_feat = self.sample_feat(triplane_feat, pts)

        return pts_feat

