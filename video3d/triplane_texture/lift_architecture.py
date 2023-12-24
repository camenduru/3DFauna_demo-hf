import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from typing import Union, List, Tuple
import os
import video3d.utils.misc as misc
import torch.nn.functional as F


class Lift_Encoder(nn.Module):
    def __init__(
        self,
        cin,
        feat_dim,
        grid_scale=7.,
        grid_size=32,
        optim_latent=False,
        img_size=256,
        with_z_feature=False,
        cam_pos_z_offset=10.
    ):
        super().__init__()

        '''
        unproject the input feature map to tri-plane, each plane is (-1, -1)*grid_scale to (1, 1)*scale
        '''
        self.cin = cin
        self.nf = feat_dim
        self.grid_scale = grid_scale
        self.grid_size = grid_size
        self.img_size = img_size
        self.with_z_feature = with_z_feature
        self.cam_pos_z_offset = cam_pos_z_offset

        self.feature_projector = nn.Linear(cin, feat_dim, bias=False)

        self.plane_latent = None
        if optim_latent:
            self.optim_latent = nn.Parameter(torch.rand(3, feat_dim, grid_size, grid_size))
        else:
            self.optim_latent = None

        if with_z_feature:
            self.conv_bottleneck = nn.Conv2d(feat_dim+1, feat_dim, kernel_size=3, stride=1, padding=1, dilation=1, padding_mode="replicate")
        else:
            self.conv_bottleneck = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, stride=1, padding=1, dilation=1, padding_mode="replicate")
        
        #TODO: implement an upsampler for input feature map here?
        self.conv_1 = nn.Conv2d(feat_dim, 4*feat_dim, kernel_size=3, stride=1, padding=1, dilation=1, padding_mode="replicate")
        self.conv_2 = nn.Conv2d(feat_dim, 4*feat_dim, kernel_size=3, stride=1, padding=1, dilation=1, padding_mode="replicate")
        self.up = nn.PixelShuffle(2)

        self.conv_enc = nn.Conv2d(feat_dim, feat_dim // 2, kernel_size=3, stride=1, padding=1, dilation=1, padding_mode="replicate")
        self.conv_dec = nn.Conv2d(feat_dim // 2, feat_dim, kernel_size=3, stride=1, padding=1, dilation=1, padding_mode="replicate")

        self.feature_fusion = nn.Linear(3*feat_dim, feat_dim, bias=False)

    def get_coords(self, grid_size):
        with torch.no_grad():
            lines = torch.arange(0, grid_size)
            grids_x, grids_y = torch.meshgrid([lines, lines], indexing="ij")
            grids = torch.stack([grids_x, grids_y], dim=-1)
            grids = (grids - self.grid_size // 2) / (self.grid_size // 2)
            grids = grids * self.grid_scale

            plane_z0 = torch.cat([grids, torch.zeros(list(grids.shape[:-1]) + [1])], dim=-1)  # [S, S, 3]
            plane_y0 = plane_z0.clone()[..., [0, 2, 1]]
            plane_x0 = plane_z0.clone()[..., [2, 0, 1]]
        
        planes = torch.stack([plane_x0, plane_y0, plane_z0], dim=0)
        return planes # [3, S, S, 3]
    
    def get_uv_z(self, pts, mvp):
        cam4 = torch.matmul(torch.nn.functional.pad(pts, pad=(0,1), mode='constant', value=1.0), torch.transpose(mvp, 1, 2))
        cam3 = cam4[..., :3] / cam4[..., 3:4]
        cam_uv = cam3[..., :2]
        # cam_uv = cam_uv.detach()
        cam_depth = cam3 + torch.FloatTensor([0, 0, self.cam_pos_z_offset]).to(pts.device).view(1, 1, 3)
        cam_depth = cam_depth / self.grid_scale * 2
        cam_depth = cam_depth[..., 2:3]

        return cam_uv, cam_depth
    
    def unproject(self, feature_map, mvp):
        '''
        feature_map: [B, C, h, w]
        mvp: [B, 4, 4]
        '''
        self.plane_latent = None
        bs, C, h, w = feature_map.shape
        device = feature_map.device
        feature_map = self.feature_projector(feature_map.permute(0, 2, 3, 1).reshape(-1, C)).reshape(bs, h, w, self.nf).permute(0, 3, 1, 2)

        feature_map = self.up(self.conv_1(feature_map))
        feature_map = self.up(self.conv_2(feature_map))

        plane_coords = self.get_coords(self.grid_size)
        plane_coords = plane_coords.unsqueeze(0).repeat(bs, 1, 1, 1, 1)
        plane_coords = plane_coords.to(device)
        
        plane_pts = plane_coords.reshape(bs, -1, 3)  # [B, N_POINTS, 3]
        plane_uv, plane_z = self.get_uv_z(plane_pts, mvp)
        plane_uv = plane_uv.detach()
        plane_z = plane_z.detach()

        nP = plane_pts.shape[1]

        plane_feature = F.grid_sample(feature_map, plane_uv.reshape(bs, 1, nP, 2), mode="bilinear", padding_mode="zeros").squeeze(dim=-2).permute(0, 2, 1)
        if self.with_z_feature:
            plane_feature = torch.cat([plane_feature, plane_z], dim=-1)
        
        plane_feature = plane_feature.reshape(plane_feature.shape[0], 3, self.grid_size, self.grid_size, plane_feature.shape[-1])
        
        return plane_feature

    def conv_plane(self, plane_feature):
        bs, _, nh, nw, nC = plane_feature.shape
        plane_feature = plane_feature.reshape(-1, nh, nw, nC).permute(0, 3, 1, 2)  # [bs*3, nC, nh, nw]
        
        plane_feature = self.conv_bottleneck(plane_feature)
        x = self.conv_dec(self.conv_enc(plane_feature))
        out = x + plane_feature
        out = out.reshape(bs, 3, out.shape[-3], out.shape[-2], out.shape[-1])

        if self.optim_latent is not None:
            optim_latent = self.optim_latent.unsqueeze(0).repeat(bs, 1, 1, 1, 1)
            out = out + optim_latent

        return out
    
    def sample_plane(self, pts, feat):
        '''
        pts: [B, K, 3]
        feat: [B, 3, C, h, w]
        '''
        pts_x, pts_y, pts_z = pts.unbind(dim=-1)

        pts_x0 = torch.stack([pts_y, pts_z], dim=-1)
        pts_y0 = torch.stack([pts_x, pts_z], dim=-1)
        pts_z0 = torch.stack([pts_x, pts_y], dim=-1)

        feat_x0 = F.grid_sample(feat[:, 0, :, :], pts_x0.unsqueeze(1), mode="bilinear", padding_mode="border").squeeze(-2).permute(0, 2, 1)
        feat_y0 = F.grid_sample(feat[:, 0, :, :], pts_y0.unsqueeze(1), mode="bilinear", padding_mode="border").squeeze(-2).permute(0, 2, 1)
        feat_z0 = F.grid_sample(feat[:, 0, :, :], pts_z0.unsqueeze(1), mode="bilinear", padding_mode="border").squeeze(-2).permute(0, 2, 1)

        pts_feat = torch.cat([feat_x0, feat_y0, feat_z0], dim=-1)
        return pts_feat

    def forward(self, feature_map, mvp, pts, inference="unproject"):
        '''
        inference = "unproject" or "sample"
        '''
        assert inference in ["unproject", "sample"]
        if inference == "unproject":
            plane_feature = self.unproject(feature_map, mvp)
            plane_feature = self.conv_plane(plane_feature)

            self.plane_latent = plane_feature.clone().detach()  # this is just for test case
        
        if inference == "unproject":
            feat_to_sample = plane_feature
        else:
            new_bs = pts.shape[0]
            feat_to_sample = self.plane_latent[:new_bs]
        
        pts_feature = self.sample_plane(pts, feat_to_sample)
        pts_feature = self.feature_fusion(pts_feature)  # [B, K, C]

        return pts_feature