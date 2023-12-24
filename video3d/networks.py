import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from typing import Union, List, Tuple
import os
import video3d.utils.misc as misc
import torch.nn.functional as F
from siren_pytorch import SirenNet
from video3d.triplane_texture.lift_architecture import Lift_Encoder
from video3d.triplane_texture.triplane_transformer import Triplane_Transformer


EPS = 1e-7


def get_activation(name, inplace=True, lrelu_param=0.2):
    if name == 'tanh':
        return nn.Tanh()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'relu':
        return nn.ReLU(inplace=inplace)
    elif name == 'lrelu':
        return nn.LeakyReLU(lrelu_param, inplace=inplace)
    else:
        raise NotImplementedError


class MLPWithPositionalEncoding(nn.Module):
    def __init__(self, 
                 cin,
                 cout, 
                 num_layers, 
                 nf=256, 
                 dropout=0, 
                 activation=None, 
                 n_harmonic_functions=10, 
                 omega0=1,
                 extra_dim=0,
                 embed_concat_pts=True,
                 symmetrize=False):
        super().__init__()
        self.extra_dim = extra_dim

        if n_harmonic_functions > 0:
            self.embedder = HarmonicEmbedding(n_harmonic_functions, omega0)
            dim_in = cin * 2 * n_harmonic_functions
            self.embed_concat_pts = embed_concat_pts
            if embed_concat_pts:
                dim_in += cin
        else:
            self.embedder = None
            dim_in = cin
        
        self.in_layer = nn.Linear(dim_in, nf)
        self.relu = nn.ReLU(inplace=True)
        self.mlp = MLP(nf + extra_dim, cout, num_layers, nf, dropout, activation)
        self.symmetrize = symmetrize

    def forward(self, x, feat=None):
        assert (feat is None and self.extra_dim == 0) or feat.shape[-1] == self.extra_dim
        if self.symmetrize:
            xs, ys, zs = x.unbind(-1)
            x = torch.stack([xs.abs(), ys, zs], -1)  # mirror -x to +x
        
        if self.embedder is not None:
            x_in = self.embedder(x)
            if self.embed_concat_pts:
                x_in = torch.cat([x, x_in], -1)
        else:
            x_in = x
        
        x_in = self.relu(self.in_layer(x_in))

        if feat is not None:
            # if len(feat.shape) == 1:
            #     for _ in range(len(x_in.shape) - 1):
            #         feat = feat.unsqueeze(0)
            #     feat = feat.repeat(*x_in.shape[:-1], 1)
            x_in = torch.concat([x_in, feat], dim=-1)
            
        return self.mlp(x_in)


class MLPWithPositionalEncoding_Style(nn.Module):
    def __init__(self, 
                 cin,
                 cout, 
                 num_layers, 
                 nf=256, 
                 dropout=0, 
                 activation=None, 
                 n_harmonic_functions=10, 
                 omega0=1,
                 extra_dim=0,
                 embed_concat_pts=True,
                 symmetrize=False,
                 style_choice='film'):
        super().__init__()
        self.extra_dim = extra_dim

        if n_harmonic_functions > 0:
            self.embedder = HarmonicEmbedding(n_harmonic_functions, omega0)
            dim_in = cin * 2 * n_harmonic_functions
            self.embed_concat_pts = embed_concat_pts
            if embed_concat_pts:
                dim_in += cin
        else:
            self.embedder = None
            dim_in = cin
        
        self.in_layer = nn.Linear(dim_in, nf)
        self.relu = nn.ReLU(inplace=True)

        if extra_dim == 0:
            self.mlp = MLP(nf + extra_dim, cout, num_layers, nf, dropout, activation)
        
        else:
            if style_choice == 'film':
                self.mlp = MLP_FiLM(nf, cout, num_layers, nf, dropout, activation)
                self.style_mlp = MLP(extra_dim, nf*2, 2, nf, dropout, None)
            
            elif style_choice == 'mod':
                self.mlp = MLP_Mod(nf, cout, num_layers, nf, dropout, activation)
                self.style_mlp = MLP(extra_dim, nf, 2, nf, dropout, None)
            
            else:
                raise NotImplementedError

            self.style_choice = style_choice
        
        self.symmetrize = symmetrize

    def forward(self, x, feat=None):
        assert (feat is None and self.extra_dim == 0) or feat.shape[-1] == self.extra_dim
        if self.symmetrize:
            xs, ys, zs = x.unbind(-1)
            x = torch.stack([xs.abs(), ys, zs], -1)  # mirror -x to +x
        
        if self.embedder is not None:
            x_in = self.embedder(x)
            if self.embed_concat_pts:
                x_in = torch.cat([x, x_in], -1)
        else:
            x_in = x
        
        x_in = self.relu(self.in_layer(x_in))

        if feat is not None:
            style = self.style_mlp(feat)

            if self.style_choice == 'film':
                style = style.reshape(style.shape[:-1] + (-1, 2))
            
            out = self.mlp(x_in, style)
        
        else:
            out = self.mlp(x_in)
            
        return out


class MLP_FiLM(nn.Module):
    def __init__(self, cin, cout, num_layers, nf=256, dropout=0, activation=None):
        # default no dropout
        super().__init__()
        assert num_layers >= 1
        self.num_layers = num_layers
        if num_layers == 1:
            self.network = Linear_FiLM(cin, cout, bias=False)
        else:
            self.relu = nn.ReLU(inplace=True)
            for i in range(num_layers):
                if i == 0:
                    setattr(self, f'linear_{i}', Linear_FiLM(cin, nf, bias=False))
                elif i == (num_layers-1):
                    setattr(self, f'linear_{i}', Linear_FiLM(nf, cout, bias=False))
                else:
                    setattr(self, f'linear_{i}', Linear_FiLM(nf, nf, bias=False))

    def forward(self, input, style):
        if self.num_layers == 1:
            out = self.network(input, style)
        else:
            x = input
            for i in range(self.num_layers):
                linear_layer = getattr(self, f'linear_{i}')
                if i == (self.num_layers - 1):
                    x = linear_layer(x, style)
                else:
                    x = linear_layer(x, style)
                    x = self.relu(x)
            
            out = x
        return out


class MLP_Mod(nn.Module):
    def __init__(self, cin, cout, num_layers, nf=256, dropout=0, activation=None):
        # default no dropout
        super().__init__()
        assert num_layers >= 1
        self.num_layers = num_layers
        if num_layers == 1:
            self.network = Linear_Mod(cin, cout, bias=False)
        else:
            self.relu = nn.ReLU(inplace=True)
            for i in range(num_layers):
                if i == 0:
                    setattr(self, f'linear_{i}', Linear_Mod(cin, nf, bias=False))
                elif i == (num_layers-1):
                    setattr(self, f'linear_{i}', Linear_Mod(nf, cout, bias=False))
                else:
                    setattr(self, f'linear_{i}', Linear_Mod(nf, nf, bias=False))

    def forward(self, input, style):
        if self.num_layers == 1:
            out = self.network(input, style)
        else:
            x = input
            for i in range(self.num_layers):
                linear_layer = getattr(self, f'linear_{i}')
                if i == (self.num_layers - 1):
                    x = linear_layer(x, style)
                else:
                    x = linear_layer(x, style)
                    x = self.relu(x)
            
            out = x
        return out


import math

class Linear_FiLM(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, style):
        # if input is [..., D], style should be [..., D, 2]
        x = input * style[..., 0] + style[..., 1]
        return torch.nn.functional.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Linear_Mod(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, style):
        # weight: [out_features, in_features]
        # style: [..., in_features]
        if len(style.shape) > 1:
            style = style.reshape(-1, style.shape[-1])
            style = style[0]
        
        weight = self.weight * style.unsqueeze(0)
        decoefs = ((weight * weight).sum(dim=-1, keepdim=True) + 1e-5).sqrt()
        weight = weight / decoefs

        return torch.nn.functional.linear(input, weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class MLPTextureSimple(nn.Module):
    def __init__(self, 
                 cin, 
                 cout, 
                 num_layers, 
                 nf=256, 
                 dropout=0, 
                 activation=None, 
                 min_max=None, 
                 n_harmonic_functions=10, 
                 omega0=1,
                 extra_dim=0,
                 embed_concat_pts=True, 
                 perturb_normal=False,
                 symmetrize=False,
                 texture_act='relu',
                 linear_bias=False):
        super().__init__()
        self.extra_dim = extra_dim

        if n_harmonic_functions > 0:
            self.embedder = HarmonicEmbedding(n_harmonic_functions, omega0)
            dim_in = cin * 2 * n_harmonic_functions
            self.embed_concat_pts = embed_concat_pts
            if embed_concat_pts:
                dim_in += cin
        else:
            self.embedder = None
            dim_in = cin
        
        self.in_layer = nn.Linear(dim_in, nf)
        self.relu = nn.ReLU(inplace=True)

        if texture_act == 'sin':
            print('using siren network for texture mlp here')
            self.mlp = SirenNet(
                dim_in=(nf + extra_dim),
                dim_hidden=nf,
                dim_out=cout,
                num_layers=num_layers,
                final_activation=get_activation(activation),
                w0_initial=30,
                use_bias=linear_bias,
                dropout=dropout
            )
        else:
            self.mlp = MLP(nf + extra_dim, cout, num_layers, nf, dropout, activation, inner_act=texture_act, linear_bias=linear_bias)
        self.perturb_normal = perturb_normal
        self.symmetrize = symmetrize
        if min_max is not None:
            self.register_buffer('min_max', min_max)
        else:
            self.min_max = None
        self.bsdf = None

    def sample(self, x, feat=None):
        assert (feat is None and self.extra_dim == 0) or (feat.shape[-1] == self.extra_dim)
        b, h, w, c = x.shape

        if self.symmetrize:
            xs, ys, zs = x.unbind(-1)
            x = torch.stack([xs.abs(), ys, zs], -1)  # mirror -x to +x
        
        x = x.view(-1, c)
        if self.embedder is not None:
            x_in = self.embedder(x)
            if self.embed_concat_pts:
                x_in = torch.cat([x, x_in], -1)
        else:
            x_in = x
        
        x_in = self.in_layer(x_in)
        if feat is not None:
            feat = feat[:,None,None].expand(b, h, w, -1).reshape(b*h*w, -1)
            x_in = torch.concat([x_in, feat], dim=-1)
        out = self.mlp(self.relu(x_in))
        if self.min_max is not None:
            out = out * (self.min_max[1][None, :] - self.min_max[0][None, :]) + self.min_max[0][None, :]
        return out.view(b, h, w, -1)


class MLPTextureTriplane(nn.Module):
    def __init__(self, 
                 cin, 
                 cout, 
                 num_layers, 
                 nf=256, 
                 dropout=0, 
                 activation=None, 
                 min_max=None, 
                 n_harmonic_functions=10, 
                 omega0=1,
                 extra_dim=0,
                 embed_concat_pts=True, 
                 perturb_normal=False,
                 symmetrize=False,
                 texture_act='relu',
                 linear_bias=False,
                 cam_pos_z_offset=10.,
                 grid_scale=7,):
        super().__init__()
        self.extra_dim = extra_dim

        if n_harmonic_functions > 0:
            self.embedder = HarmonicEmbedding(n_harmonic_functions, omega0)
            dim_in = cin * 2 * n_harmonic_functions
            self.embed_concat_pts = embed_concat_pts
            if embed_concat_pts:
                dim_in += cin
        else:
            self.embedder = None
            dim_in = cin
        
        self.in_layer = nn.Linear(dim_in, nf)
        self.relu = nn.ReLU(inplace=True)

        self.feat_net = Triplane_Transformer(
            emb_dim=256,
            num_layers=8,
            triplane_dim=80,
            triplane_scale=grid_scale
        )
        self.extra_dim -= extra_dim
        self.extra_dim += (self.feat_net.triplane_dim * 3)

        if texture_act == 'sin':
            print('using siren network for texture mlp here')
            self.mlp = SirenNet(
                dim_in=(nf + self.extra_dim),
                dim_hidden=nf,
                dim_out=cout,
                num_layers=num_layers,
                final_activation=get_activation(activation),
                w0_initial=30,
                use_bias=linear_bias,
                dropout=dropout
            )
        else:
            self.mlp = MLP(nf + self.extra_dim, cout, num_layers, nf, dropout, activation, inner_act=texture_act, linear_bias=linear_bias)
        self.perturb_normal = perturb_normal
        self.symmetrize = symmetrize
        if min_max is not None:
            self.register_buffer('min_max', min_max)
        else:
            self.min_max = None
        self.bsdf = None

    def sample(self, x, feat=None, feat_map=None, mvp=None, w2c=None, deform_xyz=None):
        # assert (feat is None and self.extra_dim == 0) or (feat.shape[-1] == self.extra_dim)
        b, h, w, c = x.shape

        if self.symmetrize:
            xs, ys, zs = x.unbind(-1)
            x = torch.stack([xs.abs(), ys, zs], -1)  # mirror -x to +x
        
        if isinstance(feat_map, dict):
            feat_map = feat_map["im_features_map"]

        feat_map = feat_map.permute(0, 2, 3, 1)
        _, ph, pw, _ = feat_map.shape
        feat_map = feat_map.reshape(feat_map.shape[0], ph*pw, feat_map.shape[-1])
        pts_feat = self.feat_net(feat_map, x.reshape(b, -1, 3))
        pts_c = pts_feat.shape[-1]
        pts_feat = pts_feat.reshape(-1, pts_c)
        
        x = x.view(-1, c)
        if self.embedder is not None:
            x_in = self.embedder(x)
            if self.embed_concat_pts:
                x_in = torch.cat([x, x_in], -1)
        else:
            x_in = x
        
        x_in = self.in_layer(x_in)
        
        x_in = torch.concat([x_in, pts_feat], dim=-1)

        out = self.mlp(self.relu(x_in))
        if self.min_max is not None:
            out = out * (self.min_max[1][None, :] - self.min_max[0][None, :]) + self.min_max[0][None, :]
        return out.view(b, h, w, -1)


class LocalFeatureBlock(nn.Module):
    def __init__(self, local_feat_dim, input_dim=384, output_dim=384, upscale_num=3):
        super().__init__()
        self.local_feat_dim = local_feat_dim
        self.conv_list = nn.ModuleList([])
        self.upscale_list = nn.ModuleList([])

        for i in range(upscale_num):
            if i == 0:
                self.conv_list.append(nn.Conv2d(input_dim, 4 * local_feat_dim, 3, stride=1, padding=1, dilation=1))
            else:
                self.conv_list.append(nn.Conv2d(local_feat_dim, 4 * local_feat_dim, 3, stride=1, padding=1, dilation=1))
            self.upscale_list.append(nn.PixelShuffle(2))
        
        self.conv_head = nn.Conv2d(local_feat_dim, output_dim, 3, stride=1, padding=1, dilation=1)
    
    def forward(self, x):
        for idx, conv in enumerate(self.conv_list):
            x = conv(x)
            x = self.upscale_list[idx](x)
        
        out = self.conv_head(x)
        return out


class MLPTextureLocal(nn.Module):
    def __init__(self, 
                 cin, 
                 cout, 
                 num_layers, 
                 nf=256, 
                 dropout=0, 
                 activation=None, 
                 min_max=None, 
                 n_harmonic_functions=10, 
                 omega0=1,
                 extra_dim=0,
                 embed_concat_pts=True, 
                 perturb_normal=False,
                 symmetrize=False,
                 texture_way=None,
                 larger_tex_dim=False,
                 cam_pos_z_offset=10.,
                 grid_scale=7.):
        super().__init__()
        self.extra_dim = extra_dim
        self.cam_pos_z_offset = cam_pos_z_offset
        self.grid_scale = grid_scale

        local_feat_dim = 64

        assert texture_way is not None
        self.texture_way = texture_way
        if 'local' in texture_way and 'global' in texture_way:
            # self.extra_dim = extra_dim + local_feat_dim
            self.extra_dim = extra_dim
        elif 'local' in texture_way and 'global' not in texture_way:
            # self.extra_dim = local_feat_dim
            self.extra_dim = extra_dim
        elif 'local' not in texture_way and 'global' in texture_way:
            self.extra_dim = extra_dim

        if n_harmonic_functions > 0:
            self.embedder = HarmonicEmbedding(n_harmonic_functions, omega0)
            dim_in = cin * 2 * n_harmonic_functions
            self.embed_concat_pts = embed_concat_pts
            if embed_concat_pts:
                dim_in += cin
        else:
            self.embedder = None
            dim_in = cin
        
        # self.local_feature_block = LocalFeatureBlock(local_feat_dim=local_feat_dim, input_dim=384, output_dim=256)
        self.local_feature_block = nn.Linear(384, nf, bias=False)

        self.in_layer = nn.Linear(dim_in, nf)
        self.relu = nn.ReLU(inplace=True)
        self.mlp = MLP(nf + self.extra_dim, cout, num_layers, nf, dropout, activation)
        self.perturb_normal = perturb_normal
        self.symmetrize = symmetrize
        if min_max is not None:
            self.register_buffer('min_max', min_max)
        else:
            self.min_max = None
        self.bsdf = None
    
    def get_uv_depth(self, xyz, mvp):
        # xyz: [b, k, 3]
        # mvp: [b, 4, 4]
        cam4 = torch.matmul(torch.nn.functional.pad(xyz, pad=(0,1), mode='constant', value=1.0), torch.transpose(mvp, 1, 2))
        cam3 = cam4[..., :3] / cam4[..., 3:4]
        cam_uv = cam3[..., :2]
        # cam_uv = cam_uv.detach()
        cam_depth = cam3 + torch.FloatTensor([0, 0, self.cam_pos_z_offset]).to(xyz.device).view(1, 1, 3)
        cam_depth = cam_depth / self.grid_scale * 2
        cam_depth = cam_depth[..., 2:3]
        # cam_depth = cam_depth.detach()
        return cam_uv, cam_depth
    
    def proj_sample_deform(self, xyz, feat_map, mvp, w2c, img_h, img_w):
        # here the xyz is deformed points
        # and we don't cast any symmtery here
        b, k, c = xyz.shape
        THRESHOLD = 1e-4
        if isinstance(feat_map, torch.Tensor):
            coordinates = xyz
            # use pre-symmetry points to get feature and record depth
            cam_uv, cam_depth = self.get_uv_depth(coordinates, mvp)
            cam_uv = cam_uv.detach()
            cam_depth = cam_depth.detach()

            # get local feature
            feature = F.grid_sample(feat_map, cam_uv.view(b, 1, k, 2), mode='bilinear').squeeze(dim=-2).permute(0, 2, 1)  # [b, k, c]

            self.input_depth = cam_depth.reshape(b, 256, 256, 1)  # [B, 256, 256, 1]
            self.input_pts = coordinates.detach()
        
        elif isinstance(feat_map, dict):
            original_mvp = feat_map['original_mvp']
            local_feat_map = feat_map['im_features_map']
            original_depth = self.input_depth[0:b]

            coordinates = xyz
            cam_uv, cam_depth = self.get_uv_depth(coordinates, original_mvp)
            cam_uv = cam_uv.detach()
            cam_depth = cam_depth.detach()

            project_feature = F.grid_sample(local_feat_map, cam_uv.view(b, 1, k, 2), mode='bilinear').squeeze(dim=-2).permute(0, 2, 1)  # [b, k, c]
            project_depth = F.grid_sample(original_depth.permute(0, 3, 1, 2), cam_uv.view(b, 1, k, 2), mode='bilinear').squeeze(dim=-2).permute(0, 2, 1)  # [b, k, 1]

            use_mask = cam_depth <= project_depth + THRESHOLD
            feature = project_feature * use_mask.repeat(1, 1, project_feature.shape[-1])
        
        ret_feature = self.local_feature_block(feature.reshape(b*k, -1))  # the linear is without bias, so 0 value feature will still get 0 value
        return ret_feature
    
    def proj_sample(self, xyz, feat_map, mvp, w2c, img_h, img_w, xyz_before_sym=None):
        # the new one with no input feature map upsampling
        # feat_map: [B, C, H, W]
        b, k, c = xyz.shape
        if isinstance(feat_map, torch.Tensor):
            if xyz_before_sym is None:
                coordinates = xyz
            else:
                coordinates = xyz_before_sym
            # use pre-symmetry points to get feature and record depth
            cam_uv, cam_depth = self.get_uv_depth(coordinates, mvp)
            cam_uv = cam_uv.detach()
            cam_depth = cam_depth.detach()

            # get local feature
            feature = F.grid_sample(feat_map, cam_uv.view(b, 1, k, 2), mode='bilinear').squeeze(dim=-2).permute(0, 2, 1)  # [b, k, c]

            self.input_depth = cam_depth.reshape(b, 256, 256, 1)  # [B, 256, 256, 1]
            self.input_pts = coordinates.detach()

        elif isinstance(feat_map, dict):
            original_mvp = feat_map['original_mvp']
            local_feat_map = feat_map['im_features_map']
            THRESHOLD = 1e-4
            original_depth = self.input_depth[0:b]
            # if b == 1:
            #    from pdb import set_trace; set_trace()
            #    tmp_mask = xyz[0].reshape(256, 256, 3).sum(dim=-1) != 0
            #    tmp_mask = tmp_mask.cpu().numpy()
            #    tmp_mask = tmp_mask * 255
            #    src_dp = self.input_depth[0,:,:,0].cpu().numpy()
            #    input_pts = self.input_pts[0].cpu().numpy()
            #    input_mask = self.input_pts[0].reshape(256, 256, 3).sum(dim=-1) != 0
            #    input_mask = input_mask.int().cpu().numpy()
            #    input_mask = input_mask * 255
            #    np.save('./tmp_save/src_dp.npy', src_dp)
            #    np.save('./tmp_save/input_pts.npy', input_pts)
            #    import cv2
            #    cv2.imwrite('./tmp_save/input_mask.png', input_mask)
            #    cv2.imwrite('./tmp_save/mask.png', tmp_mask)
            #    test_pts_pos = xyz[0].cpu().numpy()
            #    np.save('./tmp_save/test_pts_pos.npy', test_pts_pos)
            #    test_pts_raw = xyz_before_sym[0].cpu().numpy()
            #    np.save('./tmp_save/test_pts_raw.npy', test_pts_raw)
            #    mvp_now = mvp[0].detach().cpu().numpy()
            #    mvp_original = original_mvp[0].detach().cpu().numpy()
            #    np.save('./tmp_save/mvp_now.npy', mvp_now)
            #    np.save('./tmp_save/mvp_original.npy', mvp_original)
            if xyz_before_sym is None:
                # just check the project depth of xyz
                coordinates = xyz
                cam_uv, cam_depth = self.get_uv_depth(coordinates, original_mvp)
                cam_uv = cam_uv.detach()
                cam_depth = cam_depth.detach()

                project_feature = F.grid_sample(local_feat_map, cam_uv.view(b, 1, k, 2), mode='bilinear').squeeze(dim=-2).permute(0, 2, 1)  # [b, k, c]
                project_depth = F.grid_sample(original_depth.permute(0, 3, 1, 2), cam_uv.view(b, 1, k, 2), mode='bilinear').squeeze(dim=-2).permute(0, 2, 1)  # [b, k, 1]

                use_mask = cam_depth <= project_depth + THRESHOLD
                feature = project_feature * use_mask.repeat(1, 1, project_feature.shape[-1])
            else:
                # need to double check, but now we are still use symmetry! Even if the two points are all visible in input view
                coords_inp = xyz
                x_check, y_check, z_check = xyz.unbind(-1)
                xyz_check = torch.stack([-1 * x_check, y_check, z_check], -1)
                coords_rev = xyz_check      # we directly use neg-x to get the points of another side

                uv_inp, dp_inp = self.get_uv_depth(coords_inp, original_mvp)
                uv_rev, dp_rev = self.get_uv_depth(coords_rev, original_mvp)
                uv_inp = uv_inp.detach()
                uv_rev = uv_rev.detach()
                dp_inp = dp_inp.detach()
                dp_rev = dp_rev.detach()

                proj_feat_inp = F.grid_sample(local_feat_map, uv_inp.view(b, 1, k, 2), mode='bilinear').squeeze(dim=-2).permute(0, 2, 1)  # [b, k, c]
                proj_feat_rev = F.grid_sample(local_feat_map, uv_rev.view(b, 1, k, 2), mode='bilinear').squeeze(dim=-2).permute(0, 2, 1)  # [b, k, c]
                
                proj_dp_inp = F.grid_sample(original_depth.permute(0, 3, 1, 2), uv_inp.view(b, 1, k, 2), mode='bilinear').squeeze(dim=-2).permute(0, 2, 1)  # [b, k, 1]
                proj_dp_rev = F.grid_sample(original_depth.permute(0, 3, 1, 2), uv_rev.view(b, 1, k, 2), mode='bilinear').squeeze(dim=-2).permute(0, 2, 1)  # [b, k, 1]

                use_mask_inp = dp_inp <= proj_dp_inp + THRESHOLD
                use_mask_rev = dp_rev <= proj_dp_rev + THRESHOLD

                # for those points we can see in two sides, we use average
                use_mask_inp = use_mask_inp.int()
                use_mask_rev = use_mask_rev.int()
                both_vis = (use_mask_inp == 1) & (use_mask_rev == 1)
                use_mask_inp[both_vis] = 0.5
                use_mask_rev[both_vis] = 0.5

                feature = proj_feat_inp * use_mask_inp.repeat(1, 1, proj_feat_inp.shape[-1]) + proj_feat_rev * use_mask_rev.repeat(1, 1, proj_feat_rev.shape[-1])
        else:
            raise NotImplementedError
        
        ret_feature = self.local_feature_block(feature.reshape(b*k, -1))  # the linear is without bias, so 0 value feature will still get 0 value
        return ret_feature

    def sample(self, x, feat=None, feat_map=None, mvp=None, w2c=None, deform_xyz=None):
        # assert (feat is None and self.extra_dim == 0) or (feat.shape[-1] <= self.extra_dim)
        b, h, w, c = x.shape

        xyz_before_sym = None
        if self.symmetrize:
            xyz_before_sym = x.reshape(b, -1, c)
            xs, ys, zs = x.unbind(-1)
            x = torch.stack([xs.abs(), ys, zs], -1)  # mirror -x to +x
        
        mvp = mvp.detach()  # [b, 4, 4]
        w2c = w2c.detach()  # [b, 4, 4]
        
        pts_xyz = x.reshape(b, -1, c)
        deform_xyz = deform_xyz.reshape(b, -1, c)

        if 'global' in self.texture_way and 'local' in self.texture_way:
            global_feat = feat[:,None,None].expand(b, h, w, -1).reshape(b*h*w, -1)
            # local_feat = self.proj_sample(pts_xyz, feat_map, mvp, w2c, h, w, xyz_before_sym=xyz_before_sym)
            local_feat = self.proj_sample_deform(deform_xyz, feat_map, mvp, w2c, h, w)
            # feature_rep = torch.concat([global_feat, local_feat], dim=-1)
            feature_rep = global_feat + local_feat
        elif 'global' not in self.texture_way and 'local' in self.texture_way:
            # local_feat = self.proj_sample(pts_xyz, feat_map, mvp, w2c, h, w, xyz_before_sym=xyz_before_sym)
            local_feat = self.proj_sample_deform(deform_xyz, feat_map, mvp, w2c, h, w)
            feature_rep = local_feat
        elif 'global' in self.texture_way and 'local' not in self.texture_way:
            global_feat = feat[:,None,None].expand(b, h, w, -1).reshape(b*h*w, -1)
            feature_rep = global_feat
        else:
            global_feat = feat[:,None,None].expand(b, h, w, -1).reshape(b*h*w, -1)
            feature_rep = global_feat

        x = x.view(-1, c)

        if self.embedder is not None:
            x_in = self.embedder(x)
            if self.embed_concat_pts:
                x_in = torch.cat([x, x_in], -1)
        else:
            x_in = x
        
        x_in = self.in_layer(x_in)

        # if feat is not None:
        #     feat = feat[:,None,None].expand(b, h, w, -1).reshape(b*h*w, -1)
        #     x_in = torch.concat([x_in, feat], dim=-1)

        x_in = torch.concat([x_in, feature_rep], dim=-1)

        out = self.mlp(self.relu(x_in))
        if self.min_max is not None:
            out = out * (self.min_max[1][None, :] - self.min_max[0][None, :]) + self.min_max[0][None, :]
        return out.view(b, h, w, -1)


class LiftTexture(nn.Module):
    def __init__(self, 
                 cin, 
                 cout, 
                 num_layers, 
                 nf=256, 
                 dropout=0, 
                 activation=None, 
                 min_max=None, 
                 n_harmonic_functions=10, 
                 omega0=1,
                 extra_dim=0,
                 embed_concat_pts=True, 
                 perturb_normal=False,
                 symmetrize=False,
                 texture_way=None,
                 cam_pos_z_offset=10.,
                 grid_scale=7.,
                 local_feat_dim=128,
                 grid_size=32,
                 optim_latent=False):
        super().__init__()
        self.extra_dim = extra_dim
        self.cam_pos_z_offset = cam_pos_z_offset
        self.grid_scale = grid_scale

        assert texture_way is not None
        self.extra_dim = local_feat_dim + extra_dim

        if n_harmonic_functions > 0:
            self.embedder = HarmonicEmbedding(n_harmonic_functions, omega0)
            dim_in = cin * 2 * n_harmonic_functions
            self.embed_concat_pts = embed_concat_pts
            if embed_concat_pts:
                dim_in += cin
        else:
            self.embedder = None
            dim_in = cin
        
        self.encoder = Lift_Encoder(
            cin=384,
            feat_dim=local_feat_dim,
            grid_scale=grid_scale / 2,  # the dmtet is initialized in (-0.5, 0.5)
            grid_size=grid_size,
            optim_latent=optim_latent,
            with_z_feature=True,
            cam_pos_z_offset=cam_pos_z_offset
        )


        self.in_layer = nn.Linear(dim_in, nf)
        self.relu = nn.ReLU(inplace=True)
        self.mlp = MLP(nf + self.extra_dim, cout, num_layers, nf, dropout, activation)
        self.perturb_normal = perturb_normal
        self.symmetrize = symmetrize
        if min_max is not None:
            self.register_buffer('min_max', min_max)
        else:
            self.min_max = None
        self.bsdf = None
    
    def get_uv_depth(self, xyz, mvp):
        # xyz: [b, k, 3]
        # mvp: [b, 4, 4]
        cam4 = torch.matmul(torch.nn.functional.pad(xyz, pad=(0,1), mode='constant', value=1.0), torch.transpose(mvp, 1, 2))
        cam3 = cam4[..., :3] / cam4[..., 3:4]
        cam_uv = cam3[..., :2]
        # cam_uv = cam_uv.detach()
        cam_depth = cam3 + torch.FloatTensor([0, 0, self.cam_pos_z_offset]).to(xyz.device).view(1, 1, 3)
        cam_depth = cam_depth / self.grid_scale * 2
        cam_depth = cam_depth[..., 2:3]
        # cam_depth = cam_depth.detach()
        return cam_uv, cam_depth
    
    def proj_sample_deform(self, xyz, feat_map, mvp, w2c, img_h, img_w):
        # here the xyz is deformed points
        # and we don't cast any symmtery here
        if isinstance(feat_map, torch.Tensor):
            feature = self.encoder(feat_map, mvp, xyz, inference="unproject")
        
        elif isinstance(feat_map, dict):
            feature = self.encoder(feat_map['im_features_map'], mvp, xyz, inference="sample")
        C = feature.shape[-1]
        feature = feature.reshape(-1, C)
        return feature

    def sample(self, x, feat=None, feat_map=None, mvp=None, w2c=None, deform_xyz=None):
        # assert (feat is None and self.extra_dim == 0) or (feat.shape[-1] <= self.extra_dim)
        b, h, w, c = x.shape

        xyz_before_sym = None
        if self.symmetrize:
            xyz_before_sym = x.reshape(b, -1, c)
            xs, ys, zs = x.unbind(-1)
            x = torch.stack([xs.abs(), ys, zs], -1)  # mirror -x to +x
        
        mvp = mvp.detach()  # [b, 4, 4]
        w2c = w2c.detach()  # [b, 4, 4]
        
        pts_xyz = x.reshape(b, -1, c)
        deform_xyz = deform_xyz.reshape(b, -1, c)

        global_feat = feat[:,None,None].expand(b, h, w, -1).reshape(b*h*w, -1)
        local_feat = self.proj_sample_deform(deform_xyz, feat_map, mvp, w2c, h, w)
        feature_rep = torch.concat([global_feat, local_feat], dim=-1)
        x = x.view(-1, c)

        if self.embedder is not None:
            x_in = self.embedder(x)
            if self.embed_concat_pts:
                x_in = torch.cat([x, x_in], -1)
        else:
            x_in = x
        
        x_in = self.in_layer(x_in)

        # if feat is not None:
        #     feat = feat[:,None,None].expand(b, h, w, -1).reshape(b*h*w, -1)
        #     x_in = torch.concat([x_in, feat], dim=-1)

        x_in = torch.concat([x_in, feature_rep], dim=-1)

        out = self.mlp(self.relu(x_in))
        if self.min_max is not None:
            out = out * (self.min_max[1][None, :] - self.min_max[0][None, :]) + self.min_max[0][None, :]
        return out.view(b, h, w, -1)


class HarmonicEmbedding(nn.Module):
    def __init__(self, n_harmonic_functions=10, omega0=1):
        """
        Positional Embedding implementation (adapted from Pytorch3D).
        Given an input tensor `x` of shape [minibatch, ... , dim],
        the harmonic embedding layer converts each feature
        in `x` into a series of harmonic features `embedding`
        as follows:
            embedding[..., i*dim:(i+1)*dim] = [
                sin(x[..., i]),
                sin(2*x[..., i]),
                sin(4*x[..., i]),
                ...
                sin(2**self.n_harmonic_functions * x[..., i]),
                cos(x[..., i]),
                cos(2*x[..., i]),
                cos(4*x[..., i]),
                ...
                cos(2**self.n_harmonic_functions * x[..., i])
            ]
        Note that `x` is also premultiplied by `omega0` before
        evaluting the harmonic functions.
        """
        super().__init__()
        self.frequencies = omega0 * (2.0 ** torch.arange(n_harmonic_functions))

    def forward(self, x):
        """
        Args:
            x: tensor of shape [..., dim]
        Returns:
            embedding: a harmonic embedding of `x`
                of shape [..., n_harmonic_functions * dim * 2]
        """
        embed = (x[..., None] * self.frequencies.to(x.device)).view(*x.shape[:-1], -1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)


class VGGEncoder(nn.Module):
    def __init__(self, cout, pretrained=False):
        super().__init__()
        if pretrained:
            raise NotImplementedError
        vgg = models.vgg16()
        self.vgg_encoder = nn.Sequential(vgg.features, vgg.avgpool)
        self.linear1 = nn.Linear(25088, 4096)
        self.linear2 = nn.Linear(4096, cout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        out = self.relu(self.linear1(self.vgg_encoder(x).view(batch_size, -1)))
        return self.linear2(out)


class ResnetEncoder(nn.Module):
    def __init__(self, cout, pretrained=False):
        super().__init__()
        self.resnet = nn.Sequential(list(models.resnet18(weights="DEFAULT" if pretrained else None).modules())[:-1])
        self.final_linear = nn.Linear(512, cout)

    def forward(self, x):
        return self.final_linear(self.resnet(x))


class Encoder(nn.Module):
    def __init__(self, cin, cout, in_size=128, zdim=None, nf=64, activation=None):
        super().__init__()
        network = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 128x128 -> 64x64
            nn.GroupNorm(16, nf),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            nn.GroupNorm(16*2, nf*2),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.GroupNorm(16*4, nf*4),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            # nn.GroupNorm(16*8, nf*8),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        add_downsample = int(np.log2(in_size//128))
        if add_downsample > 0:
            for _ in range(add_downsample):
                network += [
                    nn.Conv2d(nf*8, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
                    # nn.GroupNorm(16*8, nf*8),
                    # nn.ReLU(inplace=True),
                    nn.LeakyReLU(0.2, inplace=True),
                ]

        network += [
            nn.Conv2d(nf*8, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
            nn.LeakyReLU(0.2, inplace=True),
        ]

        if zdim is None:
            network += [
                nn.Conv2d(nf*8, cout, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
                ]
        else:
            network += [
                nn.Conv2d(nf*8, zdim, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
                # nn.ReLU(inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(zdim, cout, kernel_size=1, stride=1, padding=0, bias=False),
                ]

        if activation is not None:
            network += [get_activation(activation)]
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input).reshape(input.size(0), -1)


class EncoderWithDINO(nn.Module):
    def __init__(self, cin_rgb, cin_dino, cout, in_size=128, zdim=None, nf=64, activation=None):
        super().__init__()
        network_rgb_in = [
            nn.Conv2d(cin_rgb, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 128x128 -> 64x64
            nn.GroupNorm(16, nf),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            nn.GroupNorm(16*2, nf*2),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.GroupNorm(16*4, nf*4),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        self.network_rgb_in = nn.Sequential(*network_rgb_in)
        network_dino_in = [
            nn.Conv2d(cin_dino, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 128x128 -> 64x64
            nn.GroupNorm(16, nf),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            nn.GroupNorm(16*2, nf*2),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.GroupNorm(16*4, nf*4),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        self.network_dino_in = nn.Sequential(*network_dino_in)

        network_fusion = [
            nn.Conv2d(nf*4*2, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            # nn.GroupNorm(16*8, nf*8),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        add_downsample = int(np.log2(in_size//128))
        if add_downsample > 0:
            for _ in range(add_downsample):
                network_fusion += [
                    nn.Conv2d(nf*8, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
                    # nn.GroupNorm(16*8, nf*8),
                    # nn.ReLU(inplace=True),
                    nn.LeakyReLU(0.2, inplace=True),
                ]

        network_fusion += [
            nn.Conv2d(nf*8, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
            nn.LeakyReLU(0.2, inplace=True),
        ]

        if zdim is None:
            network_fusion += [
                nn.Conv2d(nf*8, cout, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
                ]
        else:
            network_fusion += [
                nn.Conv2d(nf*8, zdim, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
                # nn.ReLU(inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(zdim, cout, kernel_size=1, stride=1, padding=0, bias=False),
                ]

        if activation is not None:
            network_fusion += [get_activation(activation)]
        self.network_fusion = nn.Sequential(*network_fusion)

    def forward(self, rgb_image, dino_image):
        rgb_feat = self.network_rgb_in(rgb_image)
        dino_feat = self.network_dino_in(dino_image)
        out = self.network_fusion(torch.cat([rgb_feat, dino_feat], dim=1))
        return out.reshape(rgb_image.size(0), -1)


class Encoder32(nn.Module):
    def __init__(self, cin, cout, nf=256, activation=None):
        super().__init__()
        network = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.GroupNorm(nf//4, nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            nn.GroupNorm(nf//4, nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
            nn.GroupNorm(nf//4, nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, cout, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
        ]
        if activation is not None:
            network += [get_activation(activation)]
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input).reshape(input.size(0), -1)


class MLP(nn.Module):
    def __init__(self, cin, cout, num_layers, nf=256, dropout=0, activation=None, inner_act='relu', linear_bias=False):
        super().__init__()
        assert num_layers >= 1
        layer_act = get_activation(inner_act)
        if num_layers == 1:
            network = [nn.Linear(cin, cout, bias=linear_bias)]
        else:
            # network = [nn.Linear(cin, nf, bias=False)]
            # for _ in range(num_layers-2):
            #     network += [
            #         nn.ReLU(inplace=True),
            #         nn.Linear(nf, nf, bias=False)]
            #     if dropout:
            #         network += [nn.Dropout(dropout)]
            # network += [
            #     nn.ReLU(inplace=True),
            #     nn.Linear(nf, cout, bias=False)]
            network = [nn.Linear(cin, nf, bias=linear_bias)]
            for _ in range(num_layers-2):
                network += [
                    layer_act,
                    nn.Linear(nf, nf, bias=linear_bias)]
                if dropout:
                    network += [nn.Dropout(dropout)]
            network += [
                layer_act,
                nn.Linear(nf, cout, bias=linear_bias)]
        if activation is not None:
            network += [get_activation(activation)]
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input)


class Embedding(nn.Module):
    def __init__(self, cin, cout, zdim=128, nf=64, activation=None):
        super().__init__()
        network = [
            nn.Linear(cin, nf, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(nf, zdim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(zdim, cout, bias=False)]
        if activation is not None:
            network += [get_activation(activation)]
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input.reshape(input.size(0), -1)).reshape(input.size(0), -1)


class PerceptualLoss(nn.Module):
    def __init__(self, requires_grad=False):
        super(PerceptualLoss, self).__init__()
        mean_rgb = torch.FloatTensor([0.485, 0.456, 0.406])
        std_rgb = torch.FloatTensor([0.229, 0.224, 0.225])
        self.register_buffer('mean_rgb', mean_rgb)
        self.register_buffer('std_rgb', std_rgb)

        vgg_pretrained_features = torchvision.models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def normalize(self, x):
        out = x/2 + 0.5
        out = (out - self.mean_rgb.view(1,3,1,1)) / self.std_rgb.view(1,3,1,1)
        return out

    def __call__(self, im1, im2, mask=None, conf_sigma=None):
        im = torch.cat([im1,im2], 0)
        im = self.normalize(im)  # normalize input

        ## compute features
        feats = []
        f = self.slice1(im)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice2(f)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice3(f)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice4(f)
        feats += [torch.chunk(f, 2, dim=0)]

        losses = []
        for f1, f2 in feats[2:3]:  # use relu3_3 features only
            loss = (f1-f2)**2
            if conf_sigma is not None:
                loss = loss / (2*conf_sigma**2 +EPS) + (conf_sigma +EPS).log()
            if mask is not None:
                b, c, h, w = loss.shape
                _, _, hm, wm = mask.shape
                sh, sw = hm//h, wm//w
                mask0 = nn.functional.avg_pool2d(mask, kernel_size=(sh,sw), stride=(sh,sw)).expand_as(loss)
                loss = (loss * mask0).sum() / mask0.sum()
            else:
                loss = loss.mean()
            losses += [loss]
        return sum(losses)


## from: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)

        self.norm_layer = norm_layer
        if norm_layer is not None:
            self.bn1 = norm_layer(planes)
            self.bn2 = norm_layer(planes)

        if inplanes != planes:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                norm_layer(planes),
            )
        else:
            self.downsample = None
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.norm_layer is not None:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.norm_layer is not None:
            out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResEncoder(nn.Module):
    def __init__(self, cin, cout, in_size=128, zdim=None, nf=64, activation=None):
        super().__init__()
        network = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 128x128 -> 64x64
            # nn.GroupNorm(16, nf),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            # nn.GroupNorm(16*2, nf*2),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            BasicBlock(nf*2, nf*2, norm_layer=None),
            BasicBlock(nf*2, nf*2, norm_layer=None),
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            # nn.GroupNorm(16*4, nf*4),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            BasicBlock(nf*4, nf*4, norm_layer=None),
            BasicBlock(nf*4, nf*4, norm_layer=None),
            nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            BasicBlock(nf*8, nf*8, norm_layer=None),
            BasicBlock(nf*8, nf*8, norm_layer=None),
        ]

        add_downsample = int(np.log2(in_size//64))
        if add_downsample > 0:
            for _ in range(add_downsample):
                network += [
                    nn.Conv2d(nf*8, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
                    # nn.ReLU(inplace=True),
                    nn.LeakyReLU(0.2, inplace=True),
                    BasicBlock(nf*8, nf*8, norm_layer=None),
                    BasicBlock(nf*8, nf*8, norm_layer=None),
                ]

        if zdim is None:
            network += [
                nn.Conv2d(nf*8, cout, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
                ]
        else:
            network += [
                nn.Conv2d(nf*8, zdim, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
                # nn.ReLU(inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(zdim, cout, kernel_size=1, stride=1, padding=0, bias=False),
                ]

        if activation is not None:
            network += [get_activation(activation)]
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input).reshape(input.size(0), -1)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class ViTEncoder(nn.Module):
    def __init__(self, cout, which_vit='dino_vits8', pretrained=False, frozen=False, in_size=256, final_layer_type='none', root='/root'):
        super().__init__()
        if misc.is_main_process():
            force_reload = not os.path.exists(os.path.join(root, ".cache/torch/hub/checkpoints/"))
        else:
            force_reload = False
        if "dinov2" in which_vit:
            self.ViT = torch.hub.load('facebookresearch/dinov2:main', which_vit, pretrained=pretrained, force_reload=force_reload)
        else:
            self.ViT = torch.hub.load('facebookresearch/dino:main', which_vit, pretrained=pretrained, force_reload=force_reload)
        
        if frozen:
            for p in self.ViT.parameters():
                p.requires_grad = False
        if which_vit == 'dino_vits8':
            self.vit_feat_dim = 384
            self.patch_size = 8
        elif which_vit == 'dinov2_vits14':
            self.vit_feat_dim = 384
            self.patch_size = 14
        elif which_vit == 'dino_vitb8':
            self.vit_feat_dim = 768
            self.patch_size = 8
        
        self._feats = []
        self.hook_handlers = []

        if final_layer_type == 'none':
            pass
        elif final_layer_type == 'conv':
            self.final_layer_patch_out = Encoder32(self.vit_feat_dim, cout, nf=256, activation=None)
            self.final_layer_patch_key = Encoder32(self.vit_feat_dim, cout, nf=256, activation=None)
        elif final_layer_type == 'attention':
            raise NotImplementedError
            self.final_layer = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.fc = nn.Linear(self.vit_feat_dim, cout)
        else:
            raise NotImplementedError
        self.final_layer_type = final_layer_type
    
    def _get_hook(self, facet: str):
        """
        generate a hook method for a specific block and facet.
        """
        if facet in ['attn', 'token']:
            def _hook(model, input, output):
                self._feats.append(output)
            return _hook

        if facet == 'query':
            facet_idx = 0
        elif facet == 'key':
            facet_idx = 1
        elif facet == 'value':
            facet_idx = 2
        else:
            raise TypeError(f"{facet} is not a supported facet.")

        def _inner_hook(module, input, output):
            input = input[0]
            B, N, C = input.shape
            qkv = module.qkv(input).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            self._feats.append(qkv[facet_idx]) #Bxhxtxd
        return _inner_hook
    
    def _register_hooks(self, layers: List[int], facet: str) -> None:
        """
        register hook to extract features.
        :param layers: layers from which to extract features.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        """
        for block_idx, block in enumerate(self.ViT.blocks):
            if block_idx in layers:
                if facet == 'token':
                    self.hook_handlers.append(block.register_forward_hook(self._get_hook(facet)))
                elif facet == 'attn':
                    self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(self._get_hook(facet)))
                elif facet in ['key', 'query', 'value']:
                    self.hook_handlers.append(block.attn.register_forward_hook(self._get_hook(facet)))
                else:
                    raise TypeError(f"{facet} is not a supported facet.")
    
    def _unregister_hooks(self) -> None:
        """
        unregisters the hooks. should be called after feature extraction.
        """
        for handle in self.hook_handlers:
            handle.remove()
        self.hook_handlers = []
    
    def forward(self, x, return_patches=False):
        b, c, h, w = x.shape
        self._feats = []
        self._register_hooks([11], 'key')
        #self._register_hooks([11], 'token')
        x = self.ViT.prepare_tokens(x)
        #x = self.ViT.prepare_tokens_with_masks(x)
        
        for blk in self.ViT.blocks:
            x = blk(x)
        out = self.ViT.norm(x)
        self._unregister_hooks()

        ph, pw = h // self.patch_size, w // self.patch_size
        patch_out = out[:, 1:]  # first is class token
        patch_out = patch_out.reshape(b, ph, pw, self.vit_feat_dim).permute(0, 3, 1, 2)

        patch_key = self._feats[0][:,:,1:]  # B, num_heads, num_patches, dim
        patch_key = patch_key.permute(0, 1, 3, 2).reshape(b, self.vit_feat_dim, ph, pw)

        if self.final_layer_type == 'none':
            global_feat_out = out[:, 0].reshape(b, -1)  # first is class token
            global_feat_key = self._feats[0][:, :, 0].reshape(b, -1)  # first is class token
        elif self.final_layer_type == 'conv':
            global_feat_out = self.final_layer_patch_out(patch_out).view(b, -1)
            global_feat_key = self.final_layer_patch_key(patch_key).view(b, -1)
        elif self.final_layer_type == 'attention':
            raise NotImplementedError
        else:
            raise NotImplementedError
        if not return_patches:
            patch_out = patch_key = None
        return global_feat_out, global_feat_key, patch_out, patch_key


class ArticulationNetwork(nn.Module):
    def __init__(self, net_type, feat_dim, pos_dim, num_layers, nf, n_harmonic_functions=0, omega0=1, activation=None, enable_articulation_idadd=False):
        super().__init__()
        if n_harmonic_functions > 0:
            self.posenc = HarmonicEmbedding(n_harmonic_functions=n_harmonic_functions, omega0=omega0)
            pos_dim = pos_dim * (n_harmonic_functions * 2 + 1)
        else:
            self.posenc = None
            pos_dim = 4
        cout = 3
        
        if net_type == 'mlp':
            self.network = MLP(
                feat_dim + pos_dim,  # + bone xyz pos and index
                cout,  # We represent the rotation of each bone by its Euler angles ψ, θ, and φ
                num_layers,
                nf=nf,
                dropout=0,
                activation=activation
            )
        elif net_type == 'attention':
            self.in_layer = nn.Sequential(
                nn.Linear(feat_dim + pos_dim, nf),
                nn.GELU(),
                nn.LayerNorm(nf),
            )
            self.blocks = nn.ModuleList([
            Block(
                dim=nf, num_heads=8, mlp_ratio=2., qkv_bias=False, qk_scale=None,
                drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm)
            for i in range(num_layers)])
            out_layer = [nn.Linear(nf, cout)]
            if activation:
                out_layer += [get_activation(activation)]
            self.out_layer = nn.Sequential(*out_layer)
        else:
            raise NotImplementedError
        self.net_type = net_type
        self.enable_articulation_idadd = enable_articulation_idadd
    
    def forward(self, x, pos):
        pos_inp = pos
        if self.posenc is not None:
            pos = torch.cat([pos, self.posenc(pos)], dim=-1)
        x = torch.cat([x, pos], dim=-1)
        if self.enable_articulation_idadd:
            articulation_id = pos_inp[..., -1:]
            x = x + articulation_id
        if self.net_type == 'mlp':
            out = self.network(x)
        elif self.net_type == 'attention':
            x = self.in_layer(x)
            for blk in self.blocks:
                x = blk(x)
            out = self.out_layer(x)
        else:
            raise NotImplementedError
        return out


## Attention block from ViT (https://github.com/facebookresearch/dino/blob/main/vision_transformer.py)
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class FeatureAttention(nn.Module):
    def __init__(self, vit_type, pos_dim, embedder_freq=0, zdim=128, img_size=256, activation=None):
        super().__init__()
        self.zdim = zdim
        if embedder_freq > 0:
            self.posenc = HarmonicEmbedding(n_harmonic_functions=embedder_freq, omega0=1)
            pos_dim = pos_dim * (embedder_freq * 2 + 1)
        else:
            self.posenc = None
        self.pos_dim = pos_dim

        if vit_type == 'dino_vits8':
            self.vit_feat_dim = 384
            patch_size = 8
        elif which_vit == 'dinov2_vits14':
            self.vit_feat_dim = 384
            self.patch_size = 14
        elif vit_type == 'dino_vitb8':
            self.vit_feat_dim = 768
            patch_size = 8
        else:
            raise NotImplementedError
        self.num_patches_per_dim = img_size // patch_size

        self.kv = nn.Sequential(
            nn.Linear(self.vit_feat_dim, zdim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(zdim),
            nn.Linear(zdim, zdim*2),
        )
        
        self.q = nn.Sequential(
            nn.Linear(pos_dim, zdim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(zdim),
            nn.Linear(zdim, zdim),
        )
        
        final_mlp = [
            nn.Linear(zdim, zdim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(zdim),
            nn.Linear(zdim, self.vit_feat_dim)
        ]
        if activation is not None:
            final_mlp += [get_activation(activation)]
        self.final_ln = nn.Sequential(*final_mlp)

    def forward(self, x, feat):
        _, vit_feat_dim, ph, pw = feat.shape
        assert ph == pw and ph == self.num_patches_per_dim and vit_feat_dim == self.vit_feat_dim
        
        if self.posenc is not None:
            x = torch.cat([x, self.posenc(x)], dim=-1)
        bxf, k, c = x.shape
        assert c == self.pos_dim
        
        query = self.q(x)
        feat_in = feat.view(bxf, vit_feat_dim, ph*pw).permute(0, 2, 1)  # N, K, C
        k, v = self.kv(feat_in).chunk(2, dim=-1)
        attn = torch.einsum('bnd,bpd->bnp', query, k).softmax(dim=-1)
        out = torch.einsum('bnp,bpd->bnd', attn, v)
        out = self.final_ln(out)
        return out
