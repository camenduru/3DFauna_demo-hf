import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.init as init

import torchvision.models as models
import nvdiffrast.torch as dr
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
import pickle

from video3d.render.regularizer import get_edge_length, normal_consistency, laplace_regularizer_const
from . import networks
from .renderer import *
from .utils import misc, meters, flow_viz, arap, custom_loss
from .dataloaders import get_sequence_loader, get_image_loader
from .dataloaders_ddp import get_sequence_loader_ddp, get_image_loader_ddp
from .cub_dataloaders import get_cub_loader
from .cub_dataloaders_ddp import get_cub_loader_ddp
from .utils.skinning_v4 import estimate_bones, skinning
import lpips
from einops import rearrange, repeat

# import clip
import torchvision.transforms.functional as tvf
from . import discriminator_architecture

from .geometry.dmtet import DMTetGeometry
from .geometry.dlmesh import DLMesh

from .triplane_texture.triplane_predictor import TriPlaneTex

from .render import renderutils as ru
from .render import material
from .render import mlptexture
from .render import util
from .render import mesh
from .render import light
from .render import render

EPS = 1e-7


def get_optimizer(model, lr=0.0001, betas=(0.9, 0.999), weight_decay=0):
    return torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, betas=betas, weight_decay=weight_decay)


def set_requires_grad(model, requires_grad):
    if model is not None:
        for param in model.parameters():
            param.requires_grad = requires_grad


def forward_to_matrix(vec_forward, up=[0,1,0]):
    up = torch.FloatTensor(up).to(vec_forward.device)
    # vec_forward = nn.functional.normalize(vec_forward, p=2, dim=-1)  # x right, y up, z forward
    vec_right = up.expand_as(vec_forward).cross(vec_forward, dim=-1)
    vec_right = nn.functional.normalize(vec_right, p=2, dim=-1)
    vec_up = vec_forward.cross(vec_right, dim=-1)
    vec_up = nn.functional.normalize(vec_up, p=2, dim=-1)
    rot_mat = torch.stack([vec_right, vec_up, vec_forward], -2)
    return rot_mat


def sample_pose_hypothesis_from_quad_prediction(poses_raw, total_iter, batch_size, num_frames, pose_xflip_recon=False, input_image_xflip_flag=None, rot_temp_scalar=1., num_hypos=4, naive_probs_iter=2000, best_pose_start_iter=6000, random_sample=True, temp_clip_low = 1., temp_clip_high=100.):
    rots_pred = poses_raw[..., :num_hypos*4].view(-1, num_hypos, 4)
    rots_logits = rots_pred[..., 0]  # Nx4
    # temp = 1 / np.clip(total_iter / 1000 / rot_temp_scalar, 1., 100.)
    temp = 1 / np.clip(total_iter / 1000 / rot_temp_scalar, temp_clip_low, temp_clip_high)

    rots_probs = torch.nn.functional.softmax(-rots_logits / temp, dim=1)  # N x K
    # naive_probs = torch.FloatTensor([10] + [1] * (num_hypos - 1)).to(rots_logits.device)
    naive_probs = torch.ones(num_hypos).to(rots_logits.device)
    naive_probs = naive_probs / naive_probs.sum()
    naive_probs_weight = np.clip(1 - (total_iter - naive_probs_iter) / 2000, 0, 1)
    rots_probs = naive_probs.view(1, num_hypos) * naive_probs_weight + rots_probs * (1 - naive_probs_weight)

    rots_pred = rots_pred[..., 1:4]
    trans_pred = poses_raw[..., -3:]
    best_rot_idx = torch.argmax(rots_probs, dim=1)  # N
    #print("best_rot_idx", best_rot_idx)
    #print("best_of_best", torch.argmax(rots_probs))
    #print("similar 7", torch.zeros_like(best_rot_idx) + 7)
    #print("similar 2", torch.zeros_like(best_rot_idx) + torch.argmax(rots_probs))
    
    if random_sample:
        # rand_rot_idx = torch.randint(0, 4, (batch_size * num_frames,), device=poses_raw.device)  # N
        rand_rot_idx = torch.randperm(batch_size * num_frames, device=poses_raw.device) % num_hypos  # N
        # rand_rot_idx = torch.randperm(batch_size, device=poses_raw.device)[:,None].repeat(1, num_frames).view(-1) % 4  # N
        best_flag = (torch.randperm(batch_size * num_frames, device=poses_raw.device) / (batch_size * num_frames) < np.clip((total_iter - best_pose_start_iter)/2000, 0, 0.8)).long()
        rand_flag = 1 - best_flag
        # best_flag = torch.zeros_like(best_rot_idx)
        rot_idx = best_rot_idx * best_flag + rand_rot_idx * (1 - best_flag)
    else:
        rand_flag = torch.zeros_like(best_rot_idx)
        #rot_idx = torch.full_like(torch.argmax(rots_probs, dim=1), torch.argmax(rots_probs), device=poses_raw.device)
        rot_idx = best_rot_idx


    
    rot_pred = torch.gather(rots_pred, 1, rot_idx[:, None, None].expand(-1, 1, 3))[:, 0]  # Nx3
    pose_raw = torch.cat([rot_pred, trans_pred], -1)
    rot_prob = torch.gather(rots_probs, 1, rot_idx[:, None].expand(-1, 1))[:, 0]  # N
    rot_logit = torch.gather(rots_logits, 1, rot_idx[:, None].expand(-1, 1))[:, 0]  # N

    if pose_xflip_recon:
        raise NotImplementedError

    #up = torch.FloatTensor([0, 1, 0]).to(pose_raw.device)
    rot_mat = forward_to_matrix(pose_raw[:, :3], up=[0, 1, 0])
    pose = torch.cat([rot_mat.view(batch_size * num_frames, -1), pose_raw[:, 3:]], -1)
    return pose_raw, pose, rot_idx, rot_prob, rot_logit, rots_probs, rand_flag


def get_joints_20_bones(bones, aux):
    # the bones shape is [1, 1, 20, 2, 3]
    body_bones_to_joints = aux['bones_to_joints']
    body_bones = bones[:, :, :len(body_bones_to_joints), :, :]
    body_joints = torch.empty(bones.shape[0], bones.shape[1], len(body_bones_to_joints) + 1, 3)

    for i, (a, b) in enumerate(body_bones_to_joints):
        body_joints[:, :, a, :] = body_bones[:, :, i, 0, :]
        body_joints[:, :, b, :] = body_bones[:, :, i, 1, :]
    
    leg_aux = aux['legs']
    all_leg_joints = []
    for i in range(len(leg_aux)):
        leg_bones = bones[:, :, 8+i*3:11+i*3, :, :]
        leg_joints = torch.empty(bones.shape[0], bones.shape[1], len(leg_aux[i]['leg_bones_to_joints']), 3)

        for j in range(len(leg_aux[i]['leg_bones_to_joints'])-1):
            leg_joint_idx_a = leg_aux[i]['leg_bones_to_joints'][j][0]
            leg_joint_idx_b = leg_aux[i]['leg_bones_to_joints'][j][1]

            leg_joints[:, :,  leg_joint_idx_a, :] = leg_bones[:, :, j, 0, :]
            leg_joints[:, :,  leg_joint_idx_b, :] = leg_bones[:, :, j, 1, :]
        
        all_leg_joints.append(leg_joints)
    
    all_joints = [body_joints] + all_leg_joints
    all_joints = torch.cat(all_joints, dim=2)
    return all_joints


def get_20_bones_joints(joints, aux):
    # the joints shape is [1, 1, 21, 3]
    body_bones_to_joints = aux['bones_to_joints']
    body_bones = []
    for a,b in body_bones_to_joints:
        body_bones += [torch.stack([joints[:, :, a, :], joints[:, :, b, :]], dim=2)]
    body_bones = torch.stack(body_bones, dim=2)  # [1, 1, 8, 2, 3]

    legs_bones = []
    legs_aux = aux['legs']
    for i in range(len(legs_aux)):
        leg_aux = legs_aux[i]
        leg_bones = []

        leg_bones_to_joints = leg_aux['leg_bones_to_joints']
        for j in range(len(leg_bones_to_joints)-1):
            leg_bones += [torch.stack([joints[:, :, 9+i*3+leg_bones_to_joints[j][0], :], joints[:, :, 9+i*3+leg_bones_to_joints[j][1], :]], dim=2)]
        # the last bone is attached to the body
        leg_bones += [torch.stack([
            body_bones[:, :, leg_aux['body_bone_idx'], 1, :], joints[:, :, 9+i*3+leg_bones_to_joints[-1][1], :]
        ], dim=2)]

        leg_bones = torch.stack(leg_bones, dim=2)
        legs_bones.append(leg_bones)
    
    bones = torch.cat([body_bones] + legs_bones, dim=2)
    return bones


class FixedDirectionLight(torch.nn.Module):
    def __init__(self, direction, amb, diff):
        super(FixedDirectionLight, self).__init__()
        self.light_dir = direction
        self.amb = amb
        self.diff = diff
        self.is_hacking = not (isinstance(self.amb, float)
                               or isinstance(self.amb, int))

    def forward(self, feat):
        batch_size = feat.shape[0]
        if self.is_hacking:
            return torch.concat([self.light_dir, self.amb, self.diff], -1)
        else:
            return torch.concat([self.light_dir, torch.FloatTensor([self.amb, self.diff]).to(self.light_dir.device)], -1).expand(batch_size, -1)

    def shade(self, feat, kd, normal):
        light_params = self.forward(feat)
        light_dir = light_params[..., :3][:, None, None, :]
        int_amb = light_params[..., 3:4][:, None, None, :]
        int_diff = light_params[..., 4:5][:, None, None, :]
        shading = (int_amb + int_diff *
                   torch.clamp(util.dot(light_dir, normal), min=0.0))
        shaded = shading * kd
        return shaded, shading


class SmoothLoss(nn.Module):
    def __init__(self, dim=0, smooth_type=None, loss_type="l2"):
        super(SmoothLoss, self).__init__()
        self.dim = dim
        
        supported_smooth_types = ['mid_frame', 'dislocation', 'avg']
        assert smooth_type in supported_smooth_types, f"supported smooth type: {supported_smooth_types}"
        self.smooth_type = smooth_type

        supported_loss_types = ['l2', 'mse', 'l1']
        assert loss_type in supported_loss_types, f"supported loss type: {supported_loss_types}"
        self.loss_type = loss_type

        if self.loss_type in ['l2', 'mse']:
            self.loss_fn = torch.nn.MSELoss(reduction='mean')
        elif self.loss_type in ['l1']:
            self.loss_fn = torch.nn.L1Loss()
        else:
            raise NotImplementedError

    def mid_frame_smooth(self, inputs):
        nframe = inputs.shape[self.dim]
        mid_num = (nframe-1) // 2
        # from IPython import embed; embed();
        mid_frame = torch.index_select(inputs, self.dim, torch.tensor([mid_num], device=inputs.device))
        repeat_num = self.get_repeat_num(inputs)
        smooth = mid_frame.repeat(repeat_num)
        loss = self.loss_fn(inputs, smooth)
        # print(loss)
        return loss

    def dislocation_smooth(self, inputs):
        # from IPython import embed; embed()
        nframe = inputs.shape[self.dim]
        t = torch.index_select(inputs, self.dim, torch.arange(0, nframe-1).to(inputs.device))
        t_1 = torch.index_select(inputs, self.dim, torch.arange(1, nframe).to(inputs.device))
        loss = self.loss_fn(t, t_1)
        return loss

    def avg_smooth(self, inputs):
        # nframe = inputs.shape[self.dim]
        # from IPython import embed; embed()
        avg = inputs.mean(dim=self.dim, keepdim=True)
        repeat_num = self.get_repeat_num(inputs)
        smooth = avg.repeat(repeat_num)
        loss = self.loss_fn(inputs, smooth)
        return loss

    def get_repeat_num(self, inputs):
        repeat_num = [1] * inputs.dim()
        repeat_num[self.dim] = inputs.shape[self.dim]
        return repeat_num

    def forward(self, inputs):
        print(f"smooth_type: {self.smooth_type}")
        if self.smooth_type is None:
            return 0.
        elif self.smooth_type == 'mid_frame':
            return self.mid_frame_smooth(inputs)
        elif self.smooth_type == 'dislocation':
            return self.dislocation_smooth(inputs)
        elif self.smooth_type == 'avg':
            return self.avg_smooth(inputs)
        else:
            raise NotImplementedError()


class PriorPredictor(nn.Module):
    def __init__(self, cfgs):
        super().__init__()

        #add nnParameters 
        dmtet_grid = cfgs.get('dmtet_grid', 64)
        grid_scale = cfgs.get('grid_scale', 5)
        prior_sdf_mode = cfgs.get('prior_sdf_mode', 'mlp')
        num_layers_shape = cfgs.get('num_layers_shape', 5)
        hidden_size = cfgs.get('hidden_size', 64)
        embedder_freq_shape = cfgs.get('embedder_freq_shape', 8)
        embed_concat_pts = cfgs.get('embed_concat_pts', True)
        init_sdf = cfgs.get('init_sdf', None)
        jitter_grid = cfgs.get('jitter_grid', 0.)
        perturb_sdf_iter = cfgs.get('perturb_sdf_iter', 10000)
        sym_prior_shape = cfgs.get('sym_prior_shape', False)
        train_data_dir = cfgs.get("train_data_dir", None)
        if isinstance(train_data_dir, str):
            num_of_classes = 1
        elif isinstance(train_data_dir, dict):
            self.category_id_map = {}
            num_of_classes = len(train_data_dir)
            for i, (k, _) in enumerate(train_data_dir.items()):
                self.category_id_map[k] = i
        dim_of_classes = cfgs.get('dim_of_classes', 256) if num_of_classes > 1 else 0
        condition_choice = cfgs.get('prior_condition_choice', 'concat')
        self.netShape = DMTetGeometry(dmtet_grid, grid_scale, prior_sdf_mode, num_layers=num_layers_shape, hidden_size=hidden_size, embedder_freq=embedder_freq_shape, embed_concat_pts=embed_concat_pts, init_sdf=init_sdf, jitter_grid=jitter_grid, perturb_sdf_iter=perturb_sdf_iter, sym_prior_shape=sym_prior_shape, 
                                      dim_of_classes=dim_of_classes, condition_choice=condition_choice)

        mlp_hidden_size = cfgs.get('hidden_size', 64)
        tet_bbox = self.netShape.getAABB()
        self.render_dino_mode = cfgs.get('render_dino_mode', None)
        num_layers_dino = cfgs.get("num_layers_dino", 5)
        dino_feature_recon_dim = cfgs.get('dino_feature_recon_dim', 64)

        sym_dino = cfgs.get("sym_dino", False)
        dino_min = torch.zeros(dino_feature_recon_dim) + cfgs.get('dino_min', 0.)
        dino_max = torch.zeros(dino_feature_recon_dim) + cfgs.get('dino_max', 1.)
        min_max = torch.stack((dino_min, dino_max), dim=0)
        if self.render_dino_mode is None:
            pass
        elif self.render_dino_mode == 'feature_mlpnv':
            #MLPTexture3D predict the dino for each single point. 
            self.netDINO = mlptexture.MLPTexture3D(tet_bbox, channels=dino_feature_recon_dim, internal_dims=mlp_hidden_size, hidden=num_layers_dino-1, feat_dim=0, min_max=min_max, bsdf=None, perturb_normal=False, symmetrize=sym_dino)
        elif self.render_dino_mode == 'feature_mlp':
            embedder_scaler = 2 * np.pi / grid_scale * 0.9  # originally (-0.5*s, 0.5*s) rescale to (-pi, pi) * 0.9
            embed_concat_pts = cfgs.get('embed_concat_pts', True)
            self.netDINO = networks.MLPTextureSimple(
                3,  # x, y, z coordinates
                dino_feature_recon_dim,
                num_layers_dino,
                nf=mlp_hidden_size,
                dropout=0,
                activation="sigmoid",
                min_max=min_max,
                n_harmonic_functions=cfgs.get('embedder_freq_dino', 8),
                omega0=embedder_scaler,
                extra_dim=dim_of_classes,
                embed_concat_pts=embed_concat_pts,
                perturb_normal=False,
                symmetrize=sym_dino
            )
        elif self.render_dino_mode == 'cluster':
            num_layers_dino = cfgs.get("num_layers_dino", 5)
            dino_cluster_dim = cfgs.get('dino_cluster_dim', 64)
            self.netDINO = mlptexture.MLPTexture3D(tet_bbox, channels=dino_cluster_dim, internal_dims=mlp_hidden_size, hidden=num_layers_dino-1, feat_dim=0, min_max=None, bsdf=None, perturb_normal=False, symmetrize=sym_dino)
        else:
            raise NotImplementedError

        self.classes_vectors = None
        if num_of_classes > 1:
            self.classes_vectors = torch.nn.Parameter(torch.nn.init.uniform_(torch.empty(num_of_classes, dim_of_classes), a=-0.05, b=0.05))

    def forward(self, category_name=None, perturb_sdf=False, total_iter=None, is_training=True, class_embedding=None):
        class_vector = None
        if category_name is not None:
            # print(category_name)
            if class_embedding is not None:
                class_vector = class_embedding[0]  # [128]
                return_classes_vectors = class_vector
            else:
                class_vector = self.classes_vectors[self.category_id_map[category_name]]
                return_classes_vectors = self.classes_vectors
        prior_shape = self.netShape.getMesh(perturb_sdf=perturb_sdf, total_iter=total_iter, jitter_grid=is_training, class_vector=class_vector)
        # print(prior_shape.v_pos.shape)
        # return prior_shape, self.netDINO, self.classes_vectors
        return prior_shape, self.netDINO, return_classes_vectors


class InstancePredictor(nn.Module):
    def __init__(self, cfgs, tet_bbox=None):
        super().__init__()
        self.cfgs = cfgs
        self.grid_scale = cfgs.get('grid_scale', 5)

        self.enable_encoder = cfgs.get('enable_encoder', False)
        if self.enable_encoder:
            encoder_latent_dim = cfgs.get('latent_dim', 256)
            encoder_pretrained = cfgs.get('encoder_pretrained', False)
            encoder_frozen = cfgs.get('encoder_frozen', False)
            encoder_arch = cfgs.get('encoder_arch', 'simple')
            in_image_size = cfgs.get('in_image_size', 256)
            self.dino_feature_input = cfgs.get('dino_feature_input', False)
            dino_feature_dim = cfgs.get('dino_feature_dim', 64)
            if encoder_arch == 'simple':
                if self.dino_feature_input:
                    self.netEncoder = networks.EncoderWithDINO(cin_rgb=3, cin_dino=dino_feature_dim, cout=encoder_latent_dim, in_size=in_image_size, zdim=None, nf=64, activation=None)
                else:
                    self.netEncoder = networks.Encoder(cin=3, cout=encoder_latent_dim, in_size=in_image_size, zdim=None, nf=64, activation=None)
            elif encoder_arch == 'vgg':
                self.netEncoder = networks.VGGEncoder(cout=encoder_latent_dim, pretrained=encoder_pretrained)
            elif encoder_arch == 'resnet':
                self.netEncoder = networks.ResnetEncoder(cout=encoder_latent_dim, pretrained=encoder_pretrained)
            elif encoder_arch == 'vit':
                which_vit = cfgs.get('which_vit', 'dino_vits8')
                vit_final_layer_type = cfgs.get('vit_final_layer_type', 'conv')
                root_dir = cfgs.get('root_dir', '/root')
                self.netEncoder = networks.ViTEncoder(cout=encoder_latent_dim, which_vit=which_vit, pretrained=encoder_pretrained, frozen=encoder_frozen, in_size=in_image_size, final_layer_type=vit_final_layer_type, root=root_dir)
            else:
                raise NotImplementedError
        else:
            encoder_latent_dim = 0
        
        mlp_hidden_size = cfgs.get('hidden_size', 64)
        
        bsdf = cfgs.get("bsdf", 'diffuse')
        num_layers_tex = cfgs.get("num_layers_tex", 5)
        feat_dim = cfgs.get("latent_dim", 64) if self.enable_encoder else 0
        perturb_normal = cfgs.get("perturb_normal", False)
        sym_texture = cfgs.get("sym_texture", False)
        kd_min = torch.FloatTensor(cfgs.get('kd_min', [0., 0., 0., 0.]))
        kd_max = torch.FloatTensor(cfgs.get('kd_max', [1., 1., 1., 1.]))
        ks_min = torch.FloatTensor(cfgs.get('ks_min', [0., 0., 0.]))
        ks_max = torch.FloatTensor(cfgs.get('ks_max', [0., 0., 0.]))
        nrm_min = torch.FloatTensor(cfgs.get('nrm_min', [-1., -1., 0.]))
        nrm_max = torch.FloatTensor(cfgs.get('nrm_max', [1., 1., 1.]))
        mlp_min = torch.cat((kd_min[0:3], ks_min, nrm_min), dim=0)
        mlp_max = torch.cat((kd_max[0:3], ks_max, nrm_max), dim=0)
        min_max = torch.stack((mlp_min, mlp_max), dim=0)
        out_chn = 9
        # TODO: if the tet verts are deforming, we need to recompute tet_bbox
        texture_mode = cfgs.get("texture_mode", 'mlp')
        if texture_mode == 'mlpnv':
            self.netTexture = mlptexture.MLPTexture3D(tet_bbox, channels=out_chn, internal_dims=mlp_hidden_size, hidden=num_layers_tex-1, feat_dim=feat_dim, min_max=min_max, bsdf=bsdf, perturb_normal=perturb_normal, symmetrize=sym_texture)
        elif texture_mode == 'mlp':
            embedder_scaler = 2 * np.pi / self.grid_scale * 0.9  # originally (-0.5*s, 0.5*s) rescale to (-pi, pi) * 0.9
            embed_concat_pts = cfgs.get('embed_concat_pts', True)

            self.texture_way = cfgs.get('texture_way', None)

            if self.texture_way is None:
                texture_act = cfgs.get('texture_act', 'relu')
                texture_bias = cfgs.get('texture_bias', False)
                self.netTexture = networks.MLPTextureSimple(
                    3,  # x, y, z coordinates
                    out_chn,
                    num_layers_tex,
                    nf=mlp_hidden_size,
                    dropout=0,
                    activation="sigmoid",
                    min_max=min_max,
                    n_harmonic_functions=cfgs.get('embedder_freq_tex', 10),
                    omega0=embedder_scaler,
                    extra_dim=feat_dim,
                    embed_concat_pts=embed_concat_pts,
                    perturb_normal=perturb_normal,
                    symmetrize=sym_texture,
                    texture_act=texture_act,
                    linear_bias=texture_bias
                )
            else:
                self.netTexture = networks.MLPTextureTriplane(
                        3,  # x, y, z coordinates
                        out_chn,
                        num_layers_tex,
                        nf=mlp_hidden_size,
                        dropout=0,
                        activation="sigmoid",
                        min_max=min_max,
                        n_harmonic_functions=cfgs.get('embedder_freq_tex', 10),
                        omega0=embedder_scaler,
                        extra_dim=feat_dim,
                        embed_concat_pts=embed_concat_pts,
                        perturb_normal=perturb_normal,
                        symmetrize=sym_texture,
                        texture_act='relu',
                        linear_bias=False,
                        cam_pos_z_offset=cfgs.get('cam_pos_z_offset', 10.),
                        grid_scale=self.grid_scale
                    )
                # if 'lift' in self.texture_way:
                #     # GET3D use global feature to get a tri-plane
                #     self.netTexture = TriPlaneTex(
                #         w_dim=512,
                #         img_channels=out_chn,
                #         tri_plane_resolution=256,
                #         device=cfgs.get('device', 'cpu'),
                #         mlp_latent_channel=32,
                #         n_implicit_layer=1,
                #         feat_dim=256,
                #         n_mapping_layer=8,
                #         sym_texture=sym_texture,
                #         grid_scale=self.grid_scale,
                #         min_max=min_max,
                #         perturb_normal=perturb_normal
                #     )

                #     # # project the local feature map into a grid
                #     # self.netTexture = networks.LiftTexture(
                #     #     3,  # x, y, z coordinates
                #     #     out_chn,
                #     #     num_layers_tex,
                #     #     nf=mlp_hidden_size,
                #     #     dropout=0,
                #     #     activation="sigmoid",
                #     #     min_max=min_max,
                #     #     n_harmonic_functions=cfgs.get('embedder_freq_tex', 10),
                #     #     omega0=embedder_scaler,
                #     #     extra_dim=feat_dim,
                #     #     embed_concat_pts=embed_concat_pts,
                #     #     perturb_normal=perturb_normal,
                #     #     symmetrize=sym_texture,
                #     #     texture_way=self.texture_way,
                #     #     cam_pos_z_offset=cfgs.get('cam_pos_z_offset', 10.),
                #     #     grid_scale=self.grid_scale,
                #     #     local_feat_dim=cfgs.get("lift_local_feat_dim", 128),
                #     #     grid_size=cfgs.get("lift_grid_size", 32),
                #     #     optim_latent=cfgs.get("lift_optim_latent", False)
                #     # )
                # else:
                #     # a texture mlp with local feature map from patch_out
                #     self.netTexture = networks.MLPTextureLocal(
                #         3,  # x, y, z coordinates
                #         out_chn,
                #         num_layers_tex,
                #         nf=mlp_hidden_size,
                #         dropout=0,
                #         activation="sigmoid",
                #         min_max=min_max,
                #         n_harmonic_functions=cfgs.get('embedder_freq_tex', 10),
                #         omega0=embedder_scaler,
                #         extra_dim=feat_dim,
                #         embed_concat_pts=embed_concat_pts,
                #         perturb_normal=perturb_normal,
                #         symmetrize=sym_texture,
                #         texture_way=self.texture_way,
                #         larger_tex_dim=cfgs.get('larger_tex_dim', False),
                #         cam_pos_z_offset=cfgs.get('cam_pos_z_offset', 10.),
                #         grid_scale=self.grid_scale
                #     )

        self.rot_rep = cfgs.get('rot_rep', 'euler_angle')
        self.enable_pose = cfgs.get('enable_pose', False)
        if self.enable_pose:
            cam_pos_z_offset = cfgs.get('cam_pos_z_offset', 10.)
            fov = cfgs.get('crop_fov_approx', 25)
            half_range = np.tan(fov /2 /180 * np.pi) * cam_pos_z_offset  # 2.22
            self.max_trans_xy_range = half_range * cfgs.get('max_trans_xy_range_ratio', 1.)
            self.max_trans_z_range = half_range * cfgs.get('max_trans_z_range_ratio', 1.)
            self.lookat_init = cfgs.get('lookat_init', None)
            self.lookat_zeroy = cfgs.get('lookat_zeroy', False)
            self.rot_temp_scalar = cfgs.get('rot_temp_scalar', 1.)
            self.naive_probs_iter = cfgs.get('naive_probs_iter', 2000)
            self.best_pose_start_iter = cfgs.get('best_pose_start_iter', 6000)

            if self.rot_rep == 'euler_angle':
                pose_cout = 6
            elif self.rot_rep == 'quaternion':
                pose_cout = 7
            elif self.rot_rep == 'lookat':
                pose_cout = 6
            elif self.rot_rep == 'quadlookat':
                self.num_pose_hypos = 4
                pose_cout = (3 + 1) * self.num_pose_hypos + 3  # 4 forward vectors for 4 quadrants, 4 quadrant classification logits, 3 for translation
                self.orthant_signs = torch.FloatTensor([[1,1,1], [-1,1,1], [-1,1,-1], [1,1,-1]])
            elif self.rot_rep == 'octlookat':
                self.num_pose_hypos = 8
                pose_cout = (3 + 1) * self.num_pose_hypos + 3  # 4 forward vectors for 8 octants, 8 octant classification logits, 3 for translation
                self.orthant_signs = torch.stack(torch.meshgrid([torch.arange(1, -2, -2)] *3), -1).view(-1, 3)  # 8x3
            else:
                raise NotImplementedError
            
            self.pose_arch = cfgs.get('pose_arch', 'mlp')
            if self.pose_arch == 'mlp':
                num_layers_pose = cfgs.get('num_layers_pose', 5)
                self.netPose = networks.MLP(
                    encoder_latent_dim,
                    pose_cout,
                    num_layers_pose,
                    nf=mlp_hidden_size,
                    dropout=0,
                    activation=None
                )
            elif self.pose_arch == 'encoder':
                if self.dino_feature_input:
                    dino_feature_dim = cfgs.get('dino_feature_dim', 64)
                    self.netPose = networks.EncoderWithDINO(cin_rgb=3, cin_dino=dino_feature_dim, cout=pose_cout, in_size=in_image_size, zdim=None, nf=64, activation=None)
                else:
                    self.netPose = networks.Encoder(cin=3, cout=pose_cout, in_size=in_image_size, zdim=None, nf=64, activation=None)
            elif self.pose_arch in ['encoder_dino_patch_out', 'encoder_dino_patch_key']:
                if which_vit == 'dino_vits8':
                    dino_feat_dim = 384
                elif which_vit == 'dinov2_vits14':
                    dino_feat_dim = 384
                elif which_vit == 'dino_vitb8':
                    dino_feat_dim = 768
                self.netPose = networks.Encoder32(cin=dino_feat_dim, cout=pose_cout, nf=256, activation=None)
            elif self.pose_arch == 'vit':
                encoder_pretrained = cfgs.get('encoder_pretrained', False)
                encoder_frozen = cfgs.get('encoder_frozen', False)
                which_vit = cfgs.get('which_vit', 'dino_vits8')
                vit_final_layer_type = cfgs.get('vit_final_layer_type', 'conv')
                root_dir = cfgs.get('root_dir', '/root')
                self.netPose = networks.ViTEncoder(cout=encoder_latent_dim, which_vit=which_vit, pretrained=encoder_pretrained, frozen=encoder_frozen, in_size=in_image_size, final_layer_type=vit_final_layer_type, root=root_dir)
            else:
                raise NotImplementedError
        
        self.enable_deform = cfgs.get('enable_deform', False)
        if self.enable_deform:
            embedder_scaler = 2 * np.pi / self.grid_scale * 0.9  # originally (-0.5*s, 0.5*s) rescale to (-pi, pi) * 0.9
            embed_concat_pts = cfgs.get('embed_concat_pts', True)
            num_layers_deform = cfgs.get('num_layers_deform', 5)
            self.deform_epochs = np.arange(*cfgs.get('deform_epochs', [0, 0]))
            sym_deform = cfgs.get("sym_deform", False)
            self.netDeform = networks.MLPWithPositionalEncoding(
                3,  # x, y, z coordinates
                3,  # dx, dy, dz deformation
                num_layers_deform,
                nf=mlp_hidden_size,
                dropout=0,
                activation=None,
                n_harmonic_functions=cfgs.get('embedder_freq_deform', 10),
                omega0=embedder_scaler,
                extra_dim=encoder_latent_dim,
                embed_concat_pts=embed_concat_pts,
                symmetrize=sym_deform
            )
        # self.avg_deform = cfgs.get('avg_deform', False)
        # print(f'********avg_deform: {self.avg_deform}********')

        self.enable_articulation = cfgs.get('enable_articulation', False)
        if self.enable_articulation:
            self.num_body_bones = cfgs.get('num_body_bones', 4)
            self.articulation_multiplier = cfgs.get('articulation_multiplier', 1)
            self.static_root_bones = cfgs.get('static_root_bones', False)
            self.skinning_temperature = cfgs.get('skinning_temperature', 1)
            self.articulation_epochs = np.arange(*cfgs.get('articulation_epochs', [0, 0]))
            self.num_legs = cfgs.get('num_legs', 0)
            self.num_leg_bones = cfgs.get('num_leg_bones', 0)
            self.body_bones_type = cfgs.get('body_bones_type', 'z_minmax')
            self.perturb_articulation_epochs = np.arange(*cfgs.get('perturb_articulation_epochs', [0, 0]))
            self.num_bones = self.num_body_bones + self.num_legs * self.num_leg_bones
            self.constrain_legs = cfgs.get('constrain_legs', False)
            self.attach_legs_to_body_epochs = np.arange(*cfgs.get('attach_legs_to_body_epochs', [0, 0]))
            self.max_arti_angle = cfgs.get('max_arti_angle', 60)

            num_layers_arti = cfgs.get('num_layers_arti', 5)
            which_vit = cfgs.get('which_vit', 'dino_vits8')
            if which_vit == 'dino_vits8':
                dino_feat_dim = 384
            elif which_vit == 'dino_vitb8':
                dino_feat_dim = 768
            self.articulation_arch = cfgs.get('articulation_arch', 'mlp')
            self.articulation_feature_mode = cfgs.get('articulation_feature_mode', 'sample')
            embedder_freq_arti = cfgs.get('embedder_freq_arti', 8)
            if self.articulation_feature_mode == 'global':
                feat_dim = encoder_latent_dim
            elif self.articulation_feature_mode == 'sample':
                feat_dim = dino_feat_dim
            elif self.articulation_feature_mode == 'sample+global':
                feat_dim = encoder_latent_dim + dino_feat_dim
            if self.articulation_feature_mode == 'attention':
                arti_feat_attn_zdim = cfgs.get('arti_feat_attn_zdim', 128)
                pos_dim = 1 + 2 + 3*2
                self.netFeatureAttn = networks.FeatureAttention(which_vit, pos_dim, embedder_freq_arti, arti_feat_attn_zdim, img_size=in_image_size)
            embedder_scaler = np.pi * 0.9  # originally (-1, 1) rescale to (-pi, pi) * 0.9
            enable_articulation_idadd = cfgs.get('enable_articulation_idadd', False)
            self.netArticulation = networks.ArticulationNetwork(self.articulation_arch, feat_dim, 1+2+3*2, num_layers_arti, mlp_hidden_size, n_harmonic_functions=embedder_freq_arti, omega0=embedder_scaler,
                                                                enable_articulation_idadd=enable_articulation_idadd)
            self.kinematic_tree_epoch = -1
        
        self.enable_lighting = cfgs.get('enable_lighting', False)
        if self.enable_lighting:
            num_layers_light = cfgs.get('num_layers_light', 5)
            amb_diff_min = torch.FloatTensor(cfgs.get('amb_diff_min', [0., 0.]))
            amb_diff_max = torch.FloatTensor(cfgs.get('amb_diff_max', [1., 1.]))
            intensity_min_max = torch.stack((amb_diff_min, amb_diff_max), dim=0)
            self.netLight = light.DirectionalLight(encoder_latent_dim, num_layers_light, mlp_hidden_size, intensity_min_max=intensity_min_max)

        self.cam_pos_z_offset = cfgs.get('cam_pos_z_offset', 10.)
        self.crop_fov_approx = cfgs.get("crop_fov_approx", 25)

        self.temp_clip_low = cfgs.get('temp_clip_low', 1.)
        self.temp_clip_high = cfgs.get('temp_clip_high', 100.)
        
        # if the articulation and deformation is set as iterations, then use iteration to decide, not epoch
        self.iter_articulation_start = cfgs.get('iter_articulation_start', None)
        self.iter_deformation_start = cfgs.get('iter_deformation_start', None)

        self.iter_nozeroy_start = cfgs.get('iter_nozeroy_start', None)
        self.iter_attach_leg_to_body_start = cfgs.get('iter_attach_leg_to_body_start', None)
    
    def forward_encoder(self, images, dino_features=None):
        images_in = images.view(-1, *images.shape[2:]) * 2 - 1  # rescale to (-1, 1)
        patch_out = patch_key = None
        if self.dino_feature_input and self.cfgs.get('encoder_arch', 'simple') != 'vit':
            dino_features_in = dino_features.view(-1, *dino_features.shape[2:]) * 2 - 1  # rescale to (-1, 1)
            feat_out = self.netEncoder(images_in, dino_features_in)  # Shape: (B, latent_dim)
        elif self.cfgs.get('encoder_arch', 'simple') == 'vit':
            feat_out, feat_key, patch_out, patch_key = self.netEncoder(images_in, return_patches=True)
        else:
            feat_out = self.netEncoder(images_in)  # Shape: (B, latent_dim)
        return feat_out, feat_key, patch_out, patch_key


    def forward_pose(self, images, feat, patch_out, patch_key, dino_features):
        if self.pose_arch == 'mlp':
            pose = self.netPose(feat)
        elif self.pose_arch == 'encoder':
            images_in = images.view(-1, *images.shape[2:]) * 2 - 1  # rescale to (-1, 1)
            if self.dino_feature_input:
                dino_features_in = dino_features.view(-1, *dino_features.shape[2:]) * 2 - 1  # rescale to (-1, 1)
                pose = self.netPose(images_in, dino_features_in)  # Shape: (B, latent_dim)
            else:
                pose = self.netPose(images_in)  # Shape: (B, latent_dim)
        elif self.pose_arch == 'vit':
            images_in = images.view(-1, *images.shape[2:]) * 2 - 1  # rescale to (-1, 1)
            pose = self.netPose(images_in)
        elif self.pose_arch == 'encoder_dino_patch_out':
            pose = self.netPose(patch_out)  # Shape: (B, latent_dim)
        elif self.pose_arch == 'encoder_dino_patch_key':
            pose = self.netPose(patch_key)  # Shape: (B, latent_dim)
        else:
            raise NotImplementedError
        trans_pred = pose[...,-3:].tanh() * torch.FloatTensor([self.max_trans_xy_range, self.max_trans_xy_range, self.max_trans_z_range]).to(pose.device)
        if self.rot_rep == 'euler_angle':
            multiplier = 1.
            if self.gradually_expand_yaw:
                # multiplier += (min(iteration, 20000) // 500) * 0.25
                multiplier *= 1.2 ** (min(iteration, 20000) // 500)  # 1.125^40 = 111.200
            rot_pred = torch.cat([pose[...,:1], pose[...,1:2]*multiplier, pose[...,2:3]], -1).tanh()
            rot_pred = rot_pred * torch.FloatTensor([self.max_rot_x_range, self.max_rot_y_range, self.max_rot_z_range]).to(pose.device) /180 * np.pi

        elif self.rot_rep == 'quaternion':
            quat_init = torch.FloatTensor([0.01,0,0,0]).to(pose.device)
            rot_pred = pose[...,:4] + quat_init
            rot_pred = nn.functional.normalize(rot_pred, p=2, dim=-1)
            # rot_pred = torch.cat([rot_pred[...,:1].abs(), rot_pred[...,1:]], -1)  # make real part non-negative
            rot_pred = rot_pred * rot_pred[...,:1].sign()  # make real part non-negative

        elif self.rot_rep == 'lookat':
            vec_forward_raw = pose[...,:3]
            if self.lookat_init is not None:
                vec_forward_raw = vec_forward_raw + torch.FloatTensor(self.lookat_init).to(pose.device)
            if self.lookat_zeroy:
                vec_forward_raw = vec_forward_raw * torch.FloatTensor([1,0,1]).to(pose.device)
            vec_forward_raw = nn.functional.normalize(vec_forward_raw, p=2, dim=-1)  # x right, y up, z forward
            rot_pred = vec_forward_raw

        elif self.rot_rep in ['quadlookat', 'octlookat']:
            rots_pred = pose[..., :self.num_pose_hypos*4].view(-1, self.num_pose_hypos, 4)  # (B, T, K, 4)
            rots_logits = rots_pred[..., :1]
            vec_forward_raw = rots_pred[..., 1:4]
            xs, ys, zs = vec_forward_raw.unbind(-1)
            margin = 0.
            xs = nn.functional.softplus(xs, beta=np.log(2)/(0.5+margin)) - margin  # initialize to 0.5
            if self.rot_rep == 'octlookat':
                ys = nn.functional.softplus(ys, beta=np.log(2)/(0.5+margin)) - margin  # initialize to 0.5
            if self.lookat_zeroy:
                ys = ys * 0
            zs = nn.functional.softplus(zs, beta=2*np.log(2))  # initialize to 0.5
            vec_forward_raw = torch.stack([xs, ys, zs], -1)
            vec_forward_raw = vec_forward_raw * self.orthant_signs.to(pose.device)
            vec_forward_raw = nn.functional.normalize(vec_forward_raw, p=2, dim=-1)  # x right, y up, z forward
            rot_pred = torch.cat([rots_logits, vec_forward_raw], -1).view(-1, self.num_pose_hypos*4)

        else:
            raise NotImplementedError
        
        pose = torch.cat([rot_pred, trans_pred], -1)
        return pose
    
    def forward_deformation(self, shape, feat=None, batch_size=None, num_frames=None):
        original_verts = shape.v_pos
        num_verts = original_verts.shape[1]
        if feat is not None:
            deform_feat = feat[:, None, :].repeat(1, num_verts, 1)  # Shape: (B, num_verts, latent_dim)
            original_verts = original_verts.repeat(len(feat),1,1)
        deformation = self.netDeform(original_verts, deform_feat) * 0.1  # Shape: (B, num_verts, 3)
        # if self.avg_deform:
        #     assert batch_size is not None and num_frames is not None
        #     assert deformation.shape[0] == batch_size * num_frames
        #     deformation = deformation.view(batch_size, num_frames, *deformation.shape[1:])
        #     deformation = deformation.mean(dim=1, keepdim=True)
        #     deformation = deformation.repeat(1,num_frames,*[1]*(deformation.dim()-2))
        #     deformation = deformation.view(batch_size*num_frames, *deformation.shape[2:])
        shape = shape.deform(deformation)
        return shape, deformation
    
    def forward_articulation(self, shape, feat, patch_feat, mvp, w2c, batch_size, num_frames, epoch, category, total_iter=None):
        """
        Forward propagation of articulation. For each bone, the network takes: 1) the 3D location of the bone; 2) the feature of the patch which
        the bone is projected to; and 3) an encoding of the bone's index to predict the bone's rotation (represented by an Euler angle).
        
        Args:
            shape: a Mesh object, whose v_pos has batch size BxF or 1.
            feat: the feature of the patches. Shape: (BxF, feat_dim, num_patches_per_axis, num_patches_per_axis)
            mvp: the model-view-projection matrix. Shape: (BxF, 4, 4)
        
        Returns:
            shape: a Mesh object, whose v_pos has batch size BxF (collapsed).
            articulation_angles: the predicted bone rotations. Shape: (B, F, num_bones, 3)
            aux: a dictionary containing auxiliary information.
        """
        verts = shape.v_pos
        if len(verts) == 1:
            verts = verts[None]
        else:
            verts = verts.view(batch_size, num_frames, *verts.shape[1:])
        
        if self.kinematic_tree_epoch != epoch:
        # if (epoch == self.articulation_epochs[0]) and (self.kinematic_tree_epoch != epoch):
        # if (epoch in [self.articulation_epochs[0], self.articulation_epochs[0]+2, self.articulation_epochs[0]+4]) and (self.kinematic_tree_epoch != epoch):
            if total_iter is not None and self.iter_attach_leg_to_body_start is not None:
                attach_legs_to_body = total_iter > self.iter_attach_leg_to_body_start
            else:
                attach_legs_to_body = epoch in self.attach_legs_to_body_epochs
            
            # bone_y_thresh = None if category is None or not category == "giraffe" else 0.1
            bone_y_thresh = self.cfgs.get('bone_y_thresh', None)

            # trivial set here
            body_bone_idx_preset_cfg = self.cfgs.get('body_bone_idx_preset', [0, 0, 0, 0])
            if isinstance(body_bone_idx_preset_cfg, list):
                body_bone_idx_preset = body_bone_idx_preset_cfg
            elif isinstance(body_bone_idx_preset_cfg, dict):
                iter_point = list(body_bone_idx_preset_cfg.keys())[1]
                if total_iter <= iter_point:
                    body_bone_idx_preset = body_bone_idx_preset_cfg[0]  # the first is start from 0 iter
                else:
                    body_bone_idx_preset = body_bone_idx_preset_cfg[iter_point]
            else:
                raise NotImplementedError

            bones, self.kinematic_tree, self.bone_aux = estimate_bones(verts.detach(), self.num_body_bones, n_legs=self.num_legs, n_leg_bones=self.num_leg_bones, body_bones_type=self.body_bones_type, compute_kinematic_chain=True, attach_legs_to_body=attach_legs_to_body, bone_y_threshold=bone_y_thresh, body_bone_idx_preset=body_bone_idx_preset)
            # self.kinematic_tree_epoch = epoch
        else:
            bones = estimate_bones(verts.detach(), self.num_body_bones, n_legs=self.num_legs, n_leg_bones=self.num_leg_bones, body_bones_type=self.body_bones_type, compute_kinematic_chain=False, aux=self.bone_aux)

        bones_pos = bones  # Shape: (B, F, K, 2, 3)
        if batch_size > bones_pos.shape[0] or num_frames > bones_pos.shape[1]:
            assert bones_pos.shape[0] == 1 and bones_pos.shape[1] == 1, "If there is a mismatch, then there must be only one canonical mesh."
            bones_pos = bones_pos.repeat(batch_size, num_frames, 1, 1, 1)
        num_bones = bones_pos.shape[2]
        bones_pos = bones_pos.view(batch_size*num_frames, num_bones, 2, 3)  # NxKx2x3
        bones_mid_pos = bones_pos.mean(2)  # NxKx3
        bones_idx = torch.arange(num_bones).to(bones_pos.device)

        bones_mid_pos_world4 = torch.cat([bones_mid_pos, torch.ones_like(bones_mid_pos[..., :1])], -1)  # NxKx4
        bones_mid_pos_clip4 = bones_mid_pos_world4 @ mvp.transpose(-1, -2)
        bones_mid_pos_uv = bones_mid_pos_clip4[..., :2] / bones_mid_pos_clip4[..., 3:4]
        bones_mid_pos_uv = bones_mid_pos_uv.detach()

        bones_pos_world4 = torch.cat([bones_pos, torch.ones_like(bones_pos[..., :1])], -1)  # NxKx2x4
        bones_pos_cam4 = bones_pos_world4 @ w2c[:,None].transpose(-1, -2)
        bones_pos_cam3 = bones_pos_cam4[..., :3] / bones_pos_cam4[..., 3:4]
        bones_pos_cam3 = bones_pos_cam3 + torch.FloatTensor([0, 0, self.cam_pos_z_offset]).to(bones_pos_cam3.device).view(1, 1, 1, 3)
        bones_pos_in = bones_pos_cam3.view(batch_size*num_frames, num_bones, 2*3) / self.grid_scale * 2  # (-1, 1), NxKx(2*3)
        
        bones_idx_in = ((bones_idx[None, :, None] + 0.5) / num_bones * 2 - 1).repeat(batch_size * num_frames, 1, 1)  # (-1, 1)
        bones_pos_in = torch.cat([bones_mid_pos_uv, bones_pos_in, bones_idx_in], -1).detach()

        if self.articulation_feature_mode == 'global':
            bones_patch_features = feat[:, None].repeat(1, num_bones, 1)  # (BxF, K, feat_dim)
        elif self.articulation_feature_mode == 'sample':
            bones_patch_features = F.grid_sample(patch_feat, bones_mid_pos_uv.view(batch_size * num_frames, 1, -1, 2), mode='bilinear').squeeze(dim=-2).permute(0, 2, 1)  # (BxF, K, feat_dim)
        elif self.articulation_feature_mode == 'sample+global':
            bones_patch_features = F.grid_sample(patch_feat, bones_mid_pos_uv.view(batch_size * num_frames, 1, -1, 2), mode='bilinear').squeeze(dim=-2).permute(0, 2, 1)  # (BxF, K, feat_dim)
            bones_patch_features = torch.cat([feat[:, None].repeat(1, num_bones, 1), bones_patch_features], -1)
        elif self.articulation_feature_mode == 'attention':
            bones_patch_features = self.netFeatureAttn(bones_pos_in, patch_feat)
        else:
            raise NotImplementedError

        articulation_angles = self.netArticulation(bones_patch_features, bones_pos_in).view(batch_size, num_frames, num_bones, 3) * self.articulation_multiplier

        if self.static_root_bones:
            root_bones = [self.num_body_bones // 2 - 1, self.num_body_bones - 1]
            tmp_mask = torch.ones_like(articulation_angles)
            tmp_mask[:, :, root_bones] = 0
            articulation_angles = articulation_angles * tmp_mask
        
        articulation_angles = articulation_angles.tanh()

        if self.cfgs.get('iter_leg_rotation_start', -1) > 0:
            if total_iter <= self.cfgs.get('iter_leg_rotation_start', -1):
                self.constrain_legs = True
            else:
                self.constrain_legs = False

        if self.constrain_legs:
            leg_bones_posx = [self.num_body_bones + i for i in range(self.num_leg_bones * self.num_legs // 2)]
            leg_bones_negx = [self.num_body_bones + self.num_leg_bones * self.num_legs // 2 + i for i in range(self.num_leg_bones * self.num_legs // 2)]

            tmp_mask = torch.zeros_like(articulation_angles)
            tmp_mask[:, :, leg_bones_posx + leg_bones_negx, 2] = 1
            articulation_angles = tmp_mask * (articulation_angles * 0.3) + (1 - tmp_mask) * articulation_angles  # no twist

            tmp_mask = torch.zeros_like(articulation_angles)
            tmp_mask[:, :, leg_bones_posx + leg_bones_negx, 1] = 1
            articulation_angles = tmp_mask * (articulation_angles * 0.3) + (1 - tmp_mask) * articulation_angles  # (-0.4, 0.4),  limit side bending

        # new regularizations, for bottom 2 bones of each leg, they can only rotate around x-axis, 
        # and for the toppest bone of legs, restrict its angles in a smaller range
        if (self.cfgs.get('iter_leg_rotation_start', -1) > 0) and (total_iter > self.cfgs.get('iter_leg_rotation_start', -1)):
            if self.cfgs.get('forbid_leg_rotate', False):
                if self.cfgs.get('small_leg_angle', False):
                    # regularize the rotation angle of first leg bones
                    leg_bones_top = [8, 11, 14, 17]
                    # leg_bones_top = [10, 13, 16, 19]
                    tmp_mask = torch.zeros_like(articulation_angles)
                    tmp_mask[:, :, leg_bones_top, 1] = 1
                    tmp_mask[:, :, leg_bones_top, 2] = 1
                    articulation_angles = tmp_mask * (articulation_angles * 0.05) + (1 - tmp_mask) * articulation_angles

                leg_bones_bottom = [9, 10, 12, 13, 15, 16, 18, 19]
                # leg_bones_bottom = [8, 9, 11, 12, 14, 15, 17, 18]
                tmp_mask = torch.ones_like(articulation_angles)
                tmp_mask[:, :, leg_bones_bottom, 1] = 0
                tmp_mask[:, :, leg_bones_bottom, 2] = 0
                # tmp_mask[:, :, leg_bones_bottom, 0] = 0.3
                articulation_angles = tmp_mask * articulation_angles

        if epoch in self.perturb_articulation_epochs:
            articulation_angles = articulation_angles + torch.randn_like(articulation_angles) * 0.1
        articulation_angles = articulation_angles * self.max_arti_angle / 180 * np.pi

        # check if regularize the leg-connecting body bones z-rotation first
        # then check if regularize all the body bones z-rotation
        # regularize z-rotation using 0.1 in pi-space
        body_rotate_mult = self.cfgs.get('reg_body_rotate_mult', 0.1)
        body_rotate_mult = body_rotate_mult * 180 * 1.0 / (self.max_arti_angle * np.pi)     # the max angle = mult*original_max_angle
        body_rotate_reg_mode = self.cfgs.get('body_rotate_reg_mode', 'nothing')
        if body_rotate_reg_mode == 'leg-connect':
            body_bones_mask = [2, 3, 4, 5]
            tmp_body_mask = torch.zeros_like(articulation_angles)
            tmp_body_mask[:, :, body_bones_mask, 2] = 1
            articulation_angles = tmp_body_mask * (articulation_angles * body_rotate_mult) + (1 - tmp_body_mask) * articulation_angles
            
        elif body_rotate_reg_mode == 'all-bones':
            body_bones_mask = [0, 1, 2, 3, 4, 5, 6, 7]
            tmp_body_mask = torch.zeros_like(articulation_angles)
            tmp_body_mask[:, :, body_bones_mask, 2] = 1
            articulation_angles = tmp_body_mask * (articulation_angles * body_rotate_mult) + (1 - tmp_body_mask) * articulation_angles
        
        elif body_rotate_reg_mode == 'nothing':
            articulation_angles = articulation_angles * 1.
        
        else:
            raise NotImplementedError

        verts_articulated, aux = skinning(verts, bones, self.kinematic_tree, articulation_angles, 
                                          output_posed_bones=True, temperature=self.skinning_temperature)
        verts_articulated = verts_articulated.view(batch_size*num_frames, *verts_articulated.shape[2:])
        v_tex = shape.v_tex
        if len(v_tex) != len(verts_articulated):
            v_tex = v_tex.repeat(len(verts_articulated), 1, 1)
        shape = mesh.make_mesh(
            verts_articulated,
            shape.t_pos_idx,
            v_tex,
            shape.t_tex_idx,
            shape.material)
        return shape, articulation_angles, aux
    
    def get_camera_extrinsics_from_pose(self, pose, znear=0.1, zfar=1000., crop_fov_approx=None, offset_extra=None):
        if crop_fov_approx is None:
            crop_fov_approx = self.crop_fov_approx
        N = len(pose)
        if offset_extra is not None:
            cam_pos_offset = torch.FloatTensor([0, 0, -self.cam_pos_z_offset - offset_extra]).to(pose.device)
        else:
            cam_pos_offset = torch.FloatTensor([0, 0, -self.cam_pos_z_offset]).to(pose.device)
        pose_R = pose[:, :9].view(N, 3, 3).transpose(2, 1)
        pose_T = pose[:, -3:] + cam_pos_offset[None, None, :]
        pose_T = pose_T.view(N, 3, 1)
        pose_RT = torch.cat([pose_R, pose_T], axis=2)  # Nx3x4
        w2c = torch.cat([pose_RT, torch.FloatTensor([0, 0, 0, 1]).repeat(N, 1, 1).to(pose.device)], axis=1)  # Nx4x4
        # We assume the images are perfect square.
        if isinstance(crop_fov_approx, float) or isinstance(crop_fov_approx, int):
            proj = util.perspective(crop_fov_approx / 180 * np.pi, 1, znear, zfar)[None].to(pose.device)
        elif isinstance(crop_fov_approx, torch.Tensor):
            proj = util.batched_perspective(crop_fov_approx / 180 * np.pi, 1, znear, zfar).to(pose.device)
        else:
            raise ValueError('crop_fov_approx must be float or torch.Tensor')
        mvp = torch.matmul(proj, w2c)
        campos = -torch.matmul(pose_R.transpose(2, 1), pose_T).view(N, 3)
        return mvp, w2c, campos

    def forward(self, category=None, images=None, prior_shape=None, epoch=None, dino_features=None, dino_clusters=None, total_iter=None, is_training=True):
        batch_size, num_frames = images.shape[:2]
        if self.enable_encoder:
            feat_out, feat_key, patch_out, patch_key = self.forward_encoder(images, dino_features)
        else:
            feat_out = feat_key = patch_out = patch_key = None
        shape = prior_shape
        texture = self.netTexture

        multi_hypothesis_aux = {}
        if self.iter_nozeroy_start is not None and total_iter >= self.iter_nozeroy_start:
            self.lookat_zeroy = False

        if self.enable_pose:
            poses_raw = self.forward_pose(images, feat_out, patch_out, patch_key, dino_features)
            pose_raw, pose, rot_idx, rot_prob, rot_logit, rots_probs, rand_pose_flag = sample_pose_hypothesis_from_quad_prediction(poses_raw, total_iter, batch_size, num_frames, rot_temp_scalar=self.rot_temp_scalar, num_hypos=self.num_pose_hypos, naive_probs_iter=self.naive_probs_iter, best_pose_start_iter=self.best_pose_start_iter, random_sample=is_training, temp_clip_low=self.temp_clip_low, temp_clip_high=self.temp_clip_high)
            multi_hypothesis_aux['rot_idx'] = rot_idx
            multi_hypothesis_aux['rot_prob'] = rot_prob
            multi_hypothesis_aux['rot_logit'] = rot_logit
            multi_hypothesis_aux['rots_probs'] = rots_probs
            multi_hypothesis_aux['rand_pose_flag'] = rand_pose_flag
        else:
            raise NotImplementedError
        mvp, w2c, campos = self.get_camera_extrinsics_from_pose(pose)

        deformation = None
        if self.iter_deformation_start is not None:
            if self.enable_deform and total_iter >= self.iter_deformation_start:
                shape, deformation = self.forward_deformation(shape, feat_key, batch_size, num_frames)
        else:
            if self.enable_deform and epoch in self.deform_epochs:
                shape, deformation = self.forward_deformation(shape, feat_key, batch_size, num_frames)
        
        arti_params, articulation_aux = None, {}
        if self.iter_articulation_start is not None:
            if self.enable_articulation and total_iter >= self.iter_articulation_start:
                shape, arti_params, articulation_aux = self.forward_articulation(shape, feat_key, patch_key, mvp, w2c, batch_size, num_frames, epoch, category, total_iter=total_iter)
        else:
            if self.enable_articulation and epoch in self.articulation_epochs:
                shape, arti_params, articulation_aux = self.forward_articulation(shape, feat_key, patch_key, mvp, w2c, batch_size, num_frames, epoch, category, total_iter=None)
        
        if self.enable_lighting:
            light = self.netLight
        else:
            light = None

        aux = articulation_aux
        aux.update(multi_hypothesis_aux)

        # if using texture_way to control a local texture, output patch_out
        if self.texture_way is None:
            return shape, pose_raw, pose, mvp, w2c, campos, texture, feat_out, patch_key, deformation, arti_params, light, aux
        else:
            return shape, pose_raw, pose, mvp, w2c, campos, texture, feat_out, patch_key, deformation, arti_params, light, aux, patch_out

class Unsup3DDDP:
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.device = cfgs.get('device', 'cpu')
        self.in_image_size = cfgs.get('in_image_size', 128)
        self.out_image_size = cfgs.get('out_image_size', 128)

        self.num_epochs = cfgs.get('num_epochs', 10)
        self.lr = cfgs.get('lr', 1e-4)
        self.use_scheduler = cfgs.get('use_scheduler', False)
        if self.use_scheduler:
            scheduler_milestone = cfgs.get('scheduler_milestone', [1,2,3,4,5])
            scheduler_gamma = cfgs.get('scheduler_gamma', 0.5)
            self.make_scheduler = lambda optim: torch.optim.lr_scheduler.MultiStepLR(optim, milestones=scheduler_milestone, gamma=scheduler_gamma)
        
        self.cam_pos_z_offset = cfgs.get('cam_pos_z_offset', 10.)
        self.full_size_h = cfgs.get('full_size_h', 1080)
        self.full_size_w = cfgs.get('full_size_w', 1920)
        # self.fov_w = cfgs.get('fov_w', 60)
        # self.fov_h = np.arctan(np.tan(self.fov_w /2 /180*np.pi) / self.full_size_w * self.full_size_h) *2 /np.pi*180  # 36
        self.crop_fov_approx = cfgs.get("crop_fov_approx", 25)
        self.mesh_regularization_mode = cfgs.get('mesh_regularization_mode', 'seq')

        self.enable_prior = cfgs.get('enable_prior', False)
        if self.enable_prior:
            self.netPrior = PriorPredictor(self.cfgs) #DOR - add label
            self.prior_lr = cfgs.get('prior_lr', self.lr)
            self.prior_weight_decay = cfgs.get('prior_weight_decay', 0.)
            self.prior_only_epochs = cfgs.get('prior_only_epochs', 0)
        self.netInstance = InstancePredictor(self.cfgs, tet_bbox=self.netPrior.netShape.getAABB())
        self.perturb_sdf = cfgs.get('perturb_sdf', False)
        self.blur_mask = cfgs.get('blur_mask', False)
        self.blur_mask_iter = cfgs.get('blur_mask_iter', 1)

        self.seqshape_epochs = np.arange(*cfgs.get('seqshape_epochs', [0, self.num_epochs]))
        self.avg_texture_epochs = np.arange(*cfgs.get('avg_texture_epochs', [0, 0]))
        self.swap_texture_epochs = np.arange(*cfgs.get('swap_texture_epochs', [0, 0]))
        self.swap_priorshape_epochs = np.arange(*cfgs.get('swap_priorshape_epochs', [0, 0]))
        self.avg_seqshape_epochs = np.arange(*cfgs.get('avg_seqshape_epochs', [0, 0]))
        self.swap_seqshape_epochs = np.arange(*cfgs.get('swap_seqshape_epochs', [0, 0]))
        self.pose_epochs = np.arange(*cfgs.get('pose_epochs', [0, 0]))
        self.pose_iters = cfgs.get('pose_iters', 0)
        self.deform_type = cfgs.get('deform_type', None)
        self.mesh_reg_decay_epoch = cfgs.get('mesh_reg_decay_epoch', 0)
        self.sdf_reg_decay_start_iter = cfgs.get('sdf_reg_decay_start_iter', 0)
        self.mesh_reg_decay_rate = cfgs.get('mesh_reg_decay_rate', 1)
        self.texture_epochs = np.arange(*cfgs.get('texture_epochs', [0, self.num_epochs]))
        self.zflip_epochs = np.arange(*cfgs.get('zflip_epochs', [0, self.num_epochs]))
        self.lookat_zflip_loss_epochs = np.arange(*cfgs.get('lookat_zflip_loss_epochs', [0, self.num_epochs]))
        self.lookat_zflip_no_other_losses = cfgs.get('lookat_zflip_no_other_losses', False)
        self.flow_loss_epochs = np.arange(*cfgs.get('flow_loss_epochs', [0, self.num_epochs]))
        self.sdf_inflate_reg_loss_epochs = np.arange(*cfgs.get('sdf_inflate_reg_loss_epochs', [0, self.num_epochs]))
        self.arti_reg_loss_epochs = np.arange(*cfgs.get('arti_reg_loss_epochs', [0, self.num_epochs]))
        self.background_mode = cfgs.get('background_mode', 'background')
        self.shape_prior_type = cfgs.get('shape_prior_type', 'deform')
        self.backward_prior = cfgs.get('backward_prior', True)
        self.resume_prior_optim = cfgs.get('resume_prior_optim', True)
        self.dmtet_grid_smaller_epoch = cfgs.get('dmtet_grid_smaller_epoch', 0)
        self.dmtet_grid_smaller = cfgs.get('dmtet_grid_smaller', 128)
        self.dmtet_grid = cfgs.get('dmtet_grid', 256)
        self.pose_xflip_recon_epochs = np.arange(*cfgs.get('pose_xflip_recon_epochs', [0, 0]))
        self.rot_rand_quad_epochs = np.arange(*cfgs.get('rot_rand_quad_epochs', [0, 0]))
        self.rot_all_quad_epochs = np.arange(*cfgs.get('rot_all_quad_epochs', [0, 0]))
        self.calc_dino_features = cfgs.get('calc_dino_features', False)

        # self.smooth_type = cfgs.get('smooth_type', 'None')
        # print(f"****smooth_type: {self.smooth_type}****")
        
        ## smooth losses
        # smooth articulation
        self.arti_smooth_type = cfgs.get('arti_smooth_type', None)
        self.arti_smooth_loss_type = cfgs.get('arti_smooth_loss_type', None)
        self.arti_smooth_loss_weight = cfgs.get('arti_smooth_loss_weight', 0.)
        self.using_arti_smooth_loss = self.arti_smooth_type and self.arti_smooth_loss_type and self.arti_smooth_loss_weight > 0.
        if self.using_arti_smooth_loss:
            self.arti_smooth_loss_fn = SmoothLoss(dim=1, smooth_type=self.arti_smooth_type, loss_type=self.arti_smooth_loss_type)
        else:
            self.arti_smooth_loss_fn = None
        # smooth deformation
        self.deform_smooth_type = cfgs.get('deform_smooth_type', None)
        self.deform_smooth_loss_type = cfgs.get('deform_smooth_loss_type', None)
        self.deform_smooth_loss_weight = cfgs.get('deform_smooth_loss_weight', 0.)
        self.using_deform_smooth_loss = self.deform_smooth_type and self.deform_smooth_loss_type and self.deform_smooth_loss_weight > 0.
        if self.using_deform_smooth_loss:
            self.deform_smooth_loss_fn = SmoothLoss(dim=1, smooth_type=self.deform_smooth_type, loss_type=self.deform_smooth_loss_type)
        else:
            self.deform_smooth_loss_fn = None
        # smooth camera pose
        self.campos_smooth_type = cfgs.get('campos_smooth_type', None)
        self.campos_smooth_loss_type = cfgs.get('campos_smooth_loss_type', None)
        self.campos_smooth_loss_weight = cfgs.get('campos_smooth_loss_weight', 0.)
        self.using_campos_smooth_loss = self.campos_smooth_type and self.campos_smooth_loss_type and self.campos_smooth_loss_weight > 0.
        if self.using_campos_smooth_loss:
            self.campos_smooth_loss_fn = SmoothLoss(dim=1, smooth_type=self.campos_smooth_type, loss_type=self.campos_smooth_loss_type)
        else:
            self.campos_smooth_loss_fn = None
        # smooth articulation velocity
        self.artivel_smooth_type = cfgs.get('artivel_smooth_type', None)
        self.artivel_smooth_loss_type = cfgs.get('artivel_smooth_loss_type', None)
        self.artivel_smooth_loss_weight = cfgs.get('artivel_smooth_loss_weight', 0.)
        self.using_artivel_smooth_loss = self.artivel_smooth_type and self.artivel_smooth_loss_type and self.artivel_smooth_loss_weight > 0.
        if self.using_artivel_smooth_loss:
            self.artivel_smooth_loss_fn = SmoothLoss(dim=1, smooth_type=self.artivel_smooth_type, loss_type=self.artivel_smooth_loss_type)
        else:
            self.artivel_smooth_loss_fn = None
        # smooth bone
        self.bone_smooth_type = cfgs.get('bone_smooth_type', None)
        self.bone_smooth_loss_type = cfgs.get('bone_smooth_loss_type', None)
        self.bone_smooth_loss_weight = cfgs.get('bone_smooth_loss_weight', 0.)
        self.using_bone_smooth_loss = self.bone_smooth_type and self.bone_smooth_loss_type and self.bone_smooth_loss_weight > 0.
        if self.using_bone_smooth_loss:
            self.bone_smooth_loss_fn = SmoothLoss(dim=1, smooth_type=self.bone_smooth_type, loss_type=self.bone_smooth_loss_type)
        else:
            self.bone_smooth_loss_fn = None
        # smooth bone velocity
        self.bonevel_smooth_type = cfgs.get('bonevel_smooth_type', None)
        self.bonevel_smooth_loss_type = cfgs.get('bonevel_smooth_loss_type', None)
        self.bonevel_smooth_loss_weight = cfgs.get('bonevel_smooth_loss_weight', 0.)
        self.using_bonevel_smooth_loss = self.bonevel_smooth_type and self.bonevel_smooth_loss_type and self.bonevel_smooth_loss_weight > 0.
        if self.using_bonevel_smooth_loss:
            self.bonevel_smooth_loss_fn = SmoothLoss(dim=1, smooth_type=self.bonevel_smooth_type, loss_type=self.bonevel_smooth_loss_type)
        else:
            self.bonevel_smooth_loss_fn = None


        ## perceptual loss
        if cfgs.get('perceptual_loss_weight', 0.) > 0:
            self.perceptual_loss_use_lin = cfgs.get('perceptual_loss_use_lin', True)
            self.perceptual_loss = lpips.LPIPS(net='vgg', lpips=self.perceptual_loss_use_lin)

        # self.glctx = dr.RasterizeGLContext()
        self.glctx = dr.RasterizeCudaContext()
        self.render_flow = self.cfgs.get('flow_loss_weight', 0.) > 0.
        self.extra_renders = cfgs.get('extra_renders', [])
        self.renderer_spp = cfgs.get('renderer_spp', 1)
        self.dino_feature_recon_dim = cfgs.get('dino_feature_recon_dim', 64)

        self.total_loss = 0.
        self.all_scores = torch.Tensor()
        self.checkpoint_dir = cfgs.get('checkpoint_dir', 'results')

        # iter
        self.iter_arti_reg_loss_start = cfgs.get('iter_arti_reg_loss_start', None)

        # mask distribution
        self.enable_mask_distribution = cfgs.get('enable_mask_distribution', False)
        self.enable_mask_distribution = False
        self.random_mask_law = cfgs.get('random_mask_law', 'batch_swap_noy') # batch_swap, batch_swap_noy, # random_azimuth # random_all
        self.mask_distribution_path = cfgs.get('mask_distribution_path', None)

        self.enable_clip = cfgs.get('enable_clip', False)
        self.enable_clip = False

        self.enable_disc = cfgs.get('enable_disc', False)
        self.enable_disc = False
        
        self.few_shot_gan_tex = False
        self.few_shot_clip_tex = False

        self.enable_sds = cfgs.get('enable_sds', False)
        self.enable_vsd = cfgs.get('enable_vsd', False)
        self.enable_sds = False
        self.enable_vsd = False         

    @staticmethod
    def get_data_loaders(cfgs, dataset, in_image_size=256, out_image_size=256, batch_size=64, num_workers=4, run_train=False, run_test=False, train_data_dir=None, val_data_dir=None, test_data_dir=None, flow_bool=False):
        train_loader = val_loader = test_loader = None
        color_jitter_train = cfgs.get('color_jitter_train', None)
        color_jitter_val = cfgs.get('color_jitter_val', None)
        random_flip_train = cfgs.get('random_flip_train', False)

        ## video dataset
        if dataset == 'video':
            data_loader_mode = cfgs.get('data_loader_mode', 'n_frame')
            skip_beginning = cfgs.get('skip_beginning', 4)
            skip_end = cfgs.get('skip_end', 4)
            num_sample_frames = cfgs.get('num_sample_frames', 2)
            min_seq_len = cfgs.get('min_seq_len', 10)
            max_seq_len = cfgs.get('max_seq_len', 10)
            debug_seq = cfgs.get('debug_seq', False)
            random_sample_train_frames = cfgs.get('random_sample_train_frames', False)
            shuffle_train_seqs = cfgs.get('shuffle_train_seqs', False)
            random_sample_val_frames = cfgs.get('random_sample_val_frames', False)
            load_background = cfgs.get('background_mode', 'none') == 'background'
            rgb_suffix = cfgs.get('rgb_suffix', '.png')
            load_dino_feature = cfgs.get('load_dino_feature', False)
            load_dino_cluster = cfgs.get('load_dino_cluster', False)
            dino_feature_dim = cfgs.get('dino_feature_dim', 64)
            get_loader = lambda **kwargs: get_sequence_loader(
                mode=data_loader_mode,
                batch_size=batch_size,
                num_workers=num_workers,
                in_image_size=in_image_size,
                out_image_size=out_image_size,
                debug_seq=debug_seq,
                skip_beginning=skip_beginning,
                skip_end=skip_end,
                num_sample_frames=num_sample_frames,
                min_seq_len=min_seq_len,
                max_seq_len=max_seq_len,
                load_background=load_background,
                rgb_suffix=rgb_suffix,
                load_dino_feature=load_dino_feature,
                load_dino_cluster=load_dino_cluster,
                dino_feature_dim=dino_feature_dim,
                flow_bool=flow_bool,
                **kwargs)

            if run_train:
                assert osp.isdir(train_data_dir), f"Training data directory does not exist: {train_data_dir}"
                print(f"Loading training data from {train_data_dir}")
                train_loader = get_loader(data_dir=train_data_dir, is_validation=False, random_sample=random_sample_train_frames, shuffle=shuffle_train_seqs, dense_sample=True, color_jitter=color_jitter_train, random_flip=random_flip_train)

                if val_data_dir is not None:
                    assert osp.isdir(val_data_dir), f"Validation data directory does not exist: {val_data_dir}"
                    print(f"Loading validation data from {val_data_dir}")
                    val_loader = get_loader(data_dir=val_data_dir, is_validation=True, random_sample=random_sample_val_frames, shuffle=False, dense_sample=False, color_jitter=color_jitter_val, random_flip=False)
            if run_test:
                assert osp.isdir(test_data_dir), f"Testing data directory does not exist: {test_data_dir}"
                print(f"Loading testing data from {test_data_dir}")
                test_loader = get_loader(data_dir=test_data_dir, is_validation=True, dense_sample=False, color_jitter=None, random_flip=False)

        ## CUB dataset
        elif dataset == 'cub':
            get_loader = lambda **kwargs: get_cub_loader(
                batch_size=batch_size,
                num_workers=num_workers,
                image_size=in_image_size,
                **kwargs)

            if run_train:
                assert osp.isdir(train_data_dir), f"Training data directory does not exist: {train_data_dir}"
                print(f"Loading training data from {train_data_dir}")
                train_loader = get_loader(data_dir=train_data_dir, split='train', is_validation=False)
                val_loader = get_loader(data_dir=val_data_dir, split='val', is_validation=True)

            if run_test:
                assert osp.isdir(test_data_dir), f"Testing data directory does not exist: {test_data_dir}"
                print(f"Loading testing data from {test_data_dir}")
                test_loader = get_loader(data_dir=test_data_dir, split='test', is_validation=True)

        ## other datasets
        else:
            get_loader = lambda **kwargs: get_image_loader(
                batch_size=batch_size,
                num_workers=num_workers,
                image_size=in_image_size,
                **kwargs)

            if run_train:
                assert osp.isdir(train_data_dir), f"Training data directory does not exist: {train_data_dir}"
                print(f"Loading training data from {train_data_dir}")
                train_loader = get_loader(data_dir=train_data_dir, is_validation=False, color_jitter=color_jitter_train)

                if val_data_dir is not None:
                    assert osp.isdir(val_data_dir), f"Validation data directory does not exist: {val_data_dir}"
                    print(f"Loading validation data from {val_data_dir}")
                    val_loader = get_loader(data_dir=val_data_dir, is_validation=True, color_jitter=color_jitter_val)

            if run_test:
                assert osp.isdir(test_data_dir), f"Testing data directory does not exist: {test_data_dir}"
                print(f"Loading testing data from {test_data_dir}")
                test_loader = get_loader(data_dir=test_data_dir, is_validation=True, color_jitter=None)

        return train_loader, val_loader, test_loader

    @staticmethod
    def get_data_loaders_ddp(cfgs, dataset, rank, world_size, in_image_size=256, out_image_size=256, batch_size=64, num_workers=4, run_train=False, run_test=False, train_data_dir=None, val_data_dir=None, test_data_dir=None, flow_bool=False):
        train_loader = val_loader = test_loader = None
        color_jitter_train = cfgs.get('color_jitter_train', None)
        color_jitter_val = cfgs.get('color_jitter_val', None)
        random_flip_train = cfgs.get('random_flip_train', False)

        ## video dataset
        if dataset == 'video':
            data_loader_mode = cfgs.get('data_loader_mode', 'n_frame')
            skip_beginning = cfgs.get('skip_beginning', 4)
            skip_end = cfgs.get('skip_end', 4)
            num_sample_frames = cfgs.get('num_sample_frames', 2)
            min_seq_len = cfgs.get('min_seq_len', 10)
            max_seq_len = cfgs.get('max_seq_len', 10)
            debug_seq = cfgs.get('debug_seq', False)
            random_sample_train_frames = cfgs.get('random_sample_train_frames', False)
            shuffle_train_seqs = cfgs.get('shuffle_train_seqs', False)
            random_sample_val_frames = cfgs.get('random_sample_val_frames', False)
            load_background = cfgs.get('background_mode', 'none') == 'background'
            rgb_suffix = cfgs.get('rgb_suffix', '.png')
            load_dino_feature = cfgs.get('load_dino_feature', False)
            load_dino_cluster = cfgs.get('load_dino_cluster', False)
            dino_feature_dim = cfgs.get('dino_feature_dim', 64)
            get_loader_ddp = lambda **kwargs: get_sequence_loader_ddp(
                mode=data_loader_mode,
                batch_size=batch_size,
                num_workers=num_workers,
                in_image_size=in_image_size,
                out_image_size=out_image_size,
                debug_seq=debug_seq,
                skip_beginning=skip_beginning,
                skip_end=skip_end,
                num_sample_frames=num_sample_frames,
                min_seq_len=min_seq_len,
                max_seq_len=max_seq_len,
                load_background=load_background,
                rgb_suffix=rgb_suffix,
                load_dino_feature=load_dino_feature,
                load_dino_cluster=load_dino_cluster,
                dino_feature_dim=dino_feature_dim,
                flow_bool=flow_bool,
                **kwargs)
            get_loader = lambda **kwargs: get_sequence_loader(
                mode=data_loader_mode,
                batch_size=batch_size,
                num_workers=num_workers,
                in_image_size=in_image_size,
                out_image_size=out_image_size,
                debug_seq=debug_seq,
                skip_beginning=skip_beginning,
                skip_end=skip_end,
                num_sample_frames=num_sample_frames,
                min_seq_len=min_seq_len,
                max_seq_len=max_seq_len,
                load_background=load_background,
                rgb_suffix=rgb_suffix,
                load_dino_feature=load_dino_feature,
                load_dino_cluster=load_dino_cluster,
                dino_feature_dim=dino_feature_dim,
                **kwargs)

            if run_train:
                if isinstance(train_data_dir, dict):
                    for data_path in train_data_dir.values():
                        assert osp.isdir(data_path), f"Training data directory does not exist: {data_path}"
                elif isinstance(train_data_dir, str):
                    assert osp.isdir(train_data_dir), f"Training data directory does not exist: {train_data_dir}"
                else:
                    raise ValueError("train_data_dir must be a string or a dict of strings")
                
                print(f"Loading training data...")
                train_loader = get_loader_ddp(data_dir=train_data_dir, rank=rank, world_size=world_size, is_validation=False, random_sample=random_sample_train_frames, shuffle=shuffle_train_seqs, dense_sample=True, color_jitter=color_jitter_train, random_flip=random_flip_train)

                if val_data_dir is not None:
                    if isinstance(val_data_dir, dict):
                        for data_path in val_data_dir.values():
                            assert osp.isdir(data_path), f"Training data directory does not exist: {data_path}"
                    elif isinstance(val_data_dir, str):
                        assert osp.isdir(val_data_dir), f"Training data directory does not exist: {val_data_dir}"
                    else:
                        raise ValueError("train_data_dir must be a string or a dict of strings")
                    print(f"Loading validation data...")
                    # No need for data parallel for the validation data loader.
                    val_loader = get_loader(data_dir=val_data_dir, is_validation=True, random_sample=random_sample_val_frames, shuffle=False, dense_sample=False, color_jitter=color_jitter_val, random_flip=False)
            
            if run_test:
                assert osp.isdir(test_data_dir), f"Testing data directory does not exist: {test_data_dir}"
                print(f"Loading testing data from {test_data_dir}")
                test_loader = get_loader_ddp(data_dir=test_data_dir, rank=rank, world_size=world_size, is_validation=True, dense_sample=False, color_jitter=None, random_flip=False)

        ## CUB dataset
        elif dataset == 'cub':
            get_loader = lambda **kwargs: get_cub_loader_ddp(
                batch_size=batch_size,
                num_workers=num_workers,
                image_size=in_image_size,
                **kwargs)

            if run_train:
                assert osp.isdir(train_data_dir), f"Training data directory does not exist: {train_data_dir}"
                print(f"Loading training data from {train_data_dir}")
                train_loader = get_loader(data_dir=train_data_dir, rank=rank, world_size=world_size, split='train', is_validation=False)
                val_loader = get_loader(data_dir=val_data_dir, rank=rank, world_size=world_size, split='val', is_validation=True)

            if run_test:
                assert osp.isdir(test_data_dir), f"Testing data directory does not exist: {test_data_dir}"
                print(f"Loading testing data from {test_data_dir}")
                test_loader = get_loader(data_dir=test_data_dir, rank=rank, world_size=world_size, split='test', is_validation=True)

        ## other datasets
        else:
            get_loader = lambda **kwargs: get_image_loader_ddp(
                batch_size=batch_size,
                num_workers=num_workers,
                image_size=in_image_size,
                **kwargs)

            if run_train:
                assert osp.isdir(train_data_dir), f"Training data directory does not exist: {train_data_dir}"
                print(f"Loading training data from {train_data_dir}")
                train_loader = get_loader(data_dir=train_data_dir, rank=rank, world_size=world_size, is_validation=False, color_jitter=color_jitter_train)

                if val_data_dir is not None:
                    assert osp.isdir(val_data_dir), f"Validation data directory does not exist: {val_data_dir}"
                    print(f"Loading validation data from {val_data_dir}")
                    val_loader = get_loader(data_dir=val_data_dir, rank=rank, world_size=world_size, is_validation=True, color_jitter=color_jitter_val)

            if run_test:
                assert osp.isdir(test_data_dir), f"Testing data directory does not exist: {test_data_dir}"
                print(f"Loading testing data from {test_data_dir}")
                test_loader = get_loader(data_dir=test_data_dir, rank=rank, world_size=world_size, is_validation=True, color_jitter=None)

        return train_loader, val_loader, test_loader

    def load_model_state(self, cp):
        # TODO: very hacky: if using local texture, which is also usually finetuned from global texture
        # we need to check if needs some handcrafted load in netInstance
        if (self.netInstance.texture_way is not None) or (self.cfgs.get('texture_act', 'relu') != 'relu'):
            new_netInstance_weights = {k: v for k, v in cp['netInstance'].items() if 'netTexture' not in k}
            #find the new texture weights 
            texture_weights = self.netInstance.netTexture.state_dict()
            #add the new weights to the new model weights
            for k, v in texture_weights.items():
                new_netInstance_weights['netTexture.' + k] = v
            self.netInstance.load_state_dict(new_netInstance_weights)
        else:
            self.netInstance.load_state_dict(cp["netInstance"])
        if self.enable_disc and "net_mask_disc" in cp:
            self.mask_disc.load_state_dict(cp["net_mask_disc"])
        if self.enable_prior:
            self.netPrior.load_state_dict(cp["netPrior"])


    def load_optimizer_state(self, cp):
        # TODO: also very hacky here, as the load_model_state above
        if self.netInstance.texture_way is not None:
            opt_state_dict = self.optimizerInstance.state_dict()
            param_ids = [id(p) for p in self.netInstance.netTexture.parameters()]
            new_opt_state_dict = {}
            new_opt_state_dict['state'] = {k: v for k, v in opt_state_dict['state'].items() if k not in param_ids}

            new_param_groups = []
            for param_group in opt_state_dict['param_groups']:
                new_param_group = {k: v for k, v in param_group.items() if k != 'params'}
                new_param_group['params'] = [p_id for p_id in param_group['params'] if p_id not in param_ids]
                new_param_groups.append(new_param_group)

            new_opt_state_dict['param_groups'] = new_param_groups

            self.optimizerInstance.load_state_dict(new_opt_state_dict)
        else:
            self.optimizerInstance.load_state_dict(cp["optimizerInstance"])

        # add parameters into optimizerInstance here
        # if self.enable_disc:
        #     print('add mask discriminator parameters to Instance optimizer')
        #     self.optimizerInstance.add_param_group({'params': self.mask_disc.parameters()})

        if self.use_scheduler:
            if 'schedulerInstance' in cp:
                self.schedulerInstance.load_state_dict(cp["schedulerInstance"])
        if self.enable_disc and "optimizerDiscriminator" in cp:
            self.optimizerDiscriminator.load_state_dict(cp["optimizerDiscriminator"])
        if self.enable_prior and self.resume_prior_optim:
            self.optimizerPrior.load_state_dict(cp["optimizerPrior"])
            if self.use_scheduler:
                if 'schedulerPrior' in cp:
                    self.schedulerPrior.load_state_dict(cp["schedulerPrior"])

    def get_model_state(self):
        state = {"netInstance": self.netInstance.state_dict()}
        if self.enable_disc:
            state["net_mask_disc"] = self.mask_disc.state_dict()
        if self.enable_prior:
            state["netPrior"] = self.netPrior.state_dict()
        return state

    def get_optimizer_state(self):
        state = {"optimizerInstance": self.optimizerInstance.state_dict()}
        if self.enable_disc:
            state['optimizerDiscriminator'] = self.optimizerDiscriminator.state_dict()
        if self.use_scheduler:
            state["schedulerInstance"] = self.schedulerInstance.state_dict()
        if self.enable_prior:
            state["optimizerPrior"] = self.optimizerPrior.state_dict()
            if self.use_scheduler:
                state["schedulerPrior"] = self.schedulerPrior.state_dict()
        return state

    def to(self, device):
        self.device = device
        self.netInstance.to(device)
        if self.enable_prior:
            self.netPrior.to(device)
            for v in vars(self.netPrior.netShape):
                attr = getattr(self.netPrior.netShape,v)
                if type(attr) == torch.Tensor:
                    setattr(self.netPrior.netShape, v, attr.to(device))
        if hasattr(self, 'perceptual_loss'):
            self.perceptual_loss.to(device)

    def ddp(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size

        if self.world_size > 1:
            self.netInstance_ddp = DDP(
                self.netInstance, device_ids=[rank],
                find_unused_parameters=True)
            self.netInstance_ddp._set_static_graph()
            self.netInstance = self.netInstance_ddp.module

            if self.enable_prior:
                self.netPrior_ddp = DDP(
                    self.netPrior, device_ids=[rank],
                    find_unused_parameters=True)
                self.netPrior_ddp._set_static_graph()
                self.netPrior = self.netPrior_ddp.module

            if hasattr(self, 'perceptual_loss'):
                self.perceptual_loss_ddp = DDP(
                    self.perceptual_loss, device_ids=[rank],
                    find_unused_parameters=True)
                self.perceptual_loss = self.perceptual_loss_ddp.module
        else:
            print('actually no DDP for model')

    def set_train(self):
        if self.world_size > 1:
            self.netInstance_ddp.train()
            if self.enable_prior:
                self.netPrior_ddp.train()
        else:
            self.netInstance.train()
            if self.enable_disc:
                self.mask_disc.train()
            if self.enable_prior:
                self.netPrior.train()

    def set_eval(self):
        if self.world_size > 1:
            self.netInstance_ddp.eval()
            if self.enable_prior:
                self.netPrior_ddp.eval()
        else:
            self.netInstance.eval()
            if self.enable_disc:
                self.mask_disc.eval()
            if self.enable_prior:
                self.netPrior.eval()

    def reset_optimizers(self):
        print("Resetting optimizers...")
        self.optimizerInstance = get_optimizer(self.netInstance, self.lr)

        if self.enable_disc:
            self.optimizerDiscriminator = get_optimizer(self.mask_disc, self.lr)

        if self.use_scheduler:
            self.schedulerInstance = self.make_scheduler(self.optimizerInstance)
        if self.enable_prior:
            self.optimizerPrior = get_optimizer(self.netPrior, lr=self.prior_lr, weight_decay=self.prior_weight_decay)
            if self.use_scheduler:
                self.schedulerPrior = self.make_scheduler(self.optimizerPrior)
    
    def reset_only_disc_optimizer(self):
        if self.enable_disc:
            self.optimizerDiscriminator = get_optimizer(self.mask_disc, self.lr)

    def backward(self):
        self.optimizerInstance.zero_grad()
        if self.backward_prior:
            self.optimizerPrior.zero_grad()
        # self.total_loss = self.add_unused()
        self.total_loss.backward()
        self.optimizerInstance.step()
        if self.backward_prior:
            self.optimizerPrior.step()
        self.total_loss = 0.

    def scheduler_step(self):
        if self.use_scheduler:
            self.schedulerInstance.step()
            if self.enable_prior:
                self.schedulerPrior.step()

    def zflip_pose(self, pose):
        if self.rot_rep == 'lookat':
            vec_forward = pose[:,:,6:9]
            vec_forward = vec_forward * torch.FloatTensor([1,1,-1]).view(1,1,3).to(vec_forward.device)
            up = torch.FloatTensor([0,1,0]).to(pose.device).view(1,1,3)
            vec_right = up.expand_as(vec_forward).cross(vec_forward, dim=-1)
            vec_right = nn.functional.normalize(vec_right, p=2, dim=-1)
            vec_up = vec_forward.cross(vec_right, dim=-1)
            vec_up = nn.functional.normalize(vec_up, p=2, dim=-1)
            rot_mat = torch.stack([vec_right, vec_up, vec_forward], 2)
            rot_pred = rot_mat.reshape(*pose.shape[:-1], -1)
            pose_zflip = torch.cat([rot_pred, pose[:,:,9:]], -1)
        else:
            raise NotImplementedError
        return pose_zflip

    def render(self, shape, texture, mvp, w2c, campos, resolution, background='none', im_features=None, light=None, prior_shape=None, render_flow=False, dino_pred=None, class_vector=None, render_mode='diffuse', two_sided_shading=True, num_frames=None, spp=1, bg_image=None, im_features_map=None):
        h, w = resolution
        N = len(mvp)
        if bg_image is None:
            if background in ['none', 'black']:
                bg_image = torch.zeros((N, h, w, 3), device=mvp.device)
            elif background == 'white':
                bg_image = torch.ones((N, h, w, 3), device=mvp.device)
            elif background == 'checkerboard':
                bg_image = torch.FloatTensor(util.checkerboard((h, w), 8), device=self.device).repeat(N, 1, 1, 1)  # NxHxWxC
            elif background == 'random':
                bg_image = torch.rand((N, h, w, 3), device=mvp.device)  # NxHxWxC
            elif background == 'random-pure':
                random_values = torch.rand(N)
                bg_image = random_values[..., None, None, None].repeat(1, h, w, 3).to(self.device)
            else:
                raise NotImplementedError

        #insider render_mesh -> render_layer -> shade DOR
        frame_rendered = render.render_mesh(
            self.glctx,
            shape,
            mtx_in=mvp,
            w2c=w2c,
            view_pos=campos,
            material=texture,
            lgt=light,
            resolution=resolution,
            spp=spp,
            msaa=True,
            background=bg_image,
            bsdf=render_mode,
            feat=im_features,
            prior_mesh=prior_shape,
            two_sided_shading=two_sided_shading,
            render_flow=render_flow,
            dino_pred=dino_pred,
            class_vector=class_vector,
            num_frames=num_frames,
            im_features_map=im_features_map)
        shaded = frame_rendered['shaded'].permute(0, 3, 1, 2)
        image_pred = shaded[:, :3, :, :]
        mask_pred = shaded[:, 3, :, :]
        albedo = frame_rendered['kd'].permute(0, 3, 1, 2)[:, :3, :, :]
        if 'shading' in frame_rendered:
            shading = frame_rendered['shading'].permute(0, 3, 1, 2)[:, :1, :, :]
        else:
            shading = None
        if render_flow:
            flow_pred = frame_rendered['flow']
            flow_pred = flow_pred.permute(0, 3, 1, 2)[:, :2, :, :]
        else:
            flow_pred = None
        if dino_pred is not None:
            dino_feat_im_pred = frame_rendered['dino_feat_im_pred']
            dino_feat_im_pred = dino_feat_im_pred.permute(0, 3, 1, 2)[:, :-1]
        else:
            dino_feat_im_pred = None
            
        return image_pred, mask_pred, flow_pred, dino_feat_im_pred, albedo, shading

    def compute_reconstruction_losses(self, image_pred, image_gt, mask_pred, mask_gt, mask_dt, mask_valid, flow_pred, flow_gt, dino_feat_im_gt, dino_feat_im_pred, background_mode='none', reduce=False):
        losses = {}
        batch_size, num_frames, _, h, w = image_pred.shape  # BxFxCxHxW

        # image_loss = (image_pred - image_gt) ** 2
        image_loss = (image_pred - image_gt).abs()

        ## silhouette loss
        mask_pred_valid = mask_pred * mask_valid
        # mask_pred_valid = mask_pred
        # losses["silhouette_loss"] = ((mask_pred - mask_gt) ** 2).mean()
        # mask_loss_mask = (image_loss.mean(2).detach() > 0.05).float()
        mask_loss = (mask_pred_valid - mask_gt) ** 2
        # mask_loss = nn.functional.mse_loss(mask_pred, mask_gt)
        # num_mask_pixels = mask_loss_mask.reshape(batch_size*num_frames, -1).sum(1).clamp(min=1)
        # losses["silhouette_loss"] = (mask_loss.reshape(batch_size*num_frames, -1).sum(1) / num_mask_pixels).mean()
        losses['silhouette_loss'] = mask_loss.view(batch_size, num_frames, -1).mean(2)
        losses['silhouette_dt_loss'] = (mask_pred * mask_dt[:,:,1]).view(batch_size, num_frames, -1).mean(2)
        losses['silhouette_inv_dt_loss'] = ((1-mask_pred) * mask_dt[:,:,0]).view(batch_size, num_frames, -1).mean(2)

        mask_pred_binary = (mask_pred_valid > 0.).float().detach()
        mask_both_binary = (mask_pred_binary * mask_gt).view(batch_size*num_frames, 1, *mask_pred.shape[2:])
        mask_both_binary = (nn.functional.avg_pool2d(mask_both_binary, 3, stride=1, padding=1).view(batch_size, num_frames, *mask_pred.shape[2:]) > 0.99).float().detach()  # erode by 1 pixel

        ## reconstruction loss
        # image_loss_mask = (mask_pred*mask_gt).unsqueeze(2).expand_as(image_gt)
        # image_loss = image_loss * image_loss_mask
        # num_mask_pixels = image_loss_mask.reshape(batch_size*num_frames, -1).sum(1).clamp(min=1)
        # losses["rgb_loss"] = (image_loss.reshape(batch_size*num_frames, -1).sum(1) / num_mask_pixels).mean()
        if background_mode in ['background', 'input']:
            pass
        else:
            image_loss = image_loss * mask_both_binary.unsqueeze(2)
        losses['rgb_loss'] = image_loss.reshape(batch_size, num_frames, -1).mean(2)

        if self.cfgs.get('perceptual_loss_weight', 0.) > 0:
            if background_mode in ['background', 'input']:
                perc_image_pred = image_pred
                perc_image_gt = image_gt
            else:
                perc_image_pred = image_pred * mask_pred_binary.unsqueeze(2) + 0.5 * (1-mask_pred_binary.unsqueeze(2))
                perc_image_gt = image_gt * mask_pred_binary.unsqueeze(2) + 0.5 * (1-mask_pred_binary.unsqueeze(2))
            losses['perceptual_loss'] = self.perceptual_loss(perc_image_pred.view(-1, *image_pred.shape[2:]) *2-1, perc_image_gt.view(-1, *image_gt.shape[2:]) *2-1).view(batch_size, num_frames)

        ## flow loss - between first and second frame
        if flow_pred is not None:
            flow_loss = (flow_pred - flow_gt).abs()
            flow_loss_mask = mask_both_binary[:,:-1].unsqueeze(2).expand_as(flow_gt).detach()

            ## ignore frames where GT flow is too large (likely inaccurate)
            large_flow = (flow_gt.abs() > 0.5).float() * flow_loss_mask
            large_flow = (large_flow.view(batch_size, num_frames-1, -1).sum(2) > 0).float()
            self.large_flow = large_flow

            flow_loss = flow_loss * flow_loss_mask * (1 - large_flow[:,:,None,None,None])
            num_mask_pixels = flow_loss_mask.reshape(batch_size, num_frames-1, -1).sum(2).clamp(min=1)
            losses['flow_loss'] = (flow_loss.reshape(batch_size, num_frames-1, -1).sum(2) / num_mask_pixels)
            # losses["flow_loss"] = flow_loss.mean()

        if dino_feat_im_pred is not None and dino_feat_im_gt is not None:
            dino_feat_loss = (dino_feat_im_pred - dino_feat_im_gt) ** 2
            dino_feat_loss = dino_feat_loss * mask_both_binary.unsqueeze(2)
            losses['dino_feat_im_loss'] = dino_feat_loss.reshape(batch_size, num_frames, -1).mean(2)

        if reduce:
            for k, v in losses.item():
                losses[k] = v.mean()
        return losses

    def compute_pose_xflip_reg_loss(self, input_image, dino_feat_im, pose_raw, input_image_xflip_flag=None):
        image_xflip = input_image.flip(4)
        if dino_feat_im is not None:
            dino_feat_im_xflip = dino_feat_im.flip(4)
        else:
            dino_feat_im_xflip = None
        
        if self.world_size > 1:
            netInst = self.netInstance_ddp
        else:
            netInst = self.netInstance

        # feat_xflip, _ = self.netInstance_ddp.forward_encoder(image_xflip, dino_feat_im_xflip)
        feat_xflip, _ = netInst.forward_encoder(image_xflip, dino_feat_im_xflip)
        batch_size, num_frames = input_image.shape[:2]
        # pose_xflip_raw = self.netInstance_ddp.forward_pose(image_xflip, feat_xflip, dino_feat_im_xflip)
        pose_xflip_raw = netInst.forward_pose(image_xflip, feat_xflip, dino_feat_im_xflip)

        if input_image_xflip_flag is not None:
            pose_xflip_raw_xflip = pose_xflip_raw * torch.FloatTensor([-1,1,1,-1,1,1]).to(pose_raw.device)  # forward x, trans x
            pose_xflip_raw = pose_xflip_raw * (1 - input_image_xflip_flag.view(batch_size * num_frames, 1)) + pose_xflip_raw_xflip * input_image_xflip_flag.view(batch_size * num_frames, 1)

        # rot_rep = self.netInstance_ddp.rot_rep
        rot_rep = netInst.rot_rep
        if rot_rep == 'euler_angle' or rot_rep == 'soft_calss':
            pose_xflip_xflip = pose_xflip * torch.FloatTensor([1,-1,-1,-1,1,1]).to(pose_xflip.device)  # rot y+z, trans x
            pose_xflip_reg_loss = ((pose_xflip_xflip - pose) ** 2.).mean()
        elif rot_rep == 'quaternion':
            rot_euler = pytorch3d.transforms.matrix_to_euler_angles(pytorch3d.transforms.quaternion_to_matrix(pose[...,:4]), convention='XYZ')
            pose_euler = torch.cat([rot_euler, pose[...,4:]], -1)
            rot_xflip_euler = pytorch3d.transforms.matrix_to_euler_angles(pytorch3d.transforms.quaternion_to_matrix(pose_xflip[...,:4]), convention='XYZ')
            pose_xflip_euler = torch.cat([rot_xflip_euler, pose_xflip[...,4:]], -1)
            pose_xflip_euler_xflip = pose_xflip_euler * torch.FloatTensor([1,-1,-1,-1,1,1]).to(pose_xflip.device)  # rot y+z, trans x
            pose_xflip_reg_loss = ((pose_xflip_euler_xflip - pose_euler) ** 2.).mean()
        elif rot_rep == 'lookat':
            pose_xflip_raw_xflip = pose_xflip_raw * torch.FloatTensor([-1,1,1,-1,1,1]).to(pose_raw.device)  # forward x, trans x
            pose_xflip_reg_loss = ((pose_xflip_raw_xflip - pose_raw)[...,0] ** 2.)  # compute x only
            # if epoch >= self.nolookat_zflip_loss_epochs and self.lookat_zflip_no_other_losses:
            #     pose_xflip_reg_loss = pose_xflip_reg_loss.mean(1) * is_pose_1_better
            pose_xflip_reg_loss = pose_xflip_reg_loss.mean()
        return pose_xflip_reg_loss, pose_xflip_raw
    
    def compute_edge_length_reg_loss(self, mesh, prior_mesh):
        prior_edge_lengths = get_edge_length(prior_mesh.v_pos, prior_mesh.t_pos_idx)
        max_length = prior_edge_lengths.max().detach() *1.1
        edge_lengths = get_edge_length(mesh.v_pos, mesh.t_pos_idx)
        mesh_edge_length_loss = ((edge_lengths - max_length).clamp(min=0)**2).mean()
        return mesh_edge_length_loss, edge_lengths

    def compute_regularizers(self, mesh, prior_mesh, input_image, dino_feat_im, pose_raw, input_image_xflip_flag=None, arti_params=None, deformation=None, mid_img_idx=0, posed_bones=None, class_vector=None):
        losses = {}
        aux = {}
        
        if self.enable_prior:
            losses.update(self.netPrior.netShape.get_sdf_reg_loss(class_vector=class_vector))
        
        if self.cfgs.get('pose_xflip_reg_loss_weight', 0.) > 0:
            losses["pose_xflip_reg_loss"], aux['pose_xflip_raw'] = self.compute_pose_xflip_reg_loss(input_image, dino_feat_im, pose_raw, input_image_xflip_flag)
        
        if self.using_campos_smooth_loss:
            # from IPython import embed; embed()
            pose_raw_ = pose_raw.view(self.bs, self.nf, *pose_raw.shape[1:])
            losses['campos_smooth_loss'] = self.campos_smooth_loss_fn(pose_raw_)

        b, f = input_image.shape[:2]
        if b >= 2:
            vec_forward = pose_raw[..., :3]
            losses['pose_entropy_loss'] = (vec_forward[:b//2] * vec_forward[b//2:(b//2)*2]).sum(-1).mean()
        else:
            losses['pose_entropy_loss'] = 0.

        losses['mesh_normal_consistency_loss'] = normal_consistency(mesh.v_pos, mesh.t_pos_idx)
        losses['mesh_laplacian_consistency_loss'] = laplace_regularizer_const(mesh.v_pos, mesh.t_pos_idx)
        losses['mesh_edge_length_loss'], aux['edge_lengths'] = self.compute_edge_length_reg_loss(mesh, prior_mesh)
        if arti_params is not None:
            #losses['arti_reg_loss'] = (arti_params ** 2).mean()
            losses['arti_reg_loss'] = (arti_params ** 2).mean() #TODO dor Rart

        if arti_params is not None and self.using_arti_smooth_loss:
            arti_smooth_loss = self.arti_smooth_loss_fn(arti_params)
            losses['arti_smooth_loss'] = arti_smooth_loss
        # if arti_params is not None and self.cfgs.get('arti_smooth_loss_weight', 0.) > 0:
        #     if self.smooth_type == 'loss' and mid_img_idx > 0:
        #         # print("+++++++++++++++++add smooth to *articulation* loss")
        #         # from IPython import embed; embed()
        #         arti_smooth_loss = (
        #             ((arti_params[:,mid_img_idx,:,:] - arti_params[:,0:mid_img_idx,:,:])**2)
        #             + ((arti_params[:,mid_img_idx,:,:] - arti_params[:,mid_img_idx+1:2*mid_img_idx+1,:,:])**2)
        #         ).mean()
        #         losses['arti_smooth_loss'] = arti_smooth_loss

        if arti_params is not None and self.using_artivel_smooth_loss:
            # from IPython import embed; embed()
            _, nf, _, _= arti_params.shape
            arti_vel = arti_params[:,1:nf,:,:] - arti_params[:,:(nf-1),:,:]
            artivel_smooth_loss = self.artivel_smooth_loss_fn(arti_vel)
            losses['artivel_smooth_loss'] = artivel_smooth_loss

        if deformation is not None:
            #losses['deformation_reg_loss'] = (deformation ** 2).mean()
            losses['deformation_reg_loss'] = (deformation ** 2).mean() #TODO dor - Rdef

            d1 = deformation[:, mesh.t_pos_idx[0, :, 0], :]
            d2 = deformation[:, mesh.t_pos_idx[0, :, 1], :]
            d3 = deformation[:, mesh.t_pos_idx[0, :, 2], :]

            num_samples = 5000
            sample_idx1 = torch.randperm(d1.shape[1])[:num_samples].to(self.device)
            sample_idx2 = torch.randperm(d1.shape[1])[:num_samples].to(self.device)
            sample_idx3 = torch.randperm(d1.shape[1])[:num_samples].to(self.device)

            dist1 = ((d1[:, sample_idx1, :] - d2[:, sample_idx1, :]) ** 2).mean()
            dist2 = ((d2[:, sample_idx2, :] - d3[:, sample_idx2, :]) ** 2).mean()
            dist3 = ((d3[:, sample_idx3, :] - d1[:, sample_idx3, :]) ** 2).mean()
            
            losses['smooth_deformation_loss'] = dist1 + dist2 + dist3

        if deformation is not None and self.using_deform_smooth_loss:
            deformation_ = deformation.view(self.bs, self.nf, *deformation.shape[1:])
            losses['deform_smooth_loss'] = self.deform_smooth_loss_fn(deformation_)
        # if deformation is not None and self.cfgs.get('deformation_smooth_loss_weight', 0.) > 0:
        #     if self.smooth_type == 'loss' and mid_img_idx > 0:
        #         # print("+++++++++++++++++add smooth to *deformation* loss")
        #         deformation = deformation.view(self.bs, self.nf, *deformation.shape[1:])
        #         deformation_smooth_loss = (
        #             ((deformation[:, mid_img_idx,:,:] - deformation[:, 0:mid_img_idx,:,:]) ** 2)
        #             + ((deformation[:, mid_img_idx,:,:] - deformation[:, mid_img_idx+1:2*mid_img_idx+1,:,:]) ** 2)
        #         ).mean()
        #         losses['deformation_smooth_loss'] = deformation_smooth_loss
        #         # deformation = deformation.view(self.bs * self.nf, *deformation.shape[2:])
        #     # losses['deformation_reg_loss'] = deformation.abs().mean()

        ## posed bones.
        if posed_bones is not None and self.using_bone_smooth_loss:
            bone_smooth_loss = self.bone_smooth_loss_fn(posed_bones)
            losses['bone_smooth_loss'] = bone_smooth_loss

        if posed_bones is not None and self.using_bonevel_smooth_loss:
            _, nf, _, _, _= posed_bones.shape
            bone_vel = posed_bones[:,1:nf,...] - posed_bones[:,:(nf-1),...]
            bonevel_smooth_loss = self.bonevel_smooth_loss_fn(bone_vel)
            losses['bonevel_smooth_loss'] = bonevel_smooth_loss
            
        return losses, aux
    
    def parse_dict_definition(self, dict_config, total_iter):
        '''
        The dict_config is a diction-based configuration with ascending order
        The key: value is the NUM_ITERATION_WEIGHT_BEGIN: WEIGHT
        For example,
        {0: 0.1, 1000: 0.2, 10000: 0.3}
        means at beginning, the weight is 0.1, from 1k iterations, weight is 0.2, and after 10k, weight is 0.3
        '''
        length = len(dict_config)
        all_iters = list(dict_config.keys())
        all_weights = list(dict_config.values())

        weight = all_weights[-1]

        for i in range(length-1):
            # this works for dict having at least two items, otherwise you don't need dict to set config
            iter_num = all_iters[i]
            iter_num_next = all_iters[i+1]
            if iter_num <= total_iter and total_iter < iter_num_next:
                weight = all_weights[i]
                break

        return weight

    def compute_clip_loss(self, random_image_pred, image_pred, category):
        # image preprocess for CLIP
        random_image = torch.nn.functional.interpolate(random_image_pred, (self.clip_reso, self.clip_reso), mode='bilinear')
        image_pred = torch.nn.functional.interpolate(image_pred.squeeze(1), (self.clip_reso, self.clip_reso), mode='bilinear')
        random_image = tvf.normalize(random_image, self.clip_mean, self.clip_std)
        image_pred = tvf.normalize(image_pred, self.clip_mean, self.clip_std)
        
        feat_img_1 = self.clip_model.encode_image(random_image)
        feat_img_2 = self.clip_model.encode_image(image_pred)

        clip_all_loss = torch.nn.functional.cosine_similarity(feat_img_1, feat_img_2)
        clip_all_loss = 1 - clip_all_loss.mean()

        # feat_img_1 = torch.mean(feat_img_1, dim=0)
        # feat_img_2 = torch.mean(feat_img_2, dim=0)
        # clip_all_loss = torch.nn.functional.cosine_similarity(feat_img_1, feat_img_2, dim=0)
        # clip_all_loss = 1 - clip_all_loss

        if self.enable_clip_text:
            text_feature = self.clip_text_feature[category].repeat(feat_img_1.shape[0], 1)

            text_loss_1 = torch.nn.functional.cosine_similarity(feat_img_1, text_feature).mean()
            text_loss_2 = torch.nn.functional.cosine_similarity(feat_img_2, text_feature).mean()

            # text_feature = self.clip_text_feature[category][0]

            # text_loss_1 = torch.nn.functional.cosine_similarity(feat_img_1, text_feature, dim=0)
            # text_loss_2 = torch.nn.functional.cosine_similarity(feat_img_2, text_feature, dim=0)

            clip_all_loss = clip_all_loss + (1 - text_loss_1) + (1 - text_loss_2)
        
        return {'clip_all_loss': clip_all_loss}
    
    def generate_patch_crop(self, images, masks, patch_size=128, patch_num_per_mask=1):
        b, _, H, W = masks.shape
    
        patches = []
        for i in range(masks.shape[0]):
            mask = masks[i]
            # mask: [1, H, W]
            nonzero_indices = torch.nonzero(mask > 0, as_tuple=False)  # [K', 3]
            valid_mask = (nonzero_indices[:, 1] > patch_size // 2) & (nonzero_indices[:, 1] < (H - 1 - patch_size // 2)) & (nonzero_indices[:, 2] > patch_size // 2) & (nonzero_indices[:, 2] < (W - 1 - patch_size // 2))
            valid_idx = nonzero_indices[valid_mask]  
            patch_idx = valid_idx[torch.randperm(valid_idx.shape[0])[:patch_num_per_mask]] # [K, 3]

            if patch_idx.shape[0] < patch_num_per_mask:
                patches_this_img = torch.zeros(patch_num_per_mask, 3, self.few_shot_gan_tex_patch, self.few_shot_gan_tex_patch).to(self.device)
            else:
                patches_this_img = []

                for idx in range(patch_idx.shape[0]):
                    _, y, x = patch_idx[idx]
                    
                    y_start = max(0, y - patch_size // 2)
                    y_end = min(H, y_start + patch_size)
                    x_start = max(0, x - patch_size // 2)
                    x_end = min(W, x_start + patch_size)
                    
                    patch_content = images[i, :, y_start:y_end, x_start:x_end]
                    
                    patch = F.interpolate(patch_content.unsqueeze(0), size=self.few_shot_gan_tex_patch, mode='bilinear')  # [1, 3, ps, ps]
                    patches_this_img.append(patch)
                
                patches_this_img = torch.cat(patches_this_img, dim=0)  # [K, 3, ps, ps]
            
            patches.append(patches_this_img)
        
        patches = torch.concat(patches, dim=0)  # [B*K, 3, ps, ps]
        return patches

    
    def compute_gan_tex_loss(self, category, image_gt, mask_gt, iv_image_pred, iv_mask_pred, w2c_pred, campos_pred, shape, prior_shape, texture, dino_pred, im_features, light, class_vector, num_frames, im_features_map, bins=360):
        '''
        This part is used to do gan training on texture, this is meant to only be used in fine-tuning, with local texture network
        Ideally this loss only contributes to the Texture
        '''
        delta_angle = 2 * np.pi / bins
        b = len(shape)
        rand_degree = torch.randint(120, [b])
        rand_degree = rand_degree + 120
        # rand_degree = torch.ones(b) * 180  # we want to see the reversed side
        delta_angle = delta_angle * rand_degree
        delta_rot_matrix = []
        for i in range(b):
            angle = delta_angle[i].item()
            angle_matrix = torch.FloatTensor([
                [np.cos(angle),  0, np.sin(angle), 0],
                [0,              1, 0,             0],
                [-np.sin(angle), 0, np.cos(angle), 0],
                [0,              0, 0,             1],
            ]).to(self.device)
            delta_rot_matrix.append(angle_matrix)
        delta_rot_matrix = torch.stack(delta_rot_matrix, dim=0)

        proj = util.perspective(self.crop_fov_approx / 180 * np.pi, 1, n=0.1, f=1000.0).repeat(b, 1, 1).to(self.device)

        original_mvp = torch.bmm(proj, w2c_pred)
        # original_campos = -w2c_pred[:, :3, 3]
        original_campos = campos_pred
        mvp = torch.matmul(original_mvp, delta_rot_matrix)
        campos = torch.matmul(delta_rot_matrix[:,:3,:3].transpose(2,1), original_campos[:,:,None])[:,:,0]
        w2c = w2c_pred

        resolution = (self.few_shot_gan_tex_reso, self.few_shot_gan_tex_reso)

        # only train the texture
        safe_detach = lambda x: x.detach() if x is not None else None
        mesh = safe_detach(shape)
        im_features = safe_detach(im_features)
        im_features_map = safe_detach(im_features_map)
        class_vector = safe_detach(class_vector)

        set_requires_grad(texture, True)
        set_requires_grad(dino_pred, False)
        set_requires_grad(light, False)

        background_for_reverse = 'none'
        # background_for_reverse = 'random-pure'
        
        image_pred, mask_pred, _, _, _, _ = self.render(
            mesh, 
            texture, 
            mvp, 
            w2c, 
            campos, 
            resolution, 
            background=background_for_reverse, 
            im_features=im_features, 
            light=light, 
            prior_shape=prior_shape, 
            render_flow=False, 
            dino_pred=dino_pred,
            spp=self.renderer_spp,
            class_vector=class_vector,
            render_mode='diffuse', 
            two_sided_shading=False, 
            num_frames=num_frames,
            im_features_map={"original_mvp": original_mvp, "im_features_map": im_features_map} if im_features_map is not None else None # in other views we need to pass the original mvp
        )

        mask_pred = mask_pred.unsqueeze(1)
        if self.few_shot_gan_tex_reso != self.out_image_size:
            image_pred = torch.nn.functional.interpolate(image_pred, (self.out_image_size, self.out_image_size), mode='bilinear')
            mask_pred = torch.nn.functional.interpolate(mask_pred, (self.out_image_size, self.out_image_size), mode='bilinear')

        # image_pred = image_pred.clamp(0, 1)
        # mask_pred = mask_pred.clamp(0, 1)  # [B, 1, H, W]

        if background_for_reverse == 'random':
            # as we set a random background for rendering, we also need another random background for input view
            # for background, we use the same as random view: a small resolution then upsample
            random_bg = torch.rand(self.bs, self.nf, 3, self.few_shot_gan_tex_reso, self.few_shot_gan_tex_reso).to(self.device)
            random_bg = torch.nn.functional.interpolate(random_bg.squeeze(1), (self.out_image_size, self.out_image_size), mode='bilinear').unsqueeze(1)
            iv_mask_pred = iv_mask_pred.unsqueeze(2).repeat(1, 1, 3, 1, 1)
            iv_image_pred = iv_image_pred * iv_mask_pred + random_bg * (1. - iv_mask_pred)
            iv_image_pred = iv_image_pred.squeeze(1)

            random_bg_gt = torch.rand(self.bs, self.nf, 3, self.few_shot_gan_tex_reso, self.few_shot_gan_tex_reso).to(self.device)
            random_bg_gt = torch.nn.functional.interpolate(random_bg_gt.squeeze(1), (self.out_image_size, self.out_image_size), mode='bilinear').unsqueeze(1)
            mask_gt = mask_gt.unsqueeze(2).repeat(1, 1, 3, 1, 1)
            image_gt = image_gt * mask_gt + random_bg_gt * (1. - mask_gt)
            image_gt = image_gt.squeeze(1)
        
        elif background_for_reverse == 'random-pure':
            # the background is random but with one color
            random_values = torch.rand(b)
            random_bg = random_values[..., None, None, None, None].repeat(1, 1, 3, self.few_shot_gan_tex_reso, self.few_shot_gan_tex_reso).to(self.device)
            random_bg = torch.nn.functional.interpolate(random_bg.squeeze(1), (self.out_image_size, self.out_image_size), mode='bilinear').unsqueeze(1)
            iv_mask_pred = iv_mask_pred.unsqueeze(2).repeat(1, 1, 3, 1, 1)
            iv_image_pred = iv_image_pred * iv_mask_pred + random_bg * (1. - iv_mask_pred)
            iv_image_pred = iv_image_pred.squeeze(1)

            random_values_gt = torch.rand(b)
            random_bg_gt = random_values_gt[..., None, None, None, None].repeat(1, 1, 3, self.few_shot_gan_tex_reso, self.few_shot_gan_tex_reso).to(self.device)
            random_bg_gt = torch.nn.functional.interpolate(random_bg_gt.squeeze(1), (self.out_image_size, self.out_image_size), mode='bilinear').unsqueeze(1)
            mask_gt = mask_gt.unsqueeze(2).repeat(1, 1, 3, 1, 1)
            image_gt = image_gt * mask_gt + random_bg_gt * (1. - mask_gt)
            image_gt = image_gt.squeeze(1)
        
        elif background_for_reverse == 'none':
            iv_image_pred = iv_image_pred.squeeze(1)
            iv_mask_pred = iv_mask_pred.unsqueeze(2).repeat(1, 1, 3, 1, 1)
            # image_gt = image_gt * mask_gt + random_bg_gt * (1. - mask_gt)
            mask_gt = mask_gt.unsqueeze(2).repeat(1, 1, 3, 1, 1)
            image_gt = image_gt * mask_gt
            image_gt = image_gt.squeeze(1)
        
        else:
            raise NotImplementedError

        # image_gt = torch.nn.functional.interpolate(image_gt, (32, 32), mode='bilinear')
        # image_gt = torch.nn.functional.interpolate(image_gt, (256, 256), mode='bilinear')

        # we need to let discriminator think this reverse view is Real sample
        if self.cfgs.get('few_shot_gan_tex_patch', 0) > 0:
            patch_size = torch.randint(self.few_shot_gan_tex_patch, self.few_shot_gan_tex_patch_max, (1,)).item()
            # random view
            image_pred = self.generate_patch_crop(image_pred, mask_pred, patch_size, self.few_shot_gan_tex_patch_num)
            # input view
            iv_image_pred = self.generate_patch_crop(iv_image_pred, iv_mask_pred.squeeze(1)[:, 0:1, :, :], patch_size, self.few_shot_gan_tex_patch_num)
            # gt view
            image_gt = self.generate_patch_crop(image_gt, mask_gt.squeeze(1)[:, 0:1, :, :], patch_size, self.few_shot_gan_tex_patch_num)

        return_loss = {}
        if self.few_shot_gan_tex:
            # here we compute the fake sample as real loss
            gan_tex_loss = 0.0
            if 'rv' in self.few_shot_gan_tex_fake:
                d_rv = self.discriminator_texture(image_pred)
                gan_tex_loss_rv = discriminator_architecture.bce_loss_target(d_rv, 1)
                gan_tex_loss += gan_tex_loss_rv
            
            if 'iv' in self.few_shot_gan_tex_fake:
                d_iv = self.discriminator_texture(iv_image_pred)
                gan_tex_loss_iv = discriminator_architecture.bce_loss_target(d_iv, 1)
                gan_tex_loss += gan_tex_loss_iv
            
            return_loss['gan_tex_loss'] = gan_tex_loss
        
        if self.few_shot_clip_tex:
            clip_tex_loss_rv_iv = self.compute_clip_loss(image_pred, iv_image_pred.unsqueeze(1), category='none')
            clip_tex_loss_rv_gt = self.compute_clip_loss(image_pred, image_gt.unsqueeze(1), category='none')
            clip_tex_loss = clip_tex_loss_rv_iv['clip_all_loss'] + clip_tex_loss_rv_gt['clip_all_loss']
            return_loss['clip_tex_loss'] = clip_tex_loss

        return_aux = {
            'gan_tex_render_image': image_pred.clone().clamp(0, 1),
            'gan_tex_inpview_image': iv_image_pred.clone().clamp(0, 1),
            'gan_tex_gt_image': image_gt.clone().clamp(0, 1)
        }

        with torch.no_grad():
            # self.record_image_iv = iv_image_pred.clone().clamp(0, 1)
            # self.record_image_rv = image_pred.clone().clamp(0, 1)
            # self.record_image_gt = image_gt.clone().clamp(0, 1)
            self.record_image_iv = iv_image_pred.clone()
            self.record_image_rv = image_pred.clone()
            self.record_image_gt = image_gt.clone()
        
        return return_loss, return_aux

    def compute_mask_distribution_loss(self, category, w2c_pred, shape, prior_shape, texture, dino_pred, im_features, light, class_vector, num_frames, im_features_map, bins=360):
        delta_angle = 2 * np.pi / bins
        b = len(shape)

        if self.random_mask_law == 'batch_swap':
            # shuffle in predicted poses
            rand_degree_1 = torch.randperm(int(w2c_pred.shape[0] // 2))
            rand_degree_2 = torch.randperm(w2c_pred.shape[0] - int(w2c_pred.shape[0] // 2)) + int(w2c_pred.shape[0] // 2)
            rand_degree = torch.cat([rand_degree_2, rand_degree_1], dim=0).long().to(w2c_pred.device)
            w2c = w2c_pred[rand_degree]

            proj = util.perspective(self.crop_fov_approx / 180 * np.pi, 1, n=0.1, f=1000.0).repeat(b, 1, 1).to(self.device)
            mvp = torch.bmm(proj, w2c)
            campos = -w2c[:, :3, 3]
        
        elif self.random_mask_law == 'batch_swap_noy':
            # shuffle in predicted poses
            rand_degree_1 = torch.randperm(int(w2c_pred.shape[0] // 2))
            rand_degree_2 = torch.randperm(w2c_pred.shape[0] - int(w2c_pred.shape[0] // 2)) + int(w2c_pred.shape[0] // 2)
            rand_degree = torch.cat([rand_degree_2, rand_degree_1], dim=0).long().to(w2c_pred.device)
            w2c = w2c_pred[rand_degree]
            # we don't random swap the y-translation in discriminator loss
            w2c[:, 1, 3] = w2c_pred[:, 1, 3]

            proj = util.perspective(self.crop_fov_approx / 180 * np.pi, 1, n=0.1, f=1000.0).repeat(b, 1, 1).to(self.device)
            mvp = torch.bmm(proj, w2c)
            campos = -w2c[:, :3, 3]
        
        elif self.random_mask_law == 'random_azimuth':
            # the render rotation matrix is different
            rand_degree = torch.randint(bins, [b])
            delta_angle = delta_angle * rand_degree
            delta_rot_matrix = []
            for i in range(b):
                angle = delta_angle[i].item()
                angle_matrix = torch.FloatTensor([
                    [np.cos(angle),  0, np.sin(angle), 0],
                    [0,              1, 0,             0],
                    [-np.sin(angle), 0, np.cos(angle), 0],
                    [0,              0, 0,             1],
                ]).to(self.device)
                delta_rot_matrix.append(angle_matrix)
            delta_rot_matrix = torch.stack(delta_rot_matrix, dim=0)
            
            w2c = torch.FloatTensor(np.diag([1., 1., 1., 1]))
            w2c[:3, 3] = torch.FloatTensor([0, 0, -self.cam_pos_z_offset *1.4])
            w2c = w2c.repeat(b, 1, 1).to(self.device)
            # use the predicted transition
            w2c_pred = w2c_pred.detach()
            w2c[:, :3, 3] = w2c_pred[:b][:, :3, 3]

            proj = util.perspective(self.crop_fov_approx / 180 * np.pi, 1, n=0.1, f=1000.0).repeat(b, 1, 1).to(self.device)
            mvp = torch.bmm(proj, w2c)
            campos = -w2c[:, :3, 3]

            mvp = torch.matmul(mvp, delta_rot_matrix)
            campos = torch.matmul(delta_rot_matrix[:,:3,:3].transpose(2,1), campos[:,:,None])[:,:,0]
        
        elif self.random_mask_law == 'random_all':
            # the render rotation matrix is different, and actually the translation are just pre-set
            rand_degree = torch.randint(bins, [b])
            delta_angle = delta_angle * rand_degree
            delta_rot_matrix = []
            for i in range(b):
                angle = delta_angle[i].item()
                angle_matrix = torch.FloatTensor([
                    [np.cos(angle),  0, np.sin(angle), 0],
                    [0,              1, 0,             0],
                    [-np.sin(angle), 0, np.cos(angle), 0],
                    [0,              0, 0,             1],
                ]).to(self.device)
                delta_rot_matrix.append(angle_matrix)
            delta_rot_matrix = torch.stack(delta_rot_matrix, dim=0)
            
            w2c = torch.FloatTensor(np.diag([1., 1., 1., 1]))
            w2c[:3, 3] = torch.FloatTensor([0, 0, -self.cam_pos_z_offset *1.4])
            w2c = w2c.repeat(b, 1, 1).to(self.device)

            proj = util.perspective(self.crop_fov_approx / 180 * np.pi, 1, n=0.1, f=1000.0).repeat(b, 1, 1).to(self.device)
            mvp = torch.bmm(proj, w2c)
            campos = -w2c[:, :3, 3]

            mvp = torch.matmul(mvp, delta_rot_matrix)
            campos = torch.matmul(delta_rot_matrix[:,:3,:3].transpose(2,1), campos[:,:,None])[:,:,0]
        
        else:
            raise NotImplementedError

        resolution = (self.out_image_size, self.out_image_size)
        # render the articulated shape
        mesh = shape
        if self.enable_clip:
            resolution = (self.clip_render_size, self.clip_render_size)
            set_requires_grad(texture, False)
            image_pred, mask_pred, _, _, _, _ = self.render(
                mesh, 
                texture, 
                mvp, 
                w2c, 
                campos, 
                resolution, 
                background='none', 
                im_features=im_features, 
                light=light, 
                prior_shape=prior_shape, 
                render_flow=False, 
                dino_pred=dino_pred,
                spp=self.renderer_spp,
                class_vector=class_vector,
                render_mode='diffuse', 
                two_sided_shading=False, 
                num_frames=num_frames,
                im_features_map=im_features_map
            )
            
            if resolution[0] != self.out_image_size:
                image_pred = torch.nn.functional.interpolate(image_pred, (self.out_image_size, self.out_image_size), mode='bilinear')
                mask_pred = torch.nn.functional.interpolate(mask_pred.unsqueeze(1), (self.out_image_size, self.out_image_size), mode='bilinear').squeeze(1)
        else:
            _, mask_pred, _, _, _, _ = self.render(
                mesh, 
                None, 
                mvp, 
                w2c, 
                campos, 
                resolution, 
                background='none', 
                im_features=None, 
                light=None, 
                prior_shape=prior_shape, 
                render_flow=False, 
                dino_pred=None,
                class_vector=class_vector,
                render_mode='diffuse', 
                two_sided_shading=False, 
                num_frames=num_frames,
                im_features_map=None
            )
            image_pred = None

        # TODO: disable mask distribution and isolate mask discriminator loss
        # mask_distribution = self.class_mask_distribution[category]
        # mask_distribution = torch.Tensor(mask_distribution).to(self.device).unsqueeze(0).repeat(b, 1, 1)
        mask_distribution = torch.Tensor(self.class_mask_distribution["zebra"]).to(self.device).unsqueeze(0).repeat(b, 1, 1)

        if self.mask_distribution_average:
            # if use mask_distribution_average, then first average across batch then compute the loss
            mask_pred = mask_pred.mean(dim=0).unsqueeze(0).repeat(b, 1, 1)
        
        mask_pred = mask_pred.clamp(0,1)
        mask_distribution = mask_distribution.clamp(0,1)
        distribution_loss = torch.nn.functional.binary_cross_entropy(mask_pred, mask_distribution)

        out_loss = {'mask_distribution_loss': 0 * distribution_loss}
        out_aux = {
            'mask_random_pred': mask_pred.unsqueeze(1),
            'mask_distribution': mask_distribution.unsqueeze(1),
            'rand_degree': rand_degree
        }

        if self.enable_clip:
            out_aux.update({'random_render_image': image_pred})

        return out_loss, out_aux

    def use_line_correct_valid_mask(self, mask_valid, p1, p2, mvp, mask_gt):
        line = torch.cat([p1.unsqueeze(-2), p2.unsqueeze(-2)], dim=-2) # [B, 2, 3]
        line_world4 = torch.cat([line, torch.ones_like(line[..., :1])], -1)
        line_clip4 = line_world4 @ mvp.transpose(-1, -2)
        line_uv = line_clip4[..., :2] / line_clip4[..., 3:4]
        line_uv = line_uv.detach()
        b, _, n_uv = line_uv.shape
        line_uv = line_uv * torch.Tensor([mask_valid.shape[-2] // 2, mask_valid.shape[-1] // 2]).to(line_uv.device).unsqueeze(0).unsqueeze(-1).repeat(b, 1, n_uv)
        line_uv = line_uv + torch.Tensor([mask_valid.shape[-2] // 2, mask_valid.shape[-1] // 2]).to(line_uv.device).unsqueeze(0).unsqueeze(-1).repeat(b, 1, n_uv)
        from pdb import set_trace; set_trace()
        line_slope = (line_uv[:, 0, 1] - line_uv[:, 1, 1]) / (line_uv[:, 0, 0] - line_uv[:, 1, 0])

        uv = np.mgrid[0:mask_valid.shape[-2], 0:mask_valid.shape[-1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float().unsqueeze(0).repeat(b, 1, 1, 1) # [B, 2, 256, 256]
        tmp_u = uv[:, 0, ...][mask_gt[:, 0, ...].bool()]
        tmp_v = uv[:, 1, ...][mask_gt[:, 0, ...].bool()]
        return mask_valid

    def discriminator_step(self):
        mask_gt = self.record_mask_gt
        mask_pred = self.record_mask_iv
        mask_random_pred = self.record_mask_rv
        
        self.optimizerDiscriminator.zero_grad()

        # the random view mask are False
        d_random_pred = self.mask_disc(mask_random_pred)
        disc_loss = discriminator_architecture.bce_loss_target(d_random_pred, 0)  # in gen loss, train it to be real

        grad_loss = 0.0
        count = 1

        discriminator_loss_rv = disc_loss.detach()
        discriminator_loss_gt = 0.0
        discriminator_loss_iv = 0.
        d_gt = None
        d_iv = None
        
        if self.disc_gt:
            mask_gt.requires_grad_()
            d_gt = self.mask_disc(mask_gt)
            if d_gt.requires_grad is False:
                # in the test case
                disc_gt_loss = discriminator_architecture.bce_loss_target(d_gt, 1)
            else:
                grad_penalty = self.disc_reg_mul * discriminator_architecture.compute_grad2(d_gt, mask_gt)
                disc_gt_loss = discriminator_architecture.bce_loss_target(d_gt, 1) + grad_penalty
                grad_loss += grad_penalty
            disc_loss = disc_loss + disc_gt_loss
            discriminator_loss_gt = disc_gt_loss
            count = count + 1
        
        if self.disc_iv:
            mask_pred.requires_grad_()
            d_iv = self.mask_disc(mask_pred)
            if self.disc_iv_label == 'Real':
                if d_iv.requires_grad is False:
                    # in the test case
                    disc_iv_loss = discriminator_architecture.bce_loss_target(d_iv, 1)
                else:    
                    grad_penalty = self.disc_reg_mul * discriminator_architecture.compute_grad2(d_iv, mask_pred)
                    disc_iv_loss = discriminator_architecture.bce_loss_target(d_iv, 1) + grad_penalty
                    grad_loss += grad_penalty
                
            else:
                disc_iv_loss = discriminator_architecture.bce_loss_target(d_iv, 0)
            disc_loss = disc_loss + disc_iv_loss
            count = count + 1
            discriminator_loss_iv = disc_iv_loss
        
        disc_loss = disc_loss / count
        grad_loss = grad_loss / count

        self.discriminator_loss = disc_loss * self.discriminator_loss_weight
        self.discriminator_loss.backward()
        self.optimizerDiscriminator.step()
        self.discriminator_loss = 0.
        return {
            'discriminator_loss': disc_loss,
            'discriminator_loss_rv': discriminator_loss_rv,
            'discriminator_loss_iv': discriminator_loss_iv,
            'discriminator_loss_gt': discriminator_loss_gt,
            'd_rv': d_random_pred,
            'd_iv': d_iv if d_iv is not None else None,
            'd_gt': d_gt if d_gt is not None else None,
        }, grad_loss

    def compute_mask_disc_loss_gen(self, mask_gt, mask_pred, mask_random_pred, category_name=None, condition_feat=None):
        # mask_gt[mask_gt < 1.] = 0.
        # mask_pred[mask_pred > 0.] = 1.
        # mask_random_pred[mask_random_pred > 0.] = 1.

        if not self.mask_disc_feat_condition:
            try:
                class_idx = list(self.netPrior.category_id_map.keys()).index(category_name)
            except:
                class_idx = 100
            num_classes = len(list(self.netPrior.category_id_map.keys()))
            class_idx = torch.LongTensor([class_idx])
            # class_one_hot = torch.nn.functional.one_hot(class_idx, num_classes=7).unsqueeze(-1).unsqueeze(-1).to(mask_gt.device) # [1, 7, 1, 1]
            class_one_hot = torch.nn.functional.one_hot(class_idx, num_classes=num_classes).unsqueeze(-1).unsqueeze(-1).to(mask_gt.device)
            class_one_hot = class_one_hot.repeat(mask_gt.shape[0], 1, mask_gt.shape[-2], mask_gt.shape[-1])
            # TODO: a hack try here
            class_one_hot = class_one_hot[:, :(self.mask_disc.in_dim-1), :, :]
        else:
            class_one_hot = condition_feat.detach()
            class_one_hot = class_one_hot.reshape(1, -1, 1, 1).repeat(mask_gt.shape[0], 1, mask_gt.shape[-2], mask_gt.shape[-1])

        # concat
        mask_gt = torch.cat([mask_gt, class_one_hot], dim=1)
        mask_pred = torch.cat([mask_pred, class_one_hot], dim=1)
        mask_random_pred = torch.cat([mask_random_pred, class_one_hot], dim=1)
        
        # mask shape are all [B,1,256,256]
        # the random view mask are False
        d_random_pred = self.mask_disc(mask_random_pred)
        disc_loss = discriminator_architecture.bce_loss_target(d_random_pred, 1)  # in gen loss, train it to be real
        count = 1

        disc_loss_rv = disc_loss.detach()
        disc_loss_iv = 0.0
            
        if self.disc_iv:
            if self.disc_iv_label != 'Real': # consider the input view also fake
                d_iv = self.mask_disc(mask_pred)
                disc_iv_loss = discriminator_architecture.bce_loss_target(d_iv, 1) # so now we need to train them to be real
                disc_loss = disc_loss + disc_iv_loss
                count = count + 1
                disc_loss_iv = disc_iv_loss.detach()
        
        disc_loss = disc_loss / count

        # record the masks for discriminator training
        self.record_mask_gt = mask_gt.clone().detach()
        self.record_mask_iv = mask_pred.clone().detach()
        self.record_mask_rv = mask_random_pred.clone().detach()

        return {
            'mask_disc_loss': disc_loss,
            'mask_disc_loss_rv': disc_loss_rv,
            'mask_disc_loss_iv': disc_loss_iv,
        }

    def forward(self, batch, epoch, iter, is_train=True, viz_logger=None, total_iter=None, save_results=False, save_dir=None, which_data='', logger_prefix='', is_training=True, bank_embedding=None):
        batch = [x.to(self.device) if x is not None and isinstance(x, torch.Tensor) else x for x in batch]
        input_image, mask_gt, mask_dt, mask_valid, flow_gt, bbox, bg_image, dino_feat_im, dino_cluster_im, seq_idx, frame_idx, category_name = batch
        
        # if save_results:
        #     save_for_pkl = {
        #         "image": input_image.cpu(),
        #         "mask_gt": mask_gt.cpu(),
        #         "mask_dt": mask_dt.cpu(),
        #         "mask_valid": mask_valid.cpu(),
        #         "flow_gt": None,
        #         "bbox": bbox.cpu(),
        #         "bg_image": bg_image.cpu(),
        #         "dino_feat_im": dino_feat_im.cpu(),
        #         "dino_cluster_im": dino_cluster_im.cpu(),
        #         "seq_idx": seq_idx.cpu(),
        #         "frame_idx": frame_idx.cpu(),
        #         "category_name": category_name
        #     }
        
        batch_size, num_frames, _, h0, w0 = input_image.shape  # BxFxCxHxW
        self.bs = batch_size
        self.nf = num_frames
        mid_img_idx = int((input_image.shape[1]-1)//2)
        # print(f"mid_img_idx: {mid_img_idx}")

        h = w = self.out_image_size

        def collapseF(x):
            return None if x is None else x.view(batch_size * num_frames, *x.shape[2:])
        def expandF(x):
            return None if x is None else x.view(batch_size, num_frames, *x.shape[1:])
        
        if flow_gt.dim() == 2:  # dummy tensor for not loading flow
            flow_gt = None

        if dino_cluster_im.dim() == 2:  # dummy tensor for not loading dino clusters
            dino_cluster_im = None
            dino_cluster_im_gt = None
        else:
            dino_cluster_im_gt = expandF(torch.nn.functional.interpolate(collapseF(dino_cluster_im), size=[h, w], mode="nearest"))
        
        seq_idx = seq_idx.squeeze(1)
        # seq_idx = seq_idx * 0  # single sequnce model
        frame_id, crop_x0, crop_y0, crop_w, crop_h, full_w, full_h, sharpness, label = bbox.unbind(2)  # BxFx7
        bbox = torch.stack([crop_x0, crop_y0, crop_w, crop_h], 2)
        mask_gt = (mask_gt[:, :, 0, :, :] > 0.9).float()  # BxFxHxW
        mask_dt = mask_dt / self.in_image_size

        if which_data != 'video':
            flow_gt = None

        aux_viz = {}

        ## GT
        image_gt = input_image
        if self.out_image_size != self.in_image_size:
            image_gt = expandF(torch.nn.functional.interpolate(collapseF(image_gt), size=[h, w], mode='bilinear'))
            if flow_gt is not None:
                flow_gt = torch.nn.functional.interpolate(flow_gt.view(batch_size*(num_frames-1), 2, h0, w0), size=[h, w], mode="bilinear").view(batch_size, num_frames-1, 2, h, w)

        self.train_pose_only = False
        if epoch in self.pose_epochs:
            if (total_iter // self.pose_iters) % 2 == 0:
                self.train_pose_only = True
        
        ## flip input and pose
        if epoch in self.pose_xflip_recon_epochs:
            input_image_xflip = input_image.flip(-1)
            input_image_xflip_flag = torch.randint(0, 2, (batch_size, num_frames), device=input_image.device)
            input_image = input_image * (1 - input_image_xflip_flag[:,:,None,None,None]) + input_image_xflip * input_image_xflip_flag[:,:,None,None,None]
        else:
            input_image_xflip_flag = None

        ## 1st pose hypothesis with original predictions

        # ==============================================================================================
        #  Predict prior mesh.
        # ==============================================================================================
        if self.enable_prior:
            if self.world_size > 1:
                if epoch < self.dmtet_grid_smaller_epoch:
                    if self.netPrior_ddp.module.netShape.grid_res != self.dmtet_grid_smaller:
                        self.netPrior_ddp.module.netShape.load_tets(self.dmtet_grid_smaller)
                else:
                    if self.netPrior_ddp.module.netShape.grid_res != self.dmtet_grid:
                        self.netPrior_ddp.module.netShape.load_tets(self.dmtet_grid)
            
            else:
                if epoch < self.dmtet_grid_smaller_epoch:
                    if self.netPrior.netShape.grid_res != self.dmtet_grid_smaller:
                        self.netPrior.netShape.load_tets(self.dmtet_grid_smaller)
                else:
                    if self.netPrior.netShape.grid_res != self.dmtet_grid:
                        self.netPrior.netShape.load_tets(self.dmtet_grid)
            
            perturb_sdf = self.perturb_sdf if is_train else False
            # DINO prior category specific - DOR 
            if self.world_size > 1:
                prior_shape, dino_pred, classes_vectors = self.netPrior_ddp(category_name=category_name[0], perturb_sdf=perturb_sdf, total_iter=total_iter, is_training=is_training, class_embedding=bank_embedding)
            else:
                prior_shape, dino_pred, classes_vectors = self.netPrior(category_name=category_name[0], perturb_sdf=perturb_sdf, total_iter=total_iter, is_training=is_training, class_embedding=bank_embedding)
        else:
            prior_shape = None
            raise NotImplementedError
        
        if self.world_size > 1:
            shape, pose_raw, pose, mvp, w2c, campos, texture, im_features, dino_feat_im_calc, deformation, arti_params, light, forward_aux = self.netInstance_ddp(category_name, input_image, prior_shape, epoch, dino_feat_im, dino_cluster_im, total_iter, is_training=is_training)  # frame dim collapsed N=(B*F)
        else:
            Instance_out = self.netInstance(category_name, input_image, prior_shape, epoch, dino_feat_im, dino_cluster_im, total_iter, is_training=is_training)  # frame dim collapsed N=(B*F)
            
            # if no patch_out as output from netInstance, then set im_features_map as None in following part
            if len(Instance_out) == 13:
                shape, pose_raw, pose, mvp, w2c, campos, texture, im_features, dino_feat_im_calc, deformation, arti_params, light, forward_aux = Instance_out
                im_features_map = None
            else:
                shape, pose_raw, pose, mvp, w2c, campos, texture, im_features, dino_feat_im_calc, deformation, arti_params, light, forward_aux, im_features_map = Instance_out
        
        # if save_results:
        #     save_for_pkl.update(
        #         {
        #             "pose_raw": pose_raw.cpu(),
        #             "pose": pose.cpu(),
        #             "mvp": mvp.cpu(),
        #             "w2c": w2c.cpu(),
        #             "campos": campos.cpu(),
        #             "campos_z_offset": self.netInstance.cam_pos_z_offset
        #         }
        #     )
        
        if self.calc_dino_features == True:

           # get the shape parameters of the tensor
            batch_size, height, width, channels = dino_feat_im_calc.shape #3 X 384 X 32 X 32 
            

            # reshape the tensor to have 2 dimensions, with the last dimension being preserved
            dino_feat_im = dino_feat_im_calc.reshape(batch_size , height, -1)

            # normalize the tensor using L2 normalization
            norm = torch.norm(dino_feat_im, dim=-1, keepdim=True)
            
            dino_feat_im = dino_feat_im / norm
            
            # reshape the tensor back to the original shape with an additional singleton dimension along the first dimension
            dino_feat_im = dino_feat_im.reshape(batch_size, height, width, channels)
            dino_feat_im = dino_feat_im.unsqueeze(1)
        
        
        if dino_feat_im.dim() == 2:  # dummy tensor for not loading dino features
            dino_feat_im = None
            dino_feat_im_gt = None
        else:
            dino_feat_im_gt = expandF(torch.nn.functional.interpolate(collapseF(dino_feat_im), size=[h, w], mode="bilinear"))[:, :, :self.dino_feature_recon_dim]
                
        rot_logit = forward_aux['rot_logit']
        rot_idx = forward_aux['rot_idx']
        rot_prob = forward_aux['rot_prob']

        if self.using_bonevel_smooth_loss:
            posed_bones = forward_aux['posed_bones']
        else: 
            posed_bones = None
        
        aux_viz.update(forward_aux)

        if self.train_pose_only:
            safe_detach = lambda x: x.detach() if x is not None else None
            prior_shape = safe_detach(prior_shape)
            shape = safe_detach(shape)
            im_features = safe_detach(im_features)
            arti_params = safe_detach(arti_params)
            deformation = safe_detach(deformation)
            set_requires_grad(texture, False)
            set_requires_grad(light, False)
            set_requires_grad(dino_pred, False)
        else:
            set_requires_grad(texture, True)
            set_requires_grad(light, True)
            set_requires_grad(dino_pred, True)

        render_flow = self.render_flow and num_frames > 1 #false
        # from IPython import embed; embed()

        # if num_frames > 1 and self.smooth_type == 'rend': 
        #     print("rendererr smoothness !!!!")
        #     image_pred, mask_pred, flow_pred, dino_feat_im_pred, albedo, shading = self.render(shape, texture, mvp, w2c, campos, (h, w), background=self.background_mode, im_features=im_features[torch.randperm(im_features.size(0))], light=light, prior_shape=prior_shape, render_flow=render_flow, dino_pred=dino_pred, num_frames=num_frames, spp=self.renderer_spp) #the real rendering process
        # else:
        #     print("regular render")
        #print("a cecond before rendering .... need to get the correct label and the correct vector")
        #print("label", label)
        #print("classes_vectors", classes_vectors)
        #print("im_features", im_features.shape)
        
        class_vector = None
        if classes_vectors is not None:
            if len(classes_vectors.shape) == 1:
                class_vector = classes_vectors
            else:
                class_vector = classes_vectors[self.netPrior.category_id_map[category_name[0]], :]
            
        image_pred, mask_pred, flow_pred, dino_feat_im_pred, albedo, shading = self.render(shape, texture, mvp, w2c, campos, (h, w), background=self.background_mode, im_features=im_features, light=light, prior_shape=prior_shape, render_flow=render_flow, dino_pred=dino_pred, class_vector=class_vector[None, :].expand(batch_size * num_frames, -1), num_frames=num_frames, spp=self.renderer_spp, im_features_map=im_features_map) #the real rendering process
        image_pred, mask_pred, flow_pred, dino_feat_im_pred = map(expandF, (image_pred, mask_pred, flow_pred, dino_feat_im_pred))
                
        if flow_pred is not None:
            flow_pred = flow_pred[:, :-1]  # Bx(F-1)x2xHxW

        if self.blur_mask:
            sigma = max(0.5, 3 * (1 - total_iter / self.blur_mask_iter))
            if sigma > 0.5:
                mask_gt = util.blur_image(mask_gt, kernel_size=9, sigma=sigma, mode='gaussian')
            # mask_pred = util.blur_image(mask_pred, kernel_size=7, mode='average')

        # back_line_p1 = forward_aux['posed_bones'][:, :, 3, -1].squeeze(1)  # [8, 3]
        # back_line_p2 = forward_aux['posed_bones'][:, :, 7, -1].squeeze(1)
        # mask_valid = self.use_line_correct_valid_mask(mask_valid, back_line_p1, back_line_p2, mvp, mask_gt)

        losses = self.compute_reconstruction_losses(image_pred, image_gt, mask_pred, mask_gt, mask_dt, mask_valid, flow_pred, flow_gt, dino_feat_im_gt, dino_feat_im_pred, background_mode=self.background_mode, reduce=False)
        
        ## TODO: assume flow loss is not used
        logit_loss_target = torch.zeros_like(expandF(rot_logit))
        final_losses = {}
        for name, loss in losses.items():
            if name == 'flow_loss':
                continue
            loss_weight_logit = self.cfgs.get(f"{name}_weight", 0.)
            
            if isinstance(loss_weight_logit, dict):
                loss_weight_logit = self.parse_dict_definition(loss_weight_logit, total_iter)

            # from IPython import embed; embed()
            # print("-"*10)
            # print(f"{name}_weight: {loss_weight_logit}.")
            # print(f"logit_loss_target.shape: {logit_loss_target.shape}.")
            # print(f"loss.shape: {loss.shape}.")
            # if (name in ['flow_loss'] and epoch not in self.flow_loss_epochs) or (name in ['rgb_loss', 'perceptual_loss'] and epoch not in self.texture_epochs):
            # if name in ['flow_loss', 'rgb_loss', 'perceptual_loss']:
            #     loss_weight_logit = 0.
            if name in ['sdf_bce_reg_loss', 'sdf_gradient_reg_loss', 'sdf_inflate_reg_loss']:
                if total_iter >= self.sdf_reg_decay_start_iter:
                    decay_rate = max(0, 1 - (total_iter-self.sdf_reg_decay_start_iter) / 10000)
                    loss_weight_logit = max(loss_weight_logit * decay_rate, self.cfgs.get(f"{name}_min_weight", 0.))
            if name in ['dino_feat_im_loss']:
                dino_feat_im_loss_multipler = self.cfgs.get("logit_loss_dino_feat_im_loss_multiplier", 1.)
                
                if isinstance(dino_feat_im_loss_multipler, dict):
                    dino_feat_im_loss_multipler = self.parse_dict_definition(dino_feat_im_loss_multipler, total_iter)
                
                loss_weight_logit = loss_weight_logit * dino_feat_im_loss_multipler
                # loss_weight_logit = loss_weight_logit * self.cfgs.get("logit_loss_dino_feat_im_loss_multiplier", 1.)
            if loss_weight_logit > 0:
                logit_loss_target += loss * loss_weight_logit
            
            if self.netInstance.rot_rep in ['quadlookat', 'octlookat']:
                loss = loss * rot_prob.detach().view(batch_size, num_frames)[:, :loss.shape[1]] *self.netInstance.num_pose_hypos
            if name == 'flow_loss' and num_frames > 1:
                ri = rot_idx.view(batch_size, num_frames)
                same_rot_idx = (ri[:, 1:] == ri[:, :-1]).float()
                loss = loss * same_rot_idx
            final_losses[name] = loss.mean()
        final_losses['logit_loss'] = ((expandF(rot_logit) - logit_loss_target.detach())**2.).mean()

        ## mask distribution loss
        mask_distribution_aux = None
        if self.enable_mask_distribution:
            if total_iter % self.mask_distribution_loss_freq == 0:
                mask_distribution_loss, mask_distribution_aux = self.compute_mask_distribution_loss(category_name[0], w2c, shape, prior_shape, texture, dino_pred, im_features, light, class_vector[None, :].expand(batch_size * num_frames, -1), num_frames, im_features_map)
                final_losses.update(mask_distribution_loss)
                # this also follows the iteration frequency
                if self.enable_clip:
                    random_render_image = mask_distribution_aux["random_render_image"]
                    clip_all_loss = self.compute_clip_loss(random_render_image, image_pred, category_name[0])  # a dict
                    final_losses.update(clip_all_loss)
        
        # implement the mask discriminator
        if self.enable_disc and (self.mask_discriminator_iter[0] < total_iter) and (self.mask_discriminator_iter[1] > total_iter):
            disc_loss = self.compute_mask_disc_loss_gen(mask_gt, mask_pred, mask_distribution_aux['mask_random_pred'], category_name=category_name[0], condition_feat=class_vector)
            final_losses.update(disc_loss)
        
        # implement the gan training for local texture in fine-tuning
        gan_tex_aux = None
        if (self.few_shot_gan_tex and viz_logger is None) or (self.few_shot_gan_tex and viz_logger is not None and logger_prefix == 'train_'):
            gan_tex_loss, gan_tex_aux = self.compute_gan_tex_loss(category_name[0], image_gt, mask_gt, image_pred, mask_pred, w2c, campos, shape, prior_shape, texture, dino_pred, im_features, light, class_vector[None, :].expand(batch_size * num_frames, -1), num_frames, im_features_map)
            final_losses.update(gan_tex_loss)
        
        # implement the memory bank related loss
        if bank_embedding is not None:
            batch_embedding = bank_embedding[0]     # [d]
            embeddings = bank_embedding[1]          # [B, d]
            bank_mean_dist = torch.nn.functional.mse_loss(embeddings, batch_embedding.unsqueeze(0).repeat(batch_size, 1))
            final_losses.update({'bank_mean_dist_loss': bank_mean_dist})


        ## regularizers
        regularizers, aux = self.compute_regularizers(shape, prior_shape, input_image, dino_feat_im, pose_raw, input_image_xflip_flag, arti_params, deformation, mid_img_idx, posed_bones=posed_bones, class_vector=class_vector.detach() if class_vector is not None else None)
        final_losses.update(regularizers)
        aux_viz.update(aux)

        total_loss = 0
        for name, loss in final_losses.items():
            loss_weight = self.cfgs.get(f"{name}_weight", 0.)

            if isinstance(loss_weight, dict):
                loss_weight = self.parse_dict_definition(loss_weight, total_iter)

            if loss_weight <= 0:
                continue
            
            if self.train_pose_only:
                if name not in ['silhouette_loss', 'silhouette_dt_loss', 'silhouette_inv_dt_loss', 'flow_loss', 'pose_xflip_reg_loss', 'lookat_zflip_loss', 'dino_feat_im_loss']:
                    continue
            if epoch not in self.flow_loss_epochs:
                if name in ['flow_loss']:
                    continue
            if epoch not in self.texture_epochs:
                if name in ['rgb_loss', 'perceptual_loss']:
                    continue
            if epoch not in self.lookat_zflip_loss_epochs:
                if name in ['lookat_zflip_loss']:
                    continue
            if name in ['mesh_laplacian_smoothing_loss', 'mesh_normal_consistency_loss']:
                if total_iter < self.cfgs.get('mesh_reg_start_iter', 0):
                    continue
                if epoch >= self.mesh_reg_decay_epoch:
                    decay_rate = self.mesh_reg_decay_rate ** (epoch - self.mesh_reg_decay_epoch)
                    loss_weight = max(loss_weight * decay_rate, self.cfgs.get(f"{name}_min_weight", 0.))
            if epoch not in self.sdf_inflate_reg_loss_epochs:
                if name in ['sdf_inflate_reg_loss']:
                    continue
            if self.iter_arti_reg_loss_start is not None:
                if total_iter <= self.iter_arti_reg_loss_start:
                    if name in ['arti_reg_loss']:
                        continue
            else:
                if epoch not in self.arti_reg_loss_epochs:
                    if name in ['arti_reg_loss']:
                        continue
            if name in ['sdf_bce_reg_loss', 'sdf_gradient_reg_loss', 'sdf_inflate_reg_loss']:
                if total_iter >= self.sdf_reg_decay_start_iter:
                    decay_rate = max(0, 1 - (total_iter-self.sdf_reg_decay_start_iter) / 10000)
                    loss_weight = max(loss_weight * decay_rate, self.cfgs.get(f"{name}_min_weight", 0.))
            
            total_loss += loss * loss_weight

        self.total_loss += total_loss  # reset to 0 in backward step

        if torch.isnan(self.total_loss):
            print("NaN in loss...")
            import ipdb; ipdb.set_trace()
        
        final_losses['logit_loss_target'] = logit_loss_target.mean()

        metrics = {'loss': total_loss, **final_losses}
        ## log visuals
        if viz_logger is not None:
            b0 = max(min(batch_size, 16//num_frames), 1)
            viz_logger.add_image(logger_prefix+'image/image_gt', misc.image_grid(image_gt.detach().cpu()[:b0,:].reshape(-1,*input_image.shape[2:]).clamp(0,1)), total_iter)
            viz_logger.add_image(logger_prefix+'image/image_pred', misc.image_grid(image_pred.detach().cpu()[:b0,:].reshape(-1,*image_pred.shape[2:]).clamp(0,1)), total_iter)
            # viz_logger.add_image(logger_prefix+'image/flow_loss_mask', misc.image_grid(flow_loss_mask[:b0,:,:1].reshape(-1,1,*flow_loss_mask.shape[3:]).repeat(1,3,1,1).clamp(0,1)), total_iter)
            viz_logger.add_image(logger_prefix+'image/mask_gt', misc.image_grid(mask_gt.detach().cpu()[:b0,:].reshape(-1,*mask_gt.shape[2:]).unsqueeze(1).repeat(1,3,1,1).clamp(0,1)), total_iter)
            viz_logger.add_image(logger_prefix+'image/mask_pred', misc.image_grid(mask_pred.detach().cpu()[:b0,:].reshape(-1,*mask_pred.shape[2:]).unsqueeze(1).repeat(1,3,1,1).clamp(0,1)), total_iter)

            if self.render_flow and flow_gt is not None:
            # if False:
                flow_gt = flow_gt.detach().cpu()
                flow_gt_viz = torch.cat([flow_gt[:b0], torch.zeros_like(flow_gt[:b0,:,:1])], 2) + 0.5  # -0.5~1.5
                flow_gt_viz = torch.nn.functional.pad(flow_gt_viz, pad=[0, 0, 0, 0, 0, 0, 0, 1])

                # ## draw marker on large flow frames
                # large_flow_marker_mask = torch.zeros_like(flow_gt_viz)
                # large_flow_marker_mask[:,:,:,:8,:8] = 1.
                # large_flow = torch.cat([self.large_flow, self.large_flow[:,:1] *0.], 1).detach().cpu()[:b0]
                # large_flow_marker_mask = large_flow_marker_mask * large_flow[:,:,None,None,None]
                # red = torch.FloatTensor([1,0,0])[None,None,:,None,None]
                # flow_gt_viz = large_flow_marker_mask * red + (1-large_flow_marker_mask) * flow_gt_viz
                
                viz_logger.add_image(logger_prefix+'image/flow_gt', misc.image_grid(flow_gt_viz.reshape(-1,*flow_gt_viz.shape[2:])), total_iter)
            
            if self.render_flow and flow_pred is not None:
            # if False
                flow_pred = flow_pred.detach().cpu()
                flow_pred_viz = torch.cat([flow_pred[:b0], torch.zeros_like(flow_pred[:b0,:,:1])], 2) + 0.5  # -0.5~1.5
                flow_pred_viz = torch.nn.functional.pad(flow_pred_viz, pad=[0, 0, 0, 0, 0, 0, 0, 1])
                viz_logger.add_image(logger_prefix+'image/flow_pred', misc.image_grid(flow_pred_viz.reshape(-1,*flow_pred_viz.shape[2:])), total_iter)
            
            if sds_random_images is not None:
                viz_logger.add_image(
                    logger_prefix + 'image/sds_image', 
                    self.vis_sds_image(sds_random_images, sds_aux), 
                    total_iter)
                viz_logger.add_image(
                    logger_prefix + 'image/sds_grad', 
                    self.vis_sds_grads(sds_aux), total_iter)
            
            if mask_distribution_aux is not None:
                degree_text = mask_distribution_aux['rand_degree']
                mask_random_pred = mask_distribution_aux['mask_random_pred'].detach().cpu().clamp(0, 1)
                mask_distribution_data = mask_distribution_aux['mask_distribution'].detach().cpu().clamp(0, 1)
                
                mask_random_pred_image = [misc.add_text_to_image(img, str(text.item())) for img, text in zip(mask_random_pred, degree_text)]
                mask_random_pred_image = misc.image_grid(mask_random_pred_image)
                mask_distribution_image = misc.image_grid(mask_distribution_data)

                viz_logger.add_image(
                    logger_prefix + 'image/mask_random_pred', 
                    mask_random_pred_image, 
                    total_iter)
                viz_logger.add_image(
                    logger_prefix + 'image/mask_distribution', 
                    mask_distribution_image, 
                    total_iter)
            
            if gan_tex_aux is not None:
                gan_tex_render_image = gan_tex_aux['gan_tex_render_image'].detach().cpu().clamp(0, 1)
                gan_tex_render_image = misc.image_grid(gan_tex_render_image)
                viz_logger.add_image(
                    logger_prefix + 'image/gan_tex_render_image', 
                    gan_tex_render_image, 
                    total_iter)
                
                gan_tex_render_image_iv = gan_tex_aux['gan_tex_inpview_image'].detach().cpu().clamp(0, 1)
                gan_tex_render_image_iv = misc.image_grid(gan_tex_render_image_iv)
                viz_logger.add_image(
                    logger_prefix + 'image/gan_tex_inpview_image', 
                    gan_tex_render_image_iv, 
                    total_iter)
                
                gan_tex_render_image_gt = gan_tex_aux['gan_tex_gt_image'].detach().cpu().clamp(0, 1)
                gan_tex_render_image_gt = misc.image_grid(gan_tex_render_image_gt)
                viz_logger.add_image(
                    logger_prefix + 'image/gan_tex_gt_image', 
                    gan_tex_render_image_gt, 
                    total_iter)
            
            # if self.render_flow and flow_gt is not None and flow_pred is not None:
            #     flow_gt = flow_gt.detach().cpu()
            #     # flow_gt_viz = torch.cat([flow_gt[:b0], torch.zeros_like(flow_gt[:b0,:,:1])], 2) + 0.5  # -0.5~1.5
            #     # flow_gt_viz = torch.nn.functional.pad(flow_gt_viz, pad=[0, 0, 0, 0, 0, 0, 0, 1])

            #     # ## draw marker on large flow frames
            #     # large_flow_marker_mask = torch.zeros_like(flow_gt_viz)
            #     # large_flow_marker_mask[:,:,:,:8,:8] = 1.
            #     # large_flow = torch.cat([self.large_flow, self.large_flow[:,:1] *0.], 1).detach().cpu()[:b0]
            #     # large_flow_marker_mask = large_flow_marker_mask * large_flow[:,:,None,None,None]
            #     # red = torch.FloatTensor([1,0,0])[None,None,:,None,None]
            #     # flow_gt_viz = large_flow_marker_mask * red + (1-large_flow_marker_mask) * flow_gt_viz
                
            #     # viz_logger.add_image(logger_prefix+'image/flow_gt', misc.image_grid(flow_gt_viz.reshape(-1,*flow_gt_viz.shape[2:])), total_iter)

            #     flow_pred = flow_pred.detach().cpu()
            #     # flow_pred_viz = torch.cat([flow_pred[:b0], torch.zeros_like(flow_pred[:b0,:,:1])], 2) + 0.5  # -0.5~1.5
            #     # flow_pred_viz = torch.nn.functional.pad(flow_pred_viz, pad=[0, 0, 0, 0, 0, 0, 0, 1])

            #     flow_gt_pred = torch.cat([flow_gt, flow_pred], dim=-1)
            #     flow_gt_pred = flow_gt_pred.permute(0,1,3,4,2).detach().cpu().reshape(flow_gt_pred.shape[0]*flow_gt_pred.shape[1],*flow_gt_pred.shape[2:])
            #     flow_gt_pred = flow_viz.flow_batch_to_images(flow_gt_pred)
            #     # flow_gt_pred = torch.tensor(flow_gt_pred).permute(0,3,1,2)

            #     # viz_logger.add_image(logger_prefix+'image/flow_gt_pred', misc.image_grid(flow_gt_pred.reshape(-1,*flow_gt_pred.shape[2:])), total_iter)
            #     viz_logger.add_image(logger_prefix+'image/flow_gt_pred', misc.image_grid(flow_gt_pred), total_iter)
            
            if light is not None:
                param_names = ['dir_x', 'dir_y', 'dir_z', 'int_ambient', 'int_diffuse']
                for name, param in zip(param_names, light.light_params.unbind(-1)):
                    viz_logger.add_histogram(logger_prefix+'light/'+name, param, total_iter)
                viz_logger.add_image(
                        logger_prefix + f'image/albedo',
                        misc.image_grid(expandF(albedo)[:b0, ...].view(-1, *albedo.shape[1:])),
                        total_iter)
                viz_logger.add_image(
                        logger_prefix + f'image/shading',
                        misc.image_grid(expandF(shading)[:b0, ...].view(-1, *shading.shape[1:]).repeat(1, 3, 1, 1) /2.),
                        total_iter)

            viz_logger.add_histogram(logger_prefix+'sdf', self.netPrior.netShape.get_sdf(perturb_sdf=False, class_vector=class_vector), total_iter)
            viz_logger.add_histogram(logger_prefix+'coordinates', shape.v_pos, total_iter)
            if arti_params is not None:
                viz_logger.add_histogram(logger_prefix+'arti_params', arti_params, total_iter)
                viz_logger.add_histogram(logger_prefix+'edge_lengths', aux_viz['edge_lengths'], total_iter)
            
            if deformation is not None:
                viz_logger.add_histogram(logger_prefix+'deformation', deformation, total_iter)
            
            rot_rep = self.netInstance.rot_rep
            if rot_rep == 'euler_angle' or rot_rep == 'soft_calss':
                for i, name in enumerate(['rot_x', 'rot_y', 'rot_z', 'trans_x', 'trans_y', 'trans_z']):
                    viz_logger.add_histogram(logger_prefix+'pose/'+name, pose[...,i], total_iter)
            elif rot_rep == 'quaternion':
                for i, name in enumerate(['qt_0', 'qt_1', 'qt_2', 'qt_3', 'trans_x', 'trans_y', 'trans_z']):
                    viz_logger.add_histogram(logger_prefix+'pose/'+name, pose[...,i], total_iter)
                rot_euler = pytorch3d.transforms.matrix_to_euler_angles(pytorch3d.transforms.quaternion_to_matrix(pose.detach().cpu()[...,:4]), convention='XYZ')
                for i, name in enumerate(['rot_x', 'rot_y', 'rot_z']):
                    viz_logger.add_histogram(logger_prefix+'pose/'+name, rot_euler[...,i], total_iter)
            elif rot_rep in ['lookat', 'quadlookat', 'octlookat']:
                for i, name in enumerate(['fwd_x', 'fwd_y', 'fwd_z']):
                    viz_logger.add_histogram(logger_prefix+'pose/'+name, pose_raw[...,i], total_iter)
                for i, name in enumerate(['trans_x', 'trans_y', 'trans_z']):
                    viz_logger.add_histogram(logger_prefix+'pose/'+name, pose_raw[...,-3+i], total_iter)
            
            if rot_rep in ['quadlookat', 'octlookat']:
                for i, rp in enumerate(forward_aux['rots_probs'].unbind(-1)):
                    viz_logger.add_histogram(logger_prefix+'pose/rot_prob_%d'%i, rp, total_iter)
            
            if bank_embedding is not None:
                weights_for_emb = bank_embedding[2]['weights'] # [B, k]
                for i, weight_for_emb in enumerate(weights_for_emb.unbind(-1)):
                    viz_logger.add_histogram(logger_prefix+'bank_embedding/emb_weight_%d'%i, weight_for_emb, total_iter)
                
                indices_for_emb = bank_embedding[2]['pick_idx'] # [B, k]
                for i, idx_for_emb in enumerate(indices_for_emb.unbind(-1)):
                    viz_logger.add_histogram(logger_prefix+'bank_embedding/emb_idx_%d'%i, idx_for_emb, total_iter)


            if 'pose_xflip_raw' in aux_viz:
                pose_xflip_raw = aux_viz['pose_xflip_raw']
                if rot_rep == 'euler_angle' or rot_rep == 'soft_calss':
                    for i, name in enumerate(['rot_x', 'rot_y', 'rot_z', 'trans_x', 'trans_y', 'trans_z']):
                        viz_logger.add_histogram(logger_prefix+'pose_xflip/'+name, pose_xflip[...,i], total_iter)
                elif rot_rep == 'quaternion':
                    for i, name in enumerate(['qt_0', 'qt_1', 'qt_2', 'qt_3', 'trans_x', 'trans_y', 'trans_z']):
                        viz_logger.add_histogram(logger_prefix+'pose_xflip/'+name, pose_xflip[...,i], total_iter)
                    rot_euler = pytorch3d.transforms.matrix_to_euler_angles(pytorch3d.transforms.quaternion_to_matrix(pose_xflip.detach().cpu()[...,:4]), convention='XYZ')
                    for i, name in enumerate(['rot_x', 'rot_y', 'rot_z']):
                        viz_logger.add_histogram(logger_prefix+'pose_xflip/'+name, rot_euler[...,i], total_iter)
                elif rot_rep in ['lookat', 'quadlookat', 'octlookat']:
                    for i, name in enumerate(['fwd_x', 'fwd_y', 'fwd_z']):
                        viz_logger.add_histogram(logger_prefix+'pose_xflip/'+name, pose_xflip_raw[...,i], total_iter)
                    for i, name in enumerate(['trans_x', 'trans_y', 'trans_z']):
                        viz_logger.add_histogram(logger_prefix+'pose_xflip/'+name, pose_xflip_raw[...,-3+i], total_iter)

            if dino_feat_im_gt is not None:
                dino_feat_im_gt_first3 = dino_feat_im_gt[:,:,:3]
                viz_logger.add_image(logger_prefix+'image/dino_feat_im_gt', misc.image_grid(dino_feat_im_gt_first3.detach().cpu()[:b0,:].reshape(-1,*dino_feat_im_gt_first3.shape[2:]).clamp(0,1)), total_iter)

            if dino_cluster_im_gt is not None:
                viz_logger.add_image(logger_prefix+'image/dino_cluster_im_gt', misc.image_grid(dino_cluster_im_gt.detach().cpu()[:b0,:].reshape(-1,*dino_cluster_im_gt.shape[2:]).clamp(0,1)), total_iter)
                
            if dino_feat_im_pred is not None:
                dino_feat_im_pred_first3 = dino_feat_im_pred[:,:,:3]
                viz_logger.add_image(logger_prefix+'image/dino_feat_im_pred', misc.image_grid(dino_feat_im_pred_first3.detach().cpu()[:b0,:].reshape(-1,*dino_feat_im_pred_first3.shape[2:]).clamp(0,1)), total_iter)
            
            for which_shape, modes in self.extra_renders.items():
                # This is wrong
                # if which_shape == "prior":
                #     shape_to_render = prior_shape.extend(im_features.shape[0])
                #     needed_im_features = None
                if which_shape == "instance":
                    shape_to_render = shape
                    needed_im_features = im_features
                else:
                    raise NotImplementedError
                
                for mode in modes:
                    if mode in ['gray']:
                        gray_light = FixedDirectionLight(direction=torch.FloatTensor([0, 0, 1]).to(self.device), amb=0.2, diff=0.7)
                        _, render_mask, _, _, _, rendered = self.render(shape_to_render, texture, mvp, w2c, campos, (h, w), background=self.background_mode, im_features=needed_im_features, prior_shape=prior_shape, render_mode='diffuse', light=gray_light, render_flow=False, dino_pred=None, im_features_map=im_features_map) #renderer for visualization only!!!
                        if self.background_mode == 'white':
                            # we want to render shading here, which is always black background, so modify here
                            render_mask = render_mask.unsqueeze(1)
                            rendered[render_mask == 0] = 1
                        rendered = rendered.repeat(1, 3, 1, 1)
                    else:
                        rendered, _, _, _, _, _ = self.render(shape_to_render, texture, mvp, w2c, campos, (h, w), background=self.background_mode, im_features=needed_im_features, prior_shape=prior_shape, render_mode=mode, render_flow=False, dino_pred=None, im_features_map=im_features_map) #renderer for visualization only!!!
                    if 'kd' in mode:
                        rendered = util.rgb_to_srgb(rendered)
                    rendered = rendered.detach().cpu()
                    rendered_wo_bones = rendered
                    
                    if 'posed_bones' in aux_viz:
                        rendered_bone_image = self.render_bones(mvp, aux_viz['posed_bones'], (h, w))
                        rendered_bone_image_mask = (rendered_bone_image < 1).any(1, keepdim=True).float()
                        # viz_logger.add_image(logger_prefix+'image/articulation_bones', misc.image_grid(self.render_bones(mvp, aux_viz['posed_bones'])), total_iter)
                        rendered = rendered_bone_image_mask*0.8 * rendered_bone_image + (1-rendered_bone_image_mask*0.8) * rendered

                    if rot_rep in ['quadlookat', 'octlookat']:
                        rand_pose_flag = forward_aux['rand_pose_flag'].detach().cpu()
                        rand_pose_marker_mask = torch.zeros_like(rendered)
                        rand_pose_marker_mask[:,:,:16,:16] = 1.
                        rand_pose_marker_mask = rand_pose_marker_mask * rand_pose_flag[:,None,None,None]
                        red = torch.FloatTensor([1,0,0])[None,:,None,None]
                        rendered = rand_pose_marker_mask * red + (1-rand_pose_marker_mask) * rendered

                    viz_logger.add_image(
                        logger_prefix + f'image/{which_shape}_{mode}',
                        misc.image_grid(expandF(rendered)[:b0, ...].view(-1, *rendered.shape[1:])),
                        total_iter)
                    
                    if rendered_wo_bones is not None:
                        viz_logger.add_image(
                            logger_prefix + f'image/{which_shape}_{mode}_raw',
                            misc.image_grid(expandF(rendered_wo_bones)[:b0, ...].view(-1, *rendered_wo_bones.shape[1:])),
                            total_iter)
                    
                    if mode in ['gray']:
                        viz_logger.add_video(
                            logger_prefix + f'animation/{which_shape}_{mode}',
                            self.render_rotation_frames(shape_to_render, texture, gray_light, (h, w), background=self.background_mode, im_features=needed_im_features, prior_shape=prior_shape, num_frames=15, render_mode='diffuse', b=1, im_features_map=im_features_map, original_mvp=mvp, original_w2c=w2c, original_campos=campos, render_gray=True).detach().cpu().unsqueeze(0),
                            total_iter,
                            fps=2)
                    else:
                        viz_logger.add_video(
                            logger_prefix + f'animation/{which_shape}_{mode}',
                            self.render_rotation_frames(shape_to_render, texture, light, (h, w), background=self.background_mode, im_features=needed_im_features, prior_shape=prior_shape, num_frames=15, render_mode=mode, b=1, im_features_map=im_features_map, original_mvp=mvp, original_w2c=w2c, original_campos=campos).detach().cpu().unsqueeze(0),
                            total_iter,
                            fps=2)
            
            viz_logger.add_video(
                logger_prefix+'animation/prior_image_rotation', 
                self.render_rotation_frames(prior_shape, texture, light, (h, w), background=self.background_mode, im_features=im_features, num_frames=15, b=1, text=category_name[0], im_features_map=im_features_map, original_mvp=mvp).detach().cpu().unsqueeze(0).clamp(0,1), 
                total_iter, 
                fps=2)
            
            viz_logger.add_video(
                logger_prefix+'animation/prior_normal_rotation', 
                self.render_rotation_frames(prior_shape, texture, light, (h, w), background=self.background_mode, im_features=im_features, num_frames=15, render_mode='geo_normal', b=1, text=category_name[0], im_features_map=im_features_map, original_mvp=mvp).detach().cpu().unsqueeze(0), 
                total_iter, 
                fps=2)

        if save_results and self.rank == 0:
            b0 = self.cfgs.get('num_saved_from_each_batch', batch_size*num_frames)
            # from IPython import embed; embed()
            fnames = [f'{total_iter:07d}_{fid:010d}' for fid in collapseF(frame_id.int())][:b0]

            # pkl_str = osp.join(save_dir, f'{total_iter:07d}_animal_data.pkl')
            os.makedirs(save_dir, exist_ok=True)
            # with open(pkl_str, 'wb') as fpkl:
            #     pickle.dump(save_for_pkl, fpkl)
            #     fpkl.close()

            misc.save_images(save_dir, collapseF(image_gt)[:b0].clamp(0,1).detach().cpu().numpy(), suffix='image_gt', fnames=fnames)
            misc.save_images(save_dir, collapseF(image_pred)[:b0].clamp(0,1).detach().cpu().numpy(), suffix='image_pred', fnames=fnames)
            misc.save_images(save_dir, collapseF(mask_gt)[:b0].unsqueeze(1).repeat(1,3,1,1).clamp(0,1).detach().cpu().numpy(), suffix='mask_gt', fnames=fnames)
            misc.save_images(save_dir, collapseF(mask_pred)[:b0].unsqueeze(1).repeat(1,3,1,1).clamp(0,1).detach().cpu().numpy(), suffix='mask_pred', fnames=fnames)
            # tmp_shape = shape.first_n(b0).clone()
            # tmp_shape.material = texture
            # feat = im_features[:b0] if im_features is not None else None
            # misc.save_obj(save_dir, tmp_shape, save_material=False, feat=feat, suffix="mesh", fnames=fnames)  # Save the first mesh.
            if self.render_flow and flow_gt is not None:
                flow_gt_viz = torch.cat([flow_gt, torch.zeros_like(flow_gt[:,:,:1])], 2) + 0.5  # -0.5~1.5
                flow_gt_viz = flow_gt_viz.view(-1, *flow_gt_viz.shape[2:])
                misc.save_images(save_dir, flow_gt_viz[:b0].clamp(0,1).detach().cpu().numpy(), suffix='flow_gt', fnames=fnames)
            if flow_pred is not None:
                flow_pred_viz = torch.cat([flow_pred, torch.zeros_like(flow_pred[:,:,:1])], 2) + 0.5  # -0.5~1.5
                flow_pred_viz = flow_pred_viz.view(-1, *flow_pred_viz.shape[2:])
                misc.save_images(save_dir, flow_pred_viz[:b0].clamp(0,1).detach().cpu().numpy(), suffix='flow_pred', fnames=fnames)

            misc.save_txt(save_dir, pose[:b0].detach().cpu().numpy(), suffix='pose', fnames=fnames)
        return metrics

    def save_scores(self, path):
        header = 'mask_mse, \
                  mask_iou, \
                  image_mse, \
                  flow_mse'
        mean = self.all_scores.mean(0)
        std = self.all_scores.std(0)
        header = header + '\nMean: ' + ',\t'.join(['%.8f'%x for x in mean])
        header = header + '\nStd: ' + ',\t'.join(['%.8f'%x for x in std])
        misc.save_scores(path, self.all_scores, header=header)
        print(header)

    def render_rotation_frames(self, mesh, texture, light, resolution, background='none', im_features=None, prior_shape=None, num_frames=36, render_mode='diffuse', b=None, text=None, im_features_map=None, original_mvp=None, original_w2c=None, original_campos=None, render_gray=False):
        frames = []
        if b is None:
            b = len(mesh)
        else:
            mesh = mesh.first_n(b)
            feat = im_features[:b] if im_features is not None else None
            im_features_map = im_features_map[:b] if im_features_map is not None else None
            original_mvp = original_mvp[:b] if original_mvp is not None else None  # [b, 4, 4]

            if im_features_map is not None:
                im_features_map = {'im_features_map': im_features_map, 'original_mvp':original_mvp}
        
        delta_angle = np.pi / num_frames * 2
        delta_rot_matrix = torch.FloatTensor([
            [np.cos(delta_angle),  0, np.sin(delta_angle), 0],
            [0,                    1, 0,                   0],
            [-np.sin(delta_angle), 0, np.cos(delta_angle), 0],
            [0,                    0, 0,                   1],
        ]).to(self.device).repeat(b, 1, 1)

        w2c = torch.FloatTensor(np.diag([1., 1., 1., 1]))
        w2c[:3, 3] = torch.FloatTensor([0, 0, -self.cam_pos_z_offset *1.1])
        w2c = w2c.repeat(b, 1, 1).to(self.device)
        proj = util.perspective(self.crop_fov_approx / 180 * np.pi, 1, n=0.1, f=1000.0).repeat(b, 1, 1).to(self.device)
        mvp = torch.bmm(proj, w2c)
        campos = -w2c[:, :3, 3]

        if original_w2c is not None and original_campos is not None and original_mvp is not None:
            w2c = original_w2c[:b]
            campos = original_campos[:b]
            mvp = original_mvp[:b]

        def rotate_pose(mvp, campos):
            mvp = torch.matmul(mvp, delta_rot_matrix)
            campos = torch.matmul(delta_rot_matrix[:,:3,:3].transpose(2,1), campos[:,:,None])[:,:,0]
            return mvp, campos

        for _ in range(num_frames):
            if render_gray:
                _, render_mask, _, _, _, image_pred = self.render(mesh, texture, mvp, w2c, campos, resolution, background=background, im_features=feat, light=light, prior_shape=prior_shape, render_flow=False, dino_pred=None, render_mode=render_mode, two_sided_shading=False, im_features_map=im_features_map)
                if self.background_mode == 'white':
                    # we want to render shading here, which is always black background, so modify here
                    render_mask = render_mask.unsqueeze(1)
                    image_pred[render_mask == 0] = 1
                image_pred = image_pred.repeat(1, 3, 1, 1)
            else:
                image_pred, _, _, _, _, _ = self.render(mesh, texture, mvp, w2c, campos, resolution, background=background, im_features=feat, light=light, prior_shape=prior_shape, render_flow=False, dino_pred=None, render_mode=render_mode, two_sided_shading=False, im_features_map=im_features_map) #for rotation frames only!
            image_pred = image_pred.clamp(0, 1)
            frames += [misc.image_grid(image_pred)]
            mvp, campos = rotate_pose(mvp, campos)
        
        if text is not None:
            frames = [torch.Tensor(misc.add_text_to_image(f, text)).permute(2, 0, 1) for f in frames]

        return torch.stack(frames, dim=0)  # Shape: (T, C, H, W)

    def render_bones(self, mvp, bones_pred, size=(256, 256)):
        bone_world4 = torch.concat([bones_pred, torch.ones_like(bones_pred[..., :1]).to(bones_pred.device)], dim=-1)
        b, f, num_bones = bone_world4.shape[:3]
        bones_clip4 = (bone_world4.view(b, f, num_bones*2, 1, 4) @ mvp.transpose(-1, -2).reshape(b, f, 1, 4, 4)).view(b, f, num_bones, 2, 4)
        bones_uv = bones_clip4[..., :2] / bones_clip4[..., 3:4]  # b, f, num_bones, 2, 2
        dpi = 32
        fx, fy = size[1] // dpi, size[0] // dpi

        rendered = []
        for b_idx in range(b):
            for f_idx in range(f):
                frame_bones_uv = bones_uv[b_idx, f_idx].cpu().numpy()
                fig = plt.figure(figsize=(fx, fy), dpi=dpi, frameon=False)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                for bone in frame_bones_uv:
                    ax.plot(bone[:, 0], bone[:, 1], marker='o', linewidth=8, markersize=20)
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.invert_yaxis()
                # Convert to image
                fig.add_axes(ax)
                fig.canvas.draw_idle()
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                w, h = fig.canvas.get_width_height()
                image.resize(h, w, 3)
                rendered += [image / 255.]
        return torch.from_numpy(np.stack(rendered, 0).transpose(0, 3, 1, 2))

    def render_deformation_frames(self, mesh, texture, batch_size, num_frames, resolution, background='none', im_features=None, render_mode='diffuse', b=None):
        # frames = []
        # if b is None:
        #     b = batch_size
        #     im_features = im_features[]
        # mesh = mesh.first_n(num_frames * b)
        # for i in range(b):
        #     tmp_mesh = mesh.get_m_to_n(i*num_frames:(i+1)*num_frames)
        pass

    def vis_sds_image(self, sds_image, sds_aux):
        sds_image = sds_image.detach().cpu().clamp(0, 1)
        sds_image = [misc.add_text_to_image(img, text) for img, text in zip(sds_image, sds_aux['dirs'])]
        return misc.image_grid(sds_image)

    def vis_sds_grads(self, sds_aux):
        grads = sds_aux['sd_aux']['grad']
        grads = grads.detach().cpu()
        # compute norm
        grads_norm = grads.norm(dim=1, keepdim=True)
        # interpolate to 4x size
        grads_norm = F.interpolate(grads_norm, scale_factor=4, mode='nearest')
        # add time step and weight
        t = sds_aux['sd_aux']['t']
        w = sds_aux['sd_aux']['w']
        # max norm for each sample over dim (1, 2, 3)
        n = grads_norm.view(grads_norm.shape[0], -1).max(dim=1)[0]
        texts = [f"t: {t_} w: {w_:.2f} n: {n_:.2e}" for t_, w_ , n_ in zip(t, w, n)]
        return misc.image_grid_multi_channel(grads_norm, texts=texts, font_scale=0.5)