from multiprocessing.spawn import prepare
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import nvdiffrast.torch as dr
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp

from video3d.render.regularizer import get_edge_length, normal_consistency
from . import networks
from .renderer import *
from .utils import misc, meters, flow_viz, arap, custom_loss
from .dataloaders import get_sequence_loader, get_image_loader
from .cub_dataloaders import get_cub_loader
from .utils.skinning_v4 import estimate_bones, skinning
import lpips
from einops import rearrange

from .geometry.dmtet import DMTetGeometry
from .geometry.dlmesh import DLMesh

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


def sample_pose_hypothesis_from_quad_prediction(poses_raw, total_iter, batch_size, num_frames, pose_xflip_recon=False, input_image_xflip_flag=None, rot_temp_scalar=1., num_hypos=4, naive_probs_iter=2000, best_pose_start_iter=6000, random_sample=True):
    rots_pred = poses_raw[..., :num_hypos*4].view(-1, num_hypos, 4)
    rots_logits = rots_pred[..., 0]  # Nx4
    temp = 1 / np.clip(total_iter / 1000 / rot_temp_scalar, 1., 100.)

    rots_probs = torch.nn.functional.softmax(-rots_logits / temp, dim=1)  # N x K
    # naive_probs = torch.FloatTensor([10] + [1] * (num_hypos - 1)).to(rots_logits.device)
    naive_probs = torch.ones(num_hypos).to(rots_logits.device)
    naive_probs = naive_probs / naive_probs.sum()
    naive_probs_weight = np.clip(1 - (total_iter - naive_probs_iter) / 2000, 0, 1)
    rots_probs = naive_probs.view(1, num_hypos) * naive_probs_weight + rots_probs * (1 - naive_probs_weight)

    rots_pred = rots_pred[..., 1:4]
    trans_pred = poses_raw[..., -3:]
    best_rot_idx = torch.argmax(rots_probs, dim=1)  # N
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
        rot_idx = best_rot_idx
    rot_pred = torch.gather(rots_pred, 1, rot_idx[:, None, None].expand(-1, 1, 3))[:, 0]  # Nx3
    pose_raw = torch.cat([rot_pred, trans_pred], -1)
    rot_prob = torch.gather(rots_probs, 1, rot_idx[:, None].expand(-1, 1))[:, 0]  # N
    rot_logit = torch.gather(rots_logits, 1, rot_idx[:, None].expand(-1, 1))[:, 0]  # N

    if pose_xflip_recon:
        raise NotImplementedError
    rot_mat = forward_to_matrix(pose_raw[:, :3], up=[0, 1, 0])
    pose = torch.cat([rot_mat.view(batch_size * num_frames, -1), pose_raw[:, 3:]], -1)
    return pose_raw, pose, rot_idx, rot_prob, rot_logit, rots_probs, rand_flag


class PriorPredictor(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
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
        self.netShape = DMTetGeometry(dmtet_grid, grid_scale, prior_sdf_mode, num_layers=num_layers_shape, hidden_size=hidden_size, embedder_freq=embedder_freq_shape, embed_concat_pts=embed_concat_pts, init_sdf=init_sdf, jitter_grid=jitter_grid, perturb_sdf_iter=perturb_sdf_iter, sym_prior_shape=sym_prior_shape)

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
                extra_dim=0,
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
        
    def forward(self, perturb_sdf=False, total_iter=None, is_training=True):
        prior_shape = self.netShape.getMesh(perturb_sdf=perturb_sdf, total_iter=total_iter, jitter_grid=is_training)
        return prior_shape, self.netDINO


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
                self.netEncoder = networks.ViTEncoder(cout=encoder_latent_dim, which_vit=which_vit, pretrained=encoder_pretrained, frozen=encoder_frozen, in_size=in_image_size, final_layer_type=vit_final_layer_type)
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
                symmetrize=sym_texture
            )

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
                self.netPose = networks.ViTEncoder(cout=encoder_latent_dim, which_vit=which_vit, pretrained=encoder_pretrained, frozen=encoder_frozen, in_size=in_image_size, final_layer_type=vit_final_layer_type)
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
            self.netArticulation = networks.ArticulationNetwork(self.articulation_arch, feat_dim, 1+2+3*2, num_layers_arti, mlp_hidden_size, n_harmonic_functions=embedder_freq_arti, omega0=embedder_scaler)
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
    
    def forward_deformation(self, shape, feat=None):
        original_verts = shape.v_pos
        num_verts = original_verts.shape[1]
        if feat is not None:
            deform_feat = feat[:, None, :].repeat(1, num_verts, 1)  # Shape: (B, num_verts, latent_dim)
            original_verts = original_verts.repeat(len(feat),1,1)
        deformation = self.netDeform(original_verts, deform_feat) * 0.1  # Shape: (B, num_verts, 3)
        shape = shape.deform(deformation)
        return shape, deformation
    
    def forward_articulation(self, shape, feat, patch_feat, mvp, w2c, batch_size, num_frames, epoch):
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
            attach_legs_to_body = epoch in self.attach_legs_to_body_epochs
            bones, self.kinematic_tree, self.bone_aux = estimate_bones(verts.detach(), self.num_body_bones, n_legs=self.num_legs, n_leg_bones=self.num_leg_bones, body_bones_type=self.body_bones_type, compute_kinematic_chain=True, attach_legs_to_body=attach_legs_to_body)
            self.kinematic_tree_epoch = epoch
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

        if self.constrain_legs:
            leg_bones_posx = [self.num_body_bones + i for i in range(self.num_leg_bones * self.num_legs // 2)]
            leg_bones_negx = [self.num_body_bones + self.num_leg_bones * self.num_legs // 2 + i for i in range(self.num_leg_bones * self.num_legs // 2)]

            tmp_mask = torch.zeros_like(articulation_angles)
            tmp_mask[:, :, leg_bones_posx + leg_bones_negx, 2] = 1
            articulation_angles = tmp_mask * (articulation_angles * 0.3) + (1 - tmp_mask) * articulation_angles  # no twist

            tmp_mask = torch.zeros_like(articulation_angles)
            tmp_mask[:, :, leg_bones_posx + leg_bones_negx, 1] = 1
            articulation_angles = tmp_mask * (articulation_angles * 0.3) + (1 - tmp_mask) * articulation_angles  # (-0.4, 0.4),  limit side bending
        
        if epoch in self.perturb_articulation_epochs:
            articulation_angles = articulation_angles + torch.randn_like(articulation_angles) * 0.1
        articulation_angles = articulation_angles * self.max_arti_angle / 180 * np.pi
        
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
    
    def get_camera_extrinsics_from_pose(self, pose, znear=0.1, zfar=1000.):
        N = len(pose)
        cam_pos_offset = torch.FloatTensor([0, 0, -self.cam_pos_z_offset]).to(pose.device)
        pose_R = pose[:, :9].view(N, 3, 3).transpose(2, 1)
        pose_T = pose[:, -3:] + cam_pos_offset[None, None, :]
        pose_T = pose_T.view(N, 3, 1)
        pose_RT = torch.cat([pose_R, pose_T], axis=2)  # Nx3x4
        w2c = torch.cat([pose_RT, torch.FloatTensor([0, 0, 0, 1]).repeat(N, 1, 1).to(pose.device)], axis=1)  # Nx4x4
        # We assume the images are perfect square.
        proj = util.perspective(self.crop_fov_approx / 180 * np.pi, 1, znear, zfar)[None].to(pose.device)
        mvp = torch.matmul(proj, w2c)
        campos = -torch.matmul(pose_R.transpose(2, 1), pose_T).view(N, 3)
        return mvp, w2c, campos

    def forward(self, images=None, prior_shape=None, epoch=None, dino_features=None, dino_clusters=None, total_iter=None, is_training=True):
        batch_size, num_frames = images.shape[:2]
        if self.enable_encoder:
            feat_out, feat_key, patch_out, patch_key = self.forward_encoder(images, dino_features)
        else:
            feat_out = feat_key = patch_out = patch_key = None
        shape = prior_shape
        texture = self.netTexture

        multi_hypothesis_aux = {}
        if self.enable_pose:
            poses_raw = self.forward_pose(images, feat_out, patch_out, patch_key, dino_features)
            pose_raw, pose, rot_idx, rot_prob, rot_logit, rots_probs, rand_pose_flag = sample_pose_hypothesis_from_quad_prediction(poses_raw, total_iter, batch_size, num_frames, rot_temp_scalar=self.rot_temp_scalar, num_hypos=self.num_pose_hypos, naive_probs_iter=self.naive_probs_iter, best_pose_start_iter=self.best_pose_start_iter, random_sample=is_training)
            multi_hypothesis_aux['rot_idx'] = rot_idx
            multi_hypothesis_aux['rot_prob'] = rot_prob
            multi_hypothesis_aux['rot_logit'] = rot_logit
            multi_hypothesis_aux['rots_probs'] = rots_probs
            multi_hypothesis_aux['rand_pose_flag'] = rand_pose_flag
        else:
            raise NotImplementedError
        mvp, w2c, campos = self.get_camera_extrinsics_from_pose(pose)

        deformation = None
        if self.enable_deform and epoch in self.deform_epochs:
            shape, deformation = self.forward_deformation(shape, feat_key)
        
        arti_params, articulation_aux = None, {}
        if self.enable_articulation and epoch in self.articulation_epochs:
            shape, arti_params, articulation_aux = self.forward_articulation(shape, feat_key, patch_key, mvp, w2c, batch_size, num_frames, epoch)
        
        if self.enable_lighting:
            light = self.netLight
        else:
            light = None

        aux = articulation_aux
        aux.update(multi_hypothesis_aux)

        return shape, pose_raw, pose, mvp, w2c, campos, texture, feat_out, deformation, arti_params, light, aux


class Unsup3D:
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
            self.netPrior = PriorPredictor(self.cfgs)
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

        ## perceptual loss
        if cfgs.get('perceptual_loss_weight', 0.) > 0:
            self.perceptual_loss_use_lin = cfgs.get('perceptual_loss_use_lin', True)
            self.perceptual_loss = lpips.LPIPS(net='vgg', lpips=self.perceptual_loss_use_lin)

        self.glctx = dr.RasterizeGLContext()
        self.render_flow = self.cfgs.get('flow_loss_weight', 0.) > 0.
        self.extra_renders = cfgs.get('extra_renders', [])
        self.renderer_spp = cfgs.get('renderer_spp', 1)
        self.dino_feature_recon_dim = cfgs.get('dino_feature_recon_dim', 64)

        self.total_loss = 0.
        self.all_scores = torch.Tensor()
        self.checkpoint_dir = cfgs.get('checkpoint_dir', 'results')
    
    @staticmethod
    def get_data_loaders(cfgs, dataset, in_image_size=256, out_image_size=256, batch_size=64, num_workers=4, run_train=False, run_test=False, train_data_dir=None, val_data_dir=None, test_data_dir=None):
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

    def load_model_state(self, cp):
        self.netInstance.load_state_dict(cp["netInstance"])
        if self.enable_prior:
            self.netPrior.load_state_dict(cp["netPrior"])

    def load_optimizer_state(self, cp):
        self.optimizerInstance.load_state_dict(cp["optimizerInstance"])
        if self.use_scheduler:
            if 'schedulerInstance' in cp:
                self.schedulerInstance.load_state_dict(cp["schedulerInstance"])
        if self.enable_prior and self.resume_prior_optim:
            self.optimizerPrior.load_state_dict(cp["optimizerPrior"])
            if self.use_scheduler:
                if 'schedulerPrior' in cp:
                    self.schedulerPrior.load_state_dict(cp["schedulerPrior"])

    def get_model_state(self):
        state = {"netInstance": self.netInstance.state_dict()}
        if self.enable_prior:
            state["netPrior"] = self.netPrior.state_dict()
        return state

    def get_optimizer_state(self):
        state = {"optimizerInstance": self.optimizerInstance.state_dict()}
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
        if hasattr(self, 'perceptual_loss'):
            self.perceptual_loss.to(device)

    def set_train(self):
        self.netInstance.train()
        if self.enable_prior:
            self.netPrior.train()

    def set_eval(self):
        self.netInstance.eval()
        if self.enable_prior:
            self.netPrior.eval()

    def reset_optimizers(self):
        print("Resetting optimizers...")
        self.optimizerInstance = get_optimizer(self.netInstance, self.lr)
        if self.use_scheduler:
            self.schedulerInstance = self.make_scheduler(self.optimizerInstance)
        if self.enable_prior:
            self.optimizerPrior = get_optimizer(self.netPrior, lr=self.prior_lr, weight_decay=self.prior_weight_decay)
            if self.use_scheduler:
                self.schedulerPrior = self.make_scheduler(self.optimizerPrior)

    def backward(self):
        self.optimizerInstance.zero_grad()
        if self.backward_prior:
            self.optimizerPrior.zero_grad()
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

    def render(self, shape, texture, mvp, w2c, campos, resolution, background='none', im_features=None, light=None, prior_shape=None, render_flow=True, dino_pred=None, render_mode='diffuse', two_sided_shading=True, num_frames=None, spp=1):
        h, w = resolution
        N = len(mvp)
        if background in ['none', 'black']:
            bg_image = torch.zeros((N, h, w, 3), device=mvp.device)
        elif background == 'white':
            bg_image = torch.ones((N, h, w, 3), device=mvp.device)
        elif background == 'checkerboard':
            bg_image = torch.FloatTensor(util.checkerboard((h, w), 8), device=self.device).repeat(N, 1, 1, 1)  # NxHxWxC
        else:
            raise NotImplementedError

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
            num_frames=num_frames)
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

        if dino_feat_im_pred is not None:
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
        feat_xflip, _ = self.netInstance.forward_encoder(image_xflip, dino_feat_im_xflip)
        batch_size, num_frames = input_image.shape[:2]
        pose_xflip_raw = self.netInstance.forward_pose(image_xflip, feat_xflip, dino_feat_im_xflip)

        if input_image_xflip_flag is not None:
            pose_xflip_raw_xflip = pose_xflip_raw * torch.FloatTensor([-1,1,1,-1,1,1]).to(pose_raw.device)  # forward x, trans x
            pose_xflip_raw = pose_xflip_raw * (1 - input_image_xflip_flag.view(batch_size * num_frames, 1)) + pose_xflip_raw_xflip * input_image_xflip_flag.view(batch_size * num_frames, 1)

        rot_rep = self.netInstance.rot_rep
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

    def compute_regularizers(self, mesh, prior_mesh, input_image, dino_feat_im, pose_raw, input_image_xflip_flag=None, arti_params=None, deformation=None):
        losses = {}
        aux = {}
        
        if self.enable_prior:
            losses.update(self.netPrior.netShape.get_sdf_reg_loss())
        
        if self.cfgs.get('pose_xflip_reg_loss_weight', 0.) > 0:
            losses["pose_xflip_reg_loss"], aux['pose_xflip_raw'] = self.compute_pose_xflip_reg_loss(input_image, dino_feat_im, pose_raw, input_image_xflip_flag)
        
        b, f = input_image.shape[:2]
        if b >= 2:
            vec_forward = pose_raw[..., :3]
            losses['pose_entropy_loss'] = (vec_forward[:b//2] * vec_forward[b//2:(b//2)*2]).sum(-1).mean()
        else:
            losses['pose_entropy_loss'] = 0.

        losses['mesh_normal_consistency_loss'] = normal_consistency(mesh.v_pos, mesh.t_pos_idx)
        losses['mesh_edge_length_loss'], aux['edge_lengths'] = self.compute_edge_length_reg_loss(mesh, prior_mesh)
        if arti_params is not None:
            losses['arti_reg_loss'] = (arti_params ** 2).mean()
        
        if deformation is not None:
            losses['deformation_reg_loss'] = (deformation ** 2).mean()
            # losses['deformation_reg_loss'] = deformation.abs().mean()
        
        return losses, aux
    
    def forward(self, batch, epoch, iter, is_train=True, viz_logger=None, total_iter=None, save_results=False, save_dir=None, which_data='', logger_prefix='', is_training=True):
        batch = [x.to(self.device) if x is not None else None for x in batch]
        input_image, mask_gt, mask_dt, mask_valid, flow_gt, bbox, bg_image, dino_feat_im, dino_cluster_im, seq_idx, frame_idx = batch
        batch_size, num_frames, _, h0, w0 = input_image.shape  # BxFxCxHxW
        h = w = self.out_image_size

        def collapseF(x):
            return None if x is None else x.view(batch_size * num_frames, *x.shape[2:])
        def expandF(x):
            return None if x is None else x.view(batch_size, num_frames, *x.shape[1:])
        
        if flow_gt.dim() == 2:  # dummy tensor for not loading flow
            flow_gt = None
        if dino_feat_im.dim() == 2:  # dummy tensor for not loading dino features
            dino_feat_im = None
            dino_feat_im_gt = None
        else:
            dino_feat_im_gt = expandF(torch.nn.functional.interpolate(collapseF(dino_feat_im), size=[h, w], mode="bilinear"))[:, :, :self.dino_feature_recon_dim]
        if dino_cluster_im.dim() == 2:  # dummy tensor for not loading dino clusters
            dino_cluster_im = None
            dino_cluster_im_gt = None
        else:
            dino_cluster_im_gt = expandF(torch.nn.functional.interpolate(collapseF(dino_cluster_im), size=[h, w], mode="nearest"))
        
        seq_idx = seq_idx.squeeze(1)
        # seq_idx = seq_idx * 0  # single sequnce model
        frame_id, crop_x0, crop_y0, crop_w, crop_h, full_w, full_h, sharpness = bbox.unbind(2)  # BxFx7
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
            if epoch < self.dmtet_grid_smaller_epoch:
                if self.netPrior.netShape.grid_res != self.dmtet_grid_smaller:
                    self.netPrior.netShape.load_tets(self.dmtet_grid_smaller)
            else:
                if self.netPrior.netShape.grid_res != self.dmtet_grid:
                    self.netPrior.netShape.load_tets(self.dmtet_grid)
            
            perturb_sdf = self.perturb_sdf if is_train else False
            prior_shape, dino_pred = self.netPrior(perturb_sdf=perturb_sdf, total_iter=total_iter, is_training=is_training)
        else:
            prior_shape = None
            raise NotImplementedError

        shape, pose_raw, pose, mvp, w2c, campos, texture, im_features, deformation, arti_params, light, forward_aux = self.netInstance(input_image, prior_shape, epoch, dino_feat_im, dino_cluster_im, total_iter, is_training=is_training)  # frame dim collapsed N=(B*F)
        rot_logit = forward_aux['rot_logit']
        rot_idx = forward_aux['rot_idx']
        rot_prob = forward_aux['rot_prob']
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

        render_flow = self.render_flow and num_frames > 1
        image_pred, mask_pred, flow_pred, dino_feat_im_pred, albedo, shading = self.render(shape, texture, mvp, w2c, campos, (h, w), background=self.background_mode, im_features=im_features, light=light, prior_shape=prior_shape, render_flow=render_flow, dino_pred=dino_pred, num_frames=num_frames, spp=self.renderer_spp)
        image_pred, mask_pred, flow_pred, dino_feat_im_pred = map(expandF, (image_pred, mask_pred, flow_pred, dino_feat_im_pred))
        if flow_pred is not None:
            flow_pred = flow_pred[:, :-1]  # Bx(F-1)x2xHxW

        if self.blur_mask:
            sigma = max(0.5, 3 * (1 - total_iter / self.blur_mask_iter))
            if sigma > 0.5:
                mask_gt = util.blur_image(mask_gt, kernel_size=9, sigma=sigma, mode='gaussian')
            # mask_pred = util.blur_image(mask_pred, kernel_size=7, mode='average')

        losses = self.compute_reconstruction_losses(image_pred, image_gt, mask_pred, mask_gt, mask_dt, mask_valid, flow_pred, flow_gt, dino_feat_im_gt, dino_feat_im_pred, background_mode=self.background_mode, reduce=False)
        
        ## TODO: assume flow loss is not used
        logit_loss_target = torch.zeros_like(expandF(rot_logit))
        final_losses = {}
        for name, loss in losses.items():
            loss_weight_logit = self.cfgs.get(f"{name}_weight", 0.)
            # if (name in ['flow_loss'] and epoch not in self.flow_loss_epochs) or (name in ['rgb_loss', 'perceptual_loss'] and epoch not in self.texture_epochs):
            # if name in ['flow_loss', 'rgb_loss', 'perceptual_loss']:
            #     loss_weight_logit = 0.
            if name in ['sdf_bce_reg_loss', 'sdf_gradient_reg_loss', 'sdf_inflate_reg_loss']:
                if total_iter >= self.sdf_reg_decay_start_iter:
                    decay_rate = max(0, 1 - (total_iter-self.sdf_reg_decay_start_iter) / 10000)
                    loss_weight_logit = max(loss_weight_logit * decay_rate, self.cfgs.get(f"{name}_min_weight", 0.))
            if name in ['dino_feat_im_loss']:
                loss_weight_logit = loss_weight_logit * self.cfgs.get("logit_loss_dino_feat_im_loss_multiplier", 1.)
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

        ## regularizers
        regularizers, aux = self.compute_regularizers(shape, prior_shape, input_image, dino_feat_im, pose_raw, input_image_xflip_flag, arti_params, deformation)
        final_losses.update(regularizers)
        aux_viz.update(aux)

        total_loss = 0
        for name, loss in final_losses.items():
            loss_weight = self.cfgs.get(f"{name}_weight", 0.)
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
                flow_gt = flow_gt.detach().cpu()
                flow_gt_viz = torch.cat([flow_gt[:b0], torch.zeros_like(flow_gt[:b0,:,:1])], 2) + 0.5  # -0.5~1.5
                flow_gt_viz = torch.nn.functional.pad(flow_gt_viz, pad=[0, 0, 0, 0, 0, 0, 0, 1])

                ## draw marker on large flow frames
                large_flow_marker_mask = torch.zeros_like(flow_gt_viz)
                large_flow_marker_mask[:,:,:,:8,:8] = 1.
                large_flow = torch.cat([self.large_flow, self.large_flow[:,:1] *0.], 1).detach().cpu()[:b0]
                large_flow_marker_mask = large_flow_marker_mask * large_flow[:,:,None,None,None]
                red = torch.FloatTensor([1,0,0])[None,None,:,None,None]
                flow_gt_viz = large_flow_marker_mask * red + (1-large_flow_marker_mask) * flow_gt_viz
                
                viz_logger.add_image(logger_prefix+'image/flow_gt', misc.image_grid(flow_gt_viz.reshape(-1,*flow_gt_viz.shape[2:])), total_iter)
            
            if self.render_flow and flow_pred is not None:
                flow_pred = flow_pred.detach().cpu()
                flow_pred_viz = torch.cat([flow_pred[:b0], torch.zeros_like(flow_pred[:b0,:,:1])], 2) + 0.5  # -0.5~1.5
                flow_pred_viz = torch.nn.functional.pad(flow_pred_viz, pad=[0, 0, 0, 0, 0, 0, 0, 1])
                viz_logger.add_image(logger_prefix+'image/flow_pred', misc.image_grid(flow_pred_viz.reshape(-1,*flow_pred_viz.shape[2:])), total_iter)
            
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

            viz_logger.add_histogram(logger_prefix+'sdf', self.netPrior.netShape.get_sdf(perturb_sdf=False), total_iter)
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
                    rendered, _, _, _, _, _ = self.render(shape_to_render, texture, mvp, w2c, campos, (h, w), background=self.background_mode, im_features=needed_im_features, prior_shape=prior_shape, render_mode=mode, render_flow=False, dino_pred=None)
                    if 'kd' in mode:
                        rendered = util.rgb_to_srgb(rendered)
                    rendered = rendered.detach().cpu()
                    
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
                    
                    viz_logger.add_video(
                        logger_prefix + f'animation/{which_shape}_{mode}',
                        self.render_rotation_frames(shape_to_render, texture, light, (h, w), background=self.background_mode, im_features=needed_im_features, prior_shape=prior_shape, num_frames=15, render_mode=mode, b=1).detach().cpu().unsqueeze(0),
                        total_iter,
                        fps=2)
            
            viz_logger.add_video(
                logger_prefix+'animation/prior_image_rotation', 
                self.render_rotation_frames(prior_shape, texture, light, (h, w), background=self.background_mode, im_features=im_features, num_frames=15, b=1).detach().cpu().unsqueeze(0).clamp(0,1), 
                total_iter, 
                fps=2)
            
            viz_logger.add_video(
                logger_prefix+'animation/prior_normal_rotation', 
                self.render_rotation_frames(prior_shape, texture, light, (h, w), background=self.background_mode, im_features=im_features, num_frames=15, render_mode='geo_normal', b=1).detach().cpu().unsqueeze(0), 
                total_iter, 
                fps=2)

        if save_results:
            b0 = self.cfgs.get('num_saved_from_each_batch', batch_size*num_frames)
            fnames = [f'{total_iter:07d}_{fid:10d}' for fid in collapseF(frame_id.int())][:b0]

            misc.save_images(save_dir, collapseF(image_gt)[:b0].clamp(0,1).detach().cpu().numpy(), suffix='image_gt', fnames=fnames)
            misc.save_images(save_dir, collapseF(image_pred)[:b0].clamp(0,1).detach().cpu().numpy(), suffix='image_pred', fnames=fnames)
            misc.save_images(save_dir, collapseF(mask_gt)[:b0].unsqueeze(1).repeat(1,3,1,1).clamp(0,1).detach().cpu().numpy(), suffix='mask_gt', fnames=fnames)
            misc.save_images(save_dir, collapseF(mask_pred)[:b0].unsqueeze(1).repeat(1,3,1,1).clamp(0,1).detach().cpu().numpy(), suffix='mask_pred', fnames=fnames)
            # tmp_shape = shape.first_n(b0).clone()
            # tmp_shape.material = texture
            # feat = im_features[:b0] if im_features is not None else None
            # misc.save_obj(save_dir, tmp_shape, save_material=False, feat=feat, suffix="mesh", fnames=fnames)  # Save the first mesh.
            # if self.render_flow and flow_gt is not None:
            #     flow_gt_viz = torch.cat([flow_gt, torch.zeros_like(flow_gt[:,:,:1])], 2) + 0.5  # -0.5~1.5
            #     flow_gt_viz = flow_gt_viz.view(-1, *flow_gt_viz.shape[2:])
            #     misc.save_images(save_dir, flow_gt_viz[:b0].clamp(0,1).detach().cpu().numpy(), suffix='flow_gt', fnames=fnames)
            # if flow_pred is not None:
            #     flow_pred_viz = torch.cat([flow_pred, torch.zeros_like(flow_pred[:,:,:1])], 2) + 0.5  # -0.5~1.5
            #     flow_pred_viz = flow_pred_viz.view(-1, *flow_pred_viz.shape[2:])
            #     misc.save_images(save_dir, flow_pred_viz[:b0].clamp(0,1).detach().cpu().numpy(), suffix='flow_pred', fnames=fnames)

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

    def render_rotation_frames(self, mesh, texture, light, resolution, background='none', im_features=None, prior_shape=None, num_frames=36, render_mode='diffuse', b=None):
        frames = []
        if b is None:
            b = len(mesh)
        else:
            mesh = mesh.first_n(b)
            feat = im_features[:b] if im_features is not None else None
        
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

        def rotate_pose(mvp, campos):
            mvp = torch.matmul(mvp, delta_rot_matrix)
            campos = torch.matmul(delta_rot_matrix[:,:3,:3].transpose(2,1), campos[:,:,None])[:,:,0]
            return mvp, campos

        for _ in range(num_frames):
            image_pred, _, _, _, _, _ = self.render(mesh, texture, mvp, w2c, campos, resolution, background=background, im_features=feat, light=light, prior_shape=prior_shape, render_flow=False, dino_pred=None, render_mode=render_mode, two_sided_shading=False)
            frames += [misc.image_grid(image_pred)]
            mvp, campos = rotate_pose(mvp, campos)
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
