# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import numpy as np
import torch
import nvdiffrast.torch as dr
import cv2

from video3d.render.render import render_uv

from . import util
from . import texture
from . import mlptexture
from ..utils import misc

######################################################################################
# Wrapper to make materials behave like a python dict, but register textures as 
# torch.nn.Module parameters.
######################################################################################
class Material(torch.nn.Module):
    def __init__(self, mat_dict):
        super(Material, self).__init__()
        self.mat_keys = set()
        for key in mat_dict.keys():
            self.mat_keys.add(key)
            self[key] = mat_dict[key]

    def __contains__(self, key):
        return hasattr(self, key)

    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, val):
        self.mat_keys.add(key)
        setattr(self, key, val)

    def __delitem__(self, key):
        self.mat_keys.remove(key)
        delattr(self, key)

    def keys(self):
        return self.mat_keys

######################################################################################
# .mtl material format loading / storing
######################################################################################
@torch.no_grad()
def load_mtl(fn, clear_ks=True):
    import re
    mtl_path = os.path.dirname(fn)

    # Read file
    with open(fn, 'r') as f:
        lines = f.readlines()

    # Parse materials
    materials = []
    for line in lines:
        split_line = re.split(' +|\t+|\n+', line.strip())
        prefix = split_line[0].lower()
        data = split_line[1:]
        if 'newmtl' in prefix:
            material = Material({'name' : data[0]})
            materials += [material]
        elif materials:
            if 'bsdf' in prefix or 'map_kd' in prefix or 'map_ks' in prefix or 'bump' in prefix:
                material[prefix] = data[0]
            else:
                material[prefix] = torch.tensor(tuple(float(d) for d in data), dtype=torch.float32, device='cuda')

    # Convert everything to textures. Our code expects 'kd' and 'ks' to be texture maps. So replace constants with 1x1 maps
    for mat in materials:
        if not 'bsdf' in mat:
            mat['bsdf'] = 'pbr'

        if 'map_kd' in mat:
            mat['kd'] = texture.load_texture2D(os.path.join(mtl_path, mat['map_kd']))
        else:
            mat['kd'] = texture.Texture2D(mat['kd'])
        
        if 'map_ks' in mat:
            mat['ks'] = texture.load_texture2D(os.path.join(mtl_path, mat['map_ks']), channels=3)
        else:
            mat['ks'] = texture.Texture2D(mat['ks'])

        if 'bump' in mat:
            mat['normal'] = texture.load_texture2D(os.path.join(mtl_path, mat['bump']), lambda_fn=lambda x: x * 2 - 1, channels=3)

        # Convert Kd from sRGB to linear RGB
        mat['kd'] = texture.srgb_to_rgb(mat['kd'])

        if clear_ks:
            # Override ORM occlusion (red) channel by zeros. We hijack this channel
            for mip in mat['ks'].getMips():
                mip[..., 0] = 0.0 

    return materials

@torch.no_grad()
def save_mtl(fn, material, mesh=None, feat=None, resolution=[256, 256], prior_shape=None):
    folder = os.path.dirname(fn)
    file = os.path.basename(fn)
    prefix = '_'.join(file.split('_')[:-1]) + '_'
    with open(fn, "w") as f:
        f.write('newmtl defaultMat\n')
        if material is not None:
            f.write('bsdf   %s\n' % material['bsdf'])
            if 'kd_ks_normal' in material.keys():
                assert mesh is not None
                glctx = dr.RasterizeGLContext()
                mask, kd, ks, normal = render_uv(glctx, mesh, resolution, material['kd_ks_normal'], feat=feat, prior_shape=prior_shape)
                
                hole_mask = 1. - mask
                hole_mask = hole_mask.int()[0]
                def uv_padding(image):
                    uv_padding_size = 4
                    inpaint_image = (
                        cv2.inpaint(
                            (image.detach().cpu().numpy() * 255).astype(np.uint8),
                            (hole_mask.detach().cpu().numpy() * 255).astype(np.uint8),
                            uv_padding_size,
                            cv2.INPAINT_TELEA,
                        )
                        / 255.0
                    )
                    return torch.from_numpy(inpaint_image).to(image)
                
                kd = uv_padding(kd[0])[None]

                batch_size = kd.shape[0]
                f.write(f'map_Kd {prefix}texture_kd.png\n')
                misc.save_images(folder, kd.permute(0,3,1,2).detach().cpu().numpy(), fnames=[prefix + "texture_kd"] * batch_size)
                f.write(f'map_Ks {prefix}texture_ks.png\n')
                misc.save_images(folder, ks.permute(0,3,1,2).detach().cpu().numpy(), fnames=[prefix + "texture_ks"] * batch_size)
                # disable normal
                # f.write(f'bump {prefix}texture_n.png\n')
                # misc.save_images(folder, normal.permute(0,3,1,2).detach().cpu().numpy(), fnames=[prefix + "texture_n"] * batch_size)
            if 'kd' in material.keys():
                f.write('map_Kd texture_kd.png\n')
                texture.save_texture2D(os.path.join(folder, 'texture_Kd.png'), texture.rgb_to_srgb(material['kd']))
            if 'ks' in material.keys():
                f.write('map_Ks texture_ks.png\n')
                texture.save_texture2D(os.path.join(folder, 'texture_Ks.png'), material['ks'])
            if 'normal' in material.keys():
                f.write('bump texture_n.png\n')
                texture.save_texture2D(os.path.join(folder, 'texture_n.png'), material['normal'], lambda_fn=lambda x:(util.safe_normalize(x)+1)*0.5)
        else:
            f.write('Kd 1 1 1\n')
            f.write('Ks 0 0 0\n')
            f.write('Ka 0 0 0\n')
            f.write('Tf 1 1 1\n')
            f.write('Ni 1\n')
            f.write('Ns 0\n')

######################################################################################
# Merge multiple materials into a single uber-material
######################################################################################

def _upscale_replicate(x, full_res):
    x = x.permute(0, 3, 1, 2)
    x = torch.nn.functional.pad(x, (0, full_res[1] - x.shape[3], 0, full_res[0] - x.shape[2]), 'replicate')
    return x.permute(0, 2, 3, 1).contiguous()

def merge_materials(materials, texcoords, tfaces, mfaces):
    assert len(materials) > 0
    for mat in materials:
        assert mat['bsdf'] == materials[0]['bsdf'], "All materials must have the same BSDF (uber shader)"
        assert ('normal' in mat) is ('normal' in materials[0]), "All materials must have either normal map enabled or disabled"

    uber_material = Material({
        'name' : 'uber_material',
        'bsdf' : materials[0]['bsdf'],
    })

    textures = ['kd', 'ks', 'normal']

    # Find maximum texture resolution across all materials and textures
    max_res = None
    for mat in materials:
        for tex in textures:
            tex_res = np.array(mat[tex].getRes()) if tex in mat else np.array([1, 1])
            max_res = np.maximum(max_res, tex_res) if max_res is not None else tex_res
    
    # Compute size of compund texture and round up to nearest PoT
    full_res = 2**np.ceil(np.log2(max_res * np.array([1, len(materials)]))).astype(np.int)

    # Normalize texture resolution across all materials & combine into a single large texture
    for tex in textures:
        if tex in materials[0]:
            tex_data = torch.cat(tuple(util.scale_img_nhwc(mat[tex].data, tuple(max_res)) for mat in materials), dim=2) # Lay out all textures horizontally, NHWC so dim2 is x
            tex_data = _upscale_replicate(tex_data, full_res)
            uber_material[tex] = texture.Texture2D(tex_data)

    # Compute scaling values for used / unused texture area
    s_coeff = [full_res[0] / max_res[0], full_res[1] / max_res[1]]

    # Recompute texture coordinates to cooincide with new composite texture
    new_tverts = {}
    new_tverts_data = []
    for fi in range(len(tfaces)):
        matIdx = mfaces[fi]
        for vi in range(3):
            ti = tfaces[fi][vi]
            if not (ti in new_tverts):
                new_tverts[ti] = {}
            if not (matIdx in new_tverts[ti]): # create new vertex
                new_tverts_data.append([(matIdx + texcoords[ti][0]) / s_coeff[1], texcoords[ti][1] / s_coeff[0]]) # Offset texture coodrinate (x direction) by material id & scale to local space. Note, texcoords are (u,v) but texture is stored (w,h) so the indexes swap here
                new_tverts[ti][matIdx] = len(new_tverts_data) - 1
            tfaces[fi][vi] = new_tverts[ti][matIdx] # reindex vertex

    return uber_material, new_tverts_data, tfaces

######################################################################################
# Utility functions for material
######################################################################################

def initial_guess_material(cfgs, mlp=False, init_mat=None, tet_bbox=None):
    kd_min = torch.tensor(cfgs.get('kd_min', [0., 0., 0., 0.]), dtype=torch.float32)
    kd_max = torch.tensor(cfgs.get('kd_max', [1., 1., 1., 1.]), dtype=torch.float32)
    ks_min = torch.tensor(cfgs.get('ks_min', [0., 0., 0.]), dtype=torch.float32)
    ks_max = torch.tensor(cfgs.get('ks_max', [0., 0., 0.]), dtype=torch.float32)
    nrm_min = torch.tensor(cfgs.get('nrm_min', [-1., -1., 0.]), dtype=torch.float32)
    nrm_max = torch.tensor(cfgs.get('nrm_max', [1., 1., 1.]), dtype=torch.float32)
    if mlp:
        num_layers = cfgs.get("num_layers_tex", 5)
        nf = cfgs.get("hidden_size", 128)
        enable_encoder = cfgs.get("enable_encoder", False)
        feat_dim = cfgs.get("latent_dim", 64) if enable_encoder else 0

        mlp_min = torch.cat((kd_min[0:3], ks_min, nrm_min), dim=0)
        mlp_max = torch.cat((kd_max[0:3], ks_max, nrm_max), dim=0)
        min_max = torch.stack((mlp_min, mlp_max), dim=0)
        out_chn = 9
        mlp_map_opt = mlptexture.MLPTexture3D(tet_bbox, channels=out_chn, internal_dims=nf, hidden=num_layers-1, feat_dim=feat_dim, min_max=min_max)
        mat =  Material({'kd_ks_normal' : mlp_map_opt})
    else:
        # Setup Kd (albedo) and Ks (x, roughness, metalness) textures
        if cfgs.random_textures or init_mat is None:
            num_channels = 4 if cfgs.layers > 1 else 3
            kd_init = torch.rand(size=cfgs.texture_res + [num_channels]) * (kd_max - kd_min)[None, None, 0:num_channels] + kd_min[None, None, 0:num_channels]
            kd_map_opt = texture.create_trainable(kd_init , cfgs.texture_res, not cfgs.custom_mip, [kd_min, kd_max])

            ksR = np.random.uniform(size=cfgs.texture_res + [1], low=0.0, high=0.01)
            ksG = np.random.uniform(size=cfgs.texture_res + [1], low=ks_min[1].cpu(), high=ks_max[1].cpu())
            ksB = np.random.uniform(size=cfgs.texture_res + [1], low=ks_min[2].cpu(), high=ks_max[2].cpu())

            ks_map_opt = texture.create_trainable(np.concatenate((ksR, ksG, ksB), axis=2), cfgs.texture_res, not cfgs.custom_mip, [ks_min, ks_max])
        else:
            kd_map_opt = texture.create_trainable(init_mat['kd'], cfgs.texture_res, not cfgs.custom_mip, [kd_min, kd_max])
            ks_map_opt = texture.create_trainable(init_mat['ks'], cfgs.texture_res, not cfgs.custom_mip, [ks_min, ks_max])

        # Setup normal map
        if cfgs.random_textures or init_mat is None or 'normal' not in init_mat:
            normal_map_opt = texture.create_trainable(np.array([0, 0, 1]), cfgs.texture_res, not cfgs.custom_mip, [nrm_min, nrm_max])
        else:
            normal_map_opt = texture.create_trainable(init_mat['normal'], cfgs.texture_res, not cfgs.custom_mip, [nrm_min, nrm_max])

        mat = Material({
            'kd'     : kd_map_opt,
            'ks'     : ks_map_opt,
            'normal' : normal_map_opt
        })

    if init_mat is not None:
        mat['bsdf'] = init_mat['bsdf']
    elif "bsdf" in cfgs:
        mat['bsdf'] = cfgs["bsdf"]
    else:
        mat['bsdf'] = 'pbr'

    if not cfgs.get("perturb_normal", False):
        mat['no_perturbed_nrm'] = True

    return mat