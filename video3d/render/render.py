# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import torch
import nvdiffrast.torch as dr

from . import util
from . import renderutils as ru
from . import light

# ==============================================================================================
#  Helper functions
# ==============================================================================================
def interpolate(attr, rast, attr_idx, rast_db=None):
    return dr.interpolate(attr.contiguous(), rast, attr_idx, rast_db=rast_db, diff_attrs=None if rast_db is None else 'all')

# ==============================================================================================
#  pixel shader
# ==============================================================================================
def shade(
        gb_pos,
        gb_geometric_normal,
        gb_normal,
        gb_tangent,
        gb_tex_pos,
        gb_texc,
        gb_texc_deriv,
        w2c,
        view_pos,
        lgt,
        material,
        bsdf,
        feat,
        two_sided_shading,
        delta_xy_interp=None,
        dino_pred=None,
        class_vector=None,
        im_features_map=None,
        mvp=None
    ):

    ################################################################################
    # Texture lookups
    ################################################################################
    perturbed_nrm = None
    # Combined texture, used for MLPs because lookups are expensive
    # all_tex_jitter = material.sample(gb_tex_pos + torch.normal(mean=0, std=0.01, size=gb_tex_pos.shape, device="cuda"), feat=feat)
    if material is not None:
        if im_features_map is None:
            all_tex = material.sample(gb_tex_pos, feat=feat)
        else:
            all_tex = material.sample(gb_tex_pos, feat=feat, feat_map=im_features_map, mvp=mvp, w2c=w2c, deform_xyz=gb_pos)
    else:
        all_tex = torch.ones(*gb_pos.shape[:-1], 9, device=gb_pos.device)
    kd, ks, perturbed_nrm = all_tex[..., :3], all_tex[..., 3:6], all_tex[..., 6:9]

    # Compute albedo (kd) gradient, used for material regularizer
    # kd_grad    = torch.sum(torch.abs(all_tex_jitter[..., :-6] - all_tex[..., :-6]), dim=-1, keepdim=True) / 
    
    if dino_pred is not None and class_vector is None:
        # DOR: predive the dino value using x,y,z, we would concatenate the label vector. 
        # trained together, generated image as the supervision for the one-hot-vector.
        dino_feat_im_pred = dino_pred.sample(gb_tex_pos)
        # dino_feat_im_pred = dino_pred.sample(gb_tex_pos.detach())
    if dino_pred is not None and class_vector is not None:
        dino_feat_im_pred = dino_pred.sample(gb_tex_pos, feat=class_vector)

    # else:
    #     kd_jitter  = material['kd'].sample(gb_texc + torch.normal(mean=0, std=0.005, size=gb_texc.shape, device="cuda"), gb_texc_deriv)
    #     kd = material['kd'].sample(gb_texc, gb_texc_deriv)
    #     ks = material['ks'].sample(gb_texc, gb_texc_deriv)[..., 0:3] # skip alpha
    #     if 'normal' in material:
    #         perturbed_nrm = material['normal'].sample(gb_texc, gb_texc_deriv)
    #     kd_grad    = torch.sum(torch.abs(kd_jitter[..., 0:3] - kd[..., 0:3]), dim=-1, keepdim=True) / 3

    # Separate kd into alpha and color, default alpha = 1
    # alpha = kd[..., 3:4] if kd.shape[-1] == 4 else torch.ones_like(kd[..., 0:1]) 
    # kd = kd[..., 0:3]
    alpha = torch.ones_like(kd[..., 0:1])

    ################################################################################
    # Normal perturbation & normal bend
    ################################################################################
    if material is None or not material.perturb_normal:
        perturbed_nrm = None

    gb_normal = ru.prepare_shading_normal(gb_pos, view_pos, perturbed_nrm, gb_normal, gb_tangent, gb_geometric_normal, two_sided_shading=two_sided_shading, opengl=True, use_python=True)

    # if two_sided_shading:
    #     view_vec = util.safe_normalize(view_pos - gb_pos, -1)
    #     gb_normal = torch.where(torch.sum(gb_geometric_normal * view_vec, -1, keepdim=True) > 0, gb_geometric_normal, -gb_geometric_normal)
    # else:
    #     gb_normal = gb_geometric_normal
    
    b, h, w, _ = gb_normal.shape
    cam_normal = util.safe_normalize(torch.matmul(gb_normal.view(b, -1, 3), w2c[:,:3,:3].transpose(2,1))).view(b, h, w, 3)

    ################################################################################
    # Evaluate BSDF
    ################################################################################

    assert bsdf is not None or material.bsdf is not None, "Material must specify a BSDF type"
    bsdf = bsdf if bsdf is not None else material.bsdf
    shading = None
    if bsdf == 'pbr':
        if isinstance(lgt, light.EnvironmentLight):
            shaded_col = lgt.shade(gb_pos, gb_normal, kd, ks, view_pos, specular=True)
        else:
            assert False, "Invalid light type"
    elif bsdf == 'diffuse':
        if lgt is None:
            shaded_col = kd
        elif isinstance(lgt, light.EnvironmentLight):
            shaded_col = lgt.shade(gb_pos, gb_normal, kd, ks, view_pos, specular=False)
        # elif isinstance(lgt, light.DirectionalLight):
        #     shaded_col, shading = lgt.shade(feat, kd, cam_normal)
        # else:
        #     assert False, "Invalid light type"
        else:
            shaded_col, shading = lgt.shade(feat, kd, cam_normal)
    elif bsdf == 'normal':
        shaded_col = (gb_normal + 1.0) * 0.5
    elif bsdf == 'geo_normal':
        shaded_col = (gb_geometric_normal + 1.0) * 0.5
    elif bsdf == 'tangent':
        shaded_col = (gb_tangent + 1.0) * 0.5
    elif bsdf == 'kd':
        shaded_col = kd
    elif bsdf == 'ks':
        shaded_col = ks
    else:
        assert False, "Invalid BSDF '%s'" % bsdf
    
    # Return multiple buffers
    buffers = {
        'kd'   : torch.cat((kd, alpha), dim=-1),
        'shaded'    : torch.cat((shaded_col, alpha), dim=-1),
        # 'kd_grad'   : torch.cat((kd_grad, alpha), dim=-1),
        # 'occlusion' : torch.cat((ks[..., :1], alpha), dim=-1),
    }

    if dino_pred is not None:
        buffers['dino_feat_im_pred'] = torch.cat((dino_feat_im_pred, alpha), dim=-1)

    if delta_xy_interp is not None:
        buffers['flow'] = torch.cat((delta_xy_interp, alpha), dim=-1)
    
    if shading is not None:
        buffers['shading'] = torch.cat((shading, alpha), dim=-1)
    
    return buffers

# ==============================================================================================
#  Render a depth slice of the mesh (scene), some limitations:
#  - Single light
#  - Single material
# ==============================================================================================
def render_layer(
        rast,
        rast_deriv,
        mesh,
        w2c,
        view_pos,
        material,
        lgt,
        resolution,
        spp,
        msaa,
        bsdf,
        feat,
        prior_mesh,
        two_sided_shading,
        render_flow,
        delta_xy=None,
        dino_pred=None,
        class_vector=None,
        im_features_map=None,
        mvp=None
    ):

    full_res = [resolution[0]*spp, resolution[1]*spp]

    if prior_mesh is None:
        prior_mesh = mesh

    ################################################################################
    # Rasterize
    ################################################################################

    # Scale down to shading resolution when MSAA is enabled, otherwise shade at full resolution
    if spp > 1 and msaa:
        rast_out_s = util.scale_img_nhwc(rast, resolution, mag='nearest', min='nearest')
        rast_out_deriv_s = util.scale_img_nhwc(rast_deriv, resolution, mag='nearest', min='nearest') * spp
    else:
        rast_out_s = rast
        rast_out_deriv_s = rast_deriv

    if render_flow:
        delta_xy_interp, _ = interpolate(delta_xy, rast_out_s, mesh.t_pos_idx[0].int())
    else:
        delta_xy_interp = None

    ################################################################################
    # Interpolate attributes
    ################################################################################

    # Interpolate world space position
    gb_pos, _ = interpolate(mesh.v_pos, rast_out_s, mesh.t_pos_idx[0].int())

    # Compute geometric normals. We need those because of bent normals trick (for bump mapping)
    v0 = mesh.v_pos[:, mesh.t_pos_idx[0, :, 0], :]
    v1 = mesh.v_pos[:, mesh.t_pos_idx[0, :, 1], :]
    v2 = mesh.v_pos[:, mesh.t_pos_idx[0, :, 2], :]
    face_normals = util.safe_normalize(torch.cross(v1 - v0, v2 - v0, dim=-1))
    num_faces = face_normals.shape[1]
    face_normal_indices = (torch.arange(0, num_faces, dtype=torch.int64, device='cuda')[:, None]).repeat(1, 3)
    gb_geometric_normal, _ = interpolate(face_normals, rast_out_s, face_normal_indices.int())

    # Compute tangent space
    assert mesh.v_nrm is not None and mesh.v_tng is not None
    gb_normal, _ = interpolate(mesh.v_nrm, rast_out_s, mesh.t_nrm_idx[0].int())
    gb_tangent, _ = interpolate(mesh.v_tng, rast_out_s, mesh.t_tng_idx[0].int()) # Interpolate tangents

    # Texture coordinate
    assert mesh.v_tex is not None
    gb_texc, gb_texc_deriv = interpolate(mesh.v_tex, rast_out_s, mesh.t_tex_idx[0].int(), rast_db=rast_out_deriv_s)

    ################################################################################
    # Shade
    ################################################################################
    
    gb_tex_pos, _ = interpolate(prior_mesh.v_pos, rast_out_s, mesh.t_pos_idx[0].int())
    buffers = shade(gb_pos, gb_geometric_normal, gb_normal, gb_tangent, gb_tex_pos, gb_texc, gb_texc_deriv, w2c, view_pos, lgt, material, bsdf, feat=feat, two_sided_shading=two_sided_shading, delta_xy_interp=delta_xy_interp, dino_pred=dino_pred, class_vector=class_vector, im_features_map=im_features_map, mvp=mvp)

    ################################################################################
    # Prepare output
    ################################################################################

    # Scale back up to visibility resolution if using MSAA
    if spp > 1 and msaa:
        for key in buffers.keys():
            buffers[key] = util.scale_img_nhwc(buffers[key], full_res, mag='nearest', min='nearest')

    # Return buffers
    return buffers

# ==============================================================================================
#  Render a depth peeled mesh (scene), some limitations:
#  - Single light
#  - Single material
# ==============================================================================================
def render_mesh(
        ctx,
        mesh,
        mtx_in,
        w2c,
        view_pos,
        material,
        lgt,
        resolution,
        spp         = 1,
        num_layers  = 1,
        msaa        = False,
        background  = None, 
        bsdf        = None,
        feat        = None,
        prior_mesh  = None,
        two_sided_shading = True,
        render_flow = False,
        dino_pred = None,
        class_vector = None, 
        num_frames = None,
        im_features_map = None
    ):

    def prepare_input_vector(x):
        x = torch.tensor(x, dtype=torch.float32, device='cuda') if not torch.is_tensor(x) else x
        return x[:, None, None, :] if len(x.shape) == 2 else x
    
    def composite_buffer(key, layers, background, antialias):
        accum = background
        for buffers, rast in reversed(layers):
            alpha = (rast[..., -1:] > 0).float() * buffers[key][..., -1:]
            accum = torch.lerp(accum, torch.cat((buffers[key][..., :-1], torch.ones_like(buffers[key][..., -1:])), dim=-1), alpha)
            if antialias:
                accum = dr.antialias(accum.contiguous(), rast, v_pos_clip, mesh.t_pos_idx[0].int())
        return accum

    assert mesh.t_pos_idx.shape[1] > 0, "Got empty training triangle mesh (unrecoverable discontinuity)"
    assert background is None or (background.shape[1] == resolution[0] and background.shape[2] == resolution[1])

    full_res = [resolution[0] * spp, resolution[1] * spp]

    # Convert numpy arrays to torch tensors
    mtx_in      = torch.tensor(mtx_in, dtype=torch.float32, device='cuda') if not torch.is_tensor(mtx_in) else mtx_in
    view_pos    = prepare_input_vector(view_pos)  # Shape: (B, 1, 1, 3)

    # clip space transform
    v_pos_clip = ru.xfm_points(mesh.v_pos, mtx_in, use_python=True)

    # render flow
    if render_flow:
        v_pos_clip2 = v_pos_clip[..., :2] / v_pos_clip[..., -1:]
        v_pos_clip2 = v_pos_clip2.view(-1, num_frames, *v_pos_clip2.shape[1:])
        delta_xy = v_pos_clip2[:, 1:] - v_pos_clip2[:, :-1]
        delta_xy = torch.cat([delta_xy, torch.zeros_like(delta_xy[:, :1])], dim=1)
        delta_xy = delta_xy.view(-1, *delta_xy.shape[2:])
    else:
        delta_xy = None

    # Render all layers front-to-back
    layers = []
    with dr.DepthPeeler(ctx, v_pos_clip, mesh.t_pos_idx[0].int(), full_res) as peeler:
        for _ in range(num_layers):
            rast, db = peeler.rasterize_next_layer()
            rendered = render_layer(rast, db, mesh, w2c, view_pos, material, lgt, resolution, spp, msaa, bsdf, feat=feat, prior_mesh=prior_mesh, two_sided_shading=two_sided_shading, render_flow=render_flow, delta_xy=delta_xy, dino_pred=dino_pred, class_vector=class_vector, im_features_map=im_features_map, mvp=mtx_in)
            layers += [(rendered, rast)]

    # Setup background
    if background is not None:
        if spp > 1:
            background = util.scale_img_nhwc(background, full_res, mag='nearest', min='nearest')
        background = torch.cat((background, torch.zeros_like(background[..., 0:1])), dim=-1)
    else:
        background = torch.zeros(1, full_res[0], full_res[1], 4, dtype=torch.float32, device='cuda')

    # Composite layers front-to-back
    out_buffers = {}
    for key in layers[0][0].keys():
        antialias = key in ['shaded', 'dino_feat_im_pred', 'flow']
        bg = background if key in ['shaded'] else torch.zeros_like(layers[0][0][key])
        accum = composite_buffer(key, layers, bg, antialias)

        # Downscale to framebuffer resolution. Use avg pooling 
        out_buffers[key] = util.avg_pool_nhwc(accum, spp) if spp > 1 else accum

    return out_buffers

# ==============================================================================================
#  Render UVs
# ==============================================================================================
def render_uv(ctx, mesh, resolution, mlp_texture, feat=None, prior_shape=None):

    # clip space transform 
    uv_clip = mesh.v_tex * 2.0 - 1.0

    # pad to four component coordinate
    uv_clip4 = torch.cat((uv_clip, torch.zeros_like(uv_clip[...,0:1]), torch.ones_like(uv_clip[...,0:1])), dim = -1)

    # rasterize
    rast, _ = dr.rasterize(ctx, uv_clip4, mesh.t_tex_idx[0].int(), resolution)

    # Interpolate world space position
    if prior_shape is not None:
        gb_pos, _ = interpolate(prior_shape.v_pos, rast, mesh.t_pos_idx[0].int())
    else:
        gb_pos, _ = interpolate(mesh.v_pos, rast, mesh.t_pos_idx[0].int())

    # Sample out textures from MLP
    all_tex = mlp_texture.sample(gb_pos, feat=feat)
    assert all_tex.shape[-1] == 9 or all_tex.shape[-1] == 10, "Combined kd_ks_normal must be 9 or 10 channels"
    perturbed_nrm = all_tex[..., -3:]
    return (rast[..., -1:] > 0).float(), all_tex[..., :-6], all_tex[..., -6:-3], util.safe_normalize(perturbed_nrm)
