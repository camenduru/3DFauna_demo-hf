# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

from multiprocessing.spawn import get_preparation_data
import numpy as np
import torch

from ..render import mesh
from ..render import render
from ..networks import MLPWithPositionalEncoding, MLPWithPositionalEncoding_Style

###############################################################################
# Marching tetrahedrons implementation (differentiable), adapted from
# https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/ops/conversions/tetmesh.py
#
# Note this only supports batch size = 1.
###############################################################################

class DMTet:
    def __init__(self):
        self.triangle_table = torch.tensor([
                [-1, -1, -1, -1, -1, -1],
                [ 1,  0,  2, -1, -1, -1],
                [ 4,  0,  3, -1, -1, -1],
                [ 1,  4,  2,  1,  3,  4],
                [ 3,  1,  5, -1, -1, -1],
                [ 2,  3,  0,  2,  5,  3],
                [ 1,  4,  0,  1,  5,  4],
                [ 4,  2,  5, -1, -1, -1],
                [ 4,  5,  2, -1, -1, -1],
                [ 4,  1,  0,  4,  5,  1],
                [ 3,  2,  0,  3,  5,  2],
                [ 1,  3,  5, -1, -1, -1],
                [ 4,  1,  2,  4,  3,  1],
                [ 3,  0,  4, -1, -1, -1],
                [ 2,  0,  1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1]
                ], dtype=torch.long, device='cuda')

        self.num_triangles_table = torch.tensor([0,1,1,2,1,2,2,1,1,2,2,1,2,1,1,0], dtype=torch.long, device='cuda')
        self.base_tet_edges = torch.tensor([0,1,0,2,0,3,1,2,1,3,2,3], dtype=torch.long, device='cuda')

    ###############################################################################
    # Utility functions
    ###############################################################################

    def sort_edges(self, edges_ex2):
        with torch.no_grad():
            order = (edges_ex2[:,0] > edges_ex2[:,1]).long()
            order = order.unsqueeze(dim=1)

            a = torch.gather(input=edges_ex2, index=order, dim=1)      
            b = torch.gather(input=edges_ex2, index=1-order, dim=1)  

        return torch.stack([a, b],-1)

    def map_uv(self, faces, face_gidx, max_idx):
        N = int(np.ceil(np.sqrt((max_idx+1)//2)))
        tex_y, tex_x = torch.meshgrid(
            torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device="cuda"),
            torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device="cuda"),
            indexing='ij'
        )

        pad = 0.9 / N

        uvs = torch.stack([
            tex_x      , tex_y,
            tex_x + pad, tex_y,
            tex_x + pad, tex_y + pad,
            tex_x      , tex_y + pad
        ], dim=-1).view(-1, 2)

        def _idx(tet_idx, N):
            x = tet_idx % N
            y = torch.div(tet_idx, N, rounding_mode='trunc')
            return y * N + x

        tet_idx = _idx(torch.div(face_gidx, 2, rounding_mode='trunc'), N)
        tri_idx = face_gidx % 2

        uv_idx = torch.stack((
            tet_idx * 4, tet_idx * 4 + tri_idx + 1, tet_idx * 4 + tri_idx + 2
        ), dim = -1). view(-1, 3)

        return uvs, uv_idx

    ###############################################################################
    # Marching tets implementation
    ###############################################################################

    def __call__(self, pos_nx3, sdf_n, tet_fx4):
        with torch.no_grad():
            occ_n = sdf_n > 0
            occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1,4)
            occ_sum = torch.sum(occ_fx4, -1)
            valid_tets = (occ_sum>0) & (occ_sum<4)
            occ_sum = occ_sum[valid_tets]

            # find all vertices
            all_edges = tet_fx4[valid_tets][:,self.base_tet_edges].reshape(-1,2)
            all_edges = self.sort_edges(all_edges)
            unique_edges, idx_map = torch.unique(all_edges,dim=0, return_inverse=True)  
            
            unique_edges = unique_edges.long()
            mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1,2).sum(-1) == 1
            mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device="cuda") * -1
            mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long,device="cuda")
            idx_map = mapping[idx_map] # map edges to verts

            interp_v = unique_edges[mask_edges]
        edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1,2,3)
        edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1,2,1)
        edges_to_interp_sdf[:,-1] *= -1

        denominator = edges_to_interp_sdf.sum(1,keepdim = True)

        edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1])/denominator
        verts = (edges_to_interp * edges_to_interp_sdf).sum(1)

        idx_map = idx_map.reshape(-1,6)

        v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device="cuda"))
        tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1)
        num_triangles = self.num_triangles_table[tetindex]

        # Generate triangle indices
        faces = torch.cat((
            torch.gather(input=idx_map[num_triangles == 1], dim=1, index=self.triangle_table[tetindex[num_triangles == 1]][:, :3]).reshape(-1,3),
            torch.gather(input=idx_map[num_triangles == 2], dim=1, index=self.triangle_table[tetindex[num_triangles == 2]][:, :6]).reshape(-1,3),
        ), dim=0)

        # Get global face index (static, does not depend on topology)
        num_tets = tet_fx4.shape[0]
        tet_gidx = torch.arange(num_tets, dtype=torch.long, device="cuda")[valid_tets]
        face_gidx = torch.cat((
            tet_gidx[num_triangles == 1]*2,
            torch.stack((tet_gidx[num_triangles == 2]*2, tet_gidx[num_triangles == 2]*2 + 1), dim=-1).view(-1)
        ), dim=0)

        uvs, uv_idx = self.map_uv(faces, face_gidx, num_tets*2)

        return verts, faces, uvs, uv_idx

###############################################################################
# Regularizer
###############################################################################

def sdf_bce_reg_loss(sdf, all_edges):
    sdf_f1x6x2 = sdf[all_edges.reshape(-1)].reshape(-1,2)
    mask = torch.sign(sdf_f1x6x2[...,0]) != torch.sign(sdf_f1x6x2[...,1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    sdf_diff = torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,0], (sdf_f1x6x2[...,1] > 0).float()) + \
               torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,1], (sdf_f1x6x2[...,0] > 0).float())
    if torch.isnan(sdf_diff).any():
        import ipdb; ipdb.set_trace()
    return sdf_diff

###############################################################################
#  Geometry interface
###############################################################################

class DMTetGeometry(torch.nn.Module):
    def __init__(self, grid_res, scale, sdf_mode, num_layers=None, hidden_size=None, embedder_freq=None, embed_concat_pts=True, init_sdf=None, jitter_grid=0., perturb_sdf_iter=10000, sym_prior_shape=False, dim_of_classes=0, condition_choice='concat'):
        super(DMTetGeometry, self).__init__()

        self.sdf_mode = sdf_mode
        self.grid_res      = grid_res
        self.marching_tets = DMTet()
        self.grid_scale = scale
        self.init_sdf = init_sdf
        self.jitter_grid = jitter_grid
        self.perturb_sdf_iter = perturb_sdf_iter
        self.sym_prior_shape = sym_prior_shape
        self.load_tets(self.grid_res, self.grid_scale)

        if sdf_mode == "param":
            sdf = torch.rand_like(self.verts[:,0]) - 0.1  # Random init.
            self.sdf    = torch.nn.Parameter(sdf.clone().detach(), requires_grad=True)
            self.register_parameter('sdf', self.sdf)
            self.deform = torch.nn.Parameter(torch.zeros_like(self.verts), requires_grad=True)
            self.register_parameter('deform', self.deform)
        else:
            embedder_scaler = 2 * np.pi / self.grid_scale * 0.9  # originally (-0.5*s, 0.5*s) rescale to (-pi, pi) * 0.9

            if dim_of_classes == 0 or (dim_of_classes != 0 and condition_choice == 'concat'):
                self.mlp = MLPWithPositionalEncoding(
                    3, 
                    1, 
                    num_layers, 
                    nf=hidden_size, 
                    extra_dim=dim_of_classes,
                    dropout=0, 
                    activation=None, 
                    n_harmonic_functions=embedder_freq, 
                    omega0=embedder_scaler,
                    embed_concat_pts=embed_concat_pts)
            
            elif condition_choice == 'film' or condition_choice == 'mod':
                self.mlp = MLPWithPositionalEncoding_Style(
                    3, 
                    1, 
                    num_layers, 
                    nf=hidden_size, 
                    extra_dim=dim_of_classes,
                    dropout=0, 
                    activation=None, 
                    n_harmonic_functions=embedder_freq, 
                    omega0=embedder_scaler,
                    embed_concat_pts=embed_concat_pts,
                    style_choice=condition_choice)

            else:
                raise NotImplementedError

    def load_tets(self, grid_res=None, scale=None):
        if grid_res is None:
            grid_res = self.grid_res
        else:
            self.grid_res = grid_res
        if scale is None:
            scale = self.grid_scale
        else:
            self.grid_scale = scale
        tets = np.load('./data/tets/{}_tets.npz'.format(grid_res))
        self.verts = torch.tensor(tets['vertices'], dtype=torch.float32, device='cuda') * scale  # verts original scale (-0.5, 0.5)
        self.indices = torch.tensor(tets['indices'], dtype=torch.long, device='cuda')
        self.generate_edges()

    def get_sdf(self, pts=None, perturb_sdf=False, total_iter=0, class_vector=None):
        if self.sdf_mode == 'param':
            sdf = self.sdf
        else:
            if pts is None:
                pts = self.verts
            if self.sym_prior_shape:
                xs, ys, zs = pts.unbind(-1)
                pts = torch.stack([xs.abs(), ys, zs], -1)  # mirror -x to +x
            feat = None
            if class_vector is not None:
                feat = class_vector.unsqueeze(0).repeat(pts.shape[0], 1)
            sdf = self.mlp(pts, feat=feat)

        if self.init_sdf is None:
            pass
        elif type(self.init_sdf) in [float, int]:
            sdf = sdf + self.init_sdf
        elif self.init_sdf == 'sphere':
            init_radius = self.grid_scale * 0.25
            init_sdf = init_radius - pts.norm(dim=-1, keepdim=True)  # init sdf is a sphere centered at origin
            sdf = sdf + init_sdf
        elif self.init_sdf == 'ellipsoid':
            rxy = self.grid_scale * 0.15
            xs, ys, zs = pts.unbind(-1)[:3]
            init_sdf = rxy - torch.stack([xs, ys, zs/2], -1).norm(dim=-1, keepdim=True)  # init sdf is approximately an ellipsoid centered at origin
            sdf = sdf + init_sdf
        else:
            raise NotImplementedError

        if perturb_sdf:
            sdf = sdf + torch.randn_like(sdf) * 0.1 * max(0, 1-total_iter/self.perturb_sdf_iter)
        return sdf

    def get_sdf_gradient(self, class_vector=None):
        assert self.sdf_mode == 'mlp', "Only MLP supports gradient computation."
        num_samples = 5000
        sample_points = (torch.rand(num_samples, 3, device=self.verts.device) - 0.5) * self.grid_scale
        mesh_verts = self.mesh_verts.detach() + (torch.rand_like(self.mesh_verts) -0.5) * 0.1 * self.grid_scale
        rand_idx = torch.randperm(len(mesh_verts), device=mesh_verts.device)[:5000]
        mesh_verts = mesh_verts[rand_idx]
        sample_points = torch.cat([sample_points, mesh_verts], 0)
        sample_points.requires_grad = True
        y = self.get_sdf(pts=sample_points, perturb_sdf=False, class_vector=class_vector)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        try:
            gradients = torch.autograd.grad(
                outputs=[y],
                inputs=sample_points,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
        except RuntimeError:  # For validation, we have disabled gradient calculation.
            return torch.zeros_like(sample_points)
        return gradients

    def get_sdf_reg_loss(self, class_vector=None):
        reg_loss = {"sdf_bce_reg_loss": sdf_bce_reg_loss(self.current_sdf, self.all_edges).mean()}
        if self.sdf_mode == 'mlp':
            reg_loss["sdf_gradient_reg_loss"] = ((self.get_sdf_gradient(class_vector=class_vector).norm(dim=-1) - 1) ** 2).mean()
        reg_loss['sdf_inflate_reg_loss'] = -self.current_sdf.mean()
        return reg_loss
    
    def generate_edges(self):
        with torch.no_grad():
            edges = torch.tensor([0,1,0,2,0,3,1,2,1,3,2,3], dtype = torch.long, device = "cuda")
            all_edges = self.indices[:,edges].reshape(-1,2)
            all_edges_sorted = torch.sort(all_edges, dim=1)[0]
            self.all_edges = torch.unique(all_edges_sorted, dim=0)

    @torch.no_grad()
    def getAABB(self):
        return torch.min(self.verts, dim=0).values, torch.max(self.verts, dim=0).values

    def getMesh(self, material=None, perturb_sdf=False, total_iter=0, jitter_grid=True, class_vector=None):
        # Run DM tet to get a base mesh
        v_deformed = self.verts

        # if self.FLAGS.deform_grid:
        #     v_deformed = self.verts + 2 / (self.grid_res * 2) * torch.tanh(self.deform)
        # else:
        #     v_deformed = self.verts
        if jitter_grid and self.jitter_grid > 0:
            jitter = (torch.rand(1, device=v_deformed.device)*2-1) * self.jitter_grid * self.grid_scale
            v_deformed = v_deformed + jitter

        self.current_sdf = self.get_sdf(v_deformed, perturb_sdf=perturb_sdf, total_iter=total_iter, class_vector=class_vector)
        verts, faces, uvs, uv_idx = self.marching_tets(v_deformed, self.current_sdf, self.indices)
        self.mesh_verts = verts
        return mesh.make_mesh(verts[None], faces[None], uvs[None], uv_idx[None], material)

    def render(self, glctx, target, lgt, opt_material, bsdf=None):
        opt_mesh = self.getMesh(opt_material)
        return render.render_mesh(glctx, opt_mesh, target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], msaa=True, background=target['background'], bsdf=bsdf)

    def tick(self, glctx, target, lgt, opt_material, loss_fn, iteration):
        # ==============================================================================================
        #  Render optimizable object with identical conditions
        # ==============================================================================================
        buffers = self.render(glctx, target, lgt, opt_material)

        # ==============================================================================================
        #  Compute loss
        # ==============================================================================================
        t_iter = iteration / 20000

        # Image-space loss, split into a coverage component and a color component
        color_ref = target['img']
        img_loss = torch.nn.functional.mse_loss(buffers['shaded'][..., 3:], color_ref[..., 3:]) 
        img_loss = img_loss + loss_fn(buffers['shaded'][..., 0:3] * color_ref[..., 3:], color_ref[..., 0:3] * color_ref[..., 3:])

        # SDF regularizer
        # sdf_weight = self.sdf_regularizer - (self.sdf_regularizer - 0.01) * min(1.0, 4.0 * t_iter)  # Dropoff to 0.01
        reg_loss = sum(self.get_sdf_reg_loss().values)

        # Albedo (k_d) smoothnesss regularizer
        reg_loss += torch.mean(buffers['kd_grad'][..., :-1] * buffers['kd_grad'][..., -1:]) * 0.03 * min(1.0, iteration / 500)

        # Visibility regularizer
        reg_loss += torch.mean(buffers['occlusion'][..., :-1] * buffers['occlusion'][..., -1:]) * 0.001 * min(1.0, iteration / 500)

        # Light white balance regularizer
        reg_loss = reg_loss + lgt.regularizer() * 0.005

        return img_loss, reg_loss
