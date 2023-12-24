# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import torch
import xatlas
import trimesh
import numpy as np
import cv2
import nvdiffrast.torch as dr
from video3d.render.render import render_uv
from video3d.render.mesh import Mesh
from . import texture
from . import mesh
from . import material

######################################################################################
# Utility functions
######################################################################################

def _find_mat(materials, name):
    for mat in materials:
        if mat['name'] == name:
            return mat
    return materials[0] # Materials 0 is the default

######################################################################################
# Create mesh object from objfile
######################################################################################

def load_obj(filename, clear_ks=True, mtl_override=None):
    obj_path = os.path.dirname(filename)

    # Read entire file
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Load materials
    all_materials = [
        {
            'name' : '_default_mat',
            'bsdf' : 'pbr',
            'kd'   : texture.Texture2D(torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device='cuda')),
            'ks'   : texture.Texture2D(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device='cuda'))
        }
    ]
    if mtl_override is None: 
        for line in lines:
            if len(line.split()) == 0:
                continue
            if line.split()[0] == 'mtllib':
                all_materials += material.load_mtl(os.path.join(obj_path, line.split()[1]), clear_ks) # Read in entire material library
    else:
        all_materials += material.load_mtl(mtl_override)

    # load vertices
    vertices, texcoords, normals  = [], [], []
    for line in lines:
        if len(line.split()) == 0:
            continue
        
        prefix = line.split()[0].lower()
        if prefix == 'v':
            vertices.append([float(v) for v in line.split()[1:]])
        elif prefix == 'vt':
            val = [float(v) for v in line.split()[1:]]
            texcoords.append([val[0], 1.0 - val[1]])
        elif prefix == 'vn':
            normals.append([float(v) for v in line.split()[1:]])

    # load faces
    activeMatIdx = None
    used_materials = []
    faces, tfaces, nfaces, mfaces = [], [], [], []
    for line in lines:
        if len(line.split()) == 0:
            continue

        prefix = line.split()[0].lower()
        if prefix == 'usemtl': # Track used materials
            mat = _find_mat(all_materials, line.split()[1])
            if not mat in used_materials:
                used_materials.append(mat)
            activeMatIdx = used_materials.index(mat)
        elif prefix == 'f': # Parse face
            vs = line.split()[1:]
            nv = len(vs)
            vv = vs[0].split('/')
            v0 = int(vv[0]) - 1
            t0 = int(vv[1]) - 1 if vv[1] != "" else -1
            n0 = int(vv[2]) - 1 if vv[2] != "" else -1
            for i in range(nv - 2): # Triangulate polygons
                vv = vs[i + 1].split('/')
                v1 = int(vv[0]) - 1
                t1 = int(vv[1]) - 1 if vv[1] != "" else -1
                n1 = int(vv[2]) - 1 if vv[2] != "" else -1
                vv = vs[i + 2].split('/')
                v2 = int(vv[0]) - 1
                t2 = int(vv[1]) - 1 if vv[1] != "" else -1
                n2 = int(vv[2]) - 1 if vv[2] != "" else -1
                mfaces.append(activeMatIdx)
                faces.append([v0, v1, v2])
                tfaces.append([t0, t1, t2])
                nfaces.append([n0, n1, n2])
    assert len(tfaces) == len(faces) and len(nfaces) == len (faces)

    # Create an "uber" material by combining all textures into a larger texture
    if len(used_materials) > 1:
        uber_material, texcoords, tfaces = material.merge_materials(used_materials, texcoords, tfaces, mfaces)
    else:
        uber_material = used_materials[0]

    vertices = torch.tensor(vertices, dtype=torch.float32, device='cuda')
    texcoords = torch.tensor(texcoords, dtype=torch.float32, device='cuda') if len(texcoords) > 0 else None
    normals = torch.tensor(normals, dtype=torch.float32, device='cuda') if len(normals) > 0 else None
    
    faces = torch.tensor(faces, dtype=torch.int64, device='cuda')
    tfaces = torch.tensor(tfaces, dtype=torch.int64, device='cuda') if texcoords is not None else None
    nfaces = torch.tensor(nfaces, dtype=torch.int64, device='cuda') if normals is not None else None

    return mesh.Mesh(vertices, faces, normals, nfaces, texcoords, tfaces, material=uber_material)

######################################################################################
# Save mesh object to objfile
######################################################################################

def write_obj(folder, fname, mesh, idx, save_material=True, feat=None, resolution=[256, 256]):
    obj_file = os.path.join(folder, fname + '.obj')
    print("Writing mesh: ", obj_file)
    with open(obj_file, "w") as f:
        f.write(f"mtllib {fname}.mtl\n")
        f.write("g default\n")

        v_pos = mesh.v_pos[idx].detach().cpu().numpy() if mesh.v_pos is not None else None
        v_nrm = mesh.v_nrm[idx].detach().cpu().numpy() if mesh.v_nrm is not None else None
        v_tex = mesh.v_tex[idx].detach().cpu().numpy() if mesh.v_tex is not None else None

        t_pos_idx = mesh.t_pos_idx[0].detach().cpu().numpy() if mesh.t_pos_idx is not None else None
        t_nrm_idx = mesh.t_nrm_idx[0].detach().cpu().numpy() if mesh.t_nrm_idx is not None else None
        t_tex_idx = mesh.t_tex_idx[0].detach().cpu().numpy() if mesh.t_tex_idx is not None else None

        print("    writing %d vertices" % len(v_pos))
        for v in v_pos:
            f.write('v {} {} {} \n'.format(v[0], v[1], v[2]))
       
        if v_tex is not None and save_material:
            print("    writing %d texcoords" % len(v_tex))
            assert(len(t_pos_idx) == len(t_tex_idx))
            for v in v_tex:
                f.write('vt {} {} \n'.format(v[0], 1.0 - v[1]))

        if v_nrm is not None:
            print("    writing %d normals" % len(v_nrm))
            assert(len(t_pos_idx) == len(t_nrm_idx))
            for v in v_nrm:
                f.write('vn {} {} {}\n'.format(v[0], v[1], v[2]))

        # faces
        f.write("s 1 \n")
        f.write("g pMesh1\n")
        f.write("usemtl defaultMat\n")

        # Write faces
        print("    writing %d faces" % len(t_pos_idx))
        for i in range(len(t_pos_idx)):
            f.write("f ")
            for j in range(3):
                f.write(' %s/%s/%s' % (str(t_pos_idx[i][j]+1), '' if v_tex is None else str(t_tex_idx[i][j]+1), '' if v_nrm is None else str(t_nrm_idx[i][j]+1)))
            f.write("\n")

    if save_material and mesh.material is not None:
        mtl_file = os.path.join(folder, fname + '.mtl')
        print("Writing material: ", mtl_file)
        material.save_mtl(mtl_file, mesh.material, mesh=mesh.get_n(idx), feat=feat, resolution=resolution)

    print("Done exporting mesh")


def write_textured_obj(folder, fname, mesh, idx, save_material=True, feat=None, resolution=[256, 256], prior_shape=None):
    mesh = mesh.get_n(idx)
    obj_file = os.path.join(folder, fname + '.obj')
    print("Writing mesh: ", obj_file)

    # Create uvs with xatlas
    v_pos = mesh.v_pos.detach().cpu().numpy()
    t_pos_idx = mesh.t_pos_idx.detach().cpu().numpy()

    # v_color = torch.Tensor(v_pos)[None].to("cuda")
    # v_color = mesh.material.sample(v_color, feat)
    # v_color = v_color[0,0,:,:3].detach().cpu()
    # v_color = torch.concat([v_color, torch.ones((v_color.shape[0], 1))], dim=-1)
    # v_color = v_color.numpy() * 255
    # v_color = v_color.astype(np.int32)
    # tmp = trimesh.Trimesh(vertices=v_pos[0], faces=t_pos_idx[0], vertex_colors=v_color)
    # _ = tmp.export("tmp.obj")
    # from pdb import set_trace; set_trace()

    atlas = xatlas.Atlas()
    atlas.add_mesh(
        v_pos[0],
        t_pos_idx[0],
    )
    co = xatlas.ChartOptions()
    po = xatlas.PackOptions()
    # for k, v in xatlas_chart_options.items():
    #     setattr(co, k, v)
    # for k, v in xatlas_pack_options.items():
    #     setattr(po, k, v)
    atlas.generate(co, po)
    vmapping, indices, uvs = atlas.get_mesh(0)
    # vmapping, indices, uvs = xatlas.parametrize(v_pos[0], t_pos_idx[0])

    # Convert to tensors
    indices_int64 = indices.astype(np.uint64, casting='same_kind').view(np.int64)
    
    uvs = torch.tensor(uvs, dtype=torch.float32, device='cuda')
    faces = torch.tensor(indices_int64, dtype=torch.int64, device='cuda')

    # new_mesh = Mesh(v_tex=uvs, t_tex_idx=faces, base=mesh)
    new_mesh = Mesh(v_tex=uvs[None], t_tex_idx=faces[None], base=mesh)

    # glctx = dr.RasterizeGLContext()
    # mask, kd, ks, normal = render_uv(glctx, new_mesh, resolution, mesh.material, feat=feat)

    # kd_min, kd_max = torch.tensor([ 0.0,  0.0,  0.0,  0.0], dtype=torch.float32, device='cuda'), torch.tensor([ 1.0,  1.0,  1.0,  1.0], dtype=torch.float32, device='cuda')
    # ks_min, ks_max = torch.tensor([ 0.0,  0.0,  0.0] , dtype=torch.float32, device='cuda'), torch.tensor([ 0.0,  0.0,  0.0] , dtype=torch.float32, device='cuda')
    # nrm_min, nrm_max = torch.tensor([-1.0, -1.0,  0.0], dtype=torch.float32, device='cuda'), torch.tensor([ 1.0,  1.0,  1.0], dtype=torch.float32, device='cuda')

    new_mesh.material = material.Material({
        'bsdf'   : 'diffuse',
        # 'kd'     : texture.Texture2D(kd, min_max=[kd_min, kd_max]),
        # 'ks'     : texture.Texture2D(ks, min_max=[ks_min, ks_max]),
        # 'normal' : texture.Texture2D(normal, min_max=[nrm_min, nrm_max]),
        'kd_ks_normal': mesh.material
    })

    with open(obj_file, "w") as f:
        f.write(f"mtllib {fname}.mtl\n")
        f.write("g default\n")

        v_pos = new_mesh.v_pos[idx].detach().cpu().numpy() if new_mesh.v_pos is not None else None
        v_nrm = new_mesh.v_nrm[idx].detach().cpu().numpy() if new_mesh.v_nrm is not None else None
        v_tex = new_mesh.v_tex[idx].detach().cpu().numpy() if new_mesh.v_tex is not None else None

        t_pos_idx = new_mesh.t_pos_idx[0].detach().cpu().numpy() if new_mesh.t_pos_idx is not None else None
        t_nrm_idx = new_mesh.t_nrm_idx[0].detach().cpu().numpy() if new_mesh.t_nrm_idx is not None else None
        t_tex_idx = new_mesh.t_tex_idx[0].detach().cpu().numpy() if new_mesh.t_tex_idx is not None else None

        print("    writing %d vertices" % len(v_pos))
        for v in v_pos:
            f.write('v {} {} {} \n'.format(v[0], v[1], v[2]))
       
        if v_tex is not None and save_material:
            print("    writing %d texcoords" % len(v_tex))
            assert(len(t_pos_idx) == len(t_tex_idx))
            for v in v_tex:
                f.write('vt {} {} \n'.format(v[0], 1.0 - v[1]))

        if v_nrm is not None:
            print("    writing %d normals" % len(v_nrm))
            assert(len(t_pos_idx) == len(t_nrm_idx))
            for v in v_nrm:
                f.write('vn {} {} {}\n'.format(v[0], v[1], v[2]))

        # faces
        f.write("s 1 \n")
        f.write("g pMesh1\n")
        f.write("usemtl defaultMat\n")

        # Write faces
        print("    writing %d faces" % len(t_pos_idx))
        for i in range(len(t_pos_idx)):
            f.write("f ")
            for j in range(3):
                f.write(' %s/%s/%s' % (str(t_pos_idx[i][j]+1), '' if v_tex is None else str(t_tex_idx[i][j]+1), '' if v_nrm is None else str(t_nrm_idx[i][j]+1)))
            f.write("\n")

    mtl_file = os.path.join(folder, fname + '.mtl')
    print("Writing material: ", mtl_file)
    material.save_mtl(mtl_file, new_mesh.material, mesh=new_mesh, feat=feat, resolution=resolution, prior_shape=prior_shape)

    print("Done exporting mesh")