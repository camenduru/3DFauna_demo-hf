import os
import fire
import gradio as gr
from PIL import Image
from functools import partial
import argparse
import sys
import torch

if os.getenv('SYSTEM') == 'spaces':
    os.system('pip install --global-option="--no-networks" git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch')

    os.system('pip install fvcore iopath')
    os.system('pip install "git+https://github.com/facebookresearch/pytorch3d.git"')

import cv2
import time
import numpy as np
import trimesh
from segment_anything import build_sam, SamPredictor

import random
from pytorch3d import transforms
import torchvision
import torch.distributed as dist
import nvdiffrast.torch as dr
from video3d.model_ddp import Unsup3DDDP, forward_to_matrix
from video3d.trainer_few_shot import Fewshot_Trainer
from video3d.trainer_ddp import TrainerDDP
from video3d import setup_runtime
from video3d.render.mesh import make_mesh
from video3d.utils.skinning_v4 import estimate_bones, skinning, euler_angles_to_matrix
from video3d.utils.misc import save_obj
from video3d.render import util
import matplotlib.pyplot as plt
from pytorch3d import utils, renderer, transforms, structures, io
from video3d.render.render import render_mesh
from video3d.render.material import texture as material_texture


_TITLE = '''Learning the 3D Fauna of the Web'''
_DESCRIPTION = '''
<div>
Reconstruct any quadruped animal from one image.
</div>
<div>
The demo only contains the 3D reconstruction part.
</div>
'''
_GPU_ID = 0

if not hasattr(Image, 'Resampling'):
    Image.Resampling = Image


def sam_init():
    sam_checkpoint = os.path.join(os.path.dirname(__file__), "sam_pt", "sam_vit_h_4b8939.pth")
    model_type = "vit_h"
    # sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=f"cuda:{_GPU_ID}")
    # predictor = SamPredictor(sam)
    predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to("cuda"))
    return predictor


def sam_segment(predictor, input_image, *bbox_coords):
    bbox = np.array(bbox_coords)
    image = np.asarray(input_image)

    start_time = time.time()
    predictor.set_image(image)

    masks_bbox, scores_bbox, logits_bbox = predictor.predict(
        box=bbox,
        multimask_output=True
    )

    print(f"SAM Time: {time.time() - start_time:.3f}s")
    out_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    out_image[:, :, :3] = image
    out_image_bbox = out_image.copy()
    out_image_bbox[:, :, 3] = masks_bbox[-1].astype(np.uint8) * 255
    torch.cuda.empty_cache()
    return Image.fromarray(out_image_bbox, mode='RGB')
    # return Image.fromarray(out_image_bbox, mode='RGBA')


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def preprocess(predictor, input_image, chk_group=None, segment=False):
    RES = 1024
    input_image.thumbnail([RES, RES], Image.Resampling.LANCZOS)
    if chk_group is not None:
        segment = "Use SAM to center animal" in chk_group
    if segment:
        image_rem = input_image.convert('RGB')
        arr = np.asarray(image_rem)[:,:,-1]
        x_nonzero = np.nonzero(arr.sum(axis=0))
        y_nonzero = np.nonzero(arr.sum(axis=1))
        x_min = int(x_nonzero[0].min())
        y_min = int(y_nonzero[0].min())
        x_max = int(x_nonzero[0].max())
        y_max = int(y_nonzero[0].max())
        input_image = sam_segment(predictor, input_image.convert('RGB'), x_min, y_min, x_max, y_max)
    # Rescale and recenter
    # if rescale:
    #     image_arr = np.array(input_image)
    #     in_w, in_h = image_arr.shape[:2]
    #     out_res = min(RES, max(in_w, in_h))
    #     ret, mask = cv2.threshold(np.array(input_image.split()[-1]), 0, 255, cv2.THRESH_BINARY)
    #     x, y, w, h = cv2.boundingRect(mask)
    #     max_size = max(w, h)
    #     ratio = 0.75
    #     side_len = int(max_size / ratio)
    #     padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
    #     center = side_len//2
    #     padded_image[center-h//2:center-h//2+h, center-w//2:center-w//2+w] = image_arr[y:y+h, x:x+w]
    #     rgba = Image.fromarray(padded_image).resize((out_res, out_res), Image.LANCZOS)

    #     rgba_arr = np.array(rgba) / 255.0
    #     rgb = rgba_arr[...,:3] * rgba_arr[...,-1:] + (1 - rgba_arr[...,-1:])
    #     input_image = Image.fromarray((rgb * 255).astype(np.uint8))
    # else:
    #     input_image = expand2square(input_image, (127, 127, 127, 0))
    
    input_image = expand2square(input_image, (0, 0, 0))
    return input_image, input_image.resize((256, 256), Image.Resampling.LANCZOS)


def save_images(images, mask_pred, mode="transparent"):
    img = images[0]
    mask = mask_pred[0]
    img = img.clamp(0, 1)
    if mask is not None:
        mask = mask.clamp(0, 1)
        if mode == "white":
            img = img * mask + 1 * (1 - mask)
        elif mode == "black":
            img = img * mask + 0 * (1 - mask)
        else:
            img = torch.cat([img, mask[0:1]], 0)
    
    img = img.permute(1, 2, 0).cpu().numpy()
    img = Image.fromarray(np.uint8(img * 255))
    return img


def get_bank_embedding(rgb, memory_bank_keys, memory_bank, model, memory_bank_topk=10, memory_bank_dim=128):
    images = rgb
    batch_size, num_frames, _, h0, w0 = images.shape
    images = images.reshape(batch_size*num_frames, *images.shape[2:])  # 0~1
    images_in = images * 2 - 1  # rescale to (-1, 1) for DINO
    
    x = images_in
    with torch.no_grad():
        b, c, h, w = x.shape
        model.netInstance.netEncoder._feats = []
        model.netInstance.netEncoder._register_hooks([11], 'key')
        #self._register_hooks([11], 'token')
        x = model.netInstance.netEncoder.ViT.prepare_tokens(x)
        #x = self.ViT.prepare_tokens_with_masks(x)
        
        for blk in model.netInstance.netEncoder.ViT.blocks:
            x = blk(x)
        out = model.netInstance.netEncoder.ViT.norm(x)
        model.netInstance.netEncoder._unregister_hooks()

        ph, pw = h // model.netInstance.netEncoder.patch_size, w // model.netInstance.netEncoder.patch_size
        patch_out = out[:, 1:]  # first is class token
        patch_out = patch_out.reshape(b, ph, pw, model.netInstance.netEncoder.vit_feat_dim).permute(0, 3, 1, 2)

        patch_key = model.netInstance.netEncoder._feats[0][:,:,1:]  # B, num_heads, num_patches, dim
        patch_key = patch_key.permute(0, 1, 3, 2).reshape(b, model.netInstance.netEncoder.vit_feat_dim, ph, pw)

        global_feat = out[:, 0]
    
    batch_features = global_feat

    batch_size = batch_features.shape[0]
        
    query = torch.nn.functional.normalize(batch_features.unsqueeze(1), dim=-1)      # [B, 1, d_k]
    key = torch.nn.functional.normalize(memory_bank_keys, dim=-1)              # [size, d_k]
    key = key.transpose(1, 0).unsqueeze(0).repeat(batch_size, 1, 1).to(query.device)             # [B, d_k, size]

    cos_dist = torch.bmm(query, key).squeeze(1)         # [B, size], larger the more similar
    rank_idx = torch.sort(cos_dist, dim=-1, descending=True)[1][:, :memory_bank_topk] # [B, k]
    value = memory_bank.unsqueeze(0).repeat(batch_size, 1, 1).to(query.device)                         # [B, size, d_v]

    out = torch.gather(value, dim=1, index=rank_idx[..., None].repeat(1, 1, memory_bank_dim))  # [B, k, d_v]

    weights = torch.gather(cos_dist, dim=-1, index=rank_idx)    # [B, k]
    weights = torch.nn.functional.normalize(weights, p=1.0, dim=-1).unsqueeze(-1).repeat(1, 1, memory_bank_dim)    # [B, k, d_v] weights have been normalized

    out = weights * out
    out = torch.sum(out, dim=1)
    
    batch_mean_out = torch.mean(out, dim=0)

    weight_aux = {
        'weights': weights[:, :, 0], # [B, k], weights from large to small
        'pick_idx': rank_idx, # [B, k]
    }

    batch_embedding = batch_mean_out 
    embeddings = out
    weights = weight_aux

    bank_embedding_model_input = [batch_embedding, embeddings, weights]

    return bank_embedding_model_input


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


def render_bones(mvp, bones_pred, size=(256, 256)):
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
    return torch.from_numpy(np.stack(rendered, 0).transpose(0, 3, 1, 2)).to(bones_pred.device)

def add_mesh_color(mesh, color):
    verts = mesh.verts_padded()
    color = torch.FloatTensor(color).to(verts.device).view(1,1,3) / 255
    mesh.textures = renderer.TexturesVertex(verts_features=verts*0+color)
    return mesh

def create_sphere(position, scale, device, color=[139, 149, 173]):
    mesh = utils.ico_sphere(2).to(device)
    mesh = mesh.extend(position.shape[0])

    # scale and offset
    mesh = mesh.update_padded(mesh.verts_padded() * scale + position[:, None])

    mesh = add_mesh_color(mesh, color)

    return mesh

def estimate_bone_rotation(b):
    """
    (0, 0, 1) = matmul(R^(-1), b)

    assumes x, y is a symmetry plane

    returns R
    """
    b = b / torch.norm(b, dim=-1, keepdim=True)

    n = torch.FloatTensor([[1, 0, 0]]).to(b.device)
    n = n.expand_as(b)
    v = torch.cross(b, n, dim=-1)

    R = torch.stack([n, v, b], dim=-1).transpose(-2, -1)

    return R

def estimate_vector_rotation(vector_a, vector_b):
    """
    vector_a = matmul(R, vector_b)

    returns R

    https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    """
    vector_a = vector_a / torch.norm(vector_a, dim=-1, keepdim=True)
    vector_b = vector_b / torch.norm(vector_b, dim=-1, keepdim=True)

    v = torch.cross(vector_a, vector_b, dim=-1)
    c = torch.sum(vector_a * vector_b, dim=-1)

    skew = torch.stack([
        torch.stack([torch.zeros_like(v[..., 0]), -v[..., 2], v[..., 1]], dim=-1),
        torch.stack([v[..., 2], torch.zeros_like(v[..., 0]), -v[..., 0]], dim=-1),
        torch.stack([-v[..., 1], v[..., 0], torch.zeros_like(v[..., 0])], dim=-1)],
        dim=-1)

    R = torch.eye(3, device=vector_a.device)[None] + skew + torch.matmul(skew, skew) / (1  + c[..., None, None])

    return R

def create_elipsoid(bone, scale=0.05, color=[139, 149, 173], generic_rotation_estim=True):
    length = torch.norm(bone[:, 0] - bone[:, 1], dim=-1)

    mesh = utils.ico_sphere(2).to(bone.device)
    mesh = mesh.extend(bone.shape[0])
    # scale x, y
    verts = mesh.verts_padded() * torch.FloatTensor([scale, scale, 1]).to(bone.device)
    # stretch along z axis, set the start to origin
    verts[:, :, 2] = verts[:, :, 2] * length[:, None] * 0.5 + length[:, None] * 0.5

    bone_vector = bone[:, 1] - bone[:, 0]
    z_vector = torch.FloatTensor([[0, 0, 1]]).to(bone.device)
    z_vector = z_vector.expand_as(bone_vector)
    if generic_rotation_estim:
        rot = estimate_vector_rotation(z_vector, bone_vector)
    else:
        rot = estimate_bone_rotation(bone_vector)
    tsf = transforms.Rotate(rot, device=bone.device)
    tsf = tsf.compose(transforms.Translate(bone[:, 0], device=bone.device))
    verts = tsf.transform_points(verts)

    mesh = mesh.update_padded(verts)

    mesh = add_mesh_color(mesh, color)

    return mesh

def convert_textures_vertex_to_textures_uv(meshes: structures.Meshes, color1, color2) -> renderer.TexturesUV:
    """
    Convert a TexturesVertex object to a TexturesUV object.
    """
    color1 = torch.Tensor(color1).to(meshes.device).view(1, 1, 3) / 255
    color2 = torch.Tensor(color2).to(meshes.device).view(1, 1, 3) / 255
    textures_vertex = meshes.textures
    assert isinstance(textures_vertex, renderer.TexturesVertex), "Input meshes must have TexturesVertex"
    verts_rgb = textures_vertex.verts_features_padded()
    faces_uvs = meshes.faces_padded()
    batch_size = verts_rgb.shape[0]
    maps = torch.zeros(batch_size, 128, 128, 3, device=verts_rgb.device)
    maps[:, :, :64, :] = color1
    maps[:, :, 64:, :] = color2
    is_first = (verts_rgb == color1)[..., 0]
    verts_uvs = torch.zeros(batch_size, verts_rgb.shape[1], 2, device=verts_rgb.device)
    verts_uvs[is_first] = torch.FloatTensor([0.25, 0.5]).to(verts_rgb.device)
    verts_uvs[~is_first] = torch.FloatTensor([0.75, 0.5]).to(verts_rgb.device)
    textures_uv = renderer.TexturesUV(maps=maps, faces_uvs=faces_uvs, verts_uvs=verts_uvs)
    meshes.textures = textures_uv
    return meshes
    
def create_bones_scene(bones, joint_color=[66, 91, 140], bone_color=[119, 144, 189], show_end_point=False):
    meshes = []
    for bone_i in range(bones.shape[1]):
        # points
        meshes += [create_sphere(bones[:, bone_i, 0], 0.1, bones.device, color=joint_color)]
        if show_end_point:
            meshes += [create_sphere(bones[:, bone_i, 1], 0.1, bones.device, color=joint_color)]

        # connecting ellipsoid
        meshes += [create_elipsoid(bones[:, bone_i], color=bone_color)]

    current_batch_size = bones.shape[0]
    meshes = [structures.join_meshes_as_scene([m[i] for m in meshes]) for i in range(current_batch_size)]
    mesh = structures.join_meshes_as_batch(meshes)

    return mesh


def save_mesh(mesh, file_path):
    obj_file = file_path
    idx = 0
    print("Writing mesh: ", obj_file)
    with open(obj_file, "w") as f:
        # f.write(f"mtllib {fname}.mtl\n")
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


def process_mesh(shape, name):
    mesh = shape.clone()
    output_glb = f'./{name}.glb'
    output_obj = f'./{name}.obj'

    # save the obj file for download
    save_mesh(mesh, output_obj)

    # save the glb for visualize
    mesh_tri = trimesh.Trimesh(
        vertices=mesh.v_pos[0].detach().cpu().numpy(), 
        faces=mesh.t_pos_idx[0][..., [2,1,0]].detach().cpu().numpy(), 
        process=False,  
        maintain_order=True
    )
    mesh_tri.visual.vertex_colors = (mesh.v_nrm[0][..., [2,1,0]].detach().cpu().numpy() + 1.0) * 0.5 * 255.0
    mesh_tri.export(file_obj=output_glb)

    return output_glb, output_obj


def run_pipeline(model_items, cfgs, input_img):
    epoch = 999
    total_iter = 999999
    model = model_items[0]
    memory_bank = model_items[1]
    memory_bank_keys = model_items[2]

    device = f'cuda:{_GPU_ID}'

    input_image = torch.stack([torchvision.transforms.ToTensor()(input_img)], dim=0).to(device)

    with torch.no_grad():
        model.netPrior.eval()
        model.netInstance.eval()
        input_image = torch.nn.functional.interpolate(input_image, size=(256, 256), mode='bilinear', align_corners=False)
        input_image = input_image[:, None, :, :]  # [B=1, F=1, 3, 256, 256]

        bank_embedding = get_bank_embedding(
            input_image, 
            memory_bank_keys, 
            memory_bank, 
            model, 
            memory_bank_topk=cfgs.get("memory_bank_topk", 10),
            memory_bank_dim=128
        )

        prior_shape, dino_pred, classes_vectors = model.netPrior(
            category_name='tmp',
            perturb_sdf=False, 
            total_iter=total_iter,
            is_training=False, 
            class_embedding=bank_embedding
        )
        Instance_out = model.netInstance(
            'tmp', 
            input_image, 
            prior_shape, 
            epoch, 
            dino_features=None, 
            dino_clusters=None, 
            total_iter=total_iter, 
            is_training=False
        )  # frame dim collapsed N=(B*F)
        if len(Instance_out) == 13:
            shape, pose_raw, pose, mvp, w2c, campos, texture_pred, im_features, dino_feat_im_calc, deform, all_arti_params, light, forward_aux = Instance_out
            im_features_map = None
        else:
            shape, pose_raw, pose, mvp, w2c, campos, texture_pred, im_features, dino_feat_im_calc, deform, all_arti_params, light, forward_aux, im_features_map = Instance_out

        class_vector = classes_vectors  # the bank embeddings

        gray_light = FixedDirectionLight(direction=torch.FloatTensor([0, 0, 1]).to(device), amb=0.2, diff=0.7)

        image_pred, mask_pred, _, _, _, shading = model.render(
            shape, texture_pred, mvp, w2c, campos, (256, 256), background=model.background_mode, 
            im_features=im_features, light=gray_light, prior_shape=prior_shape, render_mode='diffuse', 
            render_flow=False, dino_pred=None, im_features_map=im_features_map
        )
        mask_pred = mask_pred.expand_as(image_pred)
        shading = shading.expand_as(image_pred)
        # render bones in pytorch3D style
        posed_bones = forward_aux["posed_bones"].squeeze(1)
        jc, bc = [66, 91, 140], [119, 144, 189]
        bones_meshes = create_bones_scene(posed_bones, joint_color=jc, bone_color=bc, show_end_point=True)
        bones_meshes = convert_textures_vertex_to_textures_uv(bones_meshes, color1=jc, color2=bc)
        nv_meshes = make_mesh(verts=bones_meshes.verts_padded(), faces=bones_meshes.faces_padded()[0:1],
                                uvs=bones_meshes.textures.verts_uvs_padded(), uv_idx=bones_meshes.textures.faces_uvs_padded()[0:1],
                                material=material_texture.Texture2D(bones_meshes.textures.maps_padded()))
        # buffers = render_mesh(dr.RasterizeGLContext(), nv_meshes, mvp, w2c, campos, nv_meshes.material, lgt=gray_light, feat=im_features, dino_pred=None, resolution=(256,256), bsdf="diffuse")
        buffers = render_mesh(dr.RasterizeCudaContext(), nv_meshes, mvp, w2c, campos, nv_meshes.material, lgt=gray_light, feat=im_features, dino_pred=None, resolution=(256,256), bsdf="diffuse")
        
        shaded = buffers["shaded"].permute(0, 3, 1, 2)
        bone_image = shaded[:, :3, :, :]
        bone_mask = shaded[:, 3:, :, :]
        mask_final = mask_pred.logical_or(bone_mask)
        mask_final = mask_final.int()
        image_with_bones = bone_image * bone_mask * 0.5 + (shading * (1 - bone_mask * 0.5) + 0.5 * (mask_final.float() - mask_pred.float()))

        mesh_image = save_images(shading, mask_pred)
        mesh_bones_image = save_images(image_with_bones, mask_final)

        shape_glb, shape_obj = process_mesh(shape, 'reconstruced_shape')
        base_shape_glb, base_shape_obj = process_mesh(prior_shape, 'reconstructed_base_shape')

        return mesh_image, mesh_bones_image, shape_glb, shape_obj, base_shape_glb, base_shape_obj


def run_demo():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', type=str,
                        help='Specify a GPU device')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Specify the number of worker threads for data loaders')
    parser.add_argument('--seed', default=0, type=int,
                        help='Specify a random seed')
    parser.add_argument('--config', default='./ckpts/configs.yml',
                        type=str)  # Model config path
    parser.add_argument('--checkpoint_path', default='./ckpts/iter0800000.pth', type=str)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8088'
    dist.init_process_group("gloo", rank=_GPU_ID, world_size=1)
    torch.cuda.set_device(_GPU_ID)
    args.rank = _GPU_ID
    args.world_size = 1
    args.gpu = f'{_GPU_ID}'
    device = f'cuda:{_GPU_ID}'

    resolution = (256, 256)
    batch_size = 1
    model_cfgs = setup_runtime(args)
    bone_y_thresh = 0.4
    body_bone_idx_preset = [3, 6, 6, 3]
    model_cfgs['body_bone_idx_preset'] = body_bone_idx_preset

    model = Unsup3DDDP(model_cfgs)
    # a hack attempt
    model.netPrior.classes_vectors = torch.nn.Parameter(torch.nn.init.uniform_(torch.empty(123, 128), a=-0.05, b=0.05))
    cp = torch.load(args.checkpoint_path, map_location=device)
    model.load_model_state(cp)
    memory_bank_keys = cp['memory_bank_keys']
    memory_bank = cp['memory_bank']

    model.to(device)
    memory_bank.to(device)
    memory_bank_keys.to(device)
    model_items = [
        model,
        memory_bank,
        memory_bank_keys
    ]

    predictor = sam_init()

    custom_theme = gr.themes.Soft(primary_hue="blue").set(
                    button_secondary_background_fill="*neutral_100",
                    button_secondary_background_fill_hover="*neutral_200")
    custom_css = '''#disp_image {
        text-align: center; /* Horizontally center the content */
    }'''

    with gr.Blocks(title=_TITLE, theme=custom_theme, css=custom_css) as demo:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown('# ' + _TITLE)
        gr.Markdown(_DESCRIPTION)
        with gr.Row(variant='panel'):
            with gr.Column(scale=1):
                input_image = gr.Image(type='pil', image_mode='RGBA', height=256, label='Input image', tool=None)

                example_folder = os.path.join(os.path.dirname(__file__), "./example_images")
                example_fns = [os.path.join(example_folder, example) for example in os.listdir(example_folder)]
                gr.Examples(
                    examples=example_fns,
                    inputs=[input_image],
                    # outputs=[input_image],
                    cache_examples=False,
                    label='Examples (click one of the images below to start)',
                    examples_per_page=30
                )
            with gr.Column(scale=1):
                processed_image = gr.Image(type='pil', label="Processed Image", interactive=False, height=256, tool=None, image_mode='RGB', elem_id="disp_image")
                processed_image_highres = gr.Image(type='pil', image_mode='RGB', visible=False, tool=None)

                with gr.Accordion('Advanced options', open=True):
                    with gr.Row():
                        with gr.Column():
                            input_processing = gr.CheckboxGroup(['Use SAM to center animal'], 
                                                                label='Input Image Preprocessing',
                                                                 info='untick this, if animal is already centered, e.g. in example images')
                        # with gr.Column():
                        #     output_processing = gr.CheckboxGroup(['Background Removal'], label='Output Image Postprocessing', value=[]) 
                    # with gr.Row():
                    #     with gr.Column():
                    #         scale_slider = gr.Slider(1, 5, value=3, step=1,
                    #                                     label='Classifier Free Guidance Scale')
                    #     with gr.Column():
                    #         steps_slider = gr.Slider(15, 100, value=50, step=1,
                    #                                     label='Number of Diffusion Inference Steps')
                    # with gr.Row():
                    #     with gr.Column():
                    #         seed = gr.Number(42, label='Seed')
                    #     with gr.Column():
                    #         crop_size = gr.Number(192, label='Crop size')
                    # crop_size = 192
                run_btn = gr.Button('Reconstruct', variant='primary', interactive=True)
        with gr.Row():
            view_1 = gr.Image(interactive=False, height=256, show_label=False)
            view_2 = gr.Image(interactive=False, height=256, show_label=False)
        with gr.Row():
            shape_1 = gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0],  height=512, label="Reconstructed Model")
            shape_1_download = gr.File(label="Download Full Reconstructed Model")
        with gr.Row():
            shape_2 = gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0],  height=512, label="Bank Base Shape Model")
            shape_2_download = gr.File(label="Download Full Bank Base Shape Model")

        run_btn.click(fn=partial(preprocess, predictor), 
                        inputs=[input_image, input_processing], 
                        outputs=[processed_image_highres, processed_image], queue=True
            ).success(fn=partial(run_pipeline, model_items, model_cfgs), 
                        inputs=[processed_image],
                        outputs=[view_1, view_2, shape_1, shape_1_download, shape_2, shape_2_download]
                        )
        demo.queue().launch(share=True, max_threads=80)
        # _, local_url, share_url = demo.queue().launch(share=True, server_name="0.0.0.0", server_port=23425)
        # print('local_url: ', local_url)


if __name__ == '__main__':
    fire.Fire(run_demo)