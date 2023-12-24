import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
# import pytorch3d
# import pytorch3d.loss
# import pytorch3d.renderer
# import pytorch3d.structures
# import pytorch3d.io
# import pytorch3d.transforms
from PIL import Image
from .utils import sphere
from einops import rearrange


def update_camera_pose(cameras, position, at):
    cameras.R = pytorch3d.renderer.look_at_rotation(position, at).to(cameras.device)
    cameras.T = -torch.bmm(cameras.R.transpose(1, 2), position[:, :, None])[:, :, 0]


def get_soft_rasterizer_settings(image_size, sigma=1e-6, gamma=1e-6, faces_per_pixel=30):
    blend_params = pytorch3d.renderer.BlendParams(sigma=sigma, gamma=gamma)
    settings = pytorch3d.renderer.RasterizationSettings(
        image_size=image_size,
        blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
        faces_per_pixel=faces_per_pixel,
    )
    return settings, blend_params


class Renderer(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.cfgs = cfgs
        self.device = cfgs.get('device', 'cpu')
        self.image_size = cfgs.get('out_image_size', 64)
        self.full_size_h = cfgs.get('full_size_h', 1080)
        self.full_size_w = cfgs.get('full_size_w', 1920)
        self.fov_w = cfgs.get('fov_w', 60)
        # self.fov_h = cfgs.get('fov_h', 30)
        self.fov_h = np.arctan(np.tan(self.fov_w /2 /180*np.pi) / self.full_size_w * self.full_size_h) *2 /np.pi*180
        self.crop_fov_approx = cfgs.get('crop_fov_approx', 25)
        self.cam_pos_z_offset = cfgs.get('cam_pos_z_offset', 10.)
        self.max_range = np.tan(min(self.fov_h, self.fov_w) /2 /180 * np.pi) * self.cam_pos_z_offset
        cam_pos = torch.FloatTensor([[0, 0, self.cam_pos_z_offset]]).to(self.device)
        cam_at = torch.FloatTensor([[0, 0, 0]]).to(self.device)
        self.rot_rep = cfgs.get('rot_rep', 'euler_angle')
        # self.cameras = pytorch3d.renderer.FoVPerspectiveCameras(fov=self.crop_fov_approx).to(self.device)
        # update_camera_pose(self.cameras, position=cam_pos, at=cam_at)
        # self.full_cameras = pytorch3d.renderer.FoVPerspectiveCameras(fov=self.fov_w).to(self.device)
        # update_camera_pose(self.full_cameras, position=cam_pos, at=cam_at)
        self.image_renderer = self._create_image_renderer()
        self.ico_sphere_subdiv = cfgs.get('ico_sphere_subdiv', 2)
        self.init_shape_scale_xy = float(cfgs.get('init_shape_scale_xy', 1.))
        self.init_shape_scale_z = float(cfgs.get('init_shape_scale_z', 1.))
        # init_verts, init_faces, init_aux = pytorch3d.io.load_obj(cfgs['init_shape_obj'], create_texture_atlas=True, device=self.device)
        # self.init_verts = init_verts *self.init_shape_scale
        # self.meshes = pytorch3d.structures.Meshes(verts=[self.init_verts], faces=[init_faces.verts_idx]).to(self.device)
        # self.tex_faces_uv = init_faces.textures_idx.unsqueeze(0)
        # self.tex_verts_uv = init_aux.verts_uvs.unsqueeze(0)
        # self.texture_atlas = init_aux.texture_atlas.unsqueeze(0)
        # self.num_verts_total = init_verts.size(0)

        # cmap = plt.cm.get_cmap('hsv', self.num_verts_total)
        # verts_texture = cmap(np.random.permutation(self.num_verts_total))[:,:3]
        # self.verts_texture = torch.FloatTensor(verts_texture)
        # debug_uvtex = cfgs.get('debug_uvtex', None)
        # if debug_uvtex is not None:
        #     face_tex_map = Image.open(debug_uvtex).convert('RGB').resize((512, 512))
        #     self.face_tex_map = torch.FloatTensor(np.array(face_tex_map)).permute(2,0,1) / 255.
        # else:
        #     self.face_tex_map = None

        meshes, aux = sphere.get_symmetric_ico_sphere(subdiv=self.ico_sphere_subdiv, return_tex_uv=True, return_face_tex_map=True, device=self.device)
        init_verts = meshes.verts_padded()
        self.init_verts = init_verts * torch.FloatTensor([self.init_shape_scale_xy, self.init_shape_scale_xy, self.init_shape_scale_z]).view(1,1,3).to(init_verts.device)
        # TODO: is this needed?
        self.meshes = meshes.update_padded(init_verts * 0)
        self.tex_faces_uv = aux['face_tex_ids'].unsqueeze(0)
        self.tex_verts_uv = aux['verts_tex_uv'].unsqueeze(0)
        self.face_tex_map = aux['face_tex_map'].permute(2,0,1)
        self.tex_map_seam_mask = aux['seam_mask'].permute(2,0,1)
        self.num_verts_total = init_verts.size(1)
        self.num_verts_seam = aux['num_verts_seam']
        self.num_verts_one_side = aux['num_verts_one_side']

        # hack to turn off texture symmetry
        if cfgs.get('disable_sym_tex', False):
            tex_uv_seam1 = self.tex_verts_uv[:,:aux['num_verts_seam']].clone()
            tex_uv_seam1[:,:,0] = tex_uv_seam1[:,:,0] /2 + 0.5
            tex_uv_side1 = self.tex_verts_uv[:,aux['num_verts_seam']:aux['num_verts_seam']+aux['num_verts_one_side']].clone()
            tex_uv_side1[:,:,0] = tex_uv_side1[:,:,0] /2 + 0.5
            tex_uv_seam2 = self.tex_verts_uv[:,:aux['num_verts_seam']].clone()
            tex_uv_seam2[:,:,0] = tex_uv_seam2[:,:,0] /2
            tex_uv_side2 = self.tex_verts_uv[:,aux['num_verts_seam']+aux['num_verts_one_side']:].clone()
            tex_uv_side2[:,:,0] = tex_uv_side2[:,:,0] /2
            self.tex_verts_uv = torch.cat([tex_uv_seam1, tex_uv_side1, tex_uv_side2, tex_uv_seam2], 1)

            num_faces = self.tex_faces_uv.shape[1]
            face_tex_ids1 = self.tex_faces_uv[:, :num_faces//2].clone()
            face_tex_ids2 = self.tex_faces_uv[:, num_faces//2:].clone()
            face_tex_ids2[face_tex_ids2 < aux['num_verts_seam']] += aux['num_verts_seam'] + 2*aux['num_verts_one_side']
            self.tex_faces_uv = torch.cat([face_tex_ids1, face_tex_ids2], 1)
            self.face_tex_map = torch.cat([self.face_tex_map, self.face_tex_map.flip(2)], 2)
            self.tex_map_seam_mask = torch.cat([self.tex_map_seam_mask, self.tex_map_seam_mask.flip(2)], 2)

    def _create_silhouette_renderer(self):
        settings, blend_params = get_soft_rasterizer_settings(self.image_size)
        return pytorch3d.renderer.MeshRenderer(
            rasterizer=pytorch3d.renderer.MeshRasterizer(cameras=self.cameras, raster_settings=settings),
            shader=pytorch3d.renderer.SoftSilhouetteShader(cameras=self.cameras, blend_params=blend_params)
        )

    def _create_image_renderer(self):
        settings, blend_params = get_soft_rasterizer_settings(self.image_size)
        lights = pytorch3d.renderer.DirectionalLights(device=self.device,
                                                      ambient_color=((1., 1., 1.),),
                                                      diffuse_color=((0., 0., 0.),),
                                                      specular_color=((0., 0., 0.),),
                                                      direction=((0, 1, 0),))
        return pytorch3d.renderer.MeshRenderer(
            rasterizer=pytorch3d.renderer.MeshRasterizer(cameras=self.cameras, raster_settings=settings),
            shader=pytorch3d.renderer.SoftPhongShader(device=self.device, lights=lights, cameras=self.cameras, blend_params=blend_params)
        )

    def transform_verts(self, verts, pose):
        b, f, _ = pose.shape
        if self.rot_rep == 'euler_angle' or self.rot_rep == 'soft_calss':
            rot_mat = pytorch3d.transforms.euler_angles_to_matrix(pose[...,:3].view(-1,3), convention='XYZ')
            tsf = pytorch3d.transforms.Rotate(rot_mat, device=pose.device)
        elif self.rot_rep == 'quaternion':
            rot_mat = pytorch3d.transforms.quaternion_to_matrix(pose[...,:4].view(-1,4))
            tsf = pytorch3d.transforms.Rotate(rot_mat, device=pose.device)
        elif self.rot_rep == 'lookat':
            rot_mat = pose[...,:9].view(-1,3,3)
            tsf = pytorch3d.transforms.Rotate(rot_mat, device=pose.device)
        else:
            raise NotImplementedError
        tsf = tsf.compose(pytorch3d.transforms.Translate(pose[...,-3:].view(-1,3), device=pose.device))
        new_verts = tsf.transform_points(verts.view(b*f, *verts.shape[2:]))
        return new_verts.view(b, f, *new_verts.shape[1:])

    # def transform_mesh(self, mesh, pose):
    #     mesh_verts = mesh.verts_padded()
    #     new_mesh_verts = self.transform_verts(mesh_verts, pose)
    #     new_mesh = mesh.update_padded(new_mesh_verts)
    #     return new_mesh

    def symmetrize_shape(self, shape):
        verts_seam = shape[:,:,:self.num_verts_seam] * torch.FloatTensor([0,1,1]).to(shape.device)
        verts_one_side = shape[:,:,self.num_verts_seam:self.num_verts_seam+self.num_verts_one_side] * torch.FloatTensor([1,1,1]).to(shape.device)
        verts_other_side = verts_one_side * torch.FloatTensor([-1,1,1]).to(shape.device)
        shape = torch.cat([verts_seam, verts_one_side, verts_other_side], 2)
        return shape

    def get_deformed_mesh(self, shape, pose=None, return_shape=False):
        b, f, _, _ = shape.shape
        if pose is not None:
            shape = self.transform_verts(shape, pose)
        mesh = self.meshes.extend(b*f)
        mesh = mesh.update_padded(rearrange(shape, 'b f ... -> (b f) ...'))
        if return_shape:
            return shape, mesh
        else:
            return mesh

    def get_textures(self, tex_im):
        b, f, c, h, w = tex_im.shape

        ## top half texture map in ico_sphere.obj is unused, pad with zeros
        # if 'sym' not in self.cfgs.get('init_shape_obj', ''):
            # tex_im = torch.cat([torch.zeros_like(tex_im), tex_im], 3)
        # tex_im = nn.functional.interpolate(tex_im, (h, w), mode='bilinear', align_corners=False)
        textures = pytorch3d.renderer.TexturesUV(maps=tex_im.view(b*f, *tex_im.shape[2:]).permute(0, 2, 3, 1),  # texture maps are BxHxWx3
                                                 faces_uvs=self.tex_faces_uv.repeat(b*f, 1, 1),
                                                 verts_uvs=self.tex_verts_uv.repeat(b*f, 1, 1))
        return textures

    def render_flow(self, meshes, shape, pose, deformed_shape=None):
        # verts = meshes.verts_padded()  # (B*F)xVx3
        b, f, _, _ = shape.shape
        if f < 2:
            return None

        if deformed_shape is None:
            deformed_shape, meshes = self.get_deformed_mesh(shape.detach(), pose=pose, return_shape=True)
        im_size = torch.FloatTensor([self.image_size, self.image_size]).to(shape.device)  # (w,h)
        verts_2d = self.cameras.transform_points_screen(deformed_shape.view(b*f, *deformed_shape.shape[2:]), im_size.view(1,2).repeat(b*f,1), eps=1e-7)
        verts_2d = verts_2d.view(b, f, *verts_2d.shape[1:])
        verts_flow = verts_2d[:, 1:, :, :2] - verts_2d[:, :-1, :, :2]  # Bx(F-1)xVx(x,y)
        verts_flow = verts_flow / im_size.view(1, 1, 1, 2) * 0.5 + 0.5  # 0~1
        flow_tex = torch.nn.functional.pad(verts_flow, pad=[0, 1, 0, 0, 0, 1])  # BxFxVx3

        # meshes = meshes.detach()  # detach mesh when rendering flow (only texture has gradients)
        # meshes = self.get_deformed_mesh(shape.detach())
        meshes.textures = pytorch3d.renderer.TexturesVertex(verts_features=flow_tex.view(b*f, -1, 3))
        flow = self.image_renderer(meshes_world=meshes, cameras=self.cameras)
        # settings, blend_params = get_soft_rasterizer_settings(image_size=self.image_size, sigma=1e-6, gamma=1e-6, faces_per_pixel=5)
        # flow = self.image_renderer(meshes_world=meshes, cameras=self.cameras, raster_settings=settings, blend_params=blend_params)
        flow = flow.view(b, f, *flow.shape[1:])[:, :-1]  # Bx(F-1)xHxWx3
        flow_mask = (flow[:, :, :, :, 3:] > 0.01).float()
        return (flow[:, :, :, :, :2] - 0.5) * 2 * flow_mask  # Bx(F-1)xHxWx2

    def forward(self, pose, texture, shape, crop_bbox=None, render_flow=True):
        b, f, _ = pose.shape

        ## compensate crop with intrinsics, assuming square crops
        # x0, y0, w, h = crop_bbox.unbind(2)
        # fx = 1 / np.tan(self.fov_w / 2 /180*np.pi)
        # fy = fx
        # sx = w / self.full_size_w
        # sy = sx
        # cx = ((x0+w/2) - (self.full_size_w/2)) / (self.full_size_w/2)  # [0-w] -> [-1,1]
        # cy = ((y0+h/2) - (self.full_size_h/2)) / (self.full_size_w/2)
        # znear = 1
        # zfar = 100
        # v1 = zfar / (zfar - znear)
        # v2 = -(zfar * znear) / (zfar - znear)
        #
        # # K = [[[ fx/sx,  0.0000,  cx/sx,  0.0000],
        # #       [ 0.0000,  fy/sy,  cy/sy,  0.0000],
        # #       [ 0.0000,  0.0000,  v1, v2],
        # #       [ 0.0000,  0.0000,  1.0000,  0.0000]]]
        # zeros = torch.zeros_like(sx)
        # K_row1 = torch.stack([fx/sx,  zeros,  cx/sx,  zeros], 2)
        # K_row2 = torch.stack([zeros,  fy/sy,  cy/sy,  zeros], 2)
        # K_row3 = torch.stack([zeros,  zeros,  zeros+v1,  zeros+v2], 2)
        # K_row4 = torch.stack([zeros,  zeros,  zeros+1,  zeros], 2)
        # K = torch.stack([K_row1, K_row2, K_row3, K_row4], 2)  # BxFx4x4
        # self.crop_cameras = pytorch3d.renderer.FoVPerspectiveCameras(K=K.view(-1, 4, 4), R=self.cameras.R, T=self.cameras.T, device=self.device)
        # # reset znear, zfar to scalar to bypass broadcast bug in pytorch3d blending
        # self.crop_cameras.znear = znear
        # self.crop_cameras.zfar = zfar

        deformed_shape, mesh = self.get_deformed_mesh(shape, pose=pose, return_shape=True)
        if render_flow:
            flow = self.render_flow(mesh, shape, pose, deformed_shape=deformed_shape)  # Bx(F-1)xHxWx2
            # flow = self.render_flow(mesh, shape, pose, deformed_shape=None)  # Bx(F-1)xHxWx2
        else:
            flow = None
        mesh.textures = self.get_textures(texture)
        image = self.image_renderer(meshes_world=mesh, cameras=self.cameras)
        image = image.view(b, f, *image.shape[1:])
        return image, flow, mesh
