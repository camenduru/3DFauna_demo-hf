import torch
import numpy as np
import random
import torch.nn.functional as F

from ..render.light import DirectionalLight

def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))

def get_view_direction(thetas, phis, overhead, front, phi_offset=0):
    #                   phis [B,];          thetas: [B,]
    # front = 0         [360 - front / 2, front / 2)
    # side (left) = 1   [front / 2, 180 - front / 2)
    # back = 2          [180 - front / 2, 180 + front / 2)
    # side (right) = 3  [180 + front / 2, 360 - front / 2)
    # top = 4                               [0, overhead]
    # bottom = 5                            [180-overhead, 180]
    res = torch.zeros(thetas.shape[0], dtype=torch.long)

    # first determine by phis
    phi_offset = np.deg2rad(phi_offset)
    phis = phis + phi_offset
    phis = phis % (2 * np.pi)
    half_front = front / 2
    
    res[(phis >= (2*np.pi - half_front)) | (phis < half_front)] = 0
    res[(phis >= half_front) & (phis < (np.pi - half_front))] = 1
    res[(phis >= (np.pi - half_front)) & (phis < (np.pi + half_front))] = 2
    res[(phis >= (np.pi + half_front)) & (phis < (2*np.pi - half_front))] = 3

    # override by thetas
    res[thetas <= overhead] = 4
    res[thetas >= (np.pi - overhead)] = 5
    return res


def view_direction_id_to_text(view_direction_id):
    dir_texts = ['front', 'side', 'back', 'side', 'overhead', 'bottom']
    return [dir_texts[i] for i in view_direction_id]


def append_text_direction(prompts, dir_texts):
    return [f'{prompt}, {dir_text} view' for prompt, dir_text in zip(prompts, dir_texts)]


def rand_lights(camera_dir, fixed_ambient, fixed_diffuse):
    size = camera_dir.shape[0]
    device = camera_dir.device
    random_fixed_dir = F.normalize(torch.randn_like(camera_dir) + camera_dir, dim=-1)  # Centered around camera_dir
    random_fixed_intensity = torch.tensor([fixed_ambient, fixed_diffuse], device=device)[None, :].repeat(size, 1)  # ambient, diffuse
    return DirectionalLight(mlp_in=1, mlp_layers=1, mlp_hidden_size=1, # Dummy values
                            intensity_min_max=[0.5, 1],fixed_dir=random_fixed_dir, fixed_intensity=random_fixed_intensity).to(device)

def rand_poses(size, device, radius_range=[1, 1], theta_range=[0, 120], phi_range=[0, 360], cam_z_offset=10, return_dirs=False, angle_overhead=30, angle_front=60, phi_offset=0, jitter=False, uniform_sphere_rate=0.5):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius_range: [min, max]
        theta_range: [min, max], should be in [0, pi]
        phi_range: [min, max], should be in [0, 2 * pi]
    Return:
        poses: [size, 4, 4]
    '''

    theta_range = np.deg2rad(theta_range)
    phi_range = np.deg2rad(phi_range)
    angle_overhead = np.deg2rad(angle_overhead)
    angle_front = np.deg2rad(angle_front)
    
    radius = torch.rand(size, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]

    phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]
    if random.random() < uniform_sphere_rate:
        # based on http://corysimon.github.io/articles/uniformdistn-on-sphere/
        # acos takes in [-1, 1], first convert theta range to fit in [-1, 1] 
        theta_range = torch.from_numpy(np.array(theta_range)).to(device)
        theta_amplitude_range = torch.cos(theta_range)
        # sample uniformly in amplitude space range
        thetas_amplitude = torch.rand(size, device=device) * (theta_amplitude_range[1] - theta_amplitude_range[0]) + theta_amplitude_range[0]
        # convert back
        thetas = torch.acos(thetas_amplitude)
    else:
        thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]

    centers = -torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) # [B, 3]

    targets = 0

    # jitters
    if jitter:
        centers = centers + (torch.rand_like(centers) * 0.2 - 0.1)
        targets = targets + torch.randn_like(centers) * 0.2

    # lookat
    forward_vector = safe_normalize(targets - centers)
    up_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    right_vector = safe_normalize(torch.cross(up_vector, forward_vector, dim=-1))
    
    if jitter:
        up_noise = torch.randn_like(up_vector) * 0.02
    else:
        up_noise = 0

    up_vector = safe_normalize(torch.cross(forward_vector, right_vector, dim=-1) + up_noise)

    poses = torch.stack([right_vector, up_vector, forward_vector], dim=-1)
    radius = radius[..., None] - cam_z_offset
    translations = torch.cat([torch.zeros_like(radius), torch.zeros_like(radius), radius], dim=-1)
    poses = torch.cat([poses.view(-1, 9), translations], dim=-1)

    if return_dirs:
        dirs = get_view_direction(thetas, phis, angle_overhead, angle_front, phi_offset=phi_offset)
        dirs = view_direction_id_to_text(dirs)
    else:
        dirs = None
    
    return poses, dirs
