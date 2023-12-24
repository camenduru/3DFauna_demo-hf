# Cages code used from https://github.com/yifita/deep_cage
import torch
import numpy as np
import trimesh



def deform_with_MVC(cage, cage_deformed, cage_face, query, verbose=False):
    """
    cage (B,C,3)
    cage_deformed (B,C,3)
    cage_face (B,F,3) int64
    query (B,Q,3)
    """
    weights, weights_unnormed = mean_value_coordinates_3D(query, cage, cage_face, verbose=True)
#     weights = weights.detach()
    deformed = torch.sum(weights.unsqueeze(-1)*cage_deformed.unsqueeze(1), dim=2)
    if verbose:
        return deformed, weights, weights_unnormed
    return deformed


def loadInitCage(template):
    init_cage_V, init_cage_F = read_trimesh(template)
    init_cage_V = torch.from_numpy(init_cage_V[:,:3].astype(np.float32)).unsqueeze(0)*2.0
    init_cage_F = torch.from_numpy(init_cage_F[:,:3].astype(np.int64)).unsqueeze(0)
    return init_cage_V, init_cage_F


def read_trimesh(path):
    mesh = trimesh.load(path)
    return mesh.vertices, mesh.faces


# util functions from pytorch_points
PI = 3.1415927

def normalize_to_box(input):
    """
    normalize point cloud to unit bounding box
    center = (max - min)/2
    scale = max(abs(x))
    input: pc [N, P, dim] or [P, dim]
    output: pc, centroid, furthest_distance
    """
    if len(input.shape) == 2:
        axis = 0
        P = input.shape[0]
        D = input.shape[1]
    elif len(input.shape) == 3:
        axis = 1
        P = input.shape[1]
        D = input.shape[2]
    if isinstance(input, np.ndarray):
        maxP = np.amax(input, axis=axis, keepdims=True)
        minP = np.amin(input, axis=axis, keepdims=True)
        centroid = (maxP+minP)/2
        input = input - centroid
        furthest_distance = np.amax(np.abs(input), axis=(axis, -1), keepdims=True)
        input = input / furthest_distance
    elif isinstance(input, torch.Tensor):
        maxP = torch.max(input, dim=axis, keepdim=True)[0]
        minP = torch.min(input, dim=axis, keepdim=True)[0]
        centroid = (maxP+minP)/2
        input = input - centroid
        in_shape = list(input.shape[:axis])+[P*D]
        furthest_distance = torch.max(torch.abs(input).view(in_shape), dim=axis, keepdim=True)[0]
        furthest_distance = furthest_distance.unsqueeze(-1)
        input = input / furthest_distance

    return input, centroid, furthest_distance

def normalize(tensor, dim=-1):
    """normalize tensor in specified dimension"""
    return torch.nn.functional.normalize(tensor, p=2, dim=dim, eps=1e-12, out=None)


def check_values(tensor):
    """return true if tensor doesn't contain NaN or Inf"""
    return not (torch.any(torch.isnan(tensor)).item() or torch.any(torch.isinf(tensor)).item())


class ScatterAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, idx, dim, out_size, fill=0.0):
        out = torch.full(out_size, fill, device=src.device, dtype=src.dtype)
        ctx.save_for_backward(idx)
        out.scatter_add_(dim, idx, src)
        ctx.mark_non_differentiable(idx)
        ctx.dim = dim
        return out

    @staticmethod
    def backward(ctx, ograd):
        idx, = ctx.saved_tensors
        grad = torch.gather(ograd, ctx.dim, idx)
        return grad, None, None, None, None


_scatter_add = ScatterAdd.apply


def scatter_add(src, idx, dim, out_size=None, fill=0.0):
    if out_size is None:
        out_size = list(src.size())
        dim_size = idx.max().item()+1
        out_size[dim] = dim_size
    return _scatter_add(src, idx, dim, out_size, fill)


def mean_value_coordinates_3D(query, vertices, faces, verbose=False):
    """
    Tao Ju et.al. MVC for 3D triangle meshes
    params:
        query    (B,P,3)
        vertices (B,N,3)
        faces    (B,F,3)
    return:
        wj       (B,P,N)
    """
    B, F, _ = faces.shape
    _, P, _ = query.shape
    _, N, _ = vertices.shape
    # u_i = p_i - x (B,P,N,3)
    uj = vertices.unsqueeze(1) - query.unsqueeze(2)
    # \|u_i\| (B,P,N,1)
    dj = torch.norm(uj, dim=-1, p=2, keepdim=True)
    uj = normalize(uj, dim=-1)
    # gather triangle B,P,F,3,3
    ui = torch.gather(uj.unsqueeze(2).expand(-1,-1,F,-1,-1),
                                   3,
                                   faces.unsqueeze(1).unsqueeze(-1).expand(-1,P,-1,-1,3))
    # li = \|u_{i+1}-u_{i-1}\| (B,P,F,3)
    li = torch.norm(ui[:,:,:,[1, 2, 0],:] - ui[:, :, :,[2, 0, 1],:], dim=-1, p=2)
    eps = 2e-5
    li = torch.where(li>=2, li-(li.detach()-(2-eps)), li)
    li = torch.where(li<=-2, li-(li.detach()+(2-eps)), li)
    # asin(x) is inf at +/-1
    # θi =  2arcsin[li/2] (B,P,F,3)
    theta_i = 2*torch.asin(li/2)
    assert(check_values(theta_i))
    # B,P,F,1
    h = torch.sum(theta_i, dim=-1, keepdim=True)/2
    # wi← sin[θi]d{i−1}d{i+1}
    # (B,P,F,3) ci ← (2sin[h]sin[h−θi])/(sin[θ_{i+1}]sin[θ_{i−1}])−1
    ci = 2*torch.sin(h)*torch.sin(h-theta_i)/(torch.sin(theta_i[:,:,:,[1, 2, 0]])*torch.sin(theta_i[:,:,:,[2, 0, 1]]))-1

    # NOTE: because of floating point ci can be slightly larger than 1, causing problem with sqrt(1-ci^2)
    # NOTE: sqrt(x)' is nan for x=0, hence use eps
    eps = 1e-5
    ci = torch.where(ci>=1, ci-(ci.detach()-(1-eps)), ci)
    ci = torch.where(ci<=-1, ci-(ci.detach()+(1-eps)), ci)
    # si← sign[det[u1,u2,u3]]sqrt(1-ci^2)
    # (B,P,F)*(B,P,F,3)

    si = torch.sign(torch.det(ui)).unsqueeze(-1)*torch.sqrt(1-ci**2)  # sqrt gradient nan for 0
    assert(check_values(si))
    # (B,P,F,3)
    di = torch.gather(dj.unsqueeze(2).squeeze(-1).expand(-1,-1,F,-1), 3,
                      faces.unsqueeze(1).expand(-1,P,-1,-1))
    assert(check_values(di))
    # if si.requires_grad:
    #     vertices.register_hook(save_grad("mvc/dv"))
    #     li.register_hook(save_grad("mvc/dli"))
    #     theta_i.register_hook(save_grad("mvc/dtheta"))
    #     ci.register_hook(save_grad("mvc/dci"))
    #     si.register_hook(save_grad("mvc/dsi"))
    #     di.register_hook(save_grad("mvc/ddi"))

    # wi← (θi −c[i+1]θ[i−1] −c[i−1]θ[i+1])/(disin[θi+1]s[i−1])
    # B,P,F,3
    # CHECK is there a 2* in the denominator
    wi = (theta_i-ci[:,:,:,[1,2,0]]*theta_i[:,:,:,[2,0,1]]-ci[:,:,:,[2,0,1]]*theta_i[:,:,:,[1,2,0]])/(di*torch.sin(theta_i[:,:,:,[1,2,0]])*si[:,:,:,[2,0,1]])
    # if ∃i,|si| ≤ ε, set wi to 0. coplaner with T but outside
    # ignore coplaner outside triangle
    # alternative check
    # (B,F,3,3)
    # triangle_points = torch.gather(vertices.unsqueeze(1).expand(-1,F,-1,-1), 2, faces.unsqueeze(-1).expand(-1,-1,-1,3))
    # # (B,P,F,3), (B,1,F,3) -> (B,P,F,1)
    # determinant = dot_product(triangle_points[:,:,:,0].unsqueeze(1)-query.unsqueeze(2),
    #                           torch.cross(triangle_points[:,:,:,1]-triangle_points[:,:,:,0],
    #                                       triangle_points[:,:,:,2]-triangle_points[:,:,:,0], dim=-1).unsqueeze(1), dim=-1, keepdim=True).detach()
    # # (B,P,F,1)
    # sqrdist = determinant*determinant / (4 * sqrNorm(torch.cross(triangle_points[:,:,:,1]-triangle_points[:,:,:,0], triangle_points[:,:,:,2]-triangle_points[:,:,:,0], dim=-1), keepdim=True))

    wi = torch.where(torch.any(torch.abs(si) <= 1e-5, keepdim=True, dim=-1), torch.zeros_like(wi), wi)
    # wi = torch.where(sqrdist <= 1e-5, torch.zeros_like(wi), wi)

    # if π −h < ε, x lies on t, use 2D barycentric coordinates
    # inside triangle
    inside_triangle = (PI-h).squeeze(-1)<1e-4
    # set all F for this P to zero
    wi = torch.where(torch.any(inside_triangle, dim=-1, keepdim=True).unsqueeze(-1), torch.zeros_like(wi), wi)
    # CHECK is it di https://www.cse.wustl.edu/~taoju/research/meanvalue.pdf or li http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.516.1856&rep=rep1&type=pdf
    wi = torch.where(inside_triangle.unsqueeze(-1).expand(-1,-1,-1,wi.shape[-1]), torch.sin(theta_i)*di[:,:,:,[2,0,1]]*di[:,:,:,[1,2,0]], wi)

    # sum over all faces face -> vertex (B,P,F*3) -> (B,P,N)
    wj = scatter_add(wi.reshape(B,P,-1).contiguous(), faces.unsqueeze(1).expand(-1,P,-1,-1).reshape(B,P,-1), 2, out_size=(B,P,N))

    # close to vertex (B,P,N)
    close_to_point = dj.squeeze(-1) < 1e-8
    # set all F for this P to zero
    wj = torch.where(torch.any(close_to_point, dim=-1, keepdim=True), torch.zeros_like(wj), wj)
    wj = torch.where(close_to_point, torch.ones_like(wj), wj)

    # (B,P,1)
    sumWj = torch.sum(wj, dim=-1, keepdim=True)
    sumWj = torch.where(sumWj==0, torch.ones_like(sumWj), sumWj)

    wj_normalised = wj / sumWj
    # if wj.requires_grad:
    #     saved_variables["mvc/wi"] = wi
    #     wi.register_hook(save_grad("mvc/dwi"))
    #     wj.register_hook(save_grad("mvc/dwj"))
    if verbose:
        return wj_normalised, wi
    else:
        return wj_normalised
