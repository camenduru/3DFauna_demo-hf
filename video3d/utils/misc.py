import os
import glob
import yaml
import random
import numpy as np
import cv2
import torch
import torchvision.utils as tvutils
import zipfile
import argparse
from ..render.obj import write_obj, write_textured_obj
import einops
import torch.distributed as dist



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def setup_runtime(args):
    """Load configs, initialize CUDA, CuDNN and the random seeds."""

    # Setup CUDA
    cuda_device_id = args.gpu
    if cuda_device_id is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device_id)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # Setup random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    cv2.setRNGSeed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ## Load config
    cfgs = {}
    if args.config is not None and os.path.isfile(args.config):
        cfgs = load_yaml(args.config)

    cfgs['config'] = args.config
    cfgs['seed'] = args.seed
    cfgs['num_workers'] = args.num_workers
    cfgs['device'] = f"cuda:{args.rank}" if torch.cuda.is_available() and cuda_device_id is not None else 'cpu'

    print(f"Environment: GPU {cuda_device_id} - seed {args.seed}")
    return cfgs


def load_yaml(path):
    print(f"Loading configs from {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def dump_yaml(path, cfgs):
    print(f"Saving configs to {path}")
    xmkdir(os.path.dirname(path))
    with open(path, 'w') as f:
        return yaml.safe_dump(cfgs, f)


def xmkdir(path):
    """Create directory PATH recursively if it does not exist."""
    os.makedirs(path, exist_ok=True)


def clean_checkpoint(checkpoint_dir, keep_num=2):
    if keep_num > 0:
        names = list(sorted(
            glob.glob(os.path.join(checkpoint_dir, 'checkpoint*.pth'))
        ))
        if len(names) > keep_num:
            for name in names[:-keep_num]:
                print(f"Deleting obslete checkpoint file {name}")
                os.remove(name)


def archive_code(arc_path, filetypes=['.py']):
    print(f"Archiving code to {arc_path}")
    xmkdir(os.path.dirname(arc_path))
    zipf = zipfile.ZipFile(arc_path, 'w', zipfile.ZIP_DEFLATED)
    cur_dir = os.getcwd()
    flist = []
    for ftype in filetypes:
        flist.extend(glob.glob(os.path.join(cur_dir, '[!results]*', '**', '*'+ftype), recursive=True))  # ignore results folder
        flist.extend(glob.glob(os.path.join(cur_dir, '*'+ftype)))
    [zipf.write(f, arcname=f.replace(cur_dir,'archived_code', 1)) for f in flist]
    zipf.close()


def get_model_device(model):
    return next(model.parameters()).device


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def save_videos(out_fold, imgs, prefix='', suffix='', fnames=None, ext='.mp4', cycle=False):
    prefix = prefix + '_' if prefix else ''
    suffix = '_' + suffix if suffix else ''

    imgs = imgs.transpose(0,1,3,4,2)  # BxTxCxHxW -> BxTxHxWxC
    for i, fs in enumerate(imgs):
        if cycle:
            fs = np.concatenate([fs, fs[::-1]], 0)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # fourcc = cv2.VideoWriter_fourcc(*'avc1')

        out_fold_i = out_fold[i] if isinstance(out_fold, list) else out_fold
        xmkdir(out_fold_i)

        if fnames is None:
            idx = len(glob.glob(os.path.join(out_fold_i, prefix+'*'+suffix+ext))) +1
            fname = '%07d' % idx
        else:
            fname = fnames[i]
        fpath = os.path.join(out_fold_i, prefix+fname+suffix+ext)

        vid = cv2.VideoWriter(fpath, fourcc, 5, (fs.shape[2], fs.shape[1]))
        [vid.write(np.uint8(f[...,::-1]*255.)) for f in fs]
        vid.release()


def save_images(out_fold, imgs, prefix='', suffix='', fnames=None, ext='.png'):
    prefix = prefix + '_' if prefix else ''
    suffix = '_' + suffix if suffix else ''

    imgs = imgs.transpose(0,2,3,1)
    for i, img in enumerate(imgs):
        img = np.concatenate([np.flip(img[...,:3], -1), img[...,3:]], -1)  # RGBA to BGRA
        if 'depth' in suffix:
            im_out = np.uint16(img*65535.)
        else:
            im_out = np.uint8(img*255.)

        out_fold_i = out_fold[i] if isinstance(out_fold, list) else out_fold
        xmkdir(out_fold_i)

        if fnames is None:
            idx = len(glob.glob(os.path.join(out_fold_i, prefix+'*'+suffix+ext))) +1
            fname = '%07d' % idx
        else:
            fname = fnames[i]
        fpath = os.path.join(out_fold_i, prefix+fname+suffix+ext)

        cv2.imwrite(fpath, im_out)


def save_txt(out_fold, data, prefix='', suffix='', fnames=None, ext='.txt', fmt='%.6f'):
    prefix = prefix + '_' if prefix else ''
    suffix = '_' + suffix if suffix else ''

    for i, d in enumerate(data):
        out_fold_i = out_fold[i] if isinstance(out_fold, list) else out_fold
        xmkdir(out_fold_i)

        if fnames is None:
            idx = len(glob.glob(os.path.join(out_fold_i, prefix+'*'+suffix+ext))) +1
            fname = '%07d' % idx
        else:
            fname = fnames[i]
        fpath = os.path.join(out_fold_i, prefix+fname+suffix+ext)

        np.savetxt(fpath, d, fmt=fmt, delimiter=', ')


def save_obj(out_fold, meshes=None, save_material=True, feat=None, prefix='', suffix='', fnames=None, resolution=[256, 256], prior_shape=None):
    prefix = prefix + '_' if prefix else ''
    suffix = '_' + suffix if suffix else ''

    if meshes.v_pos is None:
        return
    
    batch_size = meshes.v_pos.shape[0]
    for i in range(batch_size):
        out_fold_i = out_fold[i] if isinstance(out_fold, list) else out_fold
        xmkdir(out_fold_i)

        if fnames is None:
            idx = len(glob.glob(os.path.join(out_fold_i, prefix+'*'+suffix+".obj"))) + 1
            fname = '%07d' % idx
        else:
            fname = fnames[i]
        if save_material:
            os.makedirs(os.path.join(out_fold_i, fname), exist_ok=True)
            write_textured_obj(out_fold_i, f'{fname}/{prefix+suffix}', meshes, i, save_material=save_material, feat=feat, resolution=resolution, prior_shape=prior_shape)
        else:
            write_obj(out_fold_i, prefix+fname+suffix, meshes, i, save_material=False, feat=feat, resolution=resolution)


def compute_sc_inv_err(d_pred, d_gt, mask=None):
    b = d_pred.size(0)
    diff = d_pred - d_gt
    if mask is not None:
        diff = diff * mask
        avg = diff.view(b, -1).sum(1) / (mask.view(b, -1).sum(1))
        score = (diff - avg.view(b,1,1))**2 * mask
    else:
        avg = diff.view(b, -1).mean(1)
        score = (diff - avg.view(b,1,1))**2
    return score  # masked error maps


def compute_angular_distance(n1, n2, mask=None):
    dist = (n1*n2).sum(3).clamp(-1,1).acos() /np.pi*180
    return dist*mask if mask is not None else dist


def save_scores(out_path, scores, header=''):
    print('Saving scores to %s' %out_path)
    np.savetxt(out_path, scores, fmt='%.8f', delimiter=',\t', header=header)


def image_grid(tensor, nrow=None):
    # check if list -> stack to numpy array
    if isinstance(tensor, list):
        tensor = np.stack(tensor, 0)
    # check if numpy array -> convert to torch tensor and swap axes
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor).permute(0, 3, 1, 2)

    b, c, h, w = tensor.shape
    if nrow is None:
        nrow = int(np.ceil(b**0.5))
    if c == 1:
        tensor = tensor.repeat(1, 3, 1, 1)
    tensor = tvutils.make_grid(tensor, nrow=nrow, normalize=False)
    return tensor


def video_grid(tensor, nrow=None):
    return torch.stack([image_grid(t, nrow=nrow) for t in tensor.unbind(1)], 0)


class LazyClass(object):
    def __init__(self, cls, *args, **kwargs):
        self.cls = cls
        self.args = args
        self.kwargs = kwargs
        self.instance = None

    def get_instance(self):
        if self.instance is None:
            self.instance = self.cls(*self.args, **self.kwargs)
        return self.instance

    def __call__(self, *args, **kwargs):
        return self.get_instance()(*args, **kwargs)

    def __getattribute__(self, name):
        if name in ['cls', 'args', 'kwargs', 'instance', 'get_instance']:
            return super().__getattribute__(name)
        else:
            return getattr(self.get_instance(), name)

def add_text_to_image(img, text, pos=(12, 12), color=(1, 1, 1), font_scale=1, thickness=2):
    if isinstance(img, torch.Tensor):
        img = img.permute(1,2,0).cpu().numpy()
    # if grayscale -> convert to RGB
    if img.shape[2] == 1:
        img = np.repeat(img, 3, 2)
    img = cv2.putText(np.ascontiguousarray(img), text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return img

def image_grid_multi_channel(tensor, pca=False, texts=None, font_scale=0.5):
    """
    visualize multi-channel image, each channel is a different greyscale image
    tensor: (b, c, h, w)
    texts: list of strings of length b
    """
    # rescale to [0, 1] for per each sample in batch
    tensor = tensor.detach().cpu()
    min_ = einops.reduce(tensor, 'b c h w -> b 1 1 1', 'min')
    max_ = einops.reduce(tensor, 'b c h w -> b 1 1 1', 'max')
    tensor = (tensor - min_) / (max_ - min_)
    if pca:
        import faiss
        (b, c, h, w) = tensor.shape
        # reshape the tensor to (b, c*h*w)
        # tensor = tensor.reshape(b, c*h*w)
        tensor_flat = einops.rearrange(tensor, 'b c h w -> (b h w) c')
        pca_mat = faiss.PCAMatrix(c, 3)
        pca_mat.train(tensor_flat.numpy())
        assert pca_mat.is_trained
        tensor_flat_pca = pca_mat.apply_py(tensor_flat.numpy())
        tensor = einops.rearrange(tensor_flat_pca, '(b h w) c -> b h w c', b=b, c=3, h=h, w=w)
    else:
        tensor = einops.rearrange(tensor, 'b c h w -> (b c) 1 h w')
    if texts is not None:
        # duplicate texts for each channel
        texts = [text for text in texts for _ in range(tensor.shape[0] // len(texts))]
        tensor = [add_text_to_image(img, text, font_scale=font_scale) for img, text in zip(tensor, texts)]
    return image_grid(tensor)


########## DDP Part Taken from: https://github.com/fundamentalvision/Deformable-DETR/blob/main/util/misc.py

def is_main_process():
    return get_rank() == 0

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


