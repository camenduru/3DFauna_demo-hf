import os
from glob import glob
import random
import numpy as np
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.datasets.folder
import torchvision.transforms as transforms
from einops import rearrange


def compute_distance_transform(mask):
    mask_dt = []
    for m in mask:
        dt = torch.FloatTensor(cv2.distanceTransform(np.uint8(m[0]), cv2.DIST_L2, cv2.DIST_MASK_PRECISE))
        inv_dt = torch.FloatTensor(cv2.distanceTransform(np.uint8(1 - m[0]), cv2.DIST_L2, cv2.DIST_MASK_PRECISE))
        mask_dt += [torch.stack([dt, inv_dt], 0)]
    return torch.stack(mask_dt, 0)  # Bx2xHxW


def crop_image(image, boxs, size):
    crops = []
    for box in boxs:
        crop_x0, crop_y0, crop_w, crop_h = box
        crop = transforms.functional.resized_crop(image, crop_y0, crop_x0, crop_h, crop_w, size)
        crop = transforms.functional.to_tensor(crop)
        crops += [crop]
    return torch.stack(crops, 0)


def box_loader(fpath):
    box = np.loadtxt(fpath, 'str')
    box[0] = box[0].split('_')[0]
    return box.astype(np.float32)


def read_feat_from_img(path, n_channels):
    feat = np.array(Image.open(path))
    return dencode_feat_from_img(feat, n_channels)


def dencode_feat_from_img(img, n_channels):
    n_addon_channels = int(np.ceil(n_channels / 3) * 3) - n_channels
    n_tiles = int((n_channels + n_addon_channels) / 3)
    feat = rearrange(img, 'h (t w) c -> h w (t c)', t=n_tiles, c=3)
    feat = feat[:, :, :-n_addon_channels]
    feat = feat.astype('float32') / 255
    return feat.transpose(2, 0, 1)


def dino_loader(fpath, n_channels):
    dino_map = read_feat_from_img(fpath, n_channels)
    return dino_map


def get_valid_mask(boxs, image_size):
    valid_masks = []
    for box in boxs:
        crop_x0, crop_y0, crop_w, crop_h, full_w, full_h = box[1:7].int().numpy()
        # Discard a small margin near the boundary.
        margin_w = int(crop_w * 0.02)
        margin_h = int(crop_h * 0.02)
        mask_full = torch.ones(full_h-margin_h*2, full_w-margin_w*2)
        mask_full_pad = torch.nn.functional.pad(mask_full, (crop_w+margin_w, crop_w+margin_w, crop_h+margin_h, crop_h+margin_h), mode='constant', value=0.0)
        mask_full_crop = mask_full_pad[crop_y0+crop_h:crop_y0+crop_h*2, crop_x0+crop_w:crop_x0+crop_w*2]
        mask_crop = torch.nn.functional.interpolate(mask_full_crop[None, None, :, :], image_size, mode='nearest')[0,0]
        valid_masks += [mask_crop]
    return torch.stack(valid_masks, 0)  # NxHxW


def horizontal_flip_box(box):
    frame_id, crop_x0, crop_y0, crop_w, crop_h, full_w, full_h, sharpness, label = box.unbind(1)
    box[:,1] = full_w - crop_x0 - crop_w  # x0
    return box


def horizontal_flip_all(images, masks, mask_dt, mask_valid, flows, bboxs, bg_images, dino_features=None, dino_clusters=None):
    images = images.flip(3)  # NxCxHxW
    masks = masks.flip(3)  # NxCxHxW
    mask_dt = mask_dt.flip(3)  # NxCxHxW
    mask_valid = mask_valid.flip(2)  # NxHxW
    if flows.dim() > 1:
        flows = flows.flip(3)  # (N-1)x(x,y)xHxW
        flows[:,0] *= -1  # invert delta x
    bboxs = horizontal_flip_box(bboxs)  # NxK
    bg_images = bg_images.flip(3)  # NxCxHxW
    if dino_features.dim() > 1:
        dino_features = dino_features.flip(3)
    if dino_clusters.dim() > 1:
        dino_clusters = dino_clusters.flip(3)
    return images, masks, mask_dt, mask_valid, flows, bboxs, bg_images, dino_features, dino_clusters


class BaseSequenceDataset(Dataset):
    def __init__(self, root, skip_beginning=4, skip_end=4, min_seq_len=10, debug_seq=False):
        super().__init__()

        self.skip_beginning = skip_beginning
        self.skip_end = skip_end
        self.min_seq_len = min_seq_len
        # self.pattern = "{:07d}_{}"
        self.sequences = self._make_sequences(root)

        if debug_seq:
            # self.sequences = [self.sequences[0][20:160]] * 100
            seq_len = 0
            while seq_len < min_seq_len:
                i = np.random.randint(len(self.sequences))
                rand_seq = self.sequences[i]
                seq_len = len(rand_seq)
            self.sequences = [rand_seq]

        self.samples = []

    def _make_sequences(self, path):
        result = []
        for d in sorted(os.scandir(path), key=lambda e: e.name):
            if d.is_dir():
                files = self._parse_folder(d)
                if len(files) >= self.min_seq_len:
                    result.append(files)
        return result

    def _parse_folder(self, path):
        result = sorted(glob(os.path.join(path, '*'+self.image_loaders[0][0])))
        result = [p.replace(self.image_loaders[0][0], '{}') for p in result]

        if len(result) <= self.skip_beginning + self.skip_end:
            return []
        if self.skip_end == 0:
            return result[self.skip_beginning:]
        return result[self.skip_beginning:-self.skip_end]

    def _load_ids(self, path_patterns, loaders, transform=None):
        result = []
        for loader in loaders:
            for p in path_patterns:
                x = loader[1](p.format(loader[0]), *loader[2:])
                if transform:
                    x = transform(x)
                result.append(x)
        return tuple(result)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        raise NotImplemented("This is a base class and should not be used directly")


class NFrameSequenceDataset(BaseSequenceDataset):
    def __init__(self, root, cat_name=None, num_sample_frames=2, skip_beginning=4, skip_end=4, min_seq_len=10, in_image_size=256, out_image_size=256, debug_seq=False, random_sample=False, shuffle=False, dense_sample=True, color_jitter=None, load_background=False, random_flip=False, rgb_suffix='.png', load_dino_feature=False, load_dino_cluster=False, dino_feature_dim=64, **kwargs):
        self.cat_name = cat_name
        self.image_loaders = [("rgb"+rgb_suffix, torchvision.datasets.folder.default_loader)]
        self.mask_loaders = [("mask.png", torchvision.datasets.folder.default_loader)]
        self.bbox_loaders = [("box.txt", box_loader)]
        super().__init__(root, skip_beginning, skip_end, min_seq_len, debug_seq)
        if num_sample_frames > 1:
            self.flow_loaders = [("flow.png", cv2.imread, cv2.IMREAD_UNCHANGED)]
        else:
            self.flow_loaders = None

        self.num_sample_frames = num_sample_frames
        self.random_sample = random_sample
        if self.random_sample:
            if shuffle:
                random.shuffle(self.sequences)
            self.samples = self.sequences
        else:
            for i, s in enumerate(self.sequences):
                stride = 1 if dense_sample else self.num_sample_frames
                self.samples += [(i, k) for k in range(0, len(s), stride)]
            if shuffle:
                random.shuffle(self.samples)

        self.in_image_size = in_image_size
        self.out_image_size = out_image_size
        self.load_background = load_background
        self.color_jitter = color_jitter
        self.image_transform = transforms.Compose([transforms.Resize(self.in_image_size), transforms.ToTensor()])
        self.mask_transform = transforms.Compose([transforms.Resize(self.out_image_size, interpolation=Image.NEAREST), transforms.ToTensor()])
        if self.flow_loaders is not None:
            self.flow_transform = lambda x: (torch.FloatTensor(x.astype(np.float32)).flip(2)[:,:,:2] / 65535. ) *2 -1
        self.random_flip = random_flip
        self.load_dino_feature = load_dino_feature
        if load_dino_feature:
            self.dino_feature_loaders = [(f"feat{dino_feature_dim}.png", dino_loader, dino_feature_dim)]
        self.load_dino_cluster = load_dino_cluster
        if load_dino_cluster:
            self.dino_cluster_loaders = [("clusters.png", torchvision.datasets.folder.default_loader)]

    def __getitem__(self, index):
        if self.random_sample:
            seq_idx = index % len(self.sequences)
            seq = self.sequences[seq_idx]
            if len(seq) < self.num_sample_frames:
                start_frame_idx = 0
            else:
                start_frame_idx = np.random.randint(len(seq)-self.num_sample_frames+1)
            paths = seq[start_frame_idx:start_frame_idx+self.num_sample_frames]
        else:
            seq_idx, start_frame_idx = self.samples[index % len(self.samples)]
            seq = self.sequences[seq_idx]
            # Handle edge case: when only last frame is left, sample last two frames, except if the sequence only has one frame
            if len(seq) <= start_frame_idx +1:
                start_frame_idx = max(0, start_frame_idx-1)
            paths = seq[start_frame_idx:start_frame_idx+self.num_sample_frames]

        masks = torch.stack(self._load_ids(paths, self.mask_loaders, transform=self.mask_transform), 0)  # load all images
        mask_dt = compute_distance_transform(masks)
        jitter = False
        if self.color_jitter is not None:
            prob, b, h = self.color_jitter
            if np.random.rand() < prob:
                jitter = True
                color_jitter_tsf_fg = transforms.ColorJitter.get_params(brightness=(1-b, 1+b), contrast=None, saturation=None, hue=(-h, h))
                image_transform_fg = transforms.Compose([transforms.Resize(self.in_image_size), color_jitter_tsf_fg, transforms.ToTensor()])
                color_jitter_tsf_bg = transforms.ColorJitter.get_params(brightness=(1-b, 1+b), contrast=None, saturation=None, hue=(-h, h))
                image_transform_bg = transforms.Compose([transforms.Resize(self.in_image_size), color_jitter_tsf_bg, transforms.ToTensor()])
        if jitter:
            images_fg = torch.stack(self._load_ids(paths, self.image_loaders, transform=image_transform_fg), 0)  # load all images
            images_bg = torch.stack(self._load_ids(paths, self.image_loaders, transform=image_transform_bg), 0)  # load all images
            images = images_fg * masks + images_bg * (1-masks)
        else:
            images = torch.stack(self._load_ids(paths, self.image_loaders, transform=self.image_transform), 0)  # load all images
        if len(paths) > 1:
            flows = torch.stack(self._load_ids(paths[:-1], self.flow_loaders, transform=self.flow_transform), 0).permute(0,3,1,2)   # load flow for first image, (N-1)x(x,y)xHxW, -1~1
            flows = torch.nn.functional.interpolate(flows, size=self.out_image_size, mode="bilinear")
        else:
            flows = torch.zeros(1)
        bboxs = torch.stack(self._load_ids(paths, self.bbox_loaders, transform=torch.FloatTensor), 0)   # load bounding boxes for all images        
        mask_valid = get_valid_mask(bboxs, (self.out_image_size, self.out_image_size))  # exclude pixels cropped outside the original image
        if self.load_background:
            bg_image = torchvision.datasets.folder.default_loader(os.path.join(os.path.dirname(paths[0]), 'background_frame.jpg'))
            if jitter:
                bg_image = color_jitter_tsf_bg(bg_image)
            bg_images = crop_image(bg_image, bboxs[:, 1:5].int().numpy(), (self.in_image_size, self.in_image_size))
        else:
            bg_images = torch.zeros_like(images)
        if self.load_dino_feature:
            dino_features = torch.stack(self._load_ids(paths, self.dino_feature_loaders, transform=torch.FloatTensor), 0)  # BxFx64x224x224
        else:
            dino_features = torch.zeros(1)
        if self.load_dino_cluster:
            dino_clusters = torch.stack(self._load_ids(paths, self.dino_cluster_loaders, transform=transforms.ToTensor()), 0)  # BxFx3x55x55
        else:
            dino_clusters = torch.zeros(1)
        seq_idx = torch.LongTensor([seq_idx])
        frame_idx = torch.arange(start_frame_idx, start_frame_idx+len(paths)).long()

        if self.random_flip and np.random.rand() < 0.5:
            images, masks, mask_dt, mask_valid, flows, bboxs, bg_images, dino_features, dino_clusters = horizontal_flip_all(images, masks, mask_dt, mask_valid, flows, bboxs, bg_images, dino_features, dino_clusters)

        ## pad shorter sequence
        if len(paths) < self.num_sample_frames:
            num_pad = self.num_sample_frames - len(paths)
            images = torch.cat([images[:1]] *num_pad + [images], 0)
            masks = torch.cat([masks[:1]] *num_pad + [masks], 0)
            mask_dt = torch.cat([mask_dt[:1]] *num_pad + [mask_dt], 0)
            mask_valid = torch.cat([mask_valid[:1]] *num_pad + [mask_valid], 0)
            if flows.dim() > 1:
                flows = torch.cat([flows[:1]*0] *num_pad + [flows], 0)
            bboxs = torch.cat([bboxs[:1]] * num_pad + [bboxs], 0)
            bg_images = torch.cat([bg_images[:1]] *num_pad + [bg_images], 0)
            if dino_features.dim() > 1:
                dino_features = torch.cat([dino_features[:1]] *num_pad + [dino_features], 0)
            if dino_clusters.dim() > 1:
                dino_clusters = torch.cat([dino_clusters[:1]] *num_pad + [dino_clusters], 0)
            frame_idx = torch.cat([frame_idx[:1]] *num_pad + [frame_idx], 0)

        return images, masks, mask_dt, mask_valid, flows, bboxs, bg_images, dino_features, dino_clusters, seq_idx, frame_idx, self.cat_name


def get_sequence_loader(data_dir, **kwargs):
    if isinstance(data_dir, dict):
        loaders = []
        for k, v in data_dir.items():
            dataset= NFrameSequenceDataset(v, cat_name=k, **kwargs)
            loader = torch.utils.data.DataLoader(dataset, batch_size=kwargs['batch_size'], shuffle=kwargs['shuffle'], num_workers=kwargs['num_workers'], pin_memory=True)
            loaders += [loader]
        return loaders
    else:
        return [get_sequence_loader_single(data_dir, **kwargs)]


def get_sequence_loader_single(data_dir, mode='all_frame', is_validation=False, batch_size=256, num_workers=4, in_image_size=256, out_image_size=256, debug_seq=False, num_sample_frames=2, skip_beginning=4, skip_end=4, min_seq_len=10, max_seq_len=256, random_sample=False, shuffle=False, dense_sample=True, color_jitter=None, load_background=False, random_flip=False, rgb_suffix='.jpg', load_dino_feature=False, load_dino_cluster=False, dino_feature_dim=64):
    if mode == 'n_frame':
        dataset = NFrameSequenceDataset(data_dir, num_sample_frames=num_sample_frames, skip_beginning=skip_beginning, skip_end=skip_end, min_seq_len=min_seq_len, in_image_size=in_image_size, out_image_size=out_image_size, debug_seq=debug_seq, random_sample=random_sample, shuffle=shuffle, dense_sample=dense_sample, color_jitter=color_jitter, load_background=load_background, random_flip=random_flip, rgb_suffix=rgb_suffix, load_dino_feature=load_dino_feature, load_dino_cluster=load_dino_cluster, dino_feature_dim=dino_feature_dim)
    else:
        raise NotImplementedError
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not is_validation,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader


class ImageDataset(Dataset):
    def __init__(self, root, is_validation=False, image_size=256, color_jitter=None):
        super().__init__()
        self.image_loader = ("rgb.jpg", torchvision.datasets.folder.default_loader)
        self.mask_loader = ("mask.png", torchvision.datasets.folder.default_loader)
        self.bbox_loader = ("box.txt", np.loadtxt, 'str')
        self.samples = self._parse_folder(root)
        self.image_size = image_size
        self.color_jitter = color_jitter
        self.image_transform = transforms.Compose([transforms.Resize(self.image_size), transforms.ToTensor()])
        self.mask_transform = transforms.Compose([transforms.Resize(self.image_size, interpolation=Image.NEAREST), transforms.ToTensor()])

    def _parse_folder(self, path):
        result = sorted(glob(os.path.join(path, '**/*'+self.image_loader[0]), recursive=True))
        result = [p.replace(self.image_loader[0], '{}') for p in result]
        return result

    def _load_ids(self, path, loader, transform=None):
        x = loader[1](path.format(loader[0]), *loader[2:])
        if transform:
            x = transform(x)
        return x

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index % len(self.samples)]
        masks = self._load_ids(path, self.mask_loader, transform=self.mask_transform).unsqueeze(0)
        mask_dt = compute_distance_transform(masks)
        jitter = False
        if self.color_jitter is not None:
            prob, b, h = self.color_jitter
            if np.random.rand() < prob:
                jitter = True
                color_jitter_tsf_fg = transforms.ColorJitter.get_params(brightness=(1-b, 1+b), contrast=None, saturation=None, hue=(-h, h))
                image_transform_fg = transforms.Compose([transforms.Resize(self.image_size), color_jitter_tsf_fg, transforms.ToTensor()])
                color_jitter_tsf_bg = transforms.ColorJitter.get_params(brightness=(1-b, 1+b), contrast=None, saturation=None, hue=(-h, h))
                image_transform_bg = transforms.Compose([transforms.Resize(self.image_size), color_jitter_tsf_bg, transforms.ToTensor()])
        if jitter:
            images_fg = self._load_ids(path, self.image_loader, transform=image_transform_fg).unsqueeze(0)
            images_bg = self._load_ids(path, self.image_loader, transform=image_transform_bg).unsqueeze(0)
            images = images_fg * masks + images_bg * (1-masks)
        else:
            images = self._load_ids(path, self.image_loader, transform=self.image_transform).unsqueeze(0)
        flows = torch.zeros(1)
        bboxs = self._load_ids(path, self.bbox_loader, transform=None)
        bboxs[0] = '0'
        bboxs = torch.FloatTensor(bboxs.astype('float')).unsqueeze(0)
        bg_fpath = os.path.join(os.path.dirname(path), 'background_frame.jpg')
        if os.path.isfile(bg_fpath):
            bg_image = torchvision.datasets.folder.default_loader(bg_fpath)
            if jitter:
                bg_image = color_jitter_tsf_bg(bg_image)
            bg_image = transforms.ToTensor()(bg_image)
        else:
            bg_image = images[0]
        seq_idx = torch.LongTensor([index])
        frame_idx = torch.LongTensor([0])
        return images, masks, mask_dt, flows, bboxs, bg_image, seq_idx, frame_idx


def get_image_loader(data_dir, is_validation=False, batch_size=256, num_workers=4, image_size=256, color_jitter=None):
    dataset = ImageDataset(data_dir, is_validation=is_validation, image_size=image_size, color_jitter=color_jitter)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader
