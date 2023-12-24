import configargparse
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.utils as tvutils
import torchvision.transforms
from video3d.utils.segmentation_transforms import *
from video3d.utils.misc import setup_runtime
from video3d import networks
from video3d.trainer import Trainer
from video3d.dataloaders import SegmentationDataset


class Segmentation:
    def __init__(self, cfgs, _):
        self.cfgs = cfgs
        self.device = cfgs.get('device', 'cpu')
        self.total_loss = None
        self.net = networks.EDDeconv(cin=3, cout=1, zdim=128, nf=64, activation=None)
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.net.parameters()),
            lr=cfgs.get('lr', 1e-4),
            betas=(0.9, 0.999),
            weight_decay=5e-4)

    def load_model_state(self, cp):
        self.net.load_state_dict(cp["net"])

    def load_optimizer_state(self, cp):
        self.net.load_state_dict(cp["optimizer"])

    @staticmethod
    def get_data_loaders(cfgs):
        batch_size = cfgs.get('batch_size', 64)
        num_workers = cfgs.get('num_workers', 4)
        data_dir = cfgs.get('data_dir', './data')
        img_size = cfgs.get('image_size', 64)
        min_size = int(img_size * cfgs.get('aug_min_resize', 0.5))
        max_size = int(img_size * cfgs.get('aug_max_resize', 2.0))
        transform = Compose([RandomResize(min_size, max_size),
                             RandomHorizontalFlip(cfgs.get("aug_horizontal_flip", 0.4)),
                             RandomCrop(img_size),
                             ImageOnly(torchvision.transforms.ColorJitter(**cfgs.get("aug_color_jitter", {}))),
                             ImageOnly(torchvision.transforms.RandomGrayscale(cfgs.get("aug_grayscale", 0.2))),
                             ToTensor()])
        train_loader = torch.utils.data.DataLoader(
            SegmentationDataset(data_dir, is_validation=False, transform=transform, sequence_range=(0, 0.5)),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        transform = Compose([ToTensor()])
        val_loader = torch.utils.data.DataLoader(
            SegmentationDataset(data_dir, is_validation=True, transform=transform, sequence_range=(0.5, 1.0)),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        return train_loader, val_loader, None

    def get_state_dict(self):
        return {
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }

    def to(self, device):
        self.device = device
        self.net.to(device)

    def set_train(self):
        self.net.train()

    def set_eval(self):
        self.net.eval()

    def backward(self):
        self.optimizer.zero_grad()
        self.total_loss.backward()
        self.optimizer.step()

    def forward(self, batch, visualize=False):
        image, target = batch
        image = image.to(self.device)*2 - 1
        target = target[:, 0, :, :].to(self.device).unsqueeze(1)
        pred = self.net(image)

        self.total_loss = nn.functional.binary_cross_entropy_with_logits(pred, target)

        metrics = {'loss': self.total_loss}

        visuals = {}
        if visualize:
            visuals['rgb'] = self.image_visual(image, normalize=True, range=(-1, 1))
            visuals['target'] = self.image_visual(target, normalize=True, range=(0, 1))
            visuals['pred'] = self.image_visual(nn.functional.sigmoid(pred), normalize=True, range=(0, 1))

            return metrics, visuals

        return metrics

    def visualize(self, logger, total_iter, max_bs=25):
        pass

    def save_results(self, save_dir):
        pass

    def save_scores(self, path):
        pass

    @staticmethod
    def image_visual(tensor, **kwargs):
        if tensor.shape[1] == 1:
            tensor = tensor.repeat(1, 3, 1, 1)
        n = int(tensor.shape[0]**0.5 + 0.5)
        tensor = tvutils.make_grid(tensor.detach(), nrow=n, **kwargs).permute(1, 2, 0)
        return torch.clamp(tensor[:, :, :3] * 255, 0, 255).byte().cpu()


if __name__ == "__main__":
    parser = configargparse.ArgumentParser(description='Training configurations.')
    parser.add_argument('--config', default="config/train_segmentation.yml", type=str, is_config_file=True,
                        help='Specify a config file path')
    parser.add_argument('--gpu', default=1, type=int, help='Specify a GPU device')
    parser.add_argument('--seed', default=0, type=int, help='Specify a random seed')
    args, _ = parser.parse_known_args()

    cfgs = setup_runtime(args)
    trainer = Trainer(cfgs, Segmentation)
    trainer.train()
