import os
import os.path as osp
from copy import deepcopy
from collections import OrderedDict
import glob
from datetime import datetime
import random
import copy
import imageio
import torch
import clip
import torchvision.transforms.functional as tvf
import video3d.utils.meters as meters
import video3d.utils.misc as misc
# from video3d.dataloaders import get_image_loader
from video3d.dataloaders_ddp import get_sequence_loader_ddp, get_sequence_loader_quadrupeds, get_test_loader_quadrupeds
from . import discriminator_architecture


def sample_frames(batch, num_sample_frames, iteration, stride=1):
    ## window slicing sampling
    images, masks, flows, bboxs, bg_image, seq_idx, frame_idx = batch
    num_seqs, total_num_frames = images.shape[:2]
    # start_frame_idx = iteration % (total_num_frames - num_sample_frames +1)

    ## forward and backward
    num_windows = total_num_frames - num_sample_frames +1
    start_frame_idx = (iteration * stride) % (2*num_windows)
    ## x' = (2n-1)/2 - |(2n-1)/2 - x| : 0,1,2,3,4,5 -> 0,1,2,2,1,0
    mid_val = (2*num_windows -1) /2
    start_frame_idx = int(mid_val - abs(mid_val -start_frame_idx))

    new_batch = images[:, start_frame_idx:start_frame_idx+num_sample_frames], \
        masks[:, start_frame_idx:start_frame_idx+num_sample_frames], \
        flows[:, start_frame_idx:start_frame_idx+num_sample_frames-1], \
        bboxs[:, start_frame_idx:start_frame_idx+num_sample_frames], \
        bg_image, \
        seq_idx, \
        frame_idx[:, start_frame_idx:start_frame_idx+num_sample_frames]
    return new_batch


def indefinite_generator(loader):
    while True:
        for x in loader:
            yield x


def indefinite_generator_from_list(loaders):
    while True:
        random_idx = random.randint(0, len(loaders)-1)
        for x in loaders[random_idx]:
            yield x
            break


def get_optimizer(model, lr=0.0001, betas=(0.9, 0.999), weight_decay=0):
    return torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, betas=betas, weight_decay=weight_decay)


class Fewshot_Trainer:
    def __init__(self, cfgs, model):
        # only now supports one gpu
        self.cfgs = cfgs
        # here should be the one gpu ddp setting
        self.rank = cfgs.get('rank', 0)
        self.world_size = cfgs.get('world_size', 1)
        self.use_ddp = cfgs.get('use_ddp', True)
        self.device = cfgs.get('device', 'cpu')

        self.num_epochs = cfgs.get('num_epochs', 1)
        self.lr = cfgs.get('few_shot_lr', 1e-4)
        self.dataset = 'image'

        self.metrics_trace = meters.MetricsTrace()
        self.make_metrics = lambda m=None: meters.StandardMetrics(m)

        self.archive_code = cfgs.get('archive_code', True)
        self.batch_size = cfgs.get('batch_size', 64)
        self.in_image_size = cfgs.get('in_image_size', 256)
        self.out_image_size = cfgs.get('out_image_size', 256)
        self.num_workers = cfgs.get('num_workers', 4)
        self.checkpoint_dir = cfgs.get('checkpoint_dir', 'results')
        misc.xmkdir(self.checkpoint_dir)
        self.few_shot_resume = cfgs.get('few_shot_resume', False)
        self.save_checkpoint_freq = cfgs.get('save_checkpoint_freq', 1)
        self.keep_num_checkpoint = cfgs.get('keep_num_checkpoint', 2)  # -1 for keeping all checkpoints

        self.few_shot_data_dir = cfgs.get('few_shot_data_dir', None)
        assert self.few_shot_data_dir is not None
        # in case we add more data source
        if isinstance(self.few_shot_data_dir, list):
            self.few_shot_data_dir_more = self.few_shot_data_dir[1:]
            self.few_shot_data_dir = self.few_shot_data_dir[0]
        else:
            self.few_shot_data_dir_more = None

        assert "data_resize_update" in self.few_shot_data_dir # TODO: a hack way to make sure not using wrong data, needs to remove
        self.few_shot_categories, self.few_shot_categories_paths = self.parse_few_shot_categories(self.few_shot_data_dir, self.few_shot_data_dir_more)

        # if we need test categories, we pop it from self.few_shot_categories and self.few_shot_categories_path
        # the test category needs to be category from few-shot, and we're using bs=1 on them, no need for back views enhancement (for now, use back view images, but don't duplicate them)
        self.test_category_num = cfgs.get('few_shot_test_category_num', 0)
        self.test_category_names = cfgs.get('few_shot_test_category_names', None)
        if self.test_category_num > 0:
            # if we have valid test_category names, then use them, the number doesn't need to be equal
            if self.test_category_names is not None:
                test_cats = self.test_category_names
            else:
                test_cats = list(self.few_shot_categories_paths.keys())[-(self.test_category_num):]
            test_categories_paths = {}
            for test_cat in test_cats:
                test_categories_paths.update({test_cat: self.few_shot_categories_paths[test_cat]})
                assert test_cat in self.few_shot_categories
                self.few_shot_categories.remove(test_cat)
                self.few_shot_categories_paths.pop(test_cat)
            
            self.test_categories_paths = test_categories_paths
        else:
            self.test_categories_paths = None

        # also load the original 7 categories
        self.original_train_data_path = cfgs.get('train_data_dir', None)
        self.original_val_data_path = cfgs.get('val_data_dir', None)
        self.original_categories = []
        self.original_categories_paths = self.original_train_data_path
        for k, v in self.original_train_data_path.items():
            self.original_categories.append(k)

        self.categories = self.original_categories + self.few_shot_categories
        self.categories_paths = self.original_train_data_path.copy()
        self.categories_paths.update(self.few_shot_categories_paths)
        
        print(f'Using {len(self.categories)} cateogires: ', self.categories)

        # initialize new things
        # self.original_classes_num = cfgs.get('few_shot_original_classes_num', 7)
        self.original_classes_num = len(self.original_categories)
        self.new_classes_num = len(self.categories) - self.original_classes_num

        self.combine_dataset = cfgs.get('combine_dataset', False)
        assert self.combine_dataset, "we should use combine dataset, it's up to date"
        if self.combine_dataset:
            self.train_loader, self.val_loader, self.test_loader = self.get_data_loaders_quadrupeds(self.cfgs, self.batch_size, self.num_workers, self.in_image_size, self.out_image_size)
        else:
            self.train_loader_few_shot, self.val_loader_few_shot = self.get_data_loaders_few_shot(self.cfgs, self.batch_size, self.num_workers, self.in_image_size, self.out_image_size)
            self.train_loader_original, self.val_loader_original = self.get_data_loaders_original(self.cfgs, self.batch_size, self.num_workers, self.in_image_size, self.out_image_size)
            self.train_loader = self.train_loader_original + self.train_loader_few_shot
            if self.val_loader_few_shot is not None and self.val_loader_original is not None:
                self.val_loader = self.val_loader_original + self.val_loader_few_shot

        self.num_iterations = cfgs.get('num_iterations', 0)
        if self.num_iterations != 0:
            self.use_total_iterations = True
        else:
            self.use_total_iterations = False
        if self.use_total_iterations:
            # reset the epoch related cfgs
            
            dataloader_length = max([len(loader) for loader in self.train_loader]) * len(self.train_loader)
            print("Total length of data loader is: ", dataloader_length)
            
            total_epoch = int(self.num_iterations / dataloader_length) + 1

            print(f'run for {total_epoch} epochs')
            
            print('is_main_process()?', misc.is_main_process())

            for k, v in cfgs.items():
                if 'epoch' in k:
                    # if isinstance(v, list):
                    #     new_v = [int(total_epoch * x / 120) + 1 for x in v]
                    #     cfgs[k] = new_v
                    # elif isinstance(v, int):
                    #     new_v = int(total_epoch * v / 120) + 1
                    #     cfgs[k] = new_v

                    # a better transformation
                    if isinstance(v, int):
                        # use the floor int
                        new_v = int(total_epoch * v / 120)
                        cfgs[k] = new_v
                    elif isinstance(v, list):
                        if v[0] == v[1]:
                            # if the values in v are the same, then we use both the floor value
                            new_v = [int(total_epoch * x / 120) for x in v]
                        else:
                            # if the values are not the same, make the first using floor value and others using ceil value
                            new_v = [int(total_epoch * x / 120) + 1 for x in v]
                            new_v[0] = new_v[0] - 1
                        cfgs[k] = new_v
                else:
                    continue
            
            self.num_epochs = total_epoch
            self.cub_start_epoch = cfgs.get('cub_start_epoch', 0)
            self.cfgs = cfgs

        # the model is with nothing now
        self.model = model(cfgs)

        self.metrics_trace = meters.MetricsTrace()
        self.make_metrics = lambda m=None: meters.StandardMetrics(m)
        
        self.use_logger = True
        self.log_freq_images = cfgs.get('log_freq_images', 1000)
        self.log_train_images = cfgs.get('log_train_images', False)
        self.log_freq_losses = cfgs.get('log_freq_losses', 100)
        self.save_result_freq = cfgs.get('save_result_freq', None)
        self.train_result_dir = osp.join(self.checkpoint_dir, 'results')
        self.fix_viz_batch = cfgs.get('fix_viz_batch', False)
        self.visualize_validation = cfgs.get('visualize_validation', False)
        # self.visualize_validation = False
        self.iteration_save = cfgs.get('few_shot_iteration_save', False)
        self.iteration_save_freq = cfgs.get('few_shot_iteration_save_freq', 2000)

        self.enable_memory_bank = cfgs.get('enable_memory_bank', False)
        if self.enable_memory_bank:
            self.memory_bank_dim = 128
            self.memory_bank_size = cfgs.get('memory_bank_size', 60)
            self.memory_bank_topk = cfgs.get('memory_bank_topk', 10)
            # assert self.memory_bank_topk < self.memory_bank_size
            assert self.memory_bank_topk <= self.memory_bank_size
            self.memory_retrieve = cfgs.get('memory_retrieve', 'cos-linear')
            
            self.memory_bank_init = cfgs.get('memory_bank_init', 'random')
            if self.memory_bank_init == 'copy':
                # use trained 7 embeddings to initialize
                num_piece = self.memory_bank_size // self.original_classes_num
                num_left = self.memory_bank_size - num_piece * self.original_classes_num

                tmp_1 = torch.empty_like(self.model.netPrior.classes_vectors)
                tmp_1 = tmp_1.copy_(self.model.netPrior.classes_vectors)
                tmp_1 = tmp_1.unsqueeze(0).repeat(num_piece, 1, 1)
                tmp_1 = tmp_1.reshape(tmp_1.shape[0] * tmp_1.shape[1], tmp_1.shape[-1])

                if num_left > 0:
                    tmp_2 = torch.empty_like(self.model.netPrior.classes_vectors)
                    tmp_2 = tmp_2.copy_(self.model.netPrior.classes_vectors)
                    tmp_2 = tmp_2[:num_left]
                    tmp = torch.cat([tmp_1, tmp_2], dim=0)
                else:
                    tmp = tmp_1
                
                self.memory_bank = torch.nn.Parameter(tmp, requires_grad=True)
            
            elif self.memory_bank_init == 'random':
                self.memory_bank = torch.nn.Parameter(torch.nn.init.uniform_(torch.empty(self.memory_bank_size, self.memory_bank_dim), a=-0.05, b=0.05), requires_grad=True)
            else:
                raise NotImplementedError
            
            self.memory_encoder = cfgs.get('memory_encoder', 'DINO')  # if DINO then just use the network encoder
            if self.memory_encoder == 'CLIP':
                self.clip_model, _ = clip.load('ViT-B/32', self.device)
                self.clip_model = self.clip_model.eval().requires_grad_(False)
                self.clip_mean = [0.48145466, 0.4578275, 0.40821073]
                self.clip_std = [0.26862954, 0.26130258, 0.27577711]
                self.clip_reso = 224

                self.memory_bank_keys_dim = 512
            
            elif self.memory_encoder == 'DINO':
                self.memory_bank_keys_dim = 384
            
            else:
                raise NotImplementedError
            
            memory_bank_keys = torch.nn.init.uniform_(torch.empty(self.memory_bank_size, self.memory_bank_keys_dim), a=-0.05, b=0.05)
            self.memory_bank_keys = torch.nn.Parameter(memory_bank_keys, requires_grad=True)
        
        else:
            print("no memory bank, just use image embedding, this is only for one experiment!")
            self.memory_encoder = cfgs.get('memory_encoder', 'DINO')  # if DINO then just use the network encoder
            if self.memory_encoder == 'CLIP':
                self.clip_model, _ = clip.load('ViT-B/32', self.device)
                self.clip_model = self.clip_model.eval().requires_grad_(False)
                self.clip_mean = [0.48145466, 0.4578275, 0.40821073]
                self.clip_std = [0.26862954, 0.26130258, 0.27577711]
                self.clip_reso = 224

                self.memory_bank_keys_dim = 512
            
            elif self.memory_encoder == 'DINO':
                self.memory_bank_keys_dim = 384
            
            else:
                raise NotImplementedError

        self.prepare_model()
    
    def parse_few_shot_categories(self, data_dir, data_dir_more=None):
        # parse the categories data_dir
        few_shot_category_num = self.cfgs.get('few_shot_category_num', -1)
        assert few_shot_category_num != 0
        categories = sorted(os.listdir(data_dir))
        cnt = 0
        category_names = []
        category_names_paths = {}
        for category in categories:
            if osp.isdir(osp.join(self.few_shot_data_dir, category, 'train')):
                category_path = osp.join(self.few_shot_data_dir, category, 'train')
                category_names.append(category)
                category_names_paths.update({category: category_path})
                cnt += 1
                if few_shot_category_num > 0 and cnt >= few_shot_category_num:
                    break
        
        # more data
        if data_dir_more is not None:
            for data_dir_one in data_dir_more:
                new_categories = os.listdir(data_dir_one)
                for new_category in new_categories:
                    '''
                    if this category is not used before, add a new item
                    if there is this category before, add the paths to original paths,
                        if its a str, make it a list
                        if its already a list, append it
                    '''
                    if new_category not in category_names:

                        #TODO: a hacky way here, if in new data there is category used in 7-cat, we just make it a new one
                        if new_category in list(self.cfgs.get('train_data_dir', None).keys()):
                            new_category = '_' + new_category

                        category_names.append(new_category)
                        category_names_paths.update({
                            new_category: osp.join(data_dir_one, new_category, 'train')
                        })
                    else:
                        old_category_path = category_names_paths[new_category]
                        if isinstance(old_category_path, str):
                            category_names_paths[new_category] = [
                                old_category_path,
                                osp.join(data_dir_one, new_category, 'train')
                            ]
                        elif isinstance(old_category_path, list):
                            old_category_path = old_category_path + [osp.join(data_dir_one, new_category, 'train')]
                            category_names_paths[new_category] = old_category_path
                        else:
                            raise NotImplementedError
            
            # category_names = sorted(category_names)

        return category_names, category_names_paths

    def prepare_model(self):
        # here we prepare the model weights at outside
        # 1. load the pretrain weight
        # 2. initialize anything new, like new class vectors
        # 3. initialize new optimizer for chosen parameters

        assert self.original_classes_num == len(self.model.netPrior.category_id_map)
        
        # load pretrain
        # if not assigned few_shot_checkpoint_name, then skip this part
        if self.cfgs.get('few_shot_checkpoint_name', None) is not None:
            original_checkpoint_path = osp.join(self.checkpoint_dir, self.cfgs.get('few_shot_checkpoint_name', 'checkpoint060.pth'))
            assert osp.exists(original_checkpoint_path)
            print(f"Loading pre-trained checkpoint from {original_checkpoint_path}")
            cp = torch.load(original_checkpoint_path, map_location=self.device)
            
            # if using local-texture network in fine-tuning, the texture in previous pre-train ckpt is global
            # here we use a hack way, we just get rid of original texture ckpt
            if (self.cfgs.get('texture_way', None) is not None) or (self.cfgs.get('texture_act', 'relu') != 'relu'):
                new_netInstance_weights = {k: v for k, v in cp['netInstance'].items() if 'netTexture' not in k}
                #find the new texture weights 
                texture_weights = self.model.netInstance.netTexture.state_dict()
                #add the new weights to the new model weights
                for k, v in texture_weights.items():
                    # for the overlapping part in netTexture, we also use them
                    # if ('netTexture.' + k) in cp['netInstance'].keys():
                    #     new_netInstance_weights['netTexture.' + k] = cp['netInstance']['netTexture.' + k]
                    # else:
                    #     new_netInstance_weights['netTexture.' + k] = v
                    new_netInstance_weights['netTexture.' + k] = v
                _ = cp.pop("netInstance")
                cp.update({"netInstance": new_netInstance_weights})

            self.model.netInstance.load_state_dict(cp["netInstance"], strict=False) # For Deform
            # self.model.netInstance.load_state_dict(cp["netInstance"])
            self.model.netPrior.load_state_dict(cp["netPrior"])

            self.original_total_iter = cp["total_iter"]
        
        else:
            print("not load any pre-train weight, the iter will start from 0, make sure you set all the needed parameters")
            self.original_total_iter = 0

        if not self.cfgs.get('disable_fewshot', False):
            for i, category in enumerate(self.few_shot_categories):
                category_id = self.original_classes_num + i
                self.model.netPrior.category_id_map.update({category: category_id})
            
            few_shot_class_vector_init = self.cfgs.get('few_shot_class_vector_init', 'random')
            if few_shot_class_vector_init == 'random':
                tmp = torch.nn.init.uniform_(torch.empty(self.new_classes_num, self.model.netPrior.classes_vectors.shape[-1]), a=-0.05, b=0.05)
                tmp = tmp.to(self.model.netPrior.classes_vectors.device)
                self.model.netPrior.classes_vectors = torch.nn.Parameter(torch.cat([self.model.netPrior.classes_vectors, tmp], dim=0))
            elif few_shot_class_vector_init == 'copy':
                num_7_cat_piece = self.new_classes_num // self.original_classes_num if self.new_classes_num > self.original_classes_num else 0
                num_left = self.new_classes_num - num_7_cat_piece * self.original_classes_num
                
                if num_7_cat_piece > 0:
                    tmp_1 = torch.empty_like(self.model.netPrior.classes_vectors)
                    tmp_1 = tmp_1.copy_(self.model.netPrior.classes_vectors)
                    tmp_1 = tmp_1.unsqueeze(0).repeat(num_7_cat_piece, 1, 1)
                    tmp_1 = tmp_1.reshape(tmp_1.shape[0] * tmp_1.shape[1], tmp_1.shape[-1])
                else:
                    tmp_1 = None
                
                if num_left > 0:
                    tmp_2 = torch.empty_like(self.model.netPrior.classes_vectors)
                    tmp_2 = tmp_2.copy_(self.model.netPrior.classes_vectors)
                    tmp_2 = tmp_2[:num_left]
                else:
                    tmp_2 = None
                
                if tmp_1 != None and tmp_2 != None:
                    tmp = torch.cat([tmp_1, tmp_2], dim=0)
                elif tmp_1 == None and tmp_2 != None:
                    tmp = tmp_2
                elif tmp_2 == None and tmp_1 != None:
                    tmp = tmp_1
                else:
                    raise NotImplementedError

                tmp = tmp.to(self.model.netPrior.classes_vectors.device)
                self.model.netPrior.classes_vectors = torch.nn.Parameter(torch.cat([self.model.netPrior.classes_vectors, tmp], dim=0))
            else:
                raise NotImplementedError
        
        else:
            print("disable few shot, not increasing embedding vectors")
        
        # initialize new optimizer
        optimize_rule = self.cfgs.get('few_shot_optimize', 'all')
        if optimize_rule == 'all':
            optimize_list = [
                {'name': 'net_Prior', 'params': list(self.model.netPrior.parameters()), 'lr': self.lr * 10.},
                {'name': 'net_Instance', 'params': list(self.model.netInstance.parameters()), 'lr': self.lr * 1.},
            ]
        elif optimize_rule == 'only-emb':
            optimize_list = [
                {'name': 'class_embeddings', 'params': list([self.model.netPrior.classes_vectors]), 'lr': self.lr * 10.}
            ]
        elif optimize_rule == 'emb-instance':
            optimize_list = [
                {'name': 'class_embeddings', 'params': list([self.model.netPrior.classes_vectors]), 'lr': self.lr * 10.},
                {'name': 'net_Instance', 'params': list(self.model.netInstance.parameters()), 'lr': self.lr * 1.},
            ]
        elif optimize_rule == 'custom':
            optimize_list = [
                {'name': 'net_Prior', 'params': list(self.model.netPrior.parameters()), 'lr': self.lr * 10.},
                {'name': 'netEncoder', 'params': list(self.model.netInstance.netEncoder.parameters()), 'lr': self.lr * 1.},
                {'name': 'netTexture', 'params': list(self.model.netInstance.netTexture.parameters()), 'lr': self.lr * 1.},
                {'name': 'netPose', 'params': list(self.model.netInstance.netPose.parameters()), 'lr': self.lr * 0.01},
                {'name': 'netArticulation', 'params': list(self.model.netInstance.netArticulation.parameters()), 'lr': self.lr * 1.},
                {'name': 'netLight', 'params': list(self.model.netInstance.netLight.parameters()), 'lr': self.lr * 1.}
            ]
        elif optimize_rule == 'custom-deform':
            optimize_list = [
                {'name': 'net_Prior', 'params': list(self.model.netPrior.parameters()), 'lr': self.lr * 10.},
                {'name': 'netEncoder', 'params': list(self.model.netInstance.netEncoder.parameters()), 'lr': self.lr * 1.},
                {'name': 'netTexture', 'params': list(self.model.netInstance.netTexture.parameters()), 'lr': self.lr * 1.},
                {'name': 'netPose', 'params': list(self.model.netInstance.netPose.parameters()), 'lr': self.lr * 0.01},
                {'name': 'netArticulation', 'params': list(self.model.netInstance.netArticulation.parameters()), 'lr': self.lr * 1.},
                {'name': 'netLight', 'params': list(self.model.netInstance.netLight.parameters()), 'lr': self.lr * 1.},
                {'name': 'netDeform', 'params': list(self.model.netInstance.netDeform.parameters()), 'lr': self.lr * 1.}
            ]
        elif optimize_rule == 'texture':
            optimize_list = [
                {'name': 'netTexture', 'params': list(self.model.netInstance.netTexture.parameters()), 'lr': self.lr * 1.}
            ]
        elif optimize_rule == 'texture-light':
            optimize_list = [
                {'name': 'netTexture', 'params': list(self.model.netInstance.netTexture.parameters()), 'lr': self.lr * 1.},
                {'name': 'netLight', 'params': list(self.model.netInstance.netLight.parameters()), 'lr': self.lr * 1.}
            ]
        elif optimize_rule == 'exp':
            optimize_list = [
                {'name': 'net_Prior', 'params': list(self.model.netPrior.parameters()), 'lr': self.lr * 10.},
                {'name': 'netEncoder', 'params': list(self.model.netInstance.netEncoder.parameters()), 'lr': self.lr * 1.},
                {'name': 'netTexture', 'params': list(self.model.netInstance.netTexture.parameters()), 'lr': self.lr * 1.},
                {'name': 'netPose', 'params': list(self.model.netInstance.netPose.parameters()), 'lr': self.lr * 1.},
                {'name': 'netArticulation', 'params': list(self.model.netInstance.netArticulation.parameters()), 'lr': self.lr * 1.},
                {'name': 'netLight', 'params': list(self.model.netInstance.netLight.parameters()), 'lr': self.lr * 1.},
                {'name': 'netDeform', 'params': list(self.model.netInstance.netDeform.parameters()), 'lr': self.lr * 1.}
            ]
        else:
            raise NotImplementedError
        
        if self.enable_memory_bank and optimize_rule != 'texture':
            
            optimize_bank_components = self.cfgs.get('few_shot_optimize_bank', 'all')
            if optimize_bank_components == 'value':
                optimize_list += [
                    {'name': 'memory_bank', 'params': list([self.memory_bank]), 'lr': self.lr * 10.}
                ]
            elif optimize_bank_components == 'key':
                optimize_list += [
                    {'name': 'memory_bank_keys', 'params': list([self.memory_bank_keys]), 'lr': self.lr * 10.}
                ]
            else:
                optimize_list += [
                    {'name': 'memory_bank', 'params': list([self.memory_bank]), 'lr': self.lr * 10.},
                    {'name': 'memory_bank_keys', 'params': list([self.memory_bank_keys]), 'lr': self.lr * 10.}
                ]

        if self.model.enable_vsd:
            optimize_list += [
                {'name': 'lora', 'params': list(self.model.stable_diffusion.parameters()), 'lr': self.lr}
            ]

        # self.optimizerFewShot = torch.optim.Adam(
        #     [
        #         # {'name': 'class_embeddings', 'params': list([self.model.netPrior.classes_vectors]), 'lr': self.lr * 1.},
        #         {'name': 'net_Prior', 'params': list(self.model.netPrior.parameters()), 'lr': self.lr * 10.},
        #         {'name': 'net_Instance', 'params': list(self.model.netInstance.parameters()), 'lr': self.lr * 1.},
        #         # {'name': 'net_articulation', 'params': list(self.model.netInstance.netArticulation.parameters()), 'lr': self.lr * 10.}
        #     ], betas=(0.9, 0.99), eps=1e-15
        # )
        self.optimizerFewShot = torch.optim.Adam(optimize_list, betas=(0.9, 0.99), eps=1e-15)

        # if self.cfgs.get('texture_way', None) is not None and self.cfgs.get('gan_tex', False):
        if self.cfgs.get('gan_tex', False):
            self.optimizerDiscTex = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.discriminator_texture.parameters()), lr=self.lr, betas=(0.9, 0.99), eps=1e-15)

    def load_checkpoint(self, optim=True, checkpoint_name=None):
        # use to load the checkpoint of model and optimizer in the finetuning
        """Search the specified/latest checkpoint in checkpoint_dir and load the model and optimizer."""
        if checkpoint_name is not None:
            checkpoint_path = osp.join(self.checkpoint_dir, checkpoint_name)
        else:
            checkpoints = sorted(glob.glob(osp.join(self.checkpoint_dir, '*.pth')))
            if len(checkpoints) == 0:
                return 0, 0
            checkpoint_path = checkpoints[-1]
            self.checkpoint_name = osp.basename(checkpoint_path)
        print(f"Loading checkpoint from {checkpoint_path}")
        cp = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_model_state(cp)  # the cp has netPrior and netInstance as keys
        if optim:
            try:
                self.optimizerFewShot.load_state_dict(cp['optimizerFewShot'])
            except:
                print('you should be using the local texture so dont need to load the previous optimizer')
        if self.enable_memory_bank:
            self.memory_bank_keys = cp['memory_bank_keys']
            self.memory_bank = cp['memory_bank']
        self.metrics_trace = cp['metrics_trace']
        epoch = cp['epoch']
        total_iter = cp['total_iter']
        return epoch, total_iter
    
    def save_checkpoint(self, epoch, total_iter=0, optim=True, use_iter=False):
        """Save model, optimizer, and metrics state to a checkpoint in checkpoint_dir for the specified epoch."""
        misc.xmkdir(self.checkpoint_dir)
        if use_iter:
            checkpoint_path = osp.join(self.checkpoint_dir, f'iter{total_iter:07}.pth')
            prefix = 'iter*.pth'
        else:
            checkpoint_path = osp.join(self.checkpoint_dir, f'checkpoint{epoch:03}.pth')
            prefix = 'checkpoint*.pth'
        state_dict = self.model.get_model_state()
        if optim:
            optimizer_state = {'optimizerFewShot': self.optimizerFewShot.state_dict()}
            state_dict = {**state_dict, **optimizer_state}
        state_dict['metrics_trace'] = self.metrics_trace
        state_dict['epoch'] = epoch
        state_dict['total_iter'] = total_iter
        if self.enable_memory_bank:
            state_dict['memory_bank_keys'] = self.memory_bank_keys
            state_dict['memory_bank'] = self.memory_bank
        print(f"Saving checkpoint to {checkpoint_path}")
        torch.save(state_dict, checkpoint_path)
        if self.keep_num_checkpoint > 0:
            self.clean_checkpoint(self.checkpoint_dir, keep_num=self.keep_num_checkpoint, prefix=prefix)

    def clean_checkpoint(self, checkpoint_dir, keep_num=2, prefix='checkpoint*.pth'):
        if keep_num > 0:
            names = list(sorted(
                glob.glob(os.path.join(checkpoint_dir, prefix))
            ))
            if len(names) > keep_num:
                for name in names[:-keep_num]:
                    print(f"Deleting obslete checkpoint file {name}")
                    os.remove(name)
    
    def get_data_loaders_few_shot(self, cfgs, batch_size, num_workers, in_image_size, out_image_size):
        # support the train_data_loaders, and also an identical val_data_loader?
        train_loader = val_loader = None

        color_jitter_train = cfgs.get('color_jitter_train', None)
        color_jitter_val = cfgs.get('color_jitter_val', None)
        random_flip_train = cfgs.get('random_flip_train', False)

        data_loader_mode = cfgs.get('data_loader_mode', 'n_frame')
        
        num_sample_frames = cfgs.get('num_sample_frames', 2)
        shuffle_train_seqs = cfgs.get('shuffle_train_seqs', False)
        load_background = cfgs.get('background_mode', 'none') == 'background'
        rgb_suffix = cfgs.get('rgb_suffix', '.png')
        load_dino_feature = cfgs.get('load_dino_feature', False)
        dino_feature_dim = cfgs.get('dino_feature_dim', 64)
        get_loader_ddp = lambda **kwargs: get_sequence_loader_ddp(
            mode=data_loader_mode,
            batch_size=batch_size,
            num_workers=num_workers,
            in_image_size=in_image_size,
            out_image_size=out_image_size,
            num_sample_frames=num_sample_frames,
            load_background=load_background,
            rgb_suffix=rgb_suffix,
            load_dino_feature=load_dino_feature,
            dino_feature_dim=dino_feature_dim,
            flow_bool=0,
            **kwargs)

        print(f"Loading training data...")
        train_loader = get_loader_ddp(data_dir=[self.original_classes_num, self.few_shot_categories_paths], rank=self.rank, world_size=self.world_size, use_few_shot=True, shuffle=False, color_jitter=color_jitter_train, random_flip=random_flip_train)
        return train_loader, val_loader

    def get_data_loaders_original(self, cfgs, batch_size, num_workers, in_image_size, out_image_size):
        train_loader = val_loader = test_loader = None
        color_jitter_train = cfgs.get('color_jitter_train', None)
        color_jitter_val = cfgs.get('color_jitter_val', None)
        random_flip_train = cfgs.get('random_flip_train', False)

        data_loader_mode = cfgs.get('data_loader_mode', 'n_frame')
        skip_beginning = cfgs.get('skip_beginning', 4)
        skip_end = cfgs.get('skip_end', 4)
        num_sample_frames = cfgs.get('num_sample_frames', 2)
        min_seq_len = cfgs.get('min_seq_len', 10)
        max_seq_len = cfgs.get('max_seq_len', 10)
        debug_seq = cfgs.get('debug_seq', False)
        random_sample_train_frames = cfgs.get('random_sample_train_frames', False)
        shuffle_train_seqs = cfgs.get('shuffle_train_seqs', False)
        random_sample_val_frames = cfgs.get('random_sample_val_frames', False)
        load_background = cfgs.get('background_mode', 'none') == 'background'
        rgb_suffix = cfgs.get('rgb_suffix', '.png')
        load_dino_feature = cfgs.get('load_dino_feature', False)
        load_dino_cluster = cfgs.get('load_dino_cluster', False)
        dino_feature_dim = cfgs.get('dino_feature_dim', 64)
        get_loader_ddp = lambda **kwargs: get_sequence_loader_ddp(
            mode=data_loader_mode,
            batch_size=batch_size,
            num_workers=num_workers,
            in_image_size=in_image_size,
            out_image_size=out_image_size,
            debug_seq=debug_seq,
            skip_beginning=skip_beginning,
            skip_end=skip_end,
            num_sample_frames=num_sample_frames,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            load_background=load_background,
            rgb_suffix=rgb_suffix,
            load_dino_feature=load_dino_feature,
            load_dino_cluster=load_dino_cluster,
            dino_feature_dim=dino_feature_dim,
            flow_bool=0,
            **kwargs)
        
        # just the train now
        train_data_dir = self.original_categories_paths
        if isinstance(train_data_dir, dict):
            for data_path in train_data_dir.values():
                assert osp.isdir(data_path), f"Training data directory does not exist: {data_path}"
        elif isinstance(train_data_dir, str):
            assert osp.isdir(train_data_dir), f"Training data directory does not exist: {train_data_dir}"
        else:
            raise ValueError("train_data_dir must be a string or a dict of strings")
        
        print(f"Loading training data...")
        # the train_data_dir is a dict and will go into the original dataset type
        train_loader = get_loader_ddp(data_dir=train_data_dir, rank=self.rank, world_size=self.world_size, is_validation=False, use_few_shot=False, random_sample=random_sample_train_frames, shuffle=shuffle_train_seqs, dense_sample=True, color_jitter=color_jitter_train, random_flip=random_flip_train)

        return train_loader, val_loader

    def get_data_loaders_quadrupeds(self, cfgs, batch_size, num_workers, in_image_size, out_image_size):
        train_loader = val_loader = test_loader = None
        color_jitter_train = cfgs.get('color_jitter_train', None)
        color_jitter_val = cfgs.get('color_jitter_val', None)
        random_flip_train = cfgs.get('random_flip_train', False)

        data_loader_mode = cfgs.get('data_loader_mode', 'n_frame')
        skip_beginning = cfgs.get('skip_beginning', 4)
        skip_end = cfgs.get('skip_end', 4)
        num_sample_frames = cfgs.get('num_sample_frames', 2)
        min_seq_len = cfgs.get('min_seq_len', 10)
        max_seq_len = cfgs.get('max_seq_len', 10)
        debug_seq = cfgs.get('debug_seq', False)
        random_sample_train_frames = cfgs.get('random_sample_train_frames', False)
        shuffle_train_seqs = cfgs.get('shuffle_train_seqs', False)
        random_sample_val_frames = cfgs.get('random_sample_val_frames', False)
        load_background = cfgs.get('background_mode', 'none') == 'background'
        rgb_suffix = cfgs.get('rgb_suffix', '.png')
        load_dino_feature = cfgs.get('load_dino_feature', False)
        load_dino_cluster = cfgs.get('load_dino_cluster', False)
        dino_feature_dim = cfgs.get('dino_feature_dim', 64)

        enhance_back_view = cfgs.get('enhance_back_view', False)
        enhance_back_view_path = cfgs.get('enhance_back_view_path', None)

        override_categories = cfgs.get('override_categories', None)

        disable_fewshot = cfgs.get('disable_fewshot', False)
        dataset_split_num = cfgs.get('dataset_split_num', -1)

        get_loader_ddp = lambda **kwargs: get_sequence_loader_quadrupeds(
            mode=data_loader_mode,
            num_workers=num_workers,
            in_image_size=in_image_size,
            out_image_size=out_image_size,
            debug_seq=debug_seq,
            skip_beginning=skip_beginning,
            skip_end=skip_end,
            num_sample_frames=num_sample_frames,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            load_background=load_background,
            rgb_suffix=rgb_suffix,
            load_dino_feature=load_dino_feature,
            load_dino_cluster=load_dino_cluster,
            dino_feature_dim=dino_feature_dim,
            flow_bool=0,
            enhance_back_view=enhance_back_view,
            enhance_back_view_path=enhance_back_view_path,
            override_categories=override_categories,
            disable_fewshot=disable_fewshot,
            dataset_split_num=dataset_split_num,
            **kwargs)
        
        # just the train now
        
        print(f"Loading training data...")
        val_image_num = cfgs.get('few_shot_val_image_num', 5)
        # the train_data_dir is a dict and will go into the original dataset type
        train_loader = get_loader_ddp(original_data_dirs=self.original_categories_paths, few_shot_data_dirs=self.few_shot_categories_paths, original_num=self.original_classes_num, few_shot_num=self.new_classes_num, rank=self.rank, world_size=self.world_size, batch_size=batch_size, is_validation=False, val_image_num=val_image_num, shuffle=shuffle_train_seqs, dense_sample=True, color_jitter=color_jitter_train, random_flip=random_flip_train)
        val_loader = get_loader_ddp(original_data_dirs=self.original_val_data_path, few_shot_data_dirs=self.few_shot_categories_paths, original_num=self.original_classes_num, few_shot_num=self.new_classes_num, rank=self.rank, world_size=self.world_size, batch_size=1, is_validation=True, val_image_num=val_image_num, shuffle=False, dense_sample=True, color_jitter=color_jitter_val, random_flip=False)

        if self.test_categories_paths is not None:
            get_test_loader_ddp = lambda **kwargs: get_test_loader_quadrupeds(
                mode=data_loader_mode,
                num_workers=num_workers,
                in_image_size=in_image_size,
                out_image_size=out_image_size,
                debug_seq=debug_seq,
                skip_beginning=skip_beginning,
                skip_end=skip_end,
                num_sample_frames=num_sample_frames,
                min_seq_len=min_seq_len,
                max_seq_len=max_seq_len,
                load_background=load_background,
                rgb_suffix=rgb_suffix,
                load_dino_feature=load_dino_feature,
                load_dino_cluster=load_dino_cluster,
                dino_feature_dim=dino_feature_dim,
                flow_bool=0,
                enhance_back_view=enhance_back_view,
                enhance_back_view_path=enhance_back_view_path,
                **kwargs)
            print(f"Loading testing data...")
            test_loader = get_test_loader_ddp(test_data_dirs=self.test_categories_paths, rank=self.rank, world_size=self.world_size, batch_size=1, is_validation=True, shuffle=False, dense_sample=True, color_jitter=color_jitter_val, random_flip=False)
        else:
            test_loader = None

        return train_loader, val_loader, test_loader
    
    def forward_frozen_ViT(self, images):
        # this part use the frozen pre-train ViT
        x = images
        with torch.no_grad():
            b, c, h, w = x.shape
            self.model.netInstance.netEncoder._feats = []
            self.model.netInstance.netEncoder._register_hooks([11], 'key')
            #self._register_hooks([11], 'token')
            x = self.model.netInstance.netEncoder.ViT.prepare_tokens(x)
            #x = self.ViT.prepare_tokens_with_masks(x)
            
            for blk in self.model.netInstance.netEncoder.ViT.blocks:
                x = blk(x)
            out = self.model.netInstance.netEncoder.ViT.norm(x)
            self.model.netInstance.netEncoder._unregister_hooks()

            ph, pw = h // self.model.netInstance.netEncoder.patch_size, w // self.model.netInstance.netEncoder.patch_size
            patch_out = out[:, 1:]  # first is class token
            patch_out = patch_out.reshape(b, ph, pw, self.model.netInstance.netEncoder.vit_feat_dim).permute(0, 3, 1, 2)

            patch_key = self.model.netInstance.netEncoder._feats[0][:,:,1:]  # B, num_heads, num_patches, dim
            patch_key = patch_key.permute(0, 1, 3, 2).reshape(b, self.model.netInstance.netEncoder.vit_feat_dim, ph, pw)

            global_feat = out[:, 0]
        
        return global_feat
    
    def forward_fix_embeddings(self, batch):
        images = batch[0]
        images = images.to(self.device)
        batch_size, num_frames, _, h0, w0 = images.shape
        images = images.reshape(batch_size*num_frames, *images.shape[2:])  # 0~1

        if self.memory_encoder == 'DINO':
            images_in = images * 2 - 1  # rescale to (-1, 1)
            batch_features = self.forward_frozen_ViT(images_in)
        elif self.memory_encoder == 'CLIP':
            images_in = torch.nn.functional.interpolate(images, (self.clip_reso, self.clip_reso), mode='bilinear')
            images_in = tvf.normalize(images_in, self.clip_mean, self.clip_std)
            batch_features = self.clip_model.encode_image(images_in).float()
        else:
            raise NotImplementedError
        return batch_features

    def retrieve_memory_bank(self, batch_features, batch):
        batch_size = batch_features.shape[0]
        
        if self.memory_retrieve == 'cos-linear':
            query = torch.nn.functional.normalize(batch_features.unsqueeze(1), dim=-1)      # [B, 1, d_k]
            key = torch.nn.functional.normalize(self.memory_bank_keys, dim=-1)              # [size, d_k]
            key = key.transpose(1, 0).unsqueeze(0).repeat(batch_size, 1, 1).to(query.device)             # [B, d_k, size]

            cos_dist = torch.bmm(query, key).squeeze(1)         # [B, size], larger the more similar
            rank_idx = torch.sort(cos_dist, dim=-1, descending=True)[1][:, :self.memory_bank_topk] # [B, k]
            value = self.memory_bank.unsqueeze(0).repeat(batch_size, 1, 1).to(query.device)                         # [B, size, d_v]

            out = torch.gather(value, dim=1, index=rank_idx[..., None].repeat(1, 1, self.memory_bank_dim))  # [B, k, d_v]

            weights = torch.gather(cos_dist, dim=-1, index=rank_idx)    # [B, k]
            weights = torch.nn.functional.normalize(weights, p=1.0, dim=-1).unsqueeze(-1).repeat(1, 1, self.memory_bank_dim)    # [B, k, d_v] weights have been normalized

            out = weights * out
            out = torch.sum(out, dim=1)

        else:
            raise NotImplementedError
        
        batch_mean_out = torch.mean(out, dim=0)

        weight_aux = {
            'weights': weights[:, :, 0], # [B, k], weights from large to small
            'pick_idx': rank_idx, # [B, k]
        }

        return batch_mean_out, out, weight_aux

    def discriminator_texture_step(self):
        image_iv = self.model.record_image_iv
        image_rv = self.model.record_image_rv
        image_gt = self.model.record_image_gt
        
        self.model.record_image_iv = None
        self.model.record_image_rv = None
        self.model.record_image_gt = None

        image_iv = image_iv.requires_grad_(True)
        image_rv = image_rv.requires_grad_(True)
        image_gt = image_gt.requires_grad_(True)

        self.optimizerDiscTex.zero_grad()
        disc_loss_gt = 0.0
        disc_loss_iv = 0.0
        disc_loss_rv = 0.0
        grad_penalty = 0.0
        # for the gt image, it can only be in real or not
        if 'gt' in self.model.few_shot_gan_tex_real:
            d_gt = self.model.discriminator_texture(image_gt)
            disc_loss_gt += discriminator_architecture.bce_loss_target(d_gt, 1)
            if image_gt.requires_grad:
                grad_penalty_gt = 10. * discriminator_architecture.compute_grad2(d_gt, image_gt)
                disc_loss_gt += grad_penalty_gt
                grad_penalty += grad_penalty_gt
        
        # for the input view image, it can be in real or fake
        if 'iv' in self.model.few_shot_gan_tex_real:
            d_iv = self.model.discriminator_texture(image_iv)
            disc_loss_iv += discriminator_architecture.bce_loss_target(d_iv, 1)
            if image_iv.requires_grad:
                grad_penalty_iv = 10. * discriminator_architecture.compute_grad2(d_iv, image_iv)
                disc_loss_iv += grad_penalty_iv
                grad_penalty += grad_penalty_iv
        elif 'iv' in self.model.few_shot_gan_tex_fake:
            d_iv = self.model.discriminator_texture(image_iv)
            disc_loss_iv += discriminator_architecture.bce_loss_target(d_iv, 0)
        
        # for the random view image, it can only be in fake
        if 'rv' in self.model.few_shot_gan_tex_fake:
            d_rv = self.model.discriminator_texture(image_rv)
            disc_loss_rv += discriminator_architecture.bce_loss_target(d_rv, 0)

        all_loss = disc_loss_iv + disc_loss_rv + disc_loss_gt

        all_loss = all_loss * self.cfgs.get('gan_tex_loss_discriminator_weight', 0.1)
        self.discriminator_texture_loss = all_loss
        self.discriminator_texture_loss.backward()
        self.optimizerDiscTex.step()
        self.discriminator_texture_loss = 0.

        return {
            'discriminator_loss': all_loss.detach(),
            'discriminator_loss_iv': disc_loss_iv.detach(),
            'discriminator_loss_rv': disc_loss_rv.detach(),
            'discriminator_loss_gt': disc_loss_gt.detach(),
            'discriminator_loss_grad': grad_penalty.detach()
        }

    def train(self):
        """Perform training."""
        # archive code and configs
        if self.archive_code:
            misc.archive_code(osp.join(self.checkpoint_dir, 'archived_code.zip'), filetypes=['.py'])
        misc.dump_yaml(osp.join(self.checkpoint_dir, 'configs.yml'), self.cfgs)

        # initialize
        start_epoch = 0
        self.total_iter = 0
        self.total_iter = self.original_total_iter
        self.metrics_trace.reset()
        self.model.to(self.device)

        if self.model.enable_disc:
            self.model.reset_only_disc_optimizer()

        if self.few_shot_resume:
            resume_model_name = self.cfgs.get('few_shot_resume_name', None)
            start_epoch, self.total_iter = self.load_checkpoint(optim=True, checkpoint_name=resume_model_name)
        
        self.model.ddp(self.rank, self.world_size)

        # use tensorboard
        if self.use_logger:
            from torch.utils.tensorboard import SummaryWriter
            self.logger = SummaryWriter(osp.join(self.checkpoint_dir, 'logs', datetime.now().strftime("%Y%m%d-%H%M%S")), flush_secs=10)
            # self.viz_data_iterator = indefinite_generator_from_list(self.val_loader) if self.visualize_validation else indefinite_generator_from_list(self.train_loader)
            self.viz_data_iterator = indefinite_generator(self.val_loader[0]) if self.visualize_validation else indefinite_generator(self.train_loader[0])
            if self.fix_viz_batch:
                self.viz_batch = next(self.viz_data_iterator)
            
            if self.test_loader is not None:
                self.viz_test_data_iterator = indefinite_generator(self.test_loader[0]) if self.visualize_validation else indefinite_generator(self.train_loader[0])
        
        # run_epochs
        epoch = 0

        for epoch in range(start_epoch, self.num_epochs):
            metrics = self.run_epoch(epoch)
            if self.combine_dataset:
                self.train_loader[0].dataset._shuffle_all()
            self.metrics_trace.append("train", metrics)
            if (epoch+1) % self.save_checkpoint_freq == 0:
                self.save_checkpoint(epoch+1, total_iter=self.total_iter, optim=True)
            # if self.cfgs.get('pyplot_metrics', True):
            #     self.metrics_trace.plot(pdf_path=osp.join(self.checkpoint_dir, 'metrics.pdf'))
            self.metrics_trace.save(osp.join(self.checkpoint_dir, 'metrics.json'))
        print(f"Training completed for all {epoch+1} epochs.")
    
    def run_epoch(self, epoch):
        """Run one training epoch."""
        metrics = self.make_metrics()

        self.model.set_train()

        max_loader_len = max([len(loader) for loader in self.train_loader])
        train_generators = [indefinite_generator(loader) for loader in self.train_loader]

        iteration = 0
        while iteration < max_loader_len * len(self.train_loader):
            for generator in train_generators:
                batch = next(generator)

                self.total_iter += 1
                num_seqs, num_frames = batch[0].shape[:2]
                total_im_num = num_seqs * num_frames

                if self.enable_memory_bank:
                    batch_features = self.forward_fix_embeddings(batch)
                    batch_embedding, embeddings, weights = self.retrieve_memory_bank(batch_features, batch)
                    bank_embedding_model_input = [batch_embedding, embeddings, weights]
                else:
                    # bank_embedding_model_input = None
                    batch_features = self.forward_fix_embeddings(batch)
                    weights = {
                        "weights": torch.rand(1,10).to(batch_features.device),
                        "pick_idx": torch.randint(low=0, high=60, size=(1, 10)).to(batch_features.device)
                    }
                    bank_embedding_model_input = [batch_features[0], batch_features, weights]
                m = self.model.forward(batch, epoch=epoch, iter=iteration, total_iter=self.total_iter, which_data=self.dataset, is_training=True, bank_embedding=bank_embedding_model_input)

                # self.model.backward()
                self.optimizerFewShot.zero_grad()
                self.model.total_loss.backward()
                self.optimizerFewShot.step()
                self.model.total_loss = 0.

                # if self.cfgs.get('texture_way', None) is not None and self.cfgs.get('gan_tex', False):
                if self.model.few_shot_gan_tex:
                    # the discriminator for local texture
                    disc_ret = self.discriminator_texture_step()
                    m.update(disc_ret)
                
                if self.model.enable_disc and (self.model.mask_discriminator_iter[0] < self.total_iter) and (self.model.mask_discriminator_iter[1] > self.total_iter):
                    # the discriminator training
                    discriminator_loss_dict, grad_loss = self.model.discriminator_step()
                    m.update(
                        {
                            'mask_disc_loss_discriminator': discriminator_loss_dict['discriminator_loss'] - grad_loss, 
                            'mask_disc_loss_discriminator_grad': grad_loss,
                            'mask_disc_loss_discriminator_rv': discriminator_loss_dict['discriminator_loss_rv'],
                            'mask_disc_loss_discriminator_iv': discriminator_loss_dict['discriminator_loss_iv'],
                            'mask_disc_loss_discriminator_gt': discriminator_loss_dict['discriminator_loss_gt']
                        }
                    )
                    self.logger.add_histogram('train_'+'discriminator_logits/random_view', discriminator_loss_dict['d_rv'], self.total_iter)
                    if discriminator_loss_dict['d_iv'] is not None:
                        self.logger.add_histogram('train_'+'discriminator_logits/input_view', discriminator_loss_dict['d_iv'], self.total_iter)
                    if discriminator_loss_dict['d_gt'] is not None:
                        self.logger.add_histogram('train_'+'discriminator_logits/gt_view', discriminator_loss_dict['d_gt'], self.total_iter)

                metrics.update(m, total_im_num)
                if self.rank == 0:
                    print(f"T{epoch:04}/{iteration:05}/{metrics}")

                if self.iteration_save and self.total_iter % self.iteration_save_freq == 0:
                    self.save_checkpoint(epoch+1, total_iter=self.total_iter, optim=True, use_iter=True)

                # ## reset optimizers
                # if self.cfgs.get('opt_reset_every_iter', 0) > 0 and self.total_iter < self.cfgs.get('opt_reset_end_iter', 0):
                #     if self.total_iter % self.cfgs.get('opt_reset_every_iter', 0) == 0:
                #         self.model.reset_optimizers()

                if misc.is_main_process() and self.use_logger:
                    if self.rank == 0 and self.total_iter % self.log_freq_losses == 0:
                        for name, loss in m.items():
                            label = f'cub_loss_train/{name[4:]}' if 'cub' in name else f'loss_train/{name}'
                            self.logger.add_scalar(label, loss, self.total_iter)
                    if self.rank == 0 and self.save_result_freq is not None and self.total_iter % self.save_result_freq == 0:
                        with torch.no_grad():
                            m = self.model.forward(batch, epoch=epoch, iter=iteration, total_iter=self.total_iter, save_results=False, save_dir=self.train_result_dir, which_data=self.dataset, is_training=False, bank_embedding=bank_embedding_model_input)
                            torch.cuda.empty_cache()
                    if self.total_iter % self.log_freq_images == 0:
                        with torch.no_grad():
                            if self.rank == 0 and self.log_train_images:
                                m = self.model.forward(batch, epoch=epoch, iter=iteration, viz_logger=self.logger, total_iter=self.total_iter, which_data=self.dataset, logger_prefix='train_', is_training=False, bank_embedding=bank_embedding_model_input)
                            if self.fix_viz_batch:
                                print(f'fix_viz_batch:{self.fix_viz_batch}')
                                batch_val = self.viz_batch
                            else:
                                batch_val = next(self.viz_data_iterator)
                            if self.visualize_validation:
                                import time
                                vis_start = time.time()
                                # batch = next(self.viz_data_iterator)
                                # try:
                                #     batch = next(self.viz_data_iterator)
                                # except:  # iterator exhausted
                                #     self.reset_viz_data_iterator()
                                #     batch = next(self.viz_data_iterator)
                                if self.enable_memory_bank:
                                    batch_features_val = self.forward_fix_embeddings(batch_val)
                                    batch_embedding_val, embeddings_val, weights_val = self.retrieve_memory_bank(batch_features_val, batch_val)
                                    bank_embedding_model_input_val = [batch_embedding_val, embeddings_val, weights_val]
                                else:
                                    # bank_embedding_model_input_val = None
                                    batch_features_val = self.forward_fix_embeddings(batch_val)
                                    weights_val = {
                                        "weights": torch.rand(1,10).to(batch_features_val.device),
                                        "pick_idx": torch.randint(low=0, high=60, size=(1, 10)).to(batch_features_val.device)
                                    }
                                    bank_embedding_model_input_val = [batch_features_val[0], batch_features_val, weights_val]
                                
                                if self.total_iter % self.save_result_freq == 0:
                                    m = self.model.forward(batch_val, epoch=epoch, iter=iteration, viz_logger=self.logger, total_iter=self.total_iter, save_results=False, save_dir=self.train_result_dir, which_data=self.dataset, logger_prefix='val_', is_training=False, bank_embedding=bank_embedding_model_input_val)
                                    torch.cuda.empty_cache()
                                
                                vis_end = time.time()
                                print(f"vis time: {vis_end - vis_start}")

                                if self.test_loader is not None:
                                    # unseen category test visualization
                                    batch_test = next(self.viz_test_data_iterator)
                                    if self.enable_memory_bank:
                                        batch_features_test = self.forward_fix_embeddings(batch_test)
                                        batch_embedding_test, embeddings_test, weights_test = self.retrieve_memory_bank(batch_features_test, batch_test)
                                        bank_embedding_model_input_test = [batch_embedding_test, embeddings_test, weights_test]
                                    else:
                                        # bank_embedding_model_input_test = None
                                        batch_features_test = self.forward_fix_embeddings(batch_test)
                                        weights_test = {
                                            "weights": torch.rand(1,10).to(batch_features_test.device),
                                            "pick_idx": torch.randint(low=0, high=60, size=(1, 10)).to(batch_features_test.device)
                                        }
                                        bank_embedding_model_input_test = [batch_features_test[0], batch_features_test, weights_test]
                                    m_test = self.model.forward(batch_test, epoch=epoch, iter=iteration, viz_logger=self.logger, total_iter=self.total_iter, which_data=self.dataset, logger_prefix='test_', is_training=False, bank_embedding=bank_embedding_model_input_test)
                                    vis_test_end = time.time()
                                    print(f"vis test time: {vis_test_end - vis_end}")
                                    for name, loss in m_test.items():
                                        if self.rank == 0:
                                            self.logger.add_scalar(f'loss_test/{name}', loss, self.total_iter)

                            for name, loss in m.items():
                                if self.rank == 0:
                                    self.logger.add_scalar(f'loss_val/{name}', loss, self.total_iter)
                        torch.cuda.empty_cache()

                iteration += 1

        self.model.scheduler_step()
        return metrics
        