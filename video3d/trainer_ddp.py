import os
import os.path as osp
import glob
from datetime import datetime
import random
import torch
import video3d.utils.meters as meters
import video3d.utils.misc as misc

from video3d.dataloaders_ddp import get_sequence_loader_quadrupeds

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

def definite_generator(loader):
    for x in loader:
        yield x
    while True:
        yield None


class TrainerDDP:
    def __init__(self, cfgs, model):
        self.cfgs = cfgs
        self.is_dry_run = cfgs.get('is_dry_run', False)

        self.rank = cfgs.get('rank', 0)
        self.world_size = cfgs.get('world_size', 1)
        self.use_ddp = cfgs.get('use_ddp', True)

        self.device = cfgs.get('device', 'cpu')
        self.num_epochs = cfgs.get('num_epochs', 1)
        
        # The logic is, if the num_iterations is set in the cfg
        # for any 'epoch' in cfg, I rescale it to (epoch / 120) * epoch_now, as in horse exp
        # for any 'iter' in cfg, I just keep them the same
        self.num_iterations = cfgs.get('num_iterations', 0)
        if self.num_iterations != 0:
            self.use_total_iterations = True
        else:
            self.use_total_iterations = False

        self.num_sample_frames = cfgs.get('num_sample_frames', 100)
        self.sample_frame_stride = cfgs.get('sample_frame_stride', 1)
        self.checkpoint_dir = cfgs.get('checkpoint_dir', 'results')
        self.save_checkpoint_freq = cfgs.get('save_checkpoint_freq', 1)
        self.keep_num_checkpoint = cfgs.get('keep_num_checkpoint', 2)  # -1 for keeping all checkpoints
        self.resume = cfgs.get('resume', True)
        self.reset_epoch = cfgs.get('reset_epoch', False)
        self.finetune_ckpt = cfgs.get('finetune_ckpt', None)
        # print('!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print(f'reset epoch: {self.reset_epoch}')
        # print('!!!!!!!!!!!!!!!!!!!!!!!!!!')
        self.use_logger = cfgs.get('use_logger', True)
        self.log_freq_images = cfgs.get('log_freq_images', 1000)
        self.log_train_images = cfgs.get('log_train_images', False)
        self.log_freq_losses = cfgs.get('log_freq_losses', 100)
        self.visualize_validation = cfgs.get('visualize_validation', False)
        self.fix_viz_batch = cfgs.get('fix_viz_batch', False)
        self.archive_code = cfgs.get('archive_code', True)
        self.checkpoint_name = cfgs.get('checkpoint_name', None)
        self.test_result_dir = cfgs.get('test_result_dir', None)
        self.validate = cfgs.get('validate', False)
        self.current_epoch = 0
        self.logger = None
        self.viz_input = None
        self.dataset = cfgs.get('dataset', 'video')
        self.train_with_cub = cfgs.get('train_with_cub', False)
        self.train_with_kaggle = cfgs.get('train_with_kaggle', False)
        self.cub_start_epoch = cfgs.get('cub_start_epoch', 0)

        self.metrics_trace = meters.MetricsTrace()
        self.make_metrics = lambda m=None: meters.StandardMetrics(m)

        self.batch_size = cfgs.get('batch_size', 64)
        self.in_image_size = cfgs.get('in_image_size', 256)
        self.out_image_size = cfgs.get('out_image_size', 256)
        self.num_workers = cfgs.get('num_workers', 4)
        self.run_train = cfgs.get('run_train', False)
        self.train_data_dir = cfgs.get('train_data_dir', None)
        self.val_data_dir = cfgs.get('val_data_dir', None)
        self.run_test = cfgs.get('run_test', False)
        self.test_data_dir = cfgs.get('test_data_dir', None)
        self.flow_bool = cfgs.get('flow_bool', 0)

        if len(self.train_data_dir) <= 10 and len(self.val_data_dir) <= 10:
            self.train_loader, self.val_loader, self.test_loader = model.get_data_loaders_ddp(cfgs, self.dataset, self.rank, self.world_size, in_image_size=self.in_image_size, out_image_size=self.out_image_size, batch_size=self.batch_size, num_workers=self.num_workers, run_train=self.run_train, run_test=self.run_test, train_data_dir=self.train_data_dir, val_data_dir=self.val_data_dir, test_data_dir=self.test_data_dir, flow_bool=self.flow_bool)
        else:
            # for 128 categories specific training
            self.train_loader, self.val_loader, self.test_loader = self.get_efficient_data_loaders_ddp(
                cfgs,
                self.batch_size,
                self.num_workers,
                self.in_image_size,
                self.out_image_size
            )

        print(self.train_loader, self.val_loader, self.test_loader)
        if self.train_with_cub:
            self.batch_size_cub = cfgs.get('batch_size_cub', 64)
            self.data_dir_cub = cfgs.get('data_dir_cub', None)
            self.train_loader_cub, self.val_loader_cub, self.test_loader_cub = model.get_data_loaders_ddp(cfgs, 'cub', self.rank, self.world_size, in_image_size=self.in_image_size, batch_size=self.batch_size_cub, num_workers=self.num_workers, run_train=self.run_train, run_test=self.run_test, train_data_dir=self.data_dir_cub, val_data_dir=self.data_dir_cub, test_data_dir=self.data_dir_cub)
        if self.train_with_kaggle:
            self.batch_size_kaggle = cfgs.get('batch_size_kaggle', 64)
            self.data_dir_kaggle = cfgs.get('data_dir_kaggle', None)
            self.train_loader_kaggle, self.val_loader_kaggle, self.test_loader_kaggle = model.get_data_loaders_ddp(cfgs, 'kaggle', self.rank, self.world_size, in_image_size=self.in_image_size, batch_size=self.batch_size_kaggle, num_workers=self.num_workers, run_train=self.run_train, run_test=self.run_test, train_data_dir=self.data_dir_kaggle, val_data_dir=self.data_dir_kaggle, test_data_dir=self.data_dir_kaggle)

        if self.use_total_iterations:
            # reset the epoch related cfgs
            
            dataloader_length = max([len(loader) for loader in self.train_loader]) * len(self.train_loader)
            print("Total length of data loader is: ", dataloader_length)
            
            total_epoch = int(self.num_iterations / dataloader_length) + 1

            print(f'run for {total_epoch} epochs')
            
            print('is_main_process()?', misc.is_main_process())

            for k, v in cfgs.items():
                if 'epoch' in k:
                    if isinstance(v, list):
                        new_v = [int(total_epoch * x / 120) + 1 for x in v]
                        cfgs[k] = new_v
                    elif isinstance(v, int):
                        new_v = int(total_epoch * v / 120) + 1
                        cfgs[k] = new_v
                else:
                    continue
            
            self.num_epochs = total_epoch
            self.cub_start_epoch = cfgs.get('cub_start_epoch', 0)
            self.cfgs = cfgs

        self.model = model(cfgs)
        self.model.trainer = self
        self.save_result_freq = cfgs.get('save_result_freq', None)
        self.train_result_dir = osp.join(self.checkpoint_dir, 'results')

        self.use_wandb = cfgs.get('use_wandb', False)

    def get_efficient_data_loaders_ddp(self, cfgs, batch_size, num_workers, in_image_size, out_image_size):
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

        override_categories = None

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
            **kwargs)
        
        # just the train now
        print(f"Loading training data...")
        val_image_num = cfgs.get('few_shot_val_image_num', 5)
        # the train_data_dir is a dict and will go into the original dataset type

        #TODO: very hack here, directly assign first 7 as original categories
        o_class = ["horse", "elephant", "zebra", "cow", "giraffe", "sheep", "bear"]
        self.original_categories_paths = {}
        self.few_shot_categories_paths = {}
        self.original_val_data_path = {}
        
        for k,v in self.train_data_dir.items():
            if k in o_class:
                self.original_categories_paths.update({k: v})
                self.original_val_data_path.update({k: self.val_data_dir[k]})
            else:
                self.few_shot_categories_paths.update({k:v})
        self.new_classes_num = len(self.few_shot_categories_paths)
        self.original_classes_num = len(self.original_categories_paths)

        train_loader = get_loader_ddp(
            original_data_dirs=self.original_categories_paths, 
            few_shot_data_dirs=self.few_shot_categories_paths, 
            original_num=self.original_classes_num, 
            few_shot_num=self.new_classes_num, 
            rank=self.rank, 
            world_size=self.world_size, 
            batch_size=batch_size, 
            is_validation=False, 
            val_image_num=val_image_num, 
            shuffle=shuffle_train_seqs, 
            dense_sample=True, 
            color_jitter=color_jitter_train, 
            random_flip=random_flip_train
        )
        val_loader = get_loader_ddp(
            original_data_dirs=self.original_val_data_path, 
            few_shot_data_dirs=self.few_shot_categories_paths, 
            original_num=self.original_classes_num, 
            few_shot_num=self.new_classes_num, 
            rank=self.rank, 
            world_size=self.world_size, 
            batch_size=1, 
            is_validation=True, 
            val_image_num=val_image_num, 
            shuffle=False, 
            dense_sample=True, 
            color_jitter=color_jitter_val, 
            random_flip=False
        )

        test_loader = None

        return train_loader, val_loader, test_loader
    
    
    def load_checkpoint(self, optim=True, ckpt_path=None):
        """Search the specified/latest checkpoint in checkpoint_dir and load the model and optimizer."""
        if ckpt_path is not None:
            checkpoint_path = ckpt_path
            self.checkpoint_name = osp.basename(checkpoint_path)
        elif self.checkpoint_name is not None:
            checkpoint_path = osp.join(self.checkpoint_dir, self.checkpoint_name)
        else:
            checkpoints = sorted(glob.glob(osp.join(self.checkpoint_dir, '*.pth')))
            if len(checkpoints) == 0:
                return 0, 0
            checkpoint_path = checkpoints[-1]
            self.checkpoint_name = osp.basename(checkpoint_path)
        
        print(f"Loading checkpoint from {checkpoint_path}")
        cp = torch.load(checkpoint_path, map_location=self.device)
        # print(cp)
        self.model.load_model_state(cp)
        if optim:
            self.model.load_optimizer_state(cp)
        self.metrics_trace = cp['metrics_trace']
        epoch = cp['epoch']
        total_iter = cp['total_iter']
        
        if 'classes_vectors' in cp:
            self.model.classes_vectors = cp['classes_vectors']
        
        return epoch, total_iter

    def save_checkpoint(self, epoch, total_iter=0, optim=True):
        """Save model, optimizer, and metrics state to a checkpoint in checkpoint_dir for the specified epoch."""
        misc.xmkdir(self.checkpoint_dir)
        checkpoint_path = osp.join(self.checkpoint_dir, f'checkpoint{epoch:03}.pth')
        state_dict = self.model.get_model_state()
        if optim:
            optimizer_state = self.model.get_optimizer_state()
            state_dict = {**state_dict, **optimizer_state}
        state_dict['metrics_trace'] = self.metrics_trace
        state_dict['epoch'] = epoch
        state_dict['total_iter'] = total_iter
        print(f"Saving checkpoint to {checkpoint_path}")
        torch.save(state_dict, checkpoint_path)
        if self.keep_num_checkpoint > 0:
            misc.clean_checkpoint(self.checkpoint_dir, keep_num=self.keep_num_checkpoint)
    
    def save_last_checkpoint(self, epoch, total_iter=0, optim=True):
        """Save model, optimizer, and metrics state to a checkpoint in checkpoint_dir for the specified epoch."""
        misc.xmkdir(self.checkpoint_dir)
        checkpoint_path = osp.join(self.checkpoint_dir, 'last.pth')
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        state_dict = self.model.get_model_state()
        if optim:
            optimizer_state = self.model.get_optimizer_state()
            state_dict = {**state_dict, **optimizer_state}
        state_dict['metrics_trace'] = self.metrics_trace
        state_dict['epoch'] = epoch
        state_dict['total_iter'] = total_iter
        print(f"Saving checkpoint to {checkpoint_path}")
        torch.save(state_dict, checkpoint_path)

    def save_clean_checkpoint(self, path):
        """Save model state only to specified path."""
        torch.save(self.model.get_model_state(), path)

    def test(self):
        """Perform testing."""
        self.model.to(self.device)
        epoch, self.total_iter = self.load_checkpoint(optim=False)

        if self.use_ddp:
            self.model.ddp(self.rank, self.world_size)
        self.model.set_eval()

        if self.test_result_dir is None:
            self.test_result_dir = osp.join(self.checkpoint_dir, f'test_results_{self.checkpoint_name}'.replace('.pth', ''))
        print(f"Saving testing results to {self.test_result_dir}")

        with torch.no_grad():
            for iteration, batch in enumerate(self.test_loader):
                m = self.model.forward(batch, epoch=epoch, iter=iteration, total_iter=self.total_iter, save_results=True, save_dir=self.test_result_dir, which_data=self.dataset, is_training=False)
                print(f"T{epoch:04}/{iteration:05}")

        score_path = osp.join(self.test_result_dir, 'all_metrics.txt')
        # self.model.save_scores(score_path)

    def train(self):
        """Perform training."""
        # archive code and configs
        if self.archive_code:
            misc.archive_code(osp.join(self.checkpoint_dir, 'archived_code.zip'), filetypes=['.py'])
        misc.dump_yaml(osp.join(self.checkpoint_dir, 'configs.yml'), self.cfgs)

        # initialize
        start_epoch = 0
        self.total_iter = 0
        self.metrics_trace.reset()
        self.model.to(self.device)
        self.model.reset_optimizers()

        # resume from checkpoint
        # from IPython import embed; embed()
        if self.resume:
            start_epoch, self.total_iter = self.load_checkpoint(optim=True)
        
        if self.reset_epoch:
            start_epoch = 0
            self.total_iter = 0

        if start_epoch == 0 and self.total_iter ==0 and self.finetune_ckpt is not None:
            _, _ = self.load_checkpoint(optim=True, ckpt_path=self.finetune_ckpt)

        # distribute model
        if self.use_ddp:
            self.model.ddp(self.rank, self.world_size)

        # train with cub
        if self.train_with_cub:
            self.cub_train_data_iterator = indefinite_generator(self.train_loader_cub)

        # initialize tensorboard logger
        if misc.is_main_process() and self.use_logger:
            if self.use_wandb:
                import wandb
                wandb.tensorboard.patch(root_logdir=osp.join(self.checkpoint_dir, 'logs', datetime.now().strftime("%Y%m%d-%H%M%S")))
                wandb.init(name=self.checkpoint_dir.split("/")[-1], project="APT36K")
            from torch.utils.tensorboard import SummaryWriter
            self.logger = SummaryWriter(osp.join(self.checkpoint_dir, 'logs', datetime.now().strftime("%Y%m%d-%H%M%S")), flush_secs=10)
            self.viz_data_iterator = indefinite_generator_from_list(self.val_loader) if self.visualize_validation else indefinite_generator_from_list(self.train_loader)
            # self.viz_data_iterator = iter(self.viz_data_iterator)
            if self.fix_viz_batch:
                self.viz_batch = next(self.viz_data_iterator)

            # train with cub
            if self.train_with_cub:
                self.cub_viz_data_iterator = indefinite_generator(self.val_loader_cub) if self.visualize_validation else indefinite_generator(self.train_loader_cub)
                if self.fix_viz_batch:
                    self.viz_batch_cub = next(self.cub_viz_data_iterator)

        # run epochs
        epoch = 0
        for epoch in range(start_epoch, self.num_epochs):
            torch.distributed.barrier()
            metrics = self.run_epoch(epoch)
            if self.rank == 0:
                self.metrics_trace.append("train", metrics)
                if (epoch+1) % self.save_checkpoint_freq == 0:
                    self.save_checkpoint(epoch+1, total_iter=self.total_iter, optim=True)
                if self.cfgs.get('pyplot_metrics', True):
                    self.metrics_trace.plot(pdf_path=osp.join(self.checkpoint_dir, 'metrics.pdf'))
                self.metrics_trace.save(osp.join(self.checkpoint_dir, 'metrics.json'))
        if self.rank == 0:
            print(f"Training completed for all {epoch+1} epochs.")

    def dry_run(self):
        print(f'rank: {self.rank}, dry_run!!!!!')
        self.dry_run_iters = self.cfgs.get('dr_iters', 2)
        self.resume = self.cfgs.get('dr_resume', True)
        self.use_logger = self.cfgs.get('dr_use_logger', True)
        self.log_freq_losses = self.cfgs.get('dr_log_freq_losses', 1)
        self.save_result_freq = self.cfgs.get('dr_save_result_freq', 1)
        self.log_freq_images = self.cfgs.get('dr_log_freq_images', 1)
        self.log_train_images = self.cfgs.get('dr_log_train_images', True)
        self.visualize_validation = self.cfgs.get('dr_visualize_validation', True)
        self.num_epochs = self.cfgs.get('dr_num_epochs', 1)
        self.train()

    def run_epoch(self, epoch):
        metrics = self.make_metrics()

        self.model.set_train()

        max_loader_len = max([len(loader) for loader in self.train_loader])
        train_generators = [indefinite_generator(loader) for loader in self.train_loader]
        
        iteration = 0
        while iteration < max_loader_len * len(self.train_loader):
            for generator in train_generators:
                batch = next(generator)

                self.total_iter += 1

                if self.total_iter % 4000 == 0:
                    self.save_last_checkpoint(epoch+1, self.total_iter, optim=True)

                num_seqs, num_frames = batch[0].shape[:2]
                total_im_num = num_seqs * num_frames
                m = self.model.forward(batch, epoch=epoch, iter=iteration, total_iter=self.total_iter, which_data=self.dataset, is_training=True)

                if self.train_with_cub and epoch >= self.cub_start_epoch:
                    batch_cub = next(self.cub_train_data_iterator)
                    num_seqs, num_frames = batch_cub[0].shape[:2]
                    total_im_num += num_seqs * num_frames
                    m_cub = self.model.forward(batch_cub, epoch=epoch, iter=iteration, total_iter=self.total_iter, which_data='cub', is_training=True)
                    m.update({'cub_'+k: v for k,v in m_cub.items()})
                    m['total_loss'] = self.model.total_loss

                self.model.backward()

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

                ## reset optimizers
                if self.cfgs.get('opt_reset_every_iter', 0) > 0 and self.total_iter < self.cfgs.get('opt_reset_end_iter', 0):
                    if self.total_iter % self.cfgs.get('opt_reset_every_iter', 0) == 0:
                        self.model.reset_optimizers()

                if misc.is_main_process() and self.use_logger:
                    if self.rank == 0 and self.total_iter % self.log_freq_losses == 0:
                        for name, loss in m.items():
                            label = f'cub_loss_train/{name[4:]}' if 'cub' in name else f'loss_train/{name}'
                            self.logger.add_scalar(label, loss, self.total_iter)
                    if self.rank == 0 and self.save_result_freq is not None and self.total_iter % self.save_result_freq == 0:
                        with torch.no_grad():
                            m = self.model.forward(batch, epoch=epoch, iter=iteration, total_iter=self.total_iter, save_results=True, save_dir=self.train_result_dir, which_data=self.dataset, is_training=False)
                            torch.cuda.empty_cache()
                    if self.total_iter % self.log_freq_images == 0:
                        with torch.no_grad():
                            if self.rank == 0 and self.log_train_images:
                                m = self.model.forward(batch, epoch=epoch, iter=iteration, viz_logger=self.logger, total_iter=self.total_iter, which_data=self.dataset, logger_prefix='train_', is_training=False)
                            if self.fix_viz_batch:
                                print(f'fix_viz_batch:{self.fix_viz_batch}')
                                batch = self.viz_batch
                            else:
                                batch = next(self.viz_data_iterator)
                            if self.visualize_validation:
                                import time
                                vis_start = time.time()
                                batch = next(self.viz_data_iterator)
                                # try:
                                #     batch = next(self.viz_data_iterator)
                                # except:  # iterator exhausted
                                #     self.reset_viz_data_iterator()
                                #     batch = next(self.viz_data_iterator)
                                m = self.model.forward(batch, epoch=epoch, iter=iteration, viz_logger=self.logger, total_iter=self.total_iter, which_data=self.dataset, logger_prefix='val_', is_training=False)
                                vis_end = time.time()
                                print(f"vis time: {vis_end - vis_start}")
                            for name, loss in m.items():
                                if self.rank == 0:
                                    self.logger.add_scalar(f'loss_val/{name}', loss, self.total_iter)

                            if self.train_with_cub and epoch >= self.cub_start_epoch:
                                if self.rank == 0 and self.log_train_images:
                                    m = self.model.forward(batch_cub, epoch=epoch, iter=iteration, viz_logger=self.logger, total_iter=self.total_iter, which_data='cub', logger_prefix='cub_train_', is_training=True)

                                if self.fix_viz_batch:
                                    batch_cub = self.viz_batch_cub
                                elif self.visualize_validation:
                                    batch_cub = next(self.cub_viz_data_iterator)
                                    # try:
                                    #     batch = next(self.viz_data_iterator)
                                    # except:  # iterator exhausted
                                    #     self.reset_viz_data_iterator()
                                    #     batch = next(self.viz_data_iterator)
                                if self.rank == 0:
                                    m = self.model.forward(batch_cub, epoch=epoch, iter=iteration, viz_logger=self.logger, total_iter=self.total_iter, which_data='cub', logger_prefix='cub_val_', is_training=False)
                                    for name, loss in m.items():
                                        self.logger.add_scalar(f'cub_loss_val/{name}', loss, self.total_iter)
                        torch.cuda.empty_cache()
                if self.is_dry_run and iteration >= self.dry_run_iters:
                    break

                iteration += 1

        self.model.scheduler_step()
        return metrics
