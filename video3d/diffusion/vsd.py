import os
os.environ['HUGGINGFACE_HUB_CACHE'] = '/viscam/u/zzli'
os.environ['HF_HOME'] = '/viscam/u/zzli'

from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler

from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.utils.import_utils import is_xformers_available

# Suppress partial model loading warning
logging.set_verbosity_error()

import gc
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn
from video3d.diffusion.sd import StableDiffusion
from torch.cuda.amp import custom_bwd, custom_fwd 


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    tcnn.free_temporary_memory()

class StableDiffusion_VSD(StableDiffusion):
    def __init__(self, device, sd_version='2.1', hf_key=None, torch_dtype=torch.float32, lora_n_timestamp_samples=1):
        super().__init__(device, sd_version=sd_version, hf_key=hf_key, torch_dtype=torch_dtype)

        # self.device = device
        # self.sd_version = sd_version
        # self.torch_dtype = torch_dtype
        
        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        # # Create model
        # self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae", torch_dtype=torch_dtype).to(self.device)
        # self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        # self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder").to(self.device)
        # self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet", torch_dtype=torch_dtype).to(self.device)
        
        # self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        # # self.scheduler = PNDMScheduler.from_pretrained(model_key, subfolder="scheduler")

        # self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        # self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        print(f'[INFO] loading stable diffusion VSD modules...')

        self.unet_lora = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet", torch_dtype=torch_dtype).to(self.device)
        cleanup()
        
        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        for p in self.unet_lora.parameters():
            p.requires_grad_(False)
        
        # set up LoRA layers
        lora_attn_procs = {}
        for name in self.unet_lora.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else self.unet_lora.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = self.unet_lora.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet_lora.config.block_out_channels))[
                    block_id
                ]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet_lora.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
            )

        self.unet_lora.set_attn_processor(lora_attn_procs)

        self.lora_layers = AttnProcsLayers(self.unet_lora.attn_processors).to(
            self.device
        )
        self.lora_layers._load_state_dict_pre_hooks.clear()
        self.lora_layers._state_dict_hooks.clear()
        self.lora_n_timestamp_samples = lora_n_timestamp_samples
        self.scheduler_lora = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")

        print(f'[INFO] loaded stable diffusion VSD modules!')

    def train_lora(
        self,
        latents,
        text_embeddings,
        camera_condition
    ):
        B = latents.shape[0]
        lora_n_timestamp_samples = self.lora_n_timestamp_samples
        latents = latents.detach().repeat(lora_n_timestamp_samples, 1, 1, 1)

        t = torch.randint(
            int(self.num_train_timesteps * 0.0),
            int(self.num_train_timesteps * 1.0),
            [B * lora_n_timestamp_samples],
            dtype=torch.long,
            device=self.device,
        )

        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler_lora.add_noise(latents, noise, t)
        if self.scheduler_lora.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler_lora.config.prediction_type == "v_prediction":
            target = self.scheduler_lora.get_velocity(latents, noise, t)
        else:
            raise ValueError(
                f"Unknown prediction type {self.scheduler_lora.config.prediction_type}"
            )
        
        # use view-independent text embeddings in LoRA
        _, text_embeddings_cond = text_embeddings.chunk(2)

        if random.random() < 0.1:
            camera_condition = torch.zeros_like(camera_condition)
        
        noise_pred = self.unet_lora(
            noisy_latents,
            t,
            encoder_hidden_states=text_embeddings_cond.repeat(
                lora_n_timestamp_samples, 1, 1
            ),
            class_labels=camera_condition.reshape(B, -1).repeat(
                lora_n_timestamp_samples, 1
            ),
            cross_attention_kwargs={"scale": 1.0}
        ).sample

        loss_lora = 0.5 * F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
        return loss_lora

    
    def train_step(
        self,
        text_embeddings,
        text_embeddings_vd, 
        pred_rgb,
        camera_condition,
        im_features,
        guidance_scale=7.5,
        guidance_scale_lora=7.5,
        loss_weight=1.0,
        min_step_pct=0.02,
        max_step_pct=0.98, 
        return_aux=False
    ):
        pred_rgb = pred_rgb.to(self.torch_dtype)
        text_embeddings = text_embeddings.to(self.torch_dtype)
        text_embeddings_vd = text_embeddings_vd.to(self.torch_dtype)
        camera_condition = camera_condition.to(self.torch_dtype)
        im_features = im_features.to(self.torch_dtype)

        # condition_label = camera_condition
        condition_label = im_features
        
        b = pred_rgb.shape[0]
        
        # interp to 512x512 to be fed into vae.
        # _t = time.time()
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: interp {time.time() - _t:.4f}s')

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        min_step = int(self.num_train_timesteps * min_step_pct)
        max_step = int(self.num_train_timesteps * max_step_pct)
        t = torch.randint(min_step, max_step + 1, [b], dtype=torch.long, device=self.device)

        # encode image into latents with vae, requires grad!
        # _t = time.time()
        latents = self.encode_imgs(pred_rgb_512)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: vae enc {time.time() - _t:.4f}s')

        # predict the noise residual with unet, NO grad!
        # _t = time.time()
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            
            # disable unet class embedding here
            cls_embedding = self.unet.class_embedding
            self.unet.class_embedding = None

            cross_attention_kwargs = None
            noise_pred_pretrain = self.unet(
                latent_model_input, 
                torch.cat([t, t]), 
                encoder_hidden_states=text_embeddings_vd, 
                class_labels=None, 
                cross_attention_kwargs=cross_attention_kwargs
            ).sample

            self.unet.class_embedding = cls_embedding
            
            # use view-independent text embeddings in LoRA
            _, text_embeddings_cond = text_embeddings.chunk(2)
            
            noise_pred_est = self.unet_lora(
                latent_model_input,
                torch.cat([t, t]),
                encoder_hidden_states=torch.cat([text_embeddings_cond] * 2),
                class_labels=torch.cat(
                    [
                        condition_label.reshape(b, -1),
                        torch.zeros_like(condition_label.reshape(b, -1)),
                    ],
                    dim=0,
                ),
                cross_attention_kwargs={"scale": 1.0},
            ).sample

        noise_pred_pretrain_uncond, noise_pred_pretrain_text = noise_pred_pretrain.chunk(2)
        
        noise_pred_pretrain = noise_pred_pretrain_uncond + guidance_scale * (
            noise_pred_pretrain_text - noise_pred_pretrain_uncond
        )

        assert self.scheduler.config.prediction_type == "epsilon"
        if self.scheduler_lora.config.prediction_type == "v_prediction":
            alphas_cumprod = self.scheduler_lora.alphas_cumprod.to(
                device=latents_noisy.device, dtype=latents_noisy.dtype
            )
            alpha_t = alphas_cumprod[t] ** 0.5
            sigma_t = (1 - alphas_cumprod[t]) ** 0.5

            noise_pred_est = latent_model_input * torch.cat([sigma_t] * 2, dim=0).reshape(
                -1, 1, 1, 1
            ) + noise_pred_est * torch.cat([alpha_t] * 2, dim=0).reshape(-1, 1, 1, 1)
        
        noise_pred_est_uncond, noise_pred_est_camera = noise_pred_est.chunk(2)

        noise_pred_est = noise_pred_est_uncond + guidance_scale_lora * (
            noise_pred_est_camera - noise_pred_est_uncond
        )

        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        grad = loss_weight * w[:, None, None, None] * (noise_pred_pretrain - noise_pred_est)

        grad = torch.nan_to_num(grad)

        targets = (latents - grad).detach()
        loss_vsd = 0.5 * F.mse_loss(latents.float(), targets, reduction='sum') / latents.shape[0]

        loss_lora = self.train_lora(latents, text_embeddings, condition_label)
        
        loss = {
            'loss_vsd': loss_vsd,
            'loss_lora': loss_lora
        }

        if return_aux:
            aux = {'grad': grad, 't': t, 'w': w}
            return loss, aux
        else:
            return loss 
    


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    sd = StableDiffusion_VSD(device, opt.sd_version, opt.hf_key)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()
    plt.savefig(f'{opt.prompt}.png')