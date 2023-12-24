import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from video3d.triplane_texture.ops import bias_act
from video3d.triplane_texture.ops import fma
from video3d.triplane_texture.ops import upfirdn2d
from video3d.triplane_texture.ops import conv2d_resample
from video3d.triplane_texture.ops import grid_sample_gradfix
from video3d.triplane_texture import misc


def modulated_conv2d(
        x,  # Input tensor of shape [batch_size, in_channels, in_height, in_width].
        weight,  # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
        styles,  # Modulation coefficients of shape [batch_size, in_channels].
        noise=None,  # Optional noise tensor to add to the output activations.
        up=1,  # Integer upsampling factor.
        down=1,  # Integer downsampling factor.
        padding=0,  # Padding with respect to the upsampled image.
        resample_filter=None,
        # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
        demodulate=True,  # Apply weight demodulation?
        flip_weight=True,  # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
        fused_modconv=True,  # Perform modulation, convolution, and demodulation as a single fused operation?
):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape
    misc.assert_shape(weight, [out_channels, in_channels, kh, kw])  # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None])  # [NIHW]
    misc.assert_shape(styles, [batch_size, in_channels])  # [NI]

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(
            float('inf'), dim=[1, 2, 3], keepdim=True))  # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True)  # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0)  # [NOIkk]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1)  # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()  # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1)  # [NOIkk]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(
            x=x, w=weight.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, flip_weight=flip_weight)
        if demodulate and noise is not None:
            x = fma.fma(x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype))
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        return x

    # Execute as one fused op using grouped convolution.
    with misc.suppress_tracer_warnings():  # this value will be treated as a constant
        batch_size = int(batch_size)
    misc.assert_shape(x, [batch_size, in_channels, None, None])
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_resample.conv2d_resample(
        x=x, w=w.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, groups=batch_size,
        flip_weight=flip_weight)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    if noise is not None:
        x = x.add_(noise)
    return x


def modulated_fc(
        x,  # Input tensor of shape [batch_size, n_feature, in_channels].
        weight,  # Weight tensor of shape [out_channels, in_channels].
        styles,  # Modulation coefficients of shape [batch_size, in_channels].
        noise=None,  # Optional noise tensor to add to the output activations.
        demodulate=True,  # Apply weight demodulation?
):
    batch_size = x.shape[0]
    n_feature = x.shape[1]
    out_channels, in_channels = weight.shape
    misc.assert_shape(weight, [out_channels, in_channels])
    misc.assert_shape(x, [batch_size, n_feature, in_channels])
    misc.assert_shape(styles, [batch_size, in_channels])
    assert demodulate
    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels) / weight.norm(float('inf'), dim=[1, 2, 3], keepdim=True))  # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True)  # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = weight.unsqueeze(0)  # [NOI]
    w = w * styles.unsqueeze(dim=1)  # [NOI]
    dcoefs = (w.square().sum(dim=[2]) + 1e-8).rsqrt()  # [NO]
    w = w * dcoefs.unsqueeze(dim=-1)  # [NOI]
    x = torch.bmm(x, w.permute(0, 2, 1))
    if noise is not None:
        x = x.add_(noise)
    return x


class FullyConnectedLayer(torch.nn.Module):
    def __init__(
            self,
            in_features,  # Number of input features.
            out_features,  # Number of output features.
            bias=True,  # Apply additive bias before the activation function?
            activation='linear',  # Activation function: 'relu', 'lrelu', etc.
            device='cuda',
            lr_multiplier=1,  # Learning rate multiplier.
            bias_init=0,  # Initial value for the additive bias.
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features], device=device) / lr_multiplier)
        self.bias = torch.nn.Parameter(
            torch.full([out_features], np.float32(bias_init), device=device)) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self):
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'


class SynthesisLayer(torch.nn.Module):
    def __init__(
            self,
            in_channels,  # Number of input channels.
            out_channels,  # Number of output channels.
            w_dim,  # Intermediate latent (W) dimensionality.
            resolution,  # Resolution of this layer.
            kernel_size=3,  # Convolution kernel size.
            up=1,  # Integer upsampling factor.
            use_noise=True,  # Enable noise input?
            activation='lrelu',  # Activation function: 'relu', 'lrelu', etc.
            device='cuda',
            resample_filter=[1, 3, 3, 1],  # Low-pass filter to apply when resampling activations.
            conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
            channels_last=False,  # Use channels_last format for the weights?
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1, device=device)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(
            torch.randn([out_channels, in_channels, kernel_size, kernel_size], device=device).to(
                memory_format=memory_format))
        if use_noise:
            self.register_buffer('noise_const', torch.randn([resolution, resolution], device=device))
            self.noise_strength = torch.nn.Parameter(torch.zeros([], device=device))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels], device=device))

    def forward(self, x, w, noise_mode='random', fused_modconv=True, gain=1):
        assert noise_mode in ['random', 'const', 'none']
        in_resolution = self.resolution // self.up
        misc.assert_shape(x, [None, self.in_channels, in_resolution, in_resolution])

        styles = self.affine(w)

        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn(
                [x.shape[0], 1, self.resolution, self.resolution], device=x.device) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        flip_weight = (self.up == 1)  # slightly faster
        x = modulated_conv2d(
            x=x, weight=self.weight, styles=styles, noise=noise, up=self.up,
            padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight,
            fused_modconv=fused_modconv)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)
        return x

    def extra_repr(self):
        return ' '.join(
            [
                f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, w_dim={self.w_dim:d},',
                f'resolution={self.resolution:d}, up={self.up}, activation={self.activation:s}'])


class ImplicitSynthesisLayer(torch.nn.Module):
    def __init__(
            self,
            in_channels,  # Number of input channels.
            out_channels,  # Number of output channels.
            w_dim,  # Intermediate latent (W) dimensionality.
            use_noise=True,  # Enable noise input?
            activation='lrelu',  # Activation function: 'relu', 'lrelu', etc.
            resample_filter=[1, 3, 3, 1],  # Low-pass filter to apply when resampling activations.
            device='cuda',
            conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1, device=device)
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels], device=device))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels], device=device))

    def forward(self, w, x, noise_mode='random', gain=1):
        # x is the feature#############
        # w is the condition
        assert noise_mode in ['random', 'const', 'none']
        styles = self.affine(w)
        noise = None  # in te beegining, we didn't use the noise
        x = modulated_fc(x=x, weight=self.weight, styles=styles, noise=noise)
        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(
            x, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp,
            dim=2)  # the last dim is the feature dim
        return x

    def extra_repr(self):
        return ' '.join(
            [
                f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, w_dim={self.w_dim:d},',
                f'activation={self.activation:s}'])



class Conv2dLayer(torch.nn.Module):
    def __init__(
            self,
            in_channels,  # Number of input channels.
            out_channels,  # Number of output channels.
            kernel_size,  # Width and height of the convolution kernel.
            device='cuda',
            bias=True,  # Apply additive bias before the activation function?
            activation='linear',  # Activation function: 'relu', 'lrelu', etc.
            up=1,  # Integer upsampling factor.
            down=1,  # Integer downsampling factor.
            resample_filter=[1, 3, 3, 1],  # Low-pass filter to apply when resampling activations.
            conv_clamp=None,  # Clamp the output to +-X, None = disable clamping.
            channels_last=False,  # Expect the input to have memory_format=channels_last?
            trainable=True,  # Update the weights of this layer during training?
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size], device=device).to(
            memory_format=memory_format)
        bias = torch.zeros([out_channels], device=device) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer('weight', weight)
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None

    def forward(self, x, gain=1):
        w = self.weight * self.weight_gain
        b = self.bias.to(x.dtype) if self.bias is not None else None
        flip_weight = (self.up == 1)  # slightly faster
        x = conv2d_resample.conv2d_resample(
            x=x, w=w.to(x.dtype), f=self.resample_filter, up=self.up, down=self.down, padding=self.padding,
            flip_weight=flip_weight)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x

    def extra_repr(self):
        return ' '.join(
            [
                f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, activation={self.activation:s},',
                f'up={self.up}, down={self.down}'])


class ToRGBLayer(torch.nn.Module):
    def __init__(
            self, in_channels, out_channels, w_dim, kernel_size=1, conv_clamp=None, channels_last=False, device='cuda'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.conv_clamp = conv_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1, device=device)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(
            torch.randn([out_channels, in_channels, kernel_size, kernel_size], device=device).to(
                memory_format=memory_format))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels], device=device))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

    def forward(self, x, w, fused_modconv=True):
        styles = self.affine(w) * self.weight_gain
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)
        return x

    def extra_repr(self):
        return f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, w_dim={self.w_dim:d}'



class SynthesisBlock(torch.nn.Module):
    def __init__(
            self,
            in_channels,  # Number of input channels, 0 = first block.
            out_channels,  # Number of output channels.
            w_dim,  # Intermediate latent (W) dimensionality.
            resolution,  # Resolution of this block.
            img_channels,  # Number of output color channels.
            is_last,  # Is this the last block?
            architecture='skip',  # Architecture: 'orig', 'skip', 'resnet'.
            resample_filter=[1, 3, 3, 1],  # Low-pass filter to apply when resampling activations.
            conv_clamp=256,  # Clamp the output of convolution layers to +-X, None = disable clamping.
            use_fp16=False,  # Use FP16 for this block?
            fp16_channels_last=False,  # Use channels-last memory format with FP16?
            fused_modconv_default=True,  # Default value of fused_modconv. 'inference_only' = True for inference, False for training.
            device='cuda',
            first_layer=False,
            **layer_kwargs,  # Arguments for SynthesisLayer.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.first_layer = first_layer
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.fused_modconv_default = fused_modconv_default
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0
        if in_channels == 0:
            self.const = torch.nn.Parameter(torch.randn([out_channels, resolution, resolution], device=device))

        if in_channels != 0:
            if self.first_layer:
                self.conv0 = SynthesisLayer(
                    in_channels, out_channels, w_dim=w_dim, resolution=resolution,
                    conv_clamp=conv_clamp, channels_last=self.channels_last, device=device, **layer_kwargs)
            else:
                self.conv0 = SynthesisLayer(
                    in_channels, out_channels, w_dim=w_dim, resolution=resolution, up=2,
                    resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last, device=device,
                    **layer_kwargs)
            self.num_conv += 1

        self.conv1 = SynthesisLayer(
            out_channels, out_channels, w_dim=w_dim, resolution=resolution,
            conv_clamp=conv_clamp, channels_last=self.channels_last, device=device, **layer_kwargs)
        self.num_conv += 1

        if is_last or architecture == 'skip':
            self.torgb = ToRGBLayer(
                out_channels, img_channels, w_dim=w_dim,
                conv_clamp=conv_clamp, channels_last=self.channels_last, device=device)
            self.num_torgb += 1

        if in_channels != 0 and architecture == 'resnet':
            self.skip = Conv2dLayer(
                in_channels, out_channels, kernel_size=1, bias=False, up=2,
                resample_filter=resample_filter, channels_last=self.channels_last, device=device)

    def forward(self, x, img, ws, force_fp32=False, fused_modconv='inference_only', update_emas=False, **layer_kwargs):
        _ = update_emas  # unused
        misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        if ws.device.type != 'cuda':
            force_fp32 = True
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32

        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            fused_modconv = self.fused_modconv_default
        if fused_modconv == 'inference_only':
            fused_modconv = (not self.training)
        ##
        # Input.
        if self.in_channels == 0:
            x = self.const.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
        else:
            # misc.assert_shape(x, [None, self.in_channels, self.resolution // 2, self.resolution // 2])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            x = y + x
        else:
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)

        if img is not None:
            misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution // 2])
            img = upfirdn2d.upsample2d(img, self.resample_filter)
        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = img + y if img is not None else y

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'



class SynthesisNetwork(torch.nn.Module):
    def __init__(
            self,
            w_dim,  # Intermediate latent (W) dimensionality.
            img_resolution,  # Output image resolution.
            img_channels,  # Number of color channels.
            channel_base=32768,  # Overall multiplier for the number of channels.
            channel_max=512,  # Maximum number of channels in any layer.
            num_fp16_res=4,  # Use FP16 for the N highest resolutions.
            device='cuda',
            **block_kwargs,  # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.num_fp16_res = num_fp16_res
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]   # [4,8,16,32,64,128]

        # {4: 512, 8: 512, 16: 512, 32: 512, 64: 512, 128: 256}
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            is_last = (res == self.img_resolution)
            use_fp16 = False
            block = SynthesisBlock(
                in_channels, out_channels, w_dim=w_dim, resolution=res,
                img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, device=device, **block_kwargs)
            self.num_ws += block.num_conv
            self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def forward(self, ws, **block_kwargs):
        block_ws = []
        misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
        ws = ws.to(torch.float32)
        w_idx = 0
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
            w_idx += (block.num_conv + block.num_torgb)
        x = img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, **block_kwargs)
        return img

    def extra_repr(self):
        return ' '.join(
            [
                f'w_dim={self.w_dim:d}, num_ws={self.num_ws:d},',
                f'img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d},',
                f'num_fp16_res={self.num_fp16_res:d}'])


class ImplicitSynthesisNetwork(torch.nn.Module):
    def __init__(
            self,
            w_dim=512,  # Intermediate latent (W) dimensionality.
            input_channel=256,
            out_channels=3,  # Number of color channels.
            latent_channel=256,
            n_layers=4,
            device='cuda'
    ):
        super().__init__()
        self.n_layer = n_layers
        self.layers = []
        self.num_ws = 0
        for i_layer in range(self.n_layer):
            layer = ImplicitSynthesisLayer(
                w_dim=w_dim,
                in_channels=input_channel if i_layer == 0 else latent_channel,
                out_channels=latent_channel, device=device)
            self.layers.append(layer)
            self.num_ws += 1

        self.layers.append(
            ImplicitSynthesisLayer(
                w_dim=w_dim, in_channels=latent_channel, out_channels=out_channels,
                activation='sigmoid', device=device)
        )
        self.num_ws += 1
        self.layers = torch.nn.ModuleList(self.layers)
        self.w_dim = w_dim
        self.out_channels = out_channels

    def forward(self, ws, position, **block_kwargs):
        out = position
        for i in range(self.n_layer):
            out = self.layers[i](ws[:, i], out)
        out = self.layers[-1](ws[:, self.n_layer], out)
        return out

    def extra_repr(self):
        return ' '.join(
            [
                f'w_dim={self.w_dim:d}'])



class TriPlaneTex(torch.nn.Module):
    def __init__(
            self,
            w_dim,  # Intermediate latent (W) dimensionality.
            img_channels,  # Number of color channels.
            tri_plane_resolution=256,
            device='cuda',
            mlp_latent_channel=256,
            n_implicit_layer=3,
            feat_dim=384, # number of feat dim from encoder
            n_mapping_layer=8,
            sym_texture=True,
            grid_scale=7.,
            min_max=None,
            perturb_normal=False,
            **block_kwargs,  # Arguments for SynthesisBlock.
    ):
        super().__init__()
        self.n_implicit_layer = n_implicit_layer
        self.img_feat_dim = 32  # The setting follows Koki's paper
        self.w_dim = w_dim
        self.tri_plane_resolution = tri_plane_resolution

        # the mapping network
        self.feat_dim = feat_dim
        self.n_mapping_layer = n_mapping_layer
        self.embed = FullyConnectedLayer(feat_dim, w_dim, device=device)
        for idx in range(n_mapping_layer):
            layer = FullyConnectedLayer(w_dim, w_dim, activation='lrelu', lr_multiplier=0.1, device=device)
            setattr(self, f'mapping{idx}', layer)

        # self.w_dim = w_dim * 2

        self.tri_plane_synthesis = SynthesisNetwork(
            w_dim=self.w_dim, img_resolution=self.tri_plane_resolution,
            img_channels=self.img_feat_dim * 3,
            device=device,
            **block_kwargs)
        self.num_ws_tri_plane = self.tri_plane_synthesis.num_ws

        mlp_input_channel = self.img_feat_dim + w_dim  #
        mlp_latent_channel = mlp_latent_channel

        mlp_input_channel -= w_dim
        self.mlp_synthesis = ImplicitSynthesisNetwork(
            out_channels=img_channels,
            n_layers=self.n_implicit_layer,
            w_dim=self.w_dim,
            latent_channel=mlp_latent_channel,
            input_channel=mlp_input_channel,
            device=device)
        self.num_ws_all = self.num_ws_tri_plane + self.mlp_synthesis.num_ws

        # texture related
        self.sym_texture = sym_texture
        self.grid_scale = grid_scale
        self.shape_min = 0.
        self.shape_lenght = grid_scale / 2.

        if min_max is not None:
            self.register_buffer('min_max', min_max)
        else:
            self.min_max = None
        
        self.perturb_normal = perturb_normal

    def old_forward(
            self, feat, position=None, **block_kwargs):
        '''
        Predict texture with given latent code
        :param feat: image global feat
        :param position: position for the surface points
        :param block_kwargs:
        :return:
        '''
        assert feat.shape[-1] == self.feat_dim
        
        # mapping global feature to ws
        ws = self.embed(feat)
        for idx in range(self.n_mapping_layer):
            layer = getattr(self, f'mapping{idx}')
            ws = layer(ws)
        
        ws = ws.unsqueeze(1).repeat(1, self.num_ws_all, 1)

        plane_feat = self.tri_plane_synthesis(ws[:, :self.num_ws_tri_plane], **block_kwargs)
        tri_plane = torch.split(plane_feat, self.img_feat_dim, dim=1)

        normalized_tex_pos = (position - self.shape_min) / self.shape_lenght   # in [-1, 1]
        normalized_tex_pos = torch.clamp(normalized_tex_pos, -1.0, 1.0)

        if self.sym_texture:
            x_pos, y_pos, z_pos = normalized_tex_pos.unbind(-1)
            normalized_tex_pos = torch.stack([x_pos.abs(), y_pos, z_pos], dim=-1)


        x_feat = grid_sample_gradfix.grid_sample(
            tri_plane[0],
            torch.cat(
                [normalized_tex_pos[:, :, 0:1], normalized_tex_pos[:, :, 1:2]],
                dim=-1).unsqueeze(dim=1).detach())
        y_feat = grid_sample_gradfix.grid_sample(
            tri_plane[1],
            torch.cat(
                [normalized_tex_pos[:, :, 1:2], normalized_tex_pos[:, :, 2:3]],
                dim=-1).unsqueeze(dim=1).detach())
        z_feat = grid_sample_gradfix.grid_sample(
            tri_plane[2],
            torch.cat(
                [normalized_tex_pos[:, :, 0:1], normalized_tex_pos[:, :, 2:3]],
                dim=-1).unsqueeze(dim=1).detach())

        final_feat = (x_feat + y_feat + z_feat)
        final_feat = final_feat.squeeze(dim=2).permute(0, 2, 1)  # 32dimension
        final_feat_tex = final_feat
        out = self.mlp_synthesis(ws[:, self.num_ws_tri_plane:], final_feat_tex)
        return out

    def sample(self, xyz, feat=None, feat_map=None, mvp=None, w2c=None, deform_xyz=None):
        # query the deformed points or canonical points
        # x = deform_xyz
        x = xyz

        b, h, w, c = x.shape
        mvp = mvp.detach()  # [b, 4, 4]
        w2c = w2c.detach()  # [b, 4, 4]
        x = x.reshape(b, -1, c)

        global_feat = feat # [b, d]
        
        out = self.old_forward(
            feat=global_feat,
            position=x
        )
        if self.min_max is not None:
            out = out * (self.min_max[1][None, :] - self.min_max[0][None, :]) + self.min_max[0][None, :]
        return out.view(b, h, w, -1)