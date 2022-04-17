import copy
import datetime
import os
import random
import time
import timeit
import warnings
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from torch.nn.modules.sparse import Embedding
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Crear generador:

self.gen = Generator(num_channels=num_channels,
                             resolution=resolution,
                             structure=self.structure,
                             conditional=self.conditional,
                             n_classes=self.n_classes,
                             **g_args).to(self.device)


# ---------------------------------------------------------------------------- #
# Options for Generator
# ---------------------------------------------------------------------------- #
cfg.model.gen = CN()
cfg.model.gen.latent_size = 512
# 8 in original paper
cfg.model.gen.mapping_layers = 4
cfg.model.gen.blur_filter = [1, 2, 1]
cfg.model.gen.truncation_psi = 0.7
cfg.model.gen.truncation_cutoff = 8


"""

""" CAPAS"""


# Conv layer with equalized learning rate
class Conv2dPropia(nn.Module):

    def __init__(self, input_channels, output_channels, gain=np.sqrt(2), kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.w_mul = gain * (input_channels * self.kernel_size ** 2) ** (-0.5)
        self.weight = torch.nn.Parameter(
            torch.randn(output_channels, input_channels, self.kernel_size, self.kernel_size))
        self.bias = torch.nn.Parameter(torch.zeros(output_channels))
        self.b_mul = 1

    def forward(self, x):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul
        x = F.conv2d(x, self.weight * self.w_mul, None, padding=self.kernel_size // 2)

        if bias is not None:
            x = x + bias.view(1, -1, 1, 1)
        return x


class Upscale2d(nn.Module):
    @staticmethod
    def upscale2d(x, factor=2, gain=1):
        assert x.dim() == 4
        if gain != 1:
            x = x * gain
        if factor != 1:
            shape = x.shape
            x = x.view(shape[0], shape[1], shape[2], 1, shape[3], 1).expand(-1, -1, -1, factor, -1, factor)
            x = x.contiguous().view(shape[0], shape[1], factor * shape[2], factor * shape[3])
        return x

    def __init__(self, factor=2, gain=1):
        super().__init__()
        assert isinstance(factor, int) and factor >= 1
        self.gain = gain
        self.factor = factor

    def forward(self, x):
        return self.upscale2d(x, factor=self.factor, gain=self.gain)



# Normalizamos el vector de caracteríctas pero en cada pixel
class PixelNormLayer(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


class EqualizedLinearPropia(nn.Module):
    def __init__(self, input_size, output_size, gain=2 ** 0.5, lrmul=1):
        super().__init__()
        he_std = gain * input_size ** (-0.5)  # He init
        # Equalized learning rate and custom learning rate multiplier.
        init_std = 1.0 / lrmul
        self.w_mul = he_std * lrmul
        self.weight = torch.nn.Parameter(torch.randn(output_size,
                                                     input_size) * init_std) ##Creamos un vector de pesos inicializados de manera random
        self.bias = torch.nn.Parameter(torch.zeros(output_size))  ##Lo mismo con el bias
        self.b_mul = lrmul

    def forward(self, x):
        bias = self.bias
        bias = bias * self.b_mul
        return F.linear(x, self.weight * self.w_mul,
                        bias)  ##Tranformacion lineal de los datos tomando en cuenta los pesos y el bias


##Agrega ruido por pixel (constante en cada canal) y en cada canal hay un peso distinto
class NoiseLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))
        self.noise = None

    def forward(self, x, noise=None):
        if noise is None and self.noise is None:  ##Creamos noise
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        elif noise is None:
            noise = self.noise
        x = x + self.weight.view(1, -1, 1, 1) * noise  ##Cambiamos la forma de los pesos y lo unimos con el ruido
        return x


##Bloque que le añade el ruido a la imagen
class StyleMod(nn.Module):
    def __init__(self, latent_size, channels, use_wscale):
        super(StyleMod, self).__init__()
        self.lin = EqualizedLinearPropia(latent_size,
                                         channels * 2,
                                         gain=1.0)

    def forward(self, x, latent):
        style = self.lin(latent)  # style => [batch_size, n_channels*2]

        shape = [-1, 2, x.size(1)] + (x.dim() - 2) * [1]
        style = style.view(shape)  # [batch_size, 2, n_channels, ...] <- Cambiamos la forma del estilo que obtuvimos
        x = x * (style[:, 0] + 1.) + style[:, 1]
        return x


##Agregarle estilo
class CapaS_StyleMode(nn.Module):
    def __init__(self, channels, dlatent_size):
        super().__init__()

        self.layers = nn.Sequential(NoiseLayer(channels),
                                    nn.ReLU(0.2),
                                    nn.InstanceNorm2d(channels))  ##ADAIN
        self.styleMode = StyleMod(dlatent_size, channels, use_wscale=True)

    def forward(self, x, dlatents_in_slice=None):
        x = self.layers(x)
        x = self.styleMode(x, dlatents_in_slice)
        return x


class BlurLayer(nn.Module):
    def __init__(self, kernel=None, normalize=True, flip=False, stride=1):
        super(BlurLayer, self).__init__()
        if kernel is None:
            kernel = [1, 2, 1]
        kernel = torch.tensor(kernel, dtype=torch.float32)
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel[None, None]
        if normalize:
            kernel = kernel / kernel.sum()
        if flip:
            kernel = kernel[:, :, ::-1, ::-1]
        self.register_buffer('kernel', kernel)
        self.stride = stride

    def forward(self, x):
        # expand kernel channels
        kernel = self.kernel.expand(x.size(1), -1, -1, -1)
        x = F.conv2d(
            x,
            kernel,
            stride=self.stride,
            padding=int((self.kernel.size(2) - 1) / 2),
            groups=x.size(1)
        )
        return x



class Conv2dUPPropia(nn.Module):
    """Conv layer + Upsample"""

    def __init__(self, input_channels, output_channels, gain=np.sqrt(2)):
        super().__init__()
        self.kernel_size = 3
        self.w_mul = gain * (input_channels * self.kernel_size ** 2) ** (-0.5)
        self.weight = torch.nn.Parameter(
            torch.randn(output_channels, input_channels, self.kernel_size, self.kernel_size))
        self.bias = torch.nn.Parameter(torch.zeros(output_channels))
        self.upsample = Upscale2d()
        self.b_mul = 1

    def forward(self, x):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul
        if min(x.shape[2:]) * 2 >= 128:
            w = self.weight * self.w_mul
            w = w.permute(1, 0, 2, 3)
            # probably applying a conv on w would be more efficient. also this quadruples the weight (average)?!
            w = F.pad(w, [1, 1, 1, 1])
            w = w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]
            x = F.conv_transpose2d(x, w, stride=2, padding=(w.size(-1) - 1) // 2)
        else:
            x = self.upsample(x)
            x = F.conv2d(x, self.weight * self.w_mul, None, padding=self.kernel_size // 2)

        if bias is not None:
            x = x + bias.view(1, -1, 1, 1)

        return x

"""BlOQUES"""


class GMapping(nn.Module):

    def __init__(self, latent_size=512, dlatent_size=512, dlatent_broadcast=None,
                 mapping_layers=8, mapping_fmaps=512, mapping_lrmul=0.01,
                 normalize_latents=True):

        super().__init__()

        self.latent_size = latent_size
        self.mapping_fmaps = mapping_fmaps
        self.dlatent_size = dlatent_size
        self.dlatent_broadcast = dlatent_broadcast

        layers = []

        if normalize_latents:  ##normalizamos el vector de características en cada píxel a la longitud de la unidad en el generador después de cada capa convolucional
            layers.append(PixelNormLayer())

        layers.append(EqualizedLinearPropia(self.latent_size, self.mapping_fmaps, gain=np.sqrt(2), lrmul=mapping_lrmul))
        layers.append(nn.LeakyReLU(negative_slope=0.2) )

        for layer in range(1, mapping_layers):  ##las otras 7 capas
            fmaps_in = self.mapping_fmaps
            fmaps_out = self.dlatent_size if layer == mapping_layers - 1 else self.mapping_fmaps
            layers.append(EqualizedLinearPropia(fmaps_in, fmaps_out, gain=np.sqrt(2), lrmul=mapping_lrmul))
            layers.append(nn.LeakyReLU(negative_slope=0.2))

        self.map = nn.Sequential(*layers)

    def forward(self, x):
        x = self.map(x)
        # Broadcast -> batch_size * dlatent_broadcast * dlatent_size VA PEOR SI LO QUITAS!!!
        if self.dlatent_broadcast is not None:
            x = x.unsqueeze(1).expand(-1, self.dlatent_broadcast,
                                      -1)  ##Cambiamos dimension con unsqueeze haciendo un tensor pasando 1 fila a de varias col a 1 col varias filas. Luego expandimos (singleton dimensions expanded to a larger size.)
        return x


class InputBlock(nn.Module):  ##Primer bloque 4x4
    def __init__(self, nf, dlatent_size):
        super().__init__()
        self.nf = nf

        self.const = nn.Parameter(torch.ones(1, nf, 4, 4))
        self.bias = nn.Parameter(torch.ones(nf) )

        self.epi1 = CapaS_StyleMode(nf, dlatent_size)

        self.conv = Conv2dPropia(nf, nf)
        self.epi2 = CapaS_StyleMode(nf, dlatent_size)

    def forward(self, dlatents_in_range):
        batch_size = dlatents_in_range.size(0)

        x = self.const.expand(batch_size, -1, -1, -1)
        x = x + self.bias.view(1, -1, 1, 1)
        x = self.epi1(x, dlatents_in_range[:, 0])
        x = self.conv(x)
        x = self.epi2(x, dlatents_in_range[:, 1])

        return x


class GSynthesisBlock(nn.Module):
    def __init__(self, in_channels, out_channels,  dlatent_size):
        super().__init__()
        self.kernel_size = 3


        ##BLUR LAYER ??? LO AGREGO??
        blur = BlurLayer([1, 2, 1])
        ##Upsample + Conv 3x3
        self.conv0_up = Conv2dUPPropia(in_channels, out_channels)

        self.capa1 = CapaS_StyleMode(out_channels, dlatent_size)

        ## Conv 3x3
        self.conv1 = Conv2dPropia(out_channels, out_channels) 

        self.capa2 = CapaS_StyleMode(out_channels, dlatent_size)

    def forward(self, x, dlatents_in_range):
        x = self.conv0_up(x)
        x = self.capa1(x, dlatents_in_range[:, 0])
        x = self.conv1(x)
        x = self.capa2(x, dlatents_in_range[:, 1])
        return x


class GSynthesis(nn.Module):

    def __init__(self, dlatent_size=512, num_channels=3, resolution=1024,
                 fmap_base=8192, fmap_decay=1.0, fmap_max=512,
                 use_styles=True, structure='linear'):

        super().__init__()

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.structure = structure

        ##Calculo de la resolucion, produndiad y capas
        resolution_log2 = int(np.log2(resolution))
        self.depth = resolution_log2 - 1
        self.num_layers = resolution_log2 * 2 - 2
        self.num_styles = self.num_layers if use_styles else 1  ##Numero de estilos posibles

        self.init_block = InputBlock(nf(1), dlatent_size) #4x4
        # create the ToRGB layers for various outputs (de articulo progressive growing)
        rgb_converters = [Conv2dPropia(nf(1), num_channels, gain=1, kernel_size=1) ]

        # Building blocks for remaining layers.
        layers = []
        for res in range(3, resolution_log2 + 1):
            last_channels = nf(res - 2)
            channels = nf(res - 1)
            layers.append(GSynthesisBlock(last_channels, channels, dlatent_size) )
            rgb_converters.append(Conv2dPropia(channels, num_channels, gain=1) )

        self.bloques = nn.ModuleList(layers)
        self.to_rgb = nn.ModuleList(
            rgb_converters)  ##ThetoRGBrepresents  a  layer  that  projects  feature  vectors  to  RGB  colors  andfromRGBdoesthe reverse

        # register the temporary upsampler
        self.temporaryUpsampler = lambda x: interpolate(x, scale_factor=2)

    def forward(self, dlatents_in, depth=0, alpha=0., labels_in=None):
        x = self.init_block(dlatents_in[:, 0:2])

        if depth > 0:
            for i, bloque in enumerate(self.bloques[:depth - 1]):
                x = bloque(x, dlatents_in[:, 2 * (i + 1):2 * (i + 2)])

            residual = self.to_rgb[depth - 1](self.temporaryUpsampler(x))
            straight = self.to_rgb[depth](self.bloques[depth - 1](x, dlatents_in[:, 2 * depth:2 * (depth + 1)]))

            images_out = (alpha * straight) + ((1 - alpha) * residual)
        else:
            images_out = self.to_rgb[0](x)

        return images_out

class Truncation(nn.Module):
    def __init__(self, avg_latent, max_layer=8, threshold=0.7, beta=0.995):
        super().__init__()
        self.max_layer = max_layer
        self.threshold = threshold
        self.beta = beta
        self.register_buffer('avg_latent', avg_latent)

    def update(self, last_avg):
        self.avg_latent.copy_(self.beta * self.avg_latent + (1. - self.beta) * last_avg)

    def forward(self, x):
        interp = torch.lerp(self.avg_latent, x, self.threshold)
        do_trunc = (torch.arange(x.size(1)) < self.max_layer).view(1, -1, 1).to(x.device)
        return torch.where(do_trunc, interp, x)

"""GENERADOR:"""


class Generator(nn.Module):

    def __init__(self, resolution, latent_size=512, dlatent_size=512,
                 conditional=False, n_classes=0, truncation_psi=0.7,
                 truncation_cutoff=8, dlatent_avg_beta=0.995,
                 style_mixing_prob=0.9, **kwargs):

        super(Generator, self).__init__()

        if conditional:  ##Agregarle parte conditional a la gan
            self.class_embedding = nn.Embedding(n_classes, latent_size)
            latent_size *= 2

        self.conditional = conditional
        self.style_mixing_prob = style_mixing_prob

        # Creamos los componentes del generador
        self.num_layers = (int(np.log2(resolution)) - 1) * 2
        self.g_mapping = GMapping(latent_size, dlatent_size, dlatent_broadcast=self.num_layers,
                                  **kwargs)  ##Bloque para el mapeo
        self.g_synthesis = GSynthesis(resolution=resolution, **kwargs)   ##  Bloque para la sintesis

        if truncation_psi > 0:
            self.truncation = Truncation(avg_latent=torch.zeros(dlatent_size),  ##Truncar los valores extremos
                                         max_layer=truncation_cutoff,
                                         threshold=truncation_psi,
                                         beta=dlatent_avg_beta)
        else:
            self.truncation = None

    def forward(self, latents_in, depth, alpha, labels_in=None):  # latents_in=noise
        if not self.conditional:
            if labels_in is not None:
                warnings.warn(
                    "Generator is unconditional, labels_in will be ignored")
        else:  ##Para gerador condicional
            embedding = self.class_embedding(labels_in)
            latents_in = torch.cat([latents_in, embedding], 1)

        dlatents_in = self.g_mapping(latents_in)

        if self.training:
            # Update moving average of W(dlatent).
            # TODO
            if self.truncation is not None:
                self.truncation.update(dlatents_in[0, 0].detach())

            # Realiza la regularización de la mezcla de estilos.
            if self.style_mixing_prob is not None and self.style_mixing_prob > 0:
                latents2 = torch.randn(latents_in.shape).to(latents_in.device)
                dlatents2 = self.g_mapping(latents2)
                layer_idx = torch.from_numpy(np.arange(self.num_layers)[np.newaxis, :, np.newaxis]).to(
                    latents_in.device)
                cur_layers = 2 * (depth + 1)
                mixing_cutoff = random.randint(1,
                                               cur_layers) if random.random() < self.style_mixing_prob else cur_layers
                dlatents_in = torch.where(layer_idx < mixing_cutoff, dlatents_in, dlatents2)

            # Apply truncation trick.
            if self.truncation is not None:
                dlatents_in = self.truncation(dlatents_in)

        fake_images = self.g_synthesis(dlatents_in, depth, alpha)

        return fake_images
