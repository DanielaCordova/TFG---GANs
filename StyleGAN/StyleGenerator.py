import sys, os



curentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(curentdir)
sys.path.append(parentdir)

import torch
import torch.nn as nn
import copy
import datetime
import os
import random
import time
import timeit
import warnings
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from torch.nn.modules.sparse import Embedding
from torchvision import models
from torchsummary import summary
from StyleGAN.Components.Layers import *
from StyleGAN.Components.Blocks import *

class GMapping(nn.Module):

    def __init__(self, latent_size=512, dlatent_size=512, dlatent_broadcast=None,
                 mapping_layers=8, mapping_fmaps=512, mapping_lrmul=0.01, mapping_nonlinearity='lrelu',
                 use_wscale=True, normalize_latents=True, **kwargs):

        super().__init__()

        self.latent_size = latent_size
        self.mapping_fmaps = mapping_fmaps
        self.dlatent_size = dlatent_size
        self.dlatent_broadcast = dlatent_broadcast

        layers = []

        if normalize_latents: ##normalizamos el vector de características en cada píxel a la longitud de la unidad en el generador después de cada capa convolucional
            layers.append(PixelNormLayer())


        layers.append(EqualizedLinearPropia(self.latent_size, self.mapping_fmaps, numMul=np.sqrt(2), lrmul=mapping_lrmul))
        layers.append(nn.LeakyReLU(negative_slope=0.2))

        for layer in range(1, mapping_layers): ##las otras 8 capas
            fmaps_in = self.mapping_fmaps
            fmaps_out = self.dlatent_size if layer == mapping_layers - 1 else self.mapping_fmaps
            layers.append(EqualizedLinearPropia(fmaps_in, fmaps_out, numMul=np.sqrt(2), lrmul=mapping_lrmul))
            layers.append(nn.LeakyReLU(negative_slope=0.2))

        self.map = nn.Sequential(*layers)

    def forward(self, x):
        x = self.map(x)

        # Broadcast -> batch_size * dlatent_broadcast * dlatent_size VA PEOR SI LO QUITAS!!!
        if self.dlatent_broadcast is not None:
            x = x.unsqueeze(1).expand(-1, self.dlatent_broadcast, -1)
        return x


class GSynthesis(nn.Module):

    def __init__(self, dlatent_size=512, num_channels=3, resolution=1024,
                 fmap_base=8192, fmap_decay=1.0, fmap_max=512,
                 use_styles=True, const_input_layer=True, use_noise=True, nonlinearity='lrelu',
                 use_wscale=True, use_pixel_norm=False, use_instance_norm=True, blur_filter=None,
                 structure='linear', **kwargs):

        super().__init__()

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.structure = structure

        ##Calculo de la resolucion, produndiad y capas
        resolution_log2 = int(np.log2(resolution))
        self.depth = resolution_log2 - 1
        self.num_layers = resolution_log2 * 2 - 2
        self.num_styles = self.num_layers if use_styles else 1  ##Numero de estilos posibles

        self.init_block = InputBlock(nf(1), dlatent_size)
        # create the ToRGB layers for various outputs
        rgb_converters = [Conv2dPropia(nf(1), num_channels, gain=1, kernel_size=1)]

        # Building blocks for remaining layers.
        layers = []
        for res in range(3, resolution_log2 + 1):
            last_channels = nf(res - 2)
            channels = nf(res - 1)
            layers.append(GSynthesisBlock(last_channels, channels, dlatent_size))
            rgb_converters.append(Conv2dPropia(channels, num_channels, gain=1))

        self.bloques = nn.ModuleList(layers)
        self.to_rgb = nn.ModuleList(rgb_converters)  ##ThetoRGBrepresents  a  layer  that  projects  feature  vectors  to  RGB  colors  andfromRGBdoesthe reverse

        # register the temporary upsampler
        self.temporaryUpsampler = lambda x: interpolate(x, scale_factor=2)

    def forward(self, dlatents_in, depth=0, alpha=0., labels_in=None):
        x = self.init_block(dlatents_in[:, 0:2])
        self.alpha=alpha
        if depth > 0:
            for i, bloque in enumerate(self.bloques[:depth - 1]):
                x = bloque(x, dlatents_in[:, 2 * (i + 1):2 * (i + 2)])

            residual = self.to_rgb[depth - 1](self.temporaryUpsampler(x))
            straight = self.to_rgb[depth](self.bloques[depth - 1](x, dlatents_in[:, 2 * depth:2 * (depth + 1)]))

            images_out = (self.alpha * straight) + ((1 - self.alpha) * residual)
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


class Generator(nn.Module):

    def __init__(self, resolution, latent_size=512, dlatent_size=512,
                 conditional=False, n_classes=0, truncation_psi=0.7,
                 truncation_cutoff=8, dlatent_avg_beta=0.995,
                 style_mixing_prob=0.9, **kwargs):

        super(Generator, self).__init__()

        if conditional:
            self.class_embedding = nn.Embedding(n_classes, latent_size)
            latent_size *= 2

        self.conditional = conditional
        self.style_mixing_prob = style_mixing_prob

        # Creamos los componentes del generador
        self.num_layers = (int(np.log2(resolution)) - 1) * 2
        self.g_mapping = GMapping(latent_size, dlatent_size, dlatent_broadcast=self.num_layers, **kwargs)
        self.g_synthesis = GSynthesis(resolution=resolution, **kwargs)

        if truncation_psi > 0:
            self.truncation = Truncation(avg_latent=torch.zeros(dlatent_size),
                                         max_layer=truncation_cutoff,
                                         threshold=truncation_psi,
                                         beta=dlatent_avg_beta)
        else:
            self.truncation = None

    def forward(self, latents_in,  depth=4, alpha=0.00047281323877068556, labels_in=None): #latents_in=noise
        if not self.conditional:
            if labels_in is not None:
                warnings.warn(
                    "Generator is unconditional, labels_in will be ignored")
        else: ##Para gerador condicional
            embedding = self.class_embedding(labels_in)
            latents_in = torch.cat([latents_in, embedding], 1)
        self.alpha=alpha
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

        fake_images = self.g_synthesis(dlatents_in, depth, self.alpha)

        return fake_images