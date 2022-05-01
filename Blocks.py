"""
-------------------------------------------------
   File Name:    Blocks.py
   Date:         2019/10/17
   Description:  Copy from: https://github.com/lernapparat/lernapparat
-------------------------------------------------
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from CustomLayers import EqualizedLinear, LayerEpilogue, EqualizedConv2d, BlurLayer, View, StddevLayer, \
    NoiseLayer, StyleMod, Downscale2d


class InputBlock(nn.Module): ##Primer bloque 4x4
    def __init__(self, nf, dlatent_size):
        super().__init__()
        self.nf = nf

        self.const = nn.Parameter(torch.ones(1, nf, 4, 4))
        self.bias = nn.Parameter(torch.ones(nf))

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


class Conv2dUPPropia(nn.Module):
    """Conv layer with equalized learning rate and custom learning rate multiplier."""

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

class Conv2DownPropia(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, gain=np.sqrt(2)):
        super().__init__()
        self.kernel_size = kernel_size
        self.w_mul = gain * (input_channels * self.kernel_size ** 2) ** (-0.5)
        self.weight = torch.nn.Parameter(
            torch.randn(output_channels, input_channels, self.kernel_size, self.kernel_size))
        self.bias = torch.nn.Parameter(torch.zeros(output_channels))
        self.downscale = Downscale2d()
        self.b_mul = 1

    def forward(self, x):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul
        have_convolution=False
        downscale = self.downscale
        intermediate = None
        if downscale is not None and min(x.shape[2:]) >= 128:
            w = self.weight * self.w_mul
            w = F.pad(w, [1, 1, 1, 1])
            # in contrast to upscale, this is a mean...
            w = (w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]) * 0.25  # avg_pool?
            x = F.conv2d(x, w, stride=2, padding=(w.size(-1) - 1) // 2)
            have_convolution = True
            downscale = None
        elif downscale is not None:
            intermediate = downscale

        if not have_convolution and intermediate is None:
            return F.conv2d(x, self.weight * self.w_mul, bias, padding=self.kernel_size // 2)
        elif not have_convolution:
            x = F.conv2d(x, self.weight * self.w_mul, None, padding=self.kernel_size // 2)

        if intermediate is not None:
            x = intermediate(x)

        if bias is not None:
            x = x + bias.view(1, -1, 1, 1)

        return x

class Conv2dPropia(nn.Module):
    """Conv layer with equalized learning rate and custom learning rate multiplier."""

    def __init__(self, input_channels, output_channels, gain=np.sqrt(2), kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.w_mul =  gain * (input_channels * self.kernel_size ** 2) ** (-0.5)
        self.weight = torch.nn.Parameter(
            torch.randn(output_channels, input_channels, self.kernel_size, self.kernel_size))
        self.bias = torch.nn.Parameter(torch.zeros(output_channels))
        self.b_mul = 1

    def forward(self, x):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul
        a, b, c, d = x.shape
        if(b%2!=0):
            m = torch.nn.Conv2d(b, b-1, 1).to("cuda")
            x = m(x)
        x = F.conv2d(x, self.weight * self.w_mul, None, padding=self.kernel_size // 2)

        if bias is not None:
            x = x + bias.view(1, -1, 1, 1)
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


class DiscriminatorTop(nn.Sequential):
    def __init__(self,
                 mbstd_group_size,
                 mbstd_num_features,
                 in_channels,
                 intermediate_channels,
                 gain, use_wscale,
                 activation_layer,
                 resolution=4,
                 in_channels2=None,
                 output_features=1,
                 last_gain=1):


        layers = []
        if mbstd_group_size > 1:
            layers.append(('stddev_layer', StddevLayer(mbstd_group_size, mbstd_num_features)))

        if in_channels2 is None:
            in_channels2 = in_channels

        layers.append(('conv', EqualizedConv2d(in_channels + mbstd_num_features, in_channels2, kernel_size=3,
                                               gain=gain, use_wscale=use_wscale)))
        layers.append(('act0', activation_layer))
        layers.append(('view', View(-1)))
        layers.append(('dense0', EqualizedLinear(in_channels2 * resolution * resolution, intermediate_channels,
                                                 gain=gain, use_wscale=use_wscale)))
        layers.append(('act1', activation_layer))
        layers.append(('dense1', EqualizedLinear(intermediate_channels, output_features,
                                                 gain=last_gain, use_wscale=use_wscale)))

        super().__init__(OrderedDict(layers))


class DiscriminatorBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, gain, activation_layer, blur_kernel):
        print(in_channels)
        print(out_channels)
        super().__init__(OrderedDict([
            ('conv0', Conv2dPropia(in_channels, in_channels, kernel_size=3, gain=gain)),
            # out channels nf(res-1)
            ('act0', nn.LeakyReLU(negative_slope=0.2)),
            ('blur', BlurLayer(kernel=blur_kernel)),
            ('conv1_down', Conv2DownPropia(in_channels, out_channels, kernel_size=3,gain=gain)),
            ('act1', activation_layer)]))


if __name__ == '__main__':
    # discriminator = DiscriminatorTop()
    print('Done.')
