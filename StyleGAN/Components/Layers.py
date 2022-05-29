import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


##Alternative Batch Normalization
class PixelNormLayer(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


class Upscale2d(nn.Module):
    @staticmethod
    def upscale2d(x, sizeIncrease=2, numMul=1):
        if numMul != 1:
            x = x * numMul
        if sizeIncrease != 1:
            shape = x.shape
            x = x.view(shape[0], shape[1], shape[2], 1, shape[3], 1).expand(-1, -1, -1, sizeIncrease, -1,
                                                                            sizeIncrease)  ##Mismo tensor pero distinta forma y expandimos las nuevas dimensaiones
            x = x.contiguous().view(shape[0], shape[1], sizeIncrease * shape[2],
                                    sizeIncrease * shape[3])  ## aumetamos as dimensiones DimxDim
        return x

    def __init__(self, sizeIncrease=2, numMul=1):
        super().__init__()
        self.numMul = numMul
        self.sizeIncrease = sizeIncrease

    def forward(self, x):
        return self.upscale2d(x, sizeIncrease=self.sizeIncrease, numMul=self.numMul)


class Downscale2d(nn.Module):
    def __init__(self, sizeDecrease=2, gain=1):
        super().__init__()
        self.sizeDecrease = sizeDecrease
        self.gain = gain
        if sizeDecrease == 2:
            f = [np.sqrt(gain) / sizeDecrease] * sizeDecrease
            self.blur = BlurConv2dLayer(kernel=f, norm=False, stride=sizeDecrease)
        else:
            self.blur = None

    def forward(self, x):
        if self.blur is not None and x.dtype == torch.float32:
            return self.blur(x)
        if self.gain != 1:
            x = x * self.gain
        if self.sizeDecrease == 1:
            return x
        return F.avg_pool2d(x, self.factor)


class EqualizedLinear(nn.Module):
    def __init__(self, input_size, output_size, numMul=2 ** 0.5, increaseWeightScale=False, lrmul=1, bias=True):
        super().__init__()
        valScale = numMul * input_size**(-0.5)

        ##Values for out_features in Linear
        if increaseWeightScale:
            numToMul = 1.0 / lrmul
            self.weightScale = valScale * lrmul
        else:
            numToMul = valScale / lrmul
            self.weightScale = lrmul

        self.weight = torch.nn.Parameter(torch.randn(output_size, input_size) * numToMul)

        ##Values for bias in Linear
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_size))
            self.b_mul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        bias = self.bias
        w= self.weight
        if bias is not None:
            bias = bias * self.b_mul
        return F.linear(x, self.weight * self.weightScale, bias)


class EqualizedLinearPropia(nn.Module):
    def __init__(self, input_size, output_size, numMul=2 ** 0.5, lrmul=1):
        super().__init__()
        valScale = numMul * input_size ** (-0.5)
        init_std = 1.0 / lrmul
        self.weightScale = valScale * lrmul
        self.weight = torch.nn.Parameter(torch.randn(output_size, input_size) * init_std)
        self.bias = torch.nn.Parameter(torch.zeros(output_size))
        self.b_mul = lrmul

    def forward(self, x):
        bias = self.bias
        bias = bias * self.b_mul
        return F.linear(x, self.weight * self.weightScale, bias)


class EqualizedConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, stride=1, numMul=2 ** 0.5,
                 increaseWeightScale=False,
                 lrmul=1, bias=True, intermediate=None, upscale=False, downscale=False):
        super().__init__()

        ##Upscale propia
        if upscale:
            self.upscale = Upscale2d()
        else:
            self.upscale = None

        # Downscale propia
        if downscale:
            self.downscale = Downscale2d()
        else:
            self.downscale = None

        valScale = pow(numMul *(input_channels * kernel_size ** 2), (-0.5))

        self.kernel_size = kernel_size

        ##Mismo que Equali.Linear:
        if increaseWeightScale:
            init_std = 1.0 / lrmul
            self.weightScale = valScale * lrmul
        else:
            init_std = valScale / lrmul
            self.weightScale = lrmul

        self.weight = torch.nn.Parameter(
            torch.randn(output_channels, input_channels, kernel_size, kernel_size) * init_std)

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_channels))
            self.b_mul = lrmul
        else:
            self.bias = None

        self.intermediate = intermediate

    def forward(self, x):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul

        ##Si deseo hacer Upsample:
        if self.upscale is not None:
            x = self.upscale(x)

        downscale = self.downscale
        intermediate = self.intermediate
        # Si deseo hacer Downsample
        if downscale is not None:
            intermediate = downscale

        if intermediate is None:
            return F.conv2d(x, self.weight * self.weightScale, bias, padding=self.kernel_size // 2)
        else:
            x = F.conv2d(x, self.weight * self.weightScale, None, padding=self.kernel_size // 2)

        if intermediate is not None:
            x = intermediate(x)

        if bias is not None:
            x = x + bias.view(1, -1, 1, 1)
        return x


class NoiseLayer(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(channels))
        self.noise = None

    def forward(self, x, noise=None):
        ##Aplicar RandomNoise:
        w=self.weight
        if noise is None and self.noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        elif noise is None:  ##Usar noise pre-establecido
            noise = self.noise
        x = x + self.weight.view(1, -1, 1, 1) * noise  ##Aplicar Ruido
        return x


##Bloque que le añade el estilo a la imagen yScale*Adain +yBias
class StyleMod(nn.Module):
    def __init__(self, latent_size, channels, use_wscale):
        super(StyleMod, self).__init__()
        self.LinearPropia = EqualizedLinear(latent_size, channels * 2, numMul=1.0, increaseWeightScale=use_wscale)

    def forward(self, x, latent):
        style = self.LinearPropia(latent)  # style => [batch_size, n_channels*2]  --> alpha en ADAIN ES PRIMERA COLUMNA (Scaling Factor) y beta: Shifting Factor (beta)

        shape = [-1, 2, x.size(1)] + (x.dim() - 2) * [1]
        style = style.view(shape)  # [batch_size, 2, n_channels, ...]

        alpha = (style[:, 0] + 1.) ##Scaling Factor
        beta= style[:, 1]           ##Shifting Factor
        x = x * alpha + beta
        return x

class BlurConv2dLayer(nn.Module):
    def __init__(self, kernel=None, norm=True, swap=False, stride=1):
        super(BlurConv2dLayer, self).__init__()
        if kernel is None:
            kernel = [1, 2, 1]
        kernel = torch.tensor(kernel, dtype=torch.float32)
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel[None, None]

        ##Normalizar anterior kernel
        if norm:
            kernel = kernel / kernel.sum()
        ##Cambiar columnas del kernel
        if swap:
            kernel = kernel[:, :, ::-1, ::-1]
        self.register_buffer('kernel', kernel)
        self.stride = stride

    def forward(self, x):
        kernel = self.kernel.expand(x.size(1), -1, -1, -1)
        x = F.conv2d(x, kernel, stride=self.stride, padding=int((self.kernel.size(2) - 1) / 2), groups=x.size(1))
        return x


class View(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)


class StandardDeviationLayer(nn.Module):
    def __init__(self, group_size=4, numNewFeatures=1):
        super().__init__()
        self.group_size = group_size
        self.numNewFeatures = numNewFeatures

    def forward(self, x):
        b, c, h, w = x.shape
        y = x.reshape([min(self.group_size, b), -1, self.numNewFeatures, c // self.numNewFeatures, h, w])
        ##Sacamos la media
        y = y - y.mean(0, keepdim=True)
        ##Elevamos a la 2 la media  y luego obtenemos la pedia
        y = (pow(y, 2)).mean(0, keepdim=True)
        ##Finalmente raiz
        y = pow((y + 1e-8),0.5)
        y = y.mean([3, 4, 5], keepdim=True).squeeze(3).expand(min(self.group_size, b), -1, -1, h, w).clone().reshape(b, self.numNewFeatures, h, w)
        z = torch.cat([x, y], dim=1)
        return z
