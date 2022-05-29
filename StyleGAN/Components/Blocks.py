from StyleGAN.Components.Layers import *


class InputBlock(nn.Module): ##Primer bloque 4x4
    def __init__(self, nf, dlatent_size):
        super().__init__()
        self.nf = nf

        self.const = nn.Parameter(torch.ones(1, nf, 4, 4))
        self.bias = nn.Parameter(torch.ones(nf))

        self.adain1 = CapaS_StyleMode(nf, dlatent_size)

        self.conv = Conv2dPropia(nf, nf)
        self.adain2 = CapaS_StyleMode(nf, dlatent_size)

    def forward(self, dlatents_in_range):
        batch_size = dlatents_in_range.size(0)

        x = self.const.expand(batch_size, -1, -1, -1)
        x = x + self.bias.view(1, -1, 1, 1)
        x = self.adain1(x, dlatents_in_range[:, 0])
        x = self.conv(x)
        x = self.adain2(x, dlatents_in_range[:, 1])

        return x


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



class Conv2dUPPropia(nn.Module):
    def __init__(self, input_channels, output_channels, numMul=np.sqrt(2)):
        super().__init__()
        self.kernel_size = 3
        self.weightScale = numMul * (input_channels * self.kernel_size**2)**(-0.5)

        self.weight = torch.nn.Parameter(
            torch.randn(output_channels, input_channels, self.kernel_size, self.kernel_size))

        self.bias = torch.nn.Parameter(torch.zeros(output_channels))
        self.upsample = Upscale2d()
        self.b_mul = 1

    def forward(self, x):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul
        w=self.weight
        ##Primero aumentamos
        x = self.upsample(x)
        #Luego Conv2d
        x = F.conv2d(x, self.weight * self.weightScale, None, padding=self.kernel_size // 2)

        if bias is not None:
            x = x + bias.view(1, -1, 1, 1)

        return x

class Conv2DownPropia(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, numMul=np.sqrt(2)):
        super().__init__()
        self.kernel_size = kernel_size
        self.weightScale = numMul * (input_channels * self.kernel_size ** 2) ** (-0.5)
        self.weight = torch.nn.Parameter(torch.randn(output_channels, input_channels, self.kernel_size, self.kernel_size))
        self.bias = torch.nn.Parameter(torch.zeros(output_channels))
        self.downscale = Downscale2d()
        self.b_mul = 1

    def forward(self, x):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul

        downscale = self.downscale
        intermediate = None
        if  downscale is not None:
            intermediate = downscale

        x = F.conv2d(x, self.weight * self.weightScale, None, padding=self.kernel_size // 2)

        if intermediate is not None:
            x = intermediate(x)

        if bias is not None:
            x = x + bias.view(1, -1, 1, 1)

        return x

class Conv2dPropia(nn.Module):

    def __init__(self, input_channels, output_channels, gain=np.sqrt(2), kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.weightScale = gain * (input_channels * self.kernel_size ** 2) ** (-0.5)
        self.weight = torch.nn.Parameter( torch.randn(output_channels, input_channels, self.kernel_size, self.kernel_size))
        self.bias = torch.nn.Parameter(torch.zeros(output_channels))
        self.b_mul = 1

    def forward(self, x):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul
        x = F.conv2d(x, self.weight * self.weightScale, None, padding=self.kernel_size // 2)
        w=self.weight
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


if __name__ == '__main__':
    # discriminator = DiscriminatorTop()
    print('Done.')
