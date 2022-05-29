import sys, os

curentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(curentdir)
sys.path.append(parentdir)

from torch.nn.modules.sparse import Embedding
from StyleGAN.Components.Blocks import *
from StyleGAN.Components.Layers import *


class DiscriminatorFinal(nn.Sequential):
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
            layers.append(StandardDeviationLayer(mbstd_group_size, mbstd_num_features))

        if in_channels2 is None:
            in_channels2 = in_channels

        layers.append(EqualizedConv2d(in_channels + mbstd_num_features, in_channels2, kernel_size=3,
                                      numMul=gain, increaseWeightScale=use_wscale))
        layers.append(activation_layer)
        layers.append(View(-1))
        layers.append(EqualizedLinear(in_channels2 * resolution * resolution, intermediate_channels,
                                      numMul=gain, increaseWeightScale=use_wscale))
        layers.append(activation_layer)
        layers.append(EqualizedLinear(intermediate_channels, output_features,
                                      numMul=last_gain, increaseWeightScale=use_wscale))

        super().__init__(nn.Sequential(*layers))


class DiscriminatorBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, gain, activation_layer, blur_kernel):
        layers = []
        layers.append(Conv2dPropia(in_channels, in_channels, kernel_size=3, gain=gain))
        layers.append( nn.LeakyReLU(negative_slope=0.2))
        layers.append(Conv2DownPropia(in_channels, out_channels, kernel_size=3, numMul=gain))
        layers.append(activation_layer)
        super().__init__(nn.Sequential(*layers))


class Discriminator(nn.Module):

    def __init__(self, resolution, num_channels=3, conditional=False,
                 n_classes=0, fmap_base=8192, fmap_decay=1.0, fmap_max=512,
                 nonlinearity='lrelu', use_wscale=True, mbstd_group_size=4,
                 mbstd_num_features=1, blur_filter=None, structure='linear'
                 ):
        super(Discriminator, self).__init__()

        if conditional:
            num_channels *= 2
            self.embeddings = []

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.conditional = conditional
        self.mbstd_num_features = mbstd_num_features
        self.mbstd_group_size = mbstd_group_size
        self.structure = structure

        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4
        self.depth = resolution_log2 - 1

        act, gain = (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))

        # create the remaining layers
        blocks = []
        from_rgb = []
        for res in range(resolution_log2, 2, -1):
            blocks.append(DiscriminatorBlock(nf(res - 1), nf(res - 2),
                                             gain=gain, activation_layer=nn.LeakyReLU(negative_slope=0.2),
                                             blur_kernel=blur_filter))
            # create the fromRGB layers for various inputs:
            from_rgb.append(Conv2dPropia(num_channels, nf(res - 1), kernel_size=1))
            # Create embeddings for various inputs:
            if conditional:
                r = 2 ** (res)
                self.embeddings.append(
                    Embedding(n_classes, (num_channels // 2) * r * r))

        if self.conditional:
            self.embeddings.append(nn.Embedding(n_classes, (num_channels // 2) * 4 * 4))
            self.embeddings = nn.ModuleList(self.embeddings)

        self.blocks = nn.ModuleList(blocks)

        # Building the final block.
        self.final_block = DiscriminatorFinal(self.mbstd_group_size, self.mbstd_num_features,
                                              in_channels=nf(2), intermediate_channels=nf(2),
                                              gain=gain, use_wscale=use_wscale, activation_layer=act)
        from_rgb.append(EqualizedConv2d(num_channels, nf(2), kernel_size=1,
                                        numMul=gain, increaseWeightScale=use_wscale))
        self.from_rgb = nn.ModuleList(from_rgb)

        # register the temporary downSampler
        self.temporaryDownsampler = nn.AvgPool2d(2)

    def forward(self, input_img, depth, alpha=1., input_labels=None):
        """ input_img: [mini_batch, channel, height, width]
            input_labels: [mini_batch, label_size].
            depth: (Progressive GAN - ProGAN)
            alpha: current value of alpha
        """
        self.sizeIn = input_img.shape
        self.alpha=alpha
        if depth > 0:
            if self.conditional:
                embedding_in = self.embeddings[self.depth -
                                               depth - 1](input_labels)
                embedding_in = embedding_in.view(input_img.shape[0], -1,
                                                 input_img.shape[2],
                                                 input_img.shape[3])
                input_img = torch.cat([input_img, embedding_in], dim=1)

            residual = self.from_rgb[self.depth - depth](self.temporaryDownsampler(input_img))
            straight = self.blocks[self.depth - depth - 1](self.from_rgb[self.depth - depth - 1](input_img))
            x = (self.alpha * straight) + ((1 - self.alpha) * residual)

            for block in self.blocks[(self.depth - depth):]:
                x = block(x)
        else:
            if self.conditional:
                embedding_in = self.embeddings[-1](input_labels)
                embedding_in = embedding_in.view(input_img.shape[0], -1,
                                                 input_img.shape[2],
                                                 input_img.shape[3])
                input_img = torch.cat([input_img, embedding_in], dim=1)
            x = self.from_rgb[-1](input_img) ##Downsampling: Ejemplo-> de 128x128 a 4x4
        scores_out = self.final_block(x)

        return scores_out
    def getinSize(self):
        return self.sizeIn