from audioop import bias
import torch
import torch.nn as nn
import torch.nn.functional as F
from StyleComponents import *
import numpy as np
import StyleGenerador as sg

class GeneradorGAN(nn.Module):
    def __init__(self, noiseDim, device = 'cuda', numChan=3, hiddenDim=64):
        super(GeneradorGAN, self).__init__()
        self.inDim = noiseDim
        self.device = device
        self.gen = nn.Sequential(
            self.generar_bloque_generador(self.inDim, hiddenDim * 8, device),
            self.generar_bloque_generador(hiddenDim * 8, hiddenDim * 4, device, kernTam=4, stride=1),
            self.generar_bloque_generador(hiddenDim * 4, hiddenDim * 2, device),
            self.generar_bloque_generador(hiddenDim * 2, hiddenDim, device ),
            self.generar_bloque_generador(hiddenDim, numChan, device, kernTam=4, ultimaCapa=True),
        )

    def forward(self, input):
        x = input.view(len(input), self.inDim, 1, 1)
        return self.gen(x)

    def getNoiseDim(self):
        return self.inDim

    def generar_bloque_generador(self, inChan, outChan, device = 'cuda', kernTam=3, stride=2, ultimaCapa=False):
        if ultimaCapa:
            return nn.Sequential(
                nn.ConvTranspose2d(inChan, outChan, kernTam, stride).to(device),
                nn.Tanh().to(device),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(inChan, outChan, kernTam, stride).to(device),
                nn.BatchNorm2d(outChan).to(device),
                nn.ReLU(inplace=True).to(device),
            )

class GeneradorCondicional(nn.Module):
    def __init__(self, noiseDim, condDim, device = 'cuda', numChan=3, hiddenDim=64):
        super(GeneradorCondicional, self).__init__()
        self.inDim = noiseDim
        self.condDim = condDim
        self.device = device
        self.gen = nn.Sequential(
            self.generar_bloque_generador(self.inDim + self.condDim, hiddenDim * 8, device), # 3 out
            self.generar_bloque_generador(hiddenDim * 8, hiddenDim * 4, device, kernTam=4, stride=1), # 6 out
            self.generar_bloque_generador(hiddenDim * 4, hiddenDim * 2, device), # 13 out
            self.generar_bloque_generador(hiddenDim * 2, hiddenDim, device ), # 27 out
            self.generar_bloque_generador(hiddenDim, numChan, device, kernTam=4, ultimaCapa=True), # 55 out
        )

    def forward(self, input):
        x = input.view(len(input), self.inDim + self.condDim, 1, 1)
        return self.gen(x)

    def getNoiseDim(self):
        return self.inDim
    
    def getCondDim(self):
        return self.condDim

    def generar_bloque_generador(self, inChan, outChan, device = 'cuda', kernTam=3, stride=2, ultimaCapa=False):
        if ultimaCapa:
            return nn.Sequential(
                nn.ConvTranspose2d(inChan, outChan, kernTam, stride).to(device),
                nn.Tanh().to(device),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(inChan, outChan, kernTam, stride).to(device),
                nn.BatchNorm2d(outChan).to(device),
                nn.ReLU(inplace=True).to(device),
            )


class GeneradorCondicionalStyle(nn.Module):
    def __init__(self, noiseDim, numChan=3, hiddenDim=64, device = 'cuda'):
        super().__init__()
        self.inDim = noiseDim  
        self.gen = nn.Sequential(
            self.generar_bloque_generador(self.inDim    , hiddenDim * 32, device),
            self.generar_bloque_generador(hiddenDim * 32, hiddenDim * 16, device, kernTam=2, stride=2),
            self.generar_bloque_generador(hiddenDim * 16, hiddenDim * 8, device, kernTam=2, stride=2),
            self.generar_bloque_generador(hiddenDim * 8 , hiddenDim * 4   , device, kernTam=2, stride=2),
            self.generar_bloque_generador(hiddenDim * 4 , hiddenDim * 2   , device, kernTam=2, stride=2),
            self.generar_bloque_generador(hiddenDim * 2 , hiddenDim    , device, kernTam=2, stride=2),
            self.generar_bloque_generador(hiddenDim     , numChan      , device, kernTam=4, ultimaCapa=True),
        )

    def forward(self, input):
        x = input.view(len(input), self.inDim, 1, 1)
        return self.gen(x)

    def generar_bloque_generador(self, inChan, outChan, device, kernTam=2, stride=2, ultimaCapa=False):
        if ultimaCapa:
            return nn.Sequential(
                nn.Conv2d(inChan, outChan,1).to(device),
                nn.Tanh().to(device),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(inChan, outChan, kernTam, stride).to(device),
                nn.BatchNorm2d(outChan).to(device),
                nn.ReLU(inplace=True).to(device),
            )



class MicroStyleGANGeneratorBlock(nn.Module):
    '''
    Micro StyleGAN Generator Block Class
    Values:
        in_chan: the number of channels in the input, a scalar
        out_chan: the number of channels wanted in the output, a scalar
        w_dim: the dimension of the intermediate noise vector, a scalar
        kernel_size: the size of the convolving kernel
        starting_size: the size of the starting image
    '''

    def __init__(self, in_chan, out_chan, w_dim, kernel_size, starting_size, device, use_upsample=True):
        super().__init__()
        self.use_upsample = use_upsample
        
        if self.use_upsample:
            self.upsample = nn.Upsample((starting_size), mode='bilinear').to(device)
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size, padding=1).to(device) # Padding is used to maintain the image size
        self.inject_noise = InyecciondeRuido(out_chan).to(device)
        self.adain = AdaIN(out_chan, w_dim).to(device)
        self.activation = nn.LeakyReLU(0.2).to(device)


    def forward(self, x, w):
        '''
        Function for completing a forward pass of MicroStyleGANGeneratorBlock: Given an x and w, 
        computes a StyleGAN generator block.
        Parameters:
            x: the input into the generator, feature map of shape (n_samples, channels, width, height)
            w: the intermediate noise vector
        '''
        if self.use_upsample:
            x = self.upsample(x)
        x = self.conv(x)
        x = self.inject_noise(x)
        x = self.activation(x)
        x = self.adain(x, w)
        return x
    
    def get_self(self):
        return self

class MicroStyleGANGenerator(nn.Module):
    '''
    Micro StyleGAN Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        map_hidden_dim: the mapping inner dimension, a scalar
        w_dim: the dimension of the intermediate noise vector, a scalar
        in_chan: the dimension of the constant input, usually w_dim, a scalar
        out_chan: the number of channels wanted in the output, a scalar
        kernel_size: the size of the convolving kernel
        hidden_chan: the inner dimension, a scalar
    '''

    def __init__(self, 
                 z_dim, 
                 map_hidden_dim,
                 w_dim,
                 in_chan,
                 out_chan, 
                 kernel_size, 
                 hidden_chan,
                 layers,
                 device):
        super().__init__()
        self.map = CapasMapeadoras(z_dim, map_hidden_dim, w_dim, layers).to(device)
        # Typically this constant is initiated to all ones, but you will initiate to a
        # Gaussian to better visualize the network's effect
        self.starting_constant = nn.Parameter(torch.randn(1, in_chan, 4, 4))
        self.block0 = MicroStyleGANGeneratorBlock(in_chan, hidden_chan, w_dim, kernel_size, 4, device, use_upsample=False)
        self.block1 = MicroStyleGANGeneratorBlock(hidden_chan, hidden_chan, w_dim, kernel_size, 8, device)
        self.block2 = MicroStyleGANGeneratorBlock(hidden_chan, hidden_chan, w_dim, kernel_size, 16, device)
        self.block3 = MicroStyleGANGeneratorBlock(hidden_chan, hidden_chan, w_dim, kernel_size, 32, device)
        self.block4 = MicroStyleGANGeneratorBlock(hidden_chan, hidden_chan, w_dim, kernel_size, 64, device)
        # You need to have a way of mapping from the output noise to an image, 
        # so you learn a 1x1 convolution to transform the e.g. 512 channels into 3 channels
        # (Note that this is simplified, with clipping used in the real StyleGAN)
        self.block1_to_image = nn.Conv2d(hidden_chan, out_chan, kernel_size=1).to(device)
        self.block2_to_image = nn.Conv2d(hidden_chan, out_chan, kernel_size=1).to(device)
        self.block3_to_image = nn.Conv2d(hidden_chan, out_chan, kernel_size=1).to(device)
        self.block4_to_image = nn.Conv2d(hidden_chan, out_chan, kernel_size=1).to(device)
        self.alfa = 0.001

    def upsample_to_match_size(self, smaller_image, bigger_image):
        '''
        Function for upsampling an image to the size of another: Given a two images (smaller and bigger), 
        upsamples the first to have the same dimensions as the second.
        Parameters:
            smaller_image: the smaller image to upsample
            bigger_image: the bigger image whose dimensions will be upsampled to
        '''
        return F.interpolate(smaller_image, size=bigger_image.shape[-2:], mode='bilinear')

    def forward(self, noise, return_intermediate=False):
        '''
        Function for completing a forward pass of MicroStyleGANGenerator: Given noise, 
        computes a StyleGAN iteration.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
            return_intermediate: a boolean, true to return the images as well (for testing) and false otherwise
        '''
        x = self.starting_constant
        w = self.map(noise)
        x = self.block0(x, w)
        x_8 = self.block1(x, w) # First generator run output
        x_8_img = self.block1_to_image(x_8)
        x_16 = self.block2(x_8, w) # Second generator run output 
        x_16_img = self.block2_to_image(x_16)
        x_8_upsample = self.upsample_to_match_size(x_8_img, x_16_img) # Upsample first generator run output to be same size as second generator run output 
        # Interpolate between the upsampled image and the image from the generator using alpha
        
        interpolation16 = self.alfa * (x_16_img) + (1-self.alfa) * (x_8_upsample)

        x_32 = self.block3(interpolation16, w)
        x_32_img = self.block3_to_image(x_32)
        x_16_upsample = self.upsample_to_match_size(interpolation16, x_32_img)
        interpolation32 = self.alfa * (x_32_img) + (1-self.alfa) * (x_16_upsample)

        x_64 = self.block4(interpolation32, w)
        x_64_img = self.block4_to_image(x_64)
        x_32_upsample = self.upsample_to_match_size(interpolation32, x_64_img)
        interpolation64 = self.alfa * (x_64_img) + (1-self.alfa) * (x_32_upsample)
        
        if return_intermediate:
            return interpolation64, x_32_upsample, x_64_img
        return interpolation64
    
    def get_self(self):
        return self

    def increaseAlfa(self, alfa):
        self.alfa = self.alfa + alfa
        if self.alfa > 1:
            self.alfa = 1



class BloqueGenerador(nn.Module):

    def __init__(self, inChan, outChan, wDim, kernel, tam, device = 'cuda', upsample = True):
        super().__init__()
        self.use_upsample = upsample
        
        if self.use_upsample:
            self.upsample = nn.Upsample((tam), mode='bilinear').to(device)
        self.conv = nn.Conv2d(inChan, outChan, kernel, padding=1).to(device) # Padding is used to maintain the image size
        self.inject_noise = InyecciondeRuido(outChan).to(device)
        self.adain = AdaIN(outChan, wDim).to(device)
        self.activation = nn.LeakyReLU(0.2).to(device)

    def forward(self, img, noise):
        x = img

        if self.use_upsample:
            x = self.upsample(x)

        x = self.conv(x)
        x = self.inject_noise(x)
        x = self.activation(x)
        x = self.adain(x, noise)
        return x

class StyleGenerador(nn.Module):

    def __init__(self, zDim, inChan, mappingLayersDim, disentNoiseDim, outChan, kernel, convHiddenChan, layers, device):
        super().__init__()
        self.mapLayers = CapasMapeadoras(zDim, mappingLayersDim, disentNoiseDim, layers).to(device)
        self.entradaPreset = torch.ones_like(torch.empty(1,inChan,4,4)).to(device)
        self.primerBloque =  BloqueGenerador(inChan, convHiddenChan, disentNoiseDim, kernel, 4, upsample=False).to(device)

        self.increasingAlfa = 0

        self.blocksToImage = []
        self.genBlocks = []
        self.alfas = []
        self.upsample = nn.Upsample(scale_factor=2, mode = 'bilinear')

        size = 4

        while size <= 64:
            
            self.blocksToImage.append(nn.Conv2d(convHiddenChan, outChan, 1).to(device))
            self.genBlocks.append(BloqueGenerador(convHiddenChan, convHiddenChan, disentNoiseDim, kernel, size))
            self.alfas.append(0)

            size = size * 2

    def forward(self, noise):

        x = self.entradaPreset

        w = self.mapLayers(noise)

        x = self.primerBloque(x,w)

        i = 0

        px = self.genBlocks[i](x,w)
        px_i = self.blocksToImage[i](x)

        i = i + 1

        while i < len(self.genBlocks) :
            
            x_b = self.genBlocks[i](px, w)
            x_u = self.upsample(px_i)
            x_i = self.blocksToImage[i](x_b)

            x = (1 - self.alfas[i]) * x_u + self.alfas[i] * x_i

            px_i = x
            px = x_b

            i = i + 1

        return x

    def increaseAlfa(self, inc):
        
        if self.increasingAlfa < len(self.alfas):

            self.alfas[self.increasingAlfa] = self.alfas[self.increasingAlfa] + inc

            if self.alfas[self.increasingAlfa] >= 1 and self.increasingAlfa < len(self.alfas):
                
                self.increasingAlfa = self.increasingAlfa + 1


class GeneratorSimple(nn.Module):
    def __init__(self, noiseDim, outChan, device):
        super().__init__()
        assert(noiseDim[1] == noiseDim[2])

        self.noiseDim = noiseDim
        self.outChan = outChan

        gen = []
        size = noiseDim[1]
        chan = noiseDim[0]

        while size < 64:
            gen.append(nn.ConvTranspose2d(chan, int(chan/2), 2, 2).to(device))
            gen.append(nn.LeakyReLU(0.02).to(device))
            gen.append(nn.BatchNorm2d(int(chan/2)))

            chan = int(chan/2)
            size = int(size*2)
        
        while chan < 4:
            gen.append(nn.Conv2d(chan, int(chan/2), 1).to(device))
            gen.append(nn.LeakyReLU(0.02).to(device))
            chan = int(chan/2)

        if chan != 3:
            gen.append(nn.Conv2d(chan, self.outChan, 1).to(device))
            

        self.gen = nn.Sequential(*gen)

    def forward(self, noise):
        return self.gen(noise)


class StyleNoProgGenerator(nn.Module):
    def __init__(self, inChan, outChan, inputDim, layers, imageDim, styleDim, device):
        super().__init__()

        self.inChan = inChan
        self.styleDim = styleDim
        self.imageDim = imageDim
        self.inputDim = inputDim
        self.outChan = outChan
        self.device = device

        self.gen = []
        self.end = []

        chan = inChan
        assert(inputDim[1] == inputDim[2])
        size = inputDim[1]

        self.mappinglayers = CapasMapeadoras(self.imageDim[0] * self.imageDim[1] * self.imageDim[2], 50, styleDim, layers).to(device)

        while size < 64:
            #conv, random noise, adain
            self.gen.append(self.generate_gen_block(styleDim, chan, device))
            chan = int(chan/2)
            size = size * 2

        while chan > 4:
            self.end.append(nn.Conv2d(chan, int(chan/2), 1, 1).to(device))
            self.end.append(nn.LeakyReLU(0.02, inplace = True).to(device))
            chan = int(chan/2)
        
        if chan != outChan:
            self.end.append(nn.Conv2d(chan, outChan, 1, 1).to(device))
            self.end.append(nn.LeakyReLU(0.02))

        

    def forward(self, style):
        x = torch.ones_like(torch.empty(1,self.inChan,4,4)).to(self.device)
        style = self.mappinglayers(style)
        for i in range(len(self.gen)):
            x = self.gen[i](x, style)
            # print("Bloque " + str(i) + " shape = " + str(x.shape))

        if len(self.end) != 0:
            for i in range(len(self.end)):
                x = self.end[i](x)
        
        return x

    def generate_gen_block(self, noiseDim, chan, device):
        return StyleNoProgGeneratorBlock(noiseDim, chan, device)
    
    def getNoiseDim(self):
        return self.imageDim[0] * self.imageDim[1] * self.imageDim[2]
        



class StyleNoProgGeneratorBlock(nn.Module):

    def __init__(self, noiseDim, chan, device):
        super().__init__()
        self.c1 = nn.ConvTranspose2d(chan, int(chan/2), 2, 2).to(device)
        self.i1 = InyecciondeRuido(int(chan/2)).to(device)
        self.adain1 = AdaIN(int(chan/2), noiseDim).to(device)
        self.c2 = nn.Conv2d(int(chan/2), int(chan/2), 1, 1).to(device)
        self.i2 = InyecciondeRuido(int(chan/2)).to(device)
        self.adain2 = AdaIN(int(chan/2), noiseDim).to(device)
        self.act = nn.LeakyReLU(0,2).to(device)
    
    def forward(self, prev, noise): 

        x = self.c1(prev)
        x = self.i1(x)
        x = self.adain1(x, noise)
        x = self.c2(x)
        x = self.i2(x)
        x = self.adain2(x, noise)
        x = self.act(x)

        return x

class EqualizedConv2d(nn.Conv2d):
    def __init__(self, inChan, outChan, kernel, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__(inChan, outChan, kernel, stride, padding, dilation, groups, bias, padding_mode)
        
        torch.nn.init.normal_(self.weight)
        if bias:
            torch.nn.init.zeros_(self.bias)
        
        self.scale = np.sqrt(2) / (np.sqrt(np.prod(self.kernel_size) * self.in_channels) + 0.0000000000001)
    
    def forward(self, x):
        return torch.conv2d(
            input = x,
            weight= self.weight * self.scale,
            bias = self.bias,
            stride = self.stride,
            padding = self.padding,
            dilation = self.dilation,
            groups = self.groups
        )

class EqualizedConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, inChan, outChan, kernel, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode="zeros"):
        super().__init__(inChan, outChan, kernel, stride, padding, output_padding, groups, bias, dilation, padding_mode)
        torch.nn.init.normal_(self.weight)
        if bias:
            torch.nn.init.zeros_(self.bias)
        
        self.scale = np.sqrt(2) / np.sqrt(np.prod(self.kernel_size) * self.in_channels) + 0.0000000000001

    def forward(self, x, output_size):
        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)
        return torch.conv_transpose2d(
            input = x,
            weight= self.weight * self.scale,
            bias = self.bias,
            stride = self.stride,
            pading = self.padding,
            output_padding = output_padding,
            groups = self.groups,
            dilation = self.dilation
        )

class PixelwiseNorm(nn.Module):
    def __init__(self):
        super(PixelwiseNorm, self).__init__()
    
    def forward(self, x, alfa = 1e-8):
        y = x.pow(2.0).mean(dim=1, keepdim=True).add(alfa).sqrt()
        y = x/y
        return y

class EqualizedLinear(nn.Linear):
    def __init__(self, in_f, out_f, bias = True):
        super().__init__(in_f, out_f, bias)

        torch.nn.init.normal_(self.weight)
        if bias :
            torch.nn.init.zeros_(self.bias)

        fan_in = self.in_features
        self.scale = np.sqrt(2) / np.sqrt(fan_in)
    
    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight * self.scale, self.bias)

class EqualizedAdaIN(nn.Module):

    def __init__(self, channels, w_dim):
        super().__init__()

        # Normalize the input per-dimension
        self.instance_norm = PixelwiseNorm()

        self.style_scale_transform = EqualizedLinear(w_dim, channels)
        self.style_shift_transform = EqualizedLinear(w_dim, channels)

    def forward(self, image, w):
        '''
        Function for completing a forward pass of AdaIN: Given an image and intermediate noise vector w, 
        returns the normalized image that has been scaled and shifted by the style.
        Parameters:
            image: the feature map of shape (n_samples, channels, width, height)
            w: the intermediate noise vector
        '''
        normalized_image = self.instance_norm(image)
        style_scale = self.style_scale_transform(w)[:, :, None, None]
        style_shift = self.style_shift_transform(w)[:, :, None, None]
        
        # Calculate the transformed image
        # print("Style = " + str(w.shape))
        # print("image = " + str(image.shape))
        transformed_image = style_scale * normalized_image + style_shift
        # print("Style scale = " + str(style_scale.shape))
        # print("Style shift = " + str(style_shift.shape))
        # print("Transformed img = " + str(transformed_image.shape))


        return transformed_image
    

    def get_style_scale_transform(self):
        return self.style_scale_transform
    

    def get_style_shift_transform(self):
        return self.style_shift_transform
    

class EqualizedStyleGenBlock(nn.Module):
    def __init__(self, inChan, outChan, wDim, kernel, padding, device = 'cuda'):
        super().__init__()
        self.inChan = inChan
        self.outChan = outChan
        self.wdim = wDim
        self.kernel_size = kernel
        self.padding = padding
        self.device = device
        self.alfa = 0

        self.c1 = EqualizedConv2d(inChan, outChan, kernel, kernel, padding).to(device)
        self.i1 = InyecciondeRuido(outChan).to(device)
        self.a1 = EqualizedAdaIN(outChan, wDim).to(device)
        self.c2 = EqualizedConv2d(outChan, outChan, 1, 1).to(device)
        self.i2 = InyecciondeRuido(outChan).to(device)
        self.a2 = EqualizedAdaIN(outChan, wDim).to(device)
        self.act = nn.LeakyReLU(0,2).to(device)
    
        self.upsample = nn.Upsample(scale_factor=2 , mode = 'bilinear')

    def forward(self, prev_tens, noise):

        x = self.c1(prev_tens)
        x = self.i1(x)
        x = self.a1(x, noise)
        x = self.c2(x)
        x = self.i2(x)
        x = self.a2(x, noise)
        x = self.act(x)
        
        return x

class EqualizedStyleGen(nn.Module):

    def __init__(self, inChan, outChan, inputDim, layers, imageDim, styleDim, device):
        #                512       3     (512,4,4)    8    (3,64,64)    64
        super().__init__()
        self.inChan = inChan
        self.outChan = outChan
        self.imageDim = imageDim
        self.inputDim = inputDim
        self.layers = layers
        self.styleDim = styleDim
        self.device = device

        self.gen_blocks    = []
        self.to_rgb_blocks = []
        self.alfas         = []
        self.end           = []
        self.act_alfa      = 1

        self.mappingLayers = CapasMapeadoras(self.imageDim[0] * self.imageDim[1] * self.imageDim[2], 64, self.styleDim, self.layers).to(device)

        size = self.inputDim[1]
        chan = self.inChan

        while size < 64 :
            self.gen_blocks.append(EqualizedStyleGenBlock(chan, int(chan/2), styleDim, 2, 2, self.device).to(device))
            self.to_rgb_blocks.append(EqualizedConv2d(int(chan/2), 3, 1).to(device))
            self.alfas.append(0)

            chan = int(chan/2)
            size = int(size/2)
        
        self.upsample = nn.Upsample(scale_factor=2 , mode = 'bilinear')

    def forward(self, noise):
        
        style = self.mappingLayers(noise)
        const_in = x = torch.ones_like(torch.empty(1,self.inChan,4,4)).to(self.device)

        x_t = self.gen_blocks[0](const_in, style)
        x_img = self.to_rgb_blocks[0](x_t)

        for i in range(1, len(self.gen_blocks)):
            x_t = self.gen_blocks[i](x_t, style)
            x_u = self.upsample(x_img)
            x_img = self.to_rgb_blocks[i](x_t)

            x_img = (1-self.alfas[i]) * x_u + self.alfas[i] * x_img

        return x_img
    
    def increaseAlfa(self, alfa):

        if self.act_alfa < len(self.alfas):

            if self.alfas[self.act_alfa] < 1:
                self.alfas[self.act_alfa] = min(self.alfas[self.act_alfa] + alfa, 1)

                if self.alfas[self.act_alfa] == 1:
                    self.act_alfa = self.act_alfa + 1


class StyleNoCondGenerator(nn.Module):

    def __init__(self, resolution, batch_size, alfa, device = 'cuda', latent_size=512, dlatent_size=512, conditional=False,
                n_classes=0, truncation_psi=0.7, truncation_cutoff=8, dlatent_avg_beta=0.995,
                style_mixing_prog=0.9, **kwargs):
        
        super().__init__()
        self.resolution = resolution
        self.gen = sg.Generator(resolution, latent_size, dlatent_size, conditional, n_classes, truncation_psi, truncation_cutoff, dlatent_avg_beta, style_mixing_prog, **kwargs).to(device)
        self.alfa = alfa
        self.depth = 0
        self.batch_size = batch_size
        self.iter = 0

        

    def forward(self, noise): 
        ret = self.gen(noise, self.depth, self.alfa)

        self.iter += 1

        assert(ret.shape[2] == ret.shape[3])

        return ret

    def increaseAlfa(self, alfa):

        self.alfa = self.alfa + alfa
        if self.alfa >= 1 :
            self.alfa = 1

    def getNoiseDim(self):
        return self.resolution
    
    def resetAlfa(self):
        self.alfa = 0
    
    def getAlfa(self):
        return self.alfa

    def getDepth(self):
        return self.depth
  
    def getinSize(self):
        return self.inSize

    def increaseDepth(self):
        self.depth = self.depth + 1
        self.resetAlfa()
        if self.depth > 4:
            self.depth = 4



