import torch
import torch.nn as nn
import torch.nn.functional as F
from StyleComponents import *


class GeneradorCondicional(nn.Module):
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
        print(input.shape)
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
                 device):
        super().__init__()
        self.map = CapasMapeadoras(z_dim, map_hidden_dim, w_dim).to(device)
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

    def __init__(self, zDim, inChan, mappingLayersDim, disentNoiseDim, outChan, kernel, convHiddenChan, device):
        super().__init__()
        self.mapLayers = CapasMapeadoras(zDim, mappingLayersDim, disentNoiseDim).to(device)
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

