import torch.nn as nn
import torch.functional as F
import torch.nn.functional as nnF
import CustomLayers as cl
import Blocks as bk
import math


class DiscriminadorGAN(nn.Module):

  def __init__(self, device = 'cuda', inChan=1, hiddenDim=64):
    super(DiscriminadorGAN, self).__init__()
    l = [self.generar_bloque_discriminador(inChan, hiddenDim, device= device),
        self.generar_bloque_discriminador(hiddenDim, hiddenDim * 2, device= device),
        self.generar_bloque_discriminador(hiddenDim*2, hiddenDim * 4, device= device),
        self.generar_bloque_discriminador(hiddenDim*4, inChan,device= device,ultimaCapa= True )]

    self.discriminador = nn.ModuleList(
        l
    )

  def generar_bloque_discriminador(self, inChan, outChan, device = 'cuda', kernel = 2, stride = 2, ultimaCapa = False):
    if ultimaCapa :
      return nn.Sequential(
          nn.Conv2d(inChan, outChan, kernel, stride).to(device),
      )
    else :
      return nn.Sequential(
          nn.Conv2d(inChan, outChan, kernel, stride).to(device),
          nn.BatchNorm2d(outChan).to(device),
          nn.LeakyReLU(0.2, inplace = True).to(device),
      )

  def forward(self, image):
    for e in range(0, len(self.discriminador)):
      image = self.discriminador[e](image)
    pred = image
    return pred.view(len(pred), -1)

## Discriminador con mejoras aprendidas del libro


class DiscriminadorCondicional(nn.Module):

  def __init__(self, device = 'cuda', inChan=1, hiddenDim=64):
    super(DiscriminadorCondicional, self).__init__()
    l = [self.generar_bloque_discriminador(inChan, hiddenDim, device= device),
        self.generar_bloque_discriminador(hiddenDim, hiddenDim * 2, device= device),
        self.generar_bloque_discriminador(hiddenDim*2, hiddenDim * 4, device= device),
        self.generar_bloque_discriminador(hiddenDim*4, inChan,device= device,ultimaCapa= True )]

    self.discriminador = nn.ModuleList(
        l
    )

  def generar_bloque_discriminador(self, inChan, outChan, device = 'cuda', kernel = 4, stride = 2, ultimaCapa = False):
    if ultimaCapa :
      return nn.Sequential(
          nn.Conv2d(inChan, outChan, kernel, stride).to(device),
      )
    else :
      return nn.Sequential(
          nn.Conv2d(inChan, outChan, kernel, stride).to(device),
          nn.BatchNorm2d(outChan).to(device),
          nn.LeakyReLU(0.2, inplace = True).to(device),
      )

  def forward(self, image):
    for e in range(0, len(self.discriminador)):
      image = self.discriminador[e](image)
    pred = image
    return pred.view(len(pred), -1)

## Discriminador con mejoras aprendidas del libro

class Primer_Bloque_Disc_Libro(nn.Module):

  def __init__(self, image_size, inChan, alfa):
    super().__init__()
    self.kernel = int(4)
    self.stride = int(2)
    self.alfa = alfa
    self.padding = int( math.pow(2, int(math.ceil(math.log(image_size[1],2)))) - image_size[1] + 1)
    print("El padding del primer bloque es " + str(self.padding))
    self.stridedDSC = nn.Conv2d(int(inChan), int(2*inChan), self.kernel, self.stride, self.padding)
    self.poolingDS  = nn.AvgPool2d(int(self.kernel), int(self.stride), self.padding)
    self.dsTensorToImg = nn.Conv2d(int(2*inChan), 3, 1, stride=1)
    self.act = nn.LeakyReLU(0.2, inplace=True)

  def forward(self, img):
    
    resSC = self.stridedDSC(img)
    resPDS = self.poolingDS(img)
    imgSC  = self.dsTensorToImg(resSC)

    inter = (1 - self.alfa) * resPDS + self.alfa * imgSC

    res = self.act(inter)

    return res

  def increaseAlfa(self, inc):
    self.alfa = self.alfa + inc
    if self.alfa > 1 :
      self.alfa = 1
  
  def getAlfa(self):
    return self.alfa




  
class Bloque_Discriminador_Libro(nn.Module):

  def __init__(self, inChan, alpha, device):
    super().__init__()
    self.kernel = 4
    self.padding = 1
    self.stride = 2
    self.alfa = alpha

    self.stridedDSConv = nn.Conv2d(inChan, 2*inChan,  self.kernel, self.stride, self.padding).to(device)
    self.poolingDS = nn.AvgPool2d(self.kernel, self.stride, self.padding).to(device)
    self.dsTensorToImage = nn.Conv2d(2*inChan, 3, 1, stride=1).to(device)
    self.batchNorm = nn.BatchNorm2d(3).to(device)
    self.act = nn.LeakyReLU(0.2, inplace = True).to(device)
    

  def forward(self, img):
    resSC  = self.stridedDSConv(img)
    # [ 2*img.chan , img.h /2 , img.w/2]
    resPDS = self.poolingDS(img)
    # [ img.chan, img.h/2, img.w/2]
    imgSC  = self.dsTensorToImage(resSC)
    # [ img.chan, img.h/2, img.w/2 ]

    inter = (1 - self.alfa) * resPDS +  self.alfa * imgSC

    inter = self.batchNorm(inter)

    res = self.act(inter)

    return res

  def increaseAlfa(self, alfa):
    self.alfa = self.alfa + alfa
    if self.alfa > 1:
      self.alfa = 1

  def getAlfa(self):
    return self.alfa

class DiscriminadorLibro(nn.Module):

  def __init__(self, image_size, alfa, device):
    super().__init__()
    self.image_size = image_size
    self.inChan = image_size[0]
    self.alfa = alfa
    self.bloques = []
    x = Primer_Bloque_Disc_Libro(image_size, 3, self.alfa).to(device)
    self.bloques.append(x)
    self.add_module('primer bloque',x)
    self.bloque_act = 0
    
    size = math.pow(2, int(math.ceil(math.log(image_size[1],2)))) / 2
    inChan = 3
    bloques = 1
    while size > 4 :
      m = Bloque_Discriminador_Libro(inChan, self.alfa, device).to(device)
      self.bloques.append(m)
      self.add_module('{i}esimo bloque', m)
      size = size / 2
      bloques = bloques + 1

    print("El discriminador tiene " + str(bloques) + " bloques ")  
    self.end = nn.Conv2d(3,1,4).to(device)
    

  def forward(self, img):
    x = img
    for i in range(len(self.bloques)):
      x = self.bloques[i](x)
    x = self.end(x)
    return x.view((-1))

  def increaseAlfa(self, inc):
    if self.bloque_act >= 0:
      if self.bloques[self.bloque_act].getAlfa() < 1:
        self.bloques[self.bloque_act].increaseAlfa(inc)
        if self.bloques[self.bloque_act].getAlfa() >= 1:
          self.bloque_act = self.bloque_act - 1


class BloqueDiscBloques(nn.Module):
  """
  :param inChan : number of channels of the input tensor
  :param device : either cuda or cpu

  DSConv1 performs a convolution with no efect on the dimension of the input tensor
  Act runs the input through a LeakyReLU
  BatchNorm normalizes the input taking into account the running mean and the std deviation of the batch
  DSConv2 performs a convolution that halves the height and the width of the input tensor
  """
  def __init__(self, inChan, outChan, device):
    super().__init__()
    kernel = 4
    stride = 2
    padding = 1

    self.DSConv1 = bk.Conv2dPropia(inChan, inChan, 3,).to(device)
    self.DSConv2 = bk.Conv2DownPropia(inChan, outChan, 3).to(device)
    self.act1 = nn.LeakyReLU(0.2, True).to(device)
    self.act2 = nn.LeakyReLU(0.2, True).to(device)
    self.blur = bk.BlurLayer(kernel=[1, 2, 1]).to(device)
  
  def forward(self, img):
    img = self.DSConv1(img)
    img = self.act1(img)
    img = self.blur(img)
    img = self.DSConv2(img)
    img = self.act2(img)
    return img

class UltimoBloqueDiscBloques(nn.Module):
  def __init__(self, inChan, device):
    super().__init__()
    inFeat = inChan * 4 * 4
    # print("INFEAT = " + str(inFeat))
    self.st    = bk.StddevLayer(4,1)
    self.conv  = bk.Conv2dPropia(inChan, inChan, 1).to(device)
    self.act1  = nn.LeakyReLU(0.02, True).to(device)
    self.lin1  = cl.EqualizedLinear(inFeat, inChan).to(device)
    self.act2  = nn.LeakyReLU(0.02, True).to(device)
    self.lin2  = cl.EqualizedLinear(inChan, 1).to(device)
  
  def forward(self, img):
    img = self.st(img)
    img = self.conv(img)
    img = self.act1(img)
    img = img.view(img.shape[0],-1)
    #img = img.view(512, 8192)
    img = self.lin1(img)
    img = self.act2(img)
    img = self.lin2(img)
    return img

class DiscriminadorPorBloques(nn.Module):
  def __init__(self, max_size, inChan, hiddenChan, device):
    super().__init__()
    self.device = device
    self.max_size = max_size
    self.inChan = inChan
    self.hiddenChan = hiddenChan
  
    from_rgb_chan = [256, 512, 512, 512, 512, 512]

    self.downsampler = nn.AvgPool2d(4,2,1).to(device)
    self.blocks = nn.ModuleList()
    self.from_rgb = nn.ModuleList()
    
    size = 4
    self.blocks.append(UltimoBloqueDiscBloques(hiddenChan,device))
    self.from_rgb.append(cl.EqualizedConv2d(inChan, hiddenChan, 1).to(device))
    i = -2
    while size < self.max_size:
      self.blocks.insert(0, BloqueDiscBloques(from_rgb_chan[i] , hiddenChan, device))
      self.from_rgb.insert(0, cl.EqualizedConv2d(inChan, from_rgb_chan[i], 1).to(device))
      size = size * 2
      i = i - 1


    self.alfa = 0
    self.depth = 0
    self.inSize = 4
  
  def forward(self, img):
    in_ds = img
    if self.depth > 0:
      res_ds = self.downsampler(in_ds)
      rgb = self.depth-1 if self.depth-1 != 0 else -1
      res_ds = self.from_rgb[-(rgb)](res_ds)

      in_ds = self.from_rgb[-(self.depth+1)](in_ds)

      in_ds = self.blocks[-(self.depth+1)](in_ds)

      in_ds = (self.alfa * in_ds) + ((1 - self.alfa) * res_ds)

      next = - self.depth

      for b in self.blocks[next: ]:
        in_ds = b(in_ds)

      return in_ds
    else :
      in_ds = self.from_rgb[-1](in_ds)
      return self.blocks[-1](in_ds)

  def increaseAlfa(self,inc):
    self.alfa = self.alfa + inc

    if self.alfa >= 1 :
      self.alfa = 1
    
  def getAlfa(self):
    return self.alfa

  def getDepth(self):
    return self.depth
  
  def getinSize(self):
    return self.inSize

  def resetAlfa(self):
    self.alfa = 0

  def increaseDepth(self):
    self.depth = self.depth + 1
    self.inSize = 2 * self.inSize
    print("Alfa antes de cambiar de bloque = " + str(self.alfa))
    self.resetAlfa()
    if self.depth >= len(self.blocks):
      self.depth = len(self.blocks) - 1
      self.inSize = int(self.inSize / 2)

    






class ConvDiscriminator(nn.Module):

  def __init__(self, image_size, device):
    super().__init__()
    self.image_size = image_size
    self.device = device
    self.inChan = image_size[0]
    size = image_size[1]
    chan = self.inChan
    bloques = []
    size = size / 2
    chan = 128
    bloques.append(nn.Conv2d(self.inChan, chan, 4, 2, 1).to(device))
    while size > 4:
      bloques.append(nn.Conv2d(chan, 2*chan, 4, 2, 1).to(device))
      bloques.append(nn.BatchNorm2d(2*chan).to(device))
      bloques.append(nn.LeakyReLU(0.2).to(device))
      chan = chan * 2
      size = size / 2

    bloques.append(nn.Conv2d(chan, 1, 4).to(device))
    
    self.disc = nn.ModuleList(bloques)

  def forward(self, image):

    for bloque in self.disc :
      image = bloque(image)

    return image.view((-1))

  def getAlfa(self):
    return 0
  
  def increaseAlfa(self, alfa):
    alfa = alfa + 1


    

class CycleDiscriminator(nn.Module):
    '''
    Discriminator Class
    Structured like the contracting path of the U-Net, the discriminator will
    output a matrix of values classifying corresponding portions of the image as real or fake. 
    Parameters:
        input_channels: the number of image input channels
        hidden_channels: the initial number of discriminator convolutional filters
    '''
    def __init__(self, input_channels, hidden_channels=32):
        super(CycleDiscriminator, self).__init__()
        self.upfeature = bk.FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = bk.ContractingBlock(hidden_channels, use_bn=False, kernel_size=4, activation='lrelu')
        self.contract2 = bk.ContractingBlock(hidden_channels * 2, kernel_size=4, activation='lrelu')
        self.contract3 = bk.ContractingBlock(hidden_channels * 4, kernel_size=4, activation='lrelu')
        self.final = nn.Conv2d(hidden_channels * 8, 1, kernel_size=1)

    def forward(self, x):
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        xn = self.final(x3)
        return xn



