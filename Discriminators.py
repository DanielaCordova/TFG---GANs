import torch.nn as nn
import torch.functional as F
import torch.nn.functional as nnF
import math



class DiscriminadorCondicional(nn.Module):

  def __init__(self, inChan=1, hiddenDim=64):
    super(DiscriminadorCondicional, self).__init__()
    self.discriminador = nn.Sequential(
        self.generar_bloque_discriminador(inChan, hiddenDim),
        self.generar_bloque_discriminador(hiddenDim, hiddenDim * 2),
        self.generar_bloque_discriminador(hiddenDim*2, inChan,ultimaCapa= True )
    )

  def generar_bloque_discriminador(self, inChan, outChan, kernel = 4, stride = 2, ultimaCapa = False):
    if ultimaCapa :
      return nn.Sequential(
          nn.Conv2d(inChan, outChan, kernel, stride),
      )
    else :
      return nn.Sequential(
          nn.Conv2d(inChan, outChan, kernel, stride),
          nn.BatchNorm2d(outChan),
          nn.LeakyReLU(0.2, inplace = True),
      )

  def forward(self, image):
    pred = self.discriminador(image)
    return pred.view(len(pred), -1)

#############################################################

class DiscriminadorStyle(nn.Module):

  def __init__(self, inChan, outChan, kernel, hiddenDim, alpha):
    super().__init__()
    self.alpha = alpha
    self.block64to32 = nn.Conv2d(inChan,outChan,kernel, stride = 2) 
    self.block32to16 = nn.Conv2d(inChan,outChan,kernel, stride = 2) 
    self.block16to8  = nn.Conv2d(inChan,outChan,kernel, stride = 2) 
    self.block8to4   = nn.Conv2d(inChan,outChan,kernel, stride = 2) 
    self.block32_to_image = nn.Conv2d(inChan,outChan, kernel_size = kernel, padding = 1)
    self.block16_to_image = nn.Conv2d(inChan,outChan, kernel_size = kernel, padding = 1)
    self.block8_to_image = nn.Conv2d(inChan,outChan, kernel_size = 2, padding = 1)
    self.block4_to_image = nn.Conv2d(inChan,outChan, kernel_size = 1, padding = 1)
    self.fq      = nn.Sequential(
        self.generar_bloque_fq_discriminador(inChan, hiddenDim),
        self.generar_bloque_fq_discriminador(hiddenDim, hiddenDim*2),
        self.generar_bloque_fq_discriminador(hiddenDim*2, inChan),
        nn.BatchNorm2d(outChan),
        nn.LeakyReLU(0.2, inplace = True)
    )

  def forward(self, image):
    x32 = self.block64to32(image)
    img32 = self.block32_to_image(x32)
    ds32  = self.downsample_to_match_size(image, img32)
    interpolation32 = self.alpha * img32 + (1-self.alpha) * ds32

    x16 = self.block32to16(interpolation32)
    img16 = self.block16_to_image(x16)
    ds16  = self.downsample_to_match_size(interpolation32, img16)
    interpolation16 = self.alpha * img16 + (1-self.alpha) * ds16

    x8 = self.block16to8(interpolation16)
    img8 = self.block8_to_image(x8)
    ds8  = self.downsample_to_match_size(interpolation16, img8)
    interpolation8 = self.alpha * img8 + (1-self.alpha) * ds8

    x4 = self.block8to4(interpolation8)
    img4 = self.block4_to_image(x4)
    ds4  = self.downsample_to_match_size(interpolation8, img4)
    interpolation4 = self.alpha * img4 + (1-self.alpha) * ds4

    ret = self.fq(interpolation4)

    return ret.view(len(ret), -1)

  def setAlpha(self, alfa):
    self.alpha = alfa

  def generar_bloque_fq_discriminador(self, inChan, outChan, kernel = 1, stride = 2, ultimaCapa = False):
    return nn.Sequential(
          nn.Conv2d(inChan, outChan, kernel, stride),
          nn.BatchNorm2d(outChan),
          nn.LeakyReLU(0.2, inplace = True),
      )
  
  def downsample_to_match_size(self, bigger_image, smaller_image):
    '''
    Function for upsampling an image to the size of another: Given a two images (smaller and bigger), 
    upsamples the first to have the same dimensions as the second.
    Parameters:
        smaller_image: the smaller image whose dimensions will be upsampled to
        bigger_image: the bigger image to downsample 
    '''
    return F.interpolate(bigger_image, size=smaller_image.shape[-2:], mode='bilinear')

#######################################################

class DiscriminadorStyleMenosListo(nn.Module):

  def __init__(self, inChan, outChan, kernel, hiddenDim, alpha):
    super().__init__()
    self.alpha = alpha
    self.block64to32 = nn.Conv2d(inChan,outChan,kernel, stride = 2) 
    self.block32to16 = nn.Conv2d(inChan,outChan,kernel, stride = 2) 
    self.block16to8  = nn.Conv2d(inChan,outChan,kernel, stride = 2) 
    self.block8to4   = nn.Conv2d(inChan,outChan,kernel, stride = 2) 
    self.block32_to_image = nn.Conv2d(inChan,outChan, kernel_size = kernel, padding = 1)
    self.block16_to_image = nn.Conv2d(inChan,outChan, kernel_size = kernel, padding = 1)
    self.block8_to_image = nn.Conv2d(inChan,outChan, kernel_size = 2, padding = 1)
    self.fq      = nn.Sequential(
        self.generar_bloque_fq_discriminador(inChan, hiddenDim),
        self.generar_bloque_fq_discriminador(hiddenDim, hiddenDim*2),
        self.generar_bloque_fq_discriminador(hiddenDim*2, inChan),
        nn.BatchNorm2d(outChan),
        nn.LeakyReLU(0.2, inplace = True)
    )

  def forward(self, image):
    x32 = self.block64to32(image)
    img32 = self.block32_to_image(x32)
    ds32  = self.downsample_to_match_size(image, img32)
    interpolation32 = self.alpha * img32 + (1-self.alpha) * ds32

    x16 = self.block32to16(interpolation32)
    img16 = self.block16_to_image(x16)
    ds16  = self.downsample_to_match_size(interpolation32, img16)
    interpolation16 = self.alpha * img16 + (1-self.alpha) * ds16

    x8 = self.block16to8(interpolation16)
    img8 = self.block8_to_image(x8)
    ds8  = self.downsample_to_match_size(interpolation16, img8)
    interpolation8 = self.alpha * img8 + (1-self.alpha) * ds8

    return self.fq(interpolation8)

  def setAlpha(self, alfa):
    self.alpha = alfa

  def generar_bloque_fq_discriminador(self, inChan, outChan, kernel = 1, stride = 2, ultimaCapa = False):
    return nn.Sequential(
          nn.Conv2d(inChan, outChan, kernel, stride),
          nn.BatchNorm2d(outChan),
          nn.LeakyReLU(0.2, inplace = True),
      )
  
  def downsample_to_match_size(self, bigger_image, smaller_image):
    '''
    Function for upsampling an image to the size of another: Given a two images (smaller and bigger), 
    upsamples the first to have the same dimensions as the second.
    Parameters:
        smaller_image: the smaller image whose dimensions will be upsampled to
        bigger_image: the bigger image to downsample 
    '''
    return F.interpolate(bigger_image, size=smaller_image.shape[-2:], mode='bilinear')

######################################################

class DiscriminadorStyleListo(nn.Module):

  def __init__(self, inChan, outChan, kernel, hiddenDim, alpha):
    super().__init__()
    self.alpha = alpha
    self.block64to32 = nn.Conv2d(inChan,outChan,kernel, stride = 2) 
    self.block32to16 = nn.Conv2d(inChan,outChan,kernel, stride = 2) 
    self.block16to8  = nn.Conv2d(inChan,outChan,kernel, stride = 2) 
    self.block8to4   = nn.Conv2d(inChan,outChan,kernel, stride = 2) 
    self.block32_to_image = nn.Conv2d(inChan,outChan, kernel_size = kernel, padding = 1)
    self.block16_to_image = nn.Conv2d(inChan,outChan, kernel_size = kernel, padding = 1)
    self.block8_to_image = nn.Conv2d(inChan,outChan, kernel_size = 2, padding = 1)
    self.block4_to_image = nn.Conv2d(inChan,outChan, kernel_size = 1, padding = 1)
    self.fq      = nn.Sequential(
        self.generar_bloque_fq_discriminador(inChan, hiddenDim),
        self.generar_bloque_fq_discriminador(hiddenDim, hiddenDim*2),
        self.generar_bloque_fq_discriminador(hiddenDim*2, hiddenDim*4),
        self.generar_bloque_fq_discriminador(hiddenDim*4, hiddenDim*8),
        self.generar_bloque_fq_discriminador(hiddenDim*8, hiddenDim*4),
        self.generar_bloque_fq_discriminador(hiddenDim*4, hiddenDim*2),
        self.generar_bloque_fq_discriminador(hiddenDim*2, inChan),
        nn.BatchNorm2d(outChan),
        nn.LeakyReLU(0.2, inplace = True)
    )

  def forward(self, image):
    x32 = self.block64to32(image)
    img32 = self.block32_to_image(x32)
    ds32  = self.downsample_to_match_size(image, img32)
    interpolation32 = self.alpha * img32 + (1-self.alpha) * ds32

    x16 = self.block32to16(interpolation32)
    img16 = self.block16_to_image(x16)
    ds16  = self.downsample_to_match_size(interpolation32, img16)
    interpolation16 = self.alpha * img16 + (1-self.alpha) * ds16

    x8 = self.block16to8(interpolation16)
    img8 = self.block8_to_image(x8)
    ds8  = self.downsample_to_match_size(interpolation16, img8)
    interpolation8 = self.alpha * img8 + (1-self.alpha) * ds8

    x4 = self.block8to4(interpolation8)
    img4 = self.block4_to_image(x4)
    ds4  = self.downsample_to_match_size(interpolation8, img4)
    interpolation4 = self.alpha * img4 + (1-self.alpha) * ds4

    return self.fq(interpolation4)

  def setAlpha(self, alfa):
    self.alpha = alfa

  def generar_bloque_fq_discriminador(self, inChan, outChan, kernel = 1, stride = 2, ultimaCapa = False):
    return nn.Sequential(
          nn.Conv2d(inChan, outChan, kernel, stride),
          nn.BatchNorm2d(outChan),
          nn.LeakyReLU(0.2, inplace = True),
      )
  
  def downsample_to_match_size(self, bigger_image, smaller_image):
    '''
    Function for upsampling an image to the size of another: Given a two images (smaller and bigger), 
    upsamples the first to have the same dimensions as the second.
    Parameters:
        smaller_image: the smaller image whose dimensions will be upsampled to
        bigger_image: the bigger image to downsample 
    '''
    return F.interpolate(bigger_image, size=smaller_image.shape[-2:], mode='bilinear')


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
    if self.bloque_act < len(self.bloques):
      if self.bloques[self.bloque_act].getAlfa() < 1 :
        self.bloques[self.bloque_act].increaseAlfa(inc)
        #print("El bloque " + str(self.bloque_act) + " tiene alfa = " + str(self.bloques[self.bloque_act].getAlfa()))
        if self.bloques[self.bloque_act].getAlfa() >= 1 :
          self.bloque_act = self.bloque_act + 1

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


    







