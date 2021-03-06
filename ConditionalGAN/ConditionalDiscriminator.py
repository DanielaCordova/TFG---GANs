import torch.nn as nn
import torch.functional as F

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