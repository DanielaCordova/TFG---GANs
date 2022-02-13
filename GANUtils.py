import torch
import torch.nn as nn
import torch.functional as F

def get_InputVector_paraEtiquetar(etiquetas, numClases):
  return F.one_hot(etiquetas,numClases)

def combinarVectores(x, y):
  return torch.cat((x.float(),y.float()), 1)

def getNoise(ejemplos, dim, device = 'cpu'):
  return torch.randn(ejemplos, dim, device=device)

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)