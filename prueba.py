import torch.nn as nn
import torch as torch
import StyleComponents as stylecomp
import CustomLayers as cl
import Discriminators as dc
import Generators as gn
from torchsummary import summary

#shape = torch.empty((1,3,1,1))
#entrada = torch.ones_like(shape)
#bloque = nn.Upsample((9000), mode = 'bilinear')
#salida = bloque(entrada)
#print(salida.shape)

# shape = torch.empty((1,512,4,4))
# entrada = torch.ones_like(shape)
# bloque = stylecomp.AdaIN(256, 496)
# salida = bloque(entrada)
# print(salida.shape)

# b = nn.Conv2d(3,512,1)
# i = torch.rand(32,3,4,4)

# r = b(i)

# print(r.shape)

# i = torch.rand(32,3,8,8)
# b = cl.EqualizedConv2d(3,512,4,2, downscale = True)

# r = b(i)
# print(r.shape)

# dis = dc.DiscriminadorPorBloques(64,3,512,'cuda')


# summary(dis, (3,4,4))

gen = gn.GeneradorCondicional(131+64, 131, 3, 64)
summary(gen, (1,131+64))