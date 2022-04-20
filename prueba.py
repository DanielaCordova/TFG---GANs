import torch.nn as nn
import torch as torch
import StyleComponents as stylecomp

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

b = nn.Conv2d(3,512,1)
i = torch.rand(32,3,4,4)

r = b(i)

print(r.shape)
