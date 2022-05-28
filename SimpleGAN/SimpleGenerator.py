

import torch
import torch.nn as nn
import torch.nn.functional as F

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
                nn.LeakyReLU(0.2, inplace=True).to(device),
            )