from audioop import bias
import torch
import torch.nn as nn
import torch.nn.functional as F

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