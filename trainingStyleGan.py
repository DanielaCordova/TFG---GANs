
import GANUtils
import Generators
import Discriminators
import Training
import torch
import torch.nn as nn
import Constants
from torchvision.datasets import ImageFolder
import ImageFunctions
from torch.utils.data import DataLoader 
import os

# Modulos

disc = Discriminators.ConvDiscriminator((3,64,64), 'cuda')
# #gen  = StyleGenerator.StyleGenerador(
#     z_dim = 128,
#     map_hidden_dim=1024,
#     w_dim=496,
#     inChan = 512,
#     out_chan = 3,
#     kernel_size = 3,
#     hidden_chan = 3,
#     device = 'cuda'
# ).to('cuda')

# gen = Generators.StyleGenerador(
#     zDim=128,
#     inChan=512,
#     mappingLayersDim=1024,
#     disentNoiseDim=496,
#     outChan=3,
#     kernel=3,
#     convHiddenChan=9,
#     device='cuda'
# ).to('cuda')

gen = Generators.GeneradorCondicional(128,3,64)

disc_opt = torch.optim.Adam(disc.parameters(), lr=Constants.LR)
gen_opt = torch.optim.Adam(gen.parameters(), lr=Constants.LR)

disc.apply(GANUtils.weights_init)
gen.apply(GANUtils.weights_init)

# Dataset

ds = torch.load('preprocessedDataset.pt')

dataLoader = DataLoader(ds, batch_size=Constants.BATCH_SIZE, shuffle=True)

# Carpeta para resultados

training_dir = 'training-Style-genCond-discCond'

os.mkdir(training_dir)

# Entrenamiento

criterion = nn.BCEWithLogitsLoss()
n_epochs = 1000000
display_step = 50
increase_alfa_step = 100
alfa_step = 0.02

Training.train_styleGAN(gen, False, gen_opt, disc, False, disc_opt, dataLoader, n_epochs, 'cuda', 
                       criterion, display_step, increase_alfa_step, alfa_step, training_dir,
                       save = True)

# Training.trainGenerator(gen, False, gen_opt, disc, False, disc_opt, dataLoader, n_epochs, 'cuda', 
#                       criterion, display_step, increase_alfa_step, alfa_step, training_dir,
#                        save = True)