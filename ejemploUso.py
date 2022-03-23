import Generators
import Constants
import Discriminators
import torch
import GANUtils
import ImageFunctions
from torch.utils.data import DataLoader
import torch.nn as nn
import Training
import os

# Inicializar agentes

gen = Generators.MicroStyleGANGenerator(Constants.Z_DIM, map_hidden_dim = 1024, w_dim = 496, in_chan = 512, out_chan = Constants.NUM_CHANNELS, kernel_size = 3, hidden_chan = 3).to(Constants.DEVICE)
disc = Discriminators.Discriminador(3,64).to(Constants.DEVICE)

gen_opt = torch.optim.Adam(gen.parameters(), lr = Constants.LR)
disc_opt = torch.optim.Adam(disc.parameters(), lr = Constants.LR)

gen = gen.apply(GANUtils.weights_init)
disc = disc.apply(GANUtils.weights_init)


# Cargar Dataset

dataset, test = ImageFunctions.getDatasets('fruits-360_dataset/fruits-360')

dl = DataLoader(dataset, batch_size=5, shuffle=True)

# Parametros de training

device = Constants.DEVICE
criterion = nn.BCEWithLogitsLoss()
display_step = 500
increase_alpha_step = 100
n_epochs = 1

os.mkdir('prueba-uso')

Training.train_styleGAN(gen, gen_opt, disc, disc_opt, dl, n_epochs, device, criterion
, display_step, increase_alpha_step, 0.2, 'prueba-uso', save=True)