import sys, os

import torch


from StyleGAN.Components import make_dataset, make_logger
import StyleGenerador

import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torchvision.utils import save_image

from StyleGAN.generateStyleMixing import adjustImgRange

curentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(curentdir)
sys.path.append(parentdir)

output_dir = curentdir + "\\Models\\"
img_dir = "C:/Users/Daniela/Documents/TFG/fruits-360_dataset/fruits-360\Training"
logger = make_logger("project", output_dir, 'log')


if __name__ == '__main__':

    loadingPrev = False
    generator_FILE = "GAN_GEN_5_1.pth"
    discriminator_FILE = "GAN_DIS_5_1.pth"
    generatorOptim_FILE = "GAN_GEN_OPTIM_5_1.pth"
    discriminatorOptim_FILE = "GAN_DIS_OPTIM_5_1.pth"
    genShadow = "GAN_GEN_SHADOW_5_1.pth"

    initialDepth = 0

    gen = StyleGenerador.Generator(
        resolution=120,
        conditional=False,
        n_classes=131).to('cuda')

    gen.load_state_dict(torch.load(curentdir + "\\Models\\" + generator_FILE))

    # path for saving the files:
    num_samples = 30
    os.makedirs(output_dir, exist_ok=True)
    latent_size = 512
    resolution = 64
    out_depth = int(np.log2(resolution)) - 2

    for img_num in tqdm(range(1, num_samples + 1)):
        # generate the images:
        with torch.no_grad():
            point = torch.randn(1, latent_size)
            point = (point / point.norm()) * (pow(latent_size, 0.5))

            ##Generamos imagen
            fakeImage = gen(point, depth=out_depth, alpha=1)
            # Ajustamos Colores
            fakeImage = adjustImgRange(fakeImage)

        # Guardar imagen
        save_image(fakeImage, os.path.join(output_dir, str(img_num) + ".png"))

    print("Generated %d images at %s" % (num_samples, output_dir))
