
import tqdm
import Constants
import StyleComponents
import torch
import ImageFunctions
import matplotlib.pyplot as plt
import os
from datetime import datetime
from tqdm.auto import tqdm
import numpy as np

def graph_GANS_losses(perdidasGenerador, perdidasDiscriminador, title, dir, save):
    step_bins = 20

    x_axis = sorted([i * step_bins for i in range(len(perdidasGenerador) // step_bins)] * step_bins)
    numEjemplos = (len(perdidasGenerador) // step_bins ) * step_bins
    plt.plot(
        range(numEjemplos // step_bins),
        torch.Tensor(perdidasGenerador[:numEjemplos]).view(-1, step_bins).mean(1),
        label = "Perdida del generador"
    )
    plt.plot(
        range(numEjemplos // step_bins),
        torch.Tensor(perdidasDiscriminador[:numEjemplos]).view(-1, step_bins).mean(1),
        label = "Perdida del Discriminador"
    )
    plt.title(title)
    plt.legend()

    if save:
        os.chdir(dir)
        plt.savefig(datetime.now().strftime("%H-%M-%S-%f %d-%m-%y")+'.png')
        os.chdir('..')

    plt.show()

def train_styleGAN(gen, gen_opt, disc, disc_opt, dataloader, n_epochs, device, criterion, display_step, increase_alpha_step, alpha_step, dir, save = True):

    perdidasGenerador = []
    perdidasDiscriminador = []
    i = 0

    for epoch in range(n_epochs):
        for real, labels in tqdm(dataloader):

            disc_opt.zero_grad()
            ruidoFake = StyleComponents.get_ruido_truncado(Constants.EJEMPLOSTEST, Constants.Z_DIM, Constants.TRUNCATION)

            fake = gen(ruidoFake.to(device))
            real = real.to(device)

            prediccionRealDisc = disc(real.detach())
            prediccionFalsaDisc = disc(fake.detach())

            perdidasFalsasDisc = criterion(prediccionFalsaDisc, torch.zeros_like(prediccionFalsaDisc))
            perdidasRealesDisc = criterion(prediccionRealDisc, torch.ones_like(prediccionRealDisc))
            perdidaDisc = (perdidasFalsasDisc + perdidasRealesDisc) / 2
            perdidaDisc.backward(retain_graph = True)
            disc_opt.step()

            perdidasDiscriminador.append(perdidaDisc.item())

            gen_opt.zero_grad()

            prediccionFalsa = disc(fake)
            perdidaGen = criterion(prediccionFalsa, torch.ones_like(prediccionFalsa))
            perdidaGen.backward()
            gen_opt.step()

            perdidasGenerador.append(perdidaGen.item())

            if i % increase_alpha_step == 0 and i > 0:
                alpha = gen.alpha
                if alpha < 1 :
                    gen.alpha = alpha + alpha_step
                    disc.alpha = alpha + alpha_step

            if i % display_step == 0 and i > 0 :
                promedioGen = sum(perdidasGenerador[-display_step :]) / display_step
                promedioDisc = sum(perdidasDiscriminador[-display_step :]) / display_step

                s = f"Step {i} : GenLoss: {promedioGen} , DiscLoss: {promedioDisc}"


                ImageFunctions.tensor_as_image(fake, "fake", dir)
                ImageFunctions.tensor_as_image(real, "real", dir)

                graph_GANS_losses(perdidasGenerador, perdidasDiscriminador, s, dir, save)
                
            
            i += 1

def train_discriminator(disc, disc_opt, dataset, n_epochs, device, criterion, display_step, increase_alfa_step, alfa_step, dir, save = True, show = False):

    perdidas = []
    ejex = []
    perdidas_plot = []

    print(next(disc.parameters()).is_cuda)

    i = 0
    for e in range(n_epochs):
        for img, tag in tqdm(dataset):

            img = img.to(device)
            tag = tag.to(device)

            #print(img.is_cuda)
            #print(tag.is_cuda)

            #print(torch.cuda.current_device())

            disc_opt.zero_grad()

            res = disc(img)

            perdida = criterion(res, tag.double())

            perdida.backward(retain_graph = True)

            perdidas.append(perdida.item())

            if i % display_step == 0 and i > 0 :

                perdidas_plot.append(sum(perdidas[-display_step :]) / display_step)
                ejex.append(i)

                print("Perdida del disc en el paso " + i + " = " + perdidas_plot[-1])

                plt.plot(np.array(ejex), np.array(perdidas_plot))
                plt.legend()

                if save :
                    os.chdir(dir)
                    plt.savefig(datetime.now().strftime("%H-%M-%S-%f %d-%m-%y")+' iter ' + str(i) + '.png')
                    os.chdir('..')
                if show :
                    plt.show()

                plt.clf()

            if i % increase_alfa_step == 0 and i > 0 and disc.alfa < 1:

                disc.increaseAlfa(alfa_step)

            disc_opt.step()

            i = i+1


