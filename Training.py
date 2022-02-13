
import tqdm
import Constants
import StyleComponents
import torch
import ImageFunctions
import matplotlib.pyplot as plt
import os
from datetime import datetime
from tqdm.auto import tqdm

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
    plt.show()
    if save:
        os.chdir(dir)
        plt.savefig(datetime.now().strftime("%H-%M-%S-%f %d-%m-%y")+'.png')
        os.chdir('..')

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
                    gen_alpha = alpha + alpha_step
                    disc_alpha = alpha + alpha_step

            if i % display_step == 0 and i > 0 :
                promedioGen = sum(perdidasGenerador[-display_step :]) / display_step
                promedioDisc = sum(perdidasDiscriminador[-display_step :]) / display_step

                s = f"Step {i} : GenLoss: {promedioGen} , DiscLoss: {promedioDisc}"


                ImageFunctions.tensor_as_image(fake, "fake", dir)
                ImageFunctions.tensor_as_image(real, "real", dir)

                graph_GANS_losses(perdidasGenerador, perdidasDiscriminador, s, dir, save)
                
            
            i += 1



