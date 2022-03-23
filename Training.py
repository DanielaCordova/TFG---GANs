
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

def graph_GANS_losses(x, ejeX, title, dir, save, show):

    for ayval , labeld in x :
        plt.plot(np.array(ejeX), np.array(ayval), label=labeld)

    plt.title(title)
    plt.legend()

    if save:
        os.chdir(dir)
        plt.savefig(datetime.now().strftime("%H-%M-%S-%f %d-%m-%y")+'.png')
        os.chdir('..')
    
    if show: 
        plt.show()

    plt.clf()

def train_styleGAN(gen, pggen, gen_opt, disc, pgdisc, disc_opt, dataloader, n_epochs, device, criterion, display_step, 
increase_alpha_step, alfa_step, dir, save = True, show = False):

    perdidasGenerador = []
    perdidasGeneradorPlot = []
    perdidasDiscriminadorReal = []
    perdidasDiscriminadorRealPlot = []
    perdidasDiscriminadorFake = []
    perdidasDiscriminadorFakePlot = []
    perdidasDiscriminadorAvg = []
    perdidasDiscriminadorAvgPlot = []
    ejeX = []
    i = 0

    for epoch in range(n_epochs):
        for real in tqdm(dataloader):

            disc_opt.zero_grad()
            ruidoFake = StyleComponents.get_ruido_truncado(Constants.EJEMPLOSTEST, Constants.Z_DIM, Constants.TRUNCATION).to(device)
            print(ruidoFake.shape)

            fake = gen(ruidoFake.to(device))
            real = real.to(device)

            prediccionRealDisc = disc(real.to(device))
            prediccionFalsaDisc = disc(fake.detach().to(device))

            perdidasFalsasDisc = criterion(prediccionFalsaDisc, torch.zeros_like(prediccionFalsaDisc))
            perdidasDiscriminadorFake.append(perdidasFalsasDisc.item())

            perdidasRealesDisc = criterion(prediccionRealDisc, torch.ones_like(prediccionRealDisc))
            perdidasDiscriminadorReal.append(perdidasRealesDisc.item())

            perdidasDiscriminadorAvg.append((perdidasRealesDisc.item() + perdidasFalsasDisc.item()) / 2)

            perdidas = (perdidasFalsasDisc + perdidasRealesDisc)/2
            perdidas.backward(retain_graph = True)
            disc_opt.step()

            gen_opt.zero_grad()

            ###
            # Cambiado fake.detach
            ###
            prediccionFalsa = disc(fake)
            ##
            # Detatch pendiente de poner para no alterar el discriminador
            ## 

            perdidaGen = criterion(prediccionFalsa, torch.ones_like(prediccionFalsa))
            perdidaGen.backward()
            gen_opt.step()

            perdidasGenerador.append(perdidaGen.item())

            if i % increase_alpha_step == 0 and i > 0 and pggen:
                gen.increaseAlfa(alfa_step)

            if i % increase_alpha_step == 0 and i > 0 and pgdisc:
                disc.increaseAlfa(alfa_step)

            if i % display_step == 0 and i > 0 :
                promedioGen = sum(perdidasGenerador[-display_step :]) / display_step
                promedioDiscFake = sum(perdidasDiscriminadorFake[-display_step :]) / display_step
                promedioDiscReal = sum(perdidasDiscriminadorReal[-display_step :]) / display_step
                promedioDiscAvg = sum(perdidasDiscriminadorAvg[-display_step :]) / display_step

                perdidasGeneradorPlot.append(promedioGen)
                perdidasDiscriminadorFakePlot.append(promedioDiscFake)
                perdidasDiscriminadorRealPlot.append(promedioDiscReal)
                perdidasDiscriminadorAvgPlot.append(promedioDiscAvg)

                ejeX.append(i)

                s = "Step " + str(i) + " : GenLoss: " + str(promedioGen) + " , DiscLoss: " + str(promedioDiscAvg)

                print(s)

                ImageFunctions.tensor_as_image(fake, i, "fake", dir, save = True, show = show)
                ImageFunctions.tensor_as_image(real, i, "real", dir, save = True, show = show)

                ar_plot = [(perdidasGeneradorPlot, "perdida generador"), (perdidasDiscriminadorAvgPlot, "perdida discriminador avg"), (perdidasDiscriminadorFakePlot,"perdida discriminador fakes"), 
                (perdidasDiscriminadorRealPlot, "perdida discriminador reales")]

                graph_GANS_losses(ar_plot, ejeX, s, dir, save, show)
                
            
            i += 1

def train_styleGAN_altern(gen, gen_opt, disc, disc_opt, dataloader, n_epochs, device, criterion, display_step, increase_alpha_step, alfa_step, first_training, dif, dir, save = True, show = False):

    perdidasGenerador = []
    perdidasGeneradorPlot = []
    perdidasDiscriminadorReal = []
    perdidasDiscriminadorRealPlot = []
    perdidasDiscriminadorFake = []
    perdidasDiscriminadorFakePlot = []
    perdidasDiscriminadorAvg = []
    perdidasDiscriminadorAvgPlot = []
    ejeX = []
    i = 0

    for epoch in range(first_training):
        for real, label in tqdm(dataloader):

            disc_opt.zero_grad()
            ruidoFake = StyleComponents.get_ruido_truncado(Constants.EJEMPLOSTEST, Constants.Z_DIM, Constants.TRUNCATION).to(device)

            fake = gen(ruidoFake.to(device))
            real = real.to(device)

            prediccionRealDisc = disc(real.to(device))
            prediccionFalsaDisc = disc(fake.detach().to(device))

            perdidasFalsasDisc = criterion(prediccionFalsaDisc, torch.zeros_like(prediccionFalsaDisc))
            perdidasDiscriminadorFake.append(perdidasFalsasDisc.item())

            perdidasRealesDisc = criterion(prediccionRealDisc, torch.ones_like(prediccionRealDisc))
            perdidasDiscriminadorReal.append(perdidasRealesDisc.item())

            perdidasDiscriminadorAvg.append((perdidasRealesDisc.item() + perdidasFalsasDisc.item()) / 2)

            perdidas = (perdidasFalsasDisc + perdidasRealesDisc)/2
            perdidas.backward(retain_graph = True)
            disc_opt.step()

            gen_opt.zero_grad()

            prediccionFalsa = disc(fake)
            perdidaGen = criterion(prediccionFalsa, torch.ones_like(prediccionFalsa))
            perdidaGen.backward()
            gen_opt.step()

            perdidasGenerador.append(perdidaGen.item())

            if i % increase_alpha_step == 0 and i > 0:
                gen.increaseAlfa(alfa_step)
                disc.increaseAlfa(alfa_step)

            if i % display_step == 0 and i > 0 :
                promedioGen = sum(perdidasGenerador[-display_step :]) / display_step
                promedioDiscFake = sum(perdidasDiscriminadorFake[-display_step :]) / display_step
                promedioDiscReal = sum(perdidasDiscriminadorReal[-display_step :]) / display_step
                promedioDiscAvg = sum(perdidasDiscriminadorAvg[-display_step :]) / display_step

                perdidasGeneradorPlot.append(promedioGen)
                perdidasDiscriminadorFakePlot.append(promedioDiscFake)
                perdidasDiscriminadorRealPlot.append(promedioDiscReal)
                perdidasDiscriminadorAvgPlot.append(promedioDiscAvg)

                ejeX.append(i)

                s = "Step " + str(i) + " : GenLoss: " + str(promedioGen) + " , DiscLoss: " + str(promedioDiscAvg)

                print(s)

                ImageFunctions.tensor_as_image(fake, i, "fake", dir, save = True, show = show)
                ImageFunctions.tensor_as_image(real, i, "real", dir, save = True, show = show)

                ar_plot = [(perdidasGeneradorPlot, "perdida generador"), (perdidasDiscriminadorAvgPlot, "perdida discriminador avg"), (perdidasDiscriminadorFakePlot,"perdida discriminador fakes"), 
                (perdidasDiscriminadorRealPlot, "perdida discriminador reales")]

                graph_GANS_losses(ar_plot, ejeX, s, dir, save, show)
                
            
            i += 1
    
    it = iter(dataloader)

    for epoch in range(n_epochs - first_training):

        if perdidasDiscriminadorAvg[-1] > dif + perdidasGenerador[-1]:
            # Entrenar discriminador

            real, labels = next(it)

            disc_opt.zero_grad()
            ruidoFake = StyleComponents.get_ruido_truncado(Constants.EJEMPLOSTEST, Constants.Z_DIM, Constants.TRUNCATION).to(device)

            fake = gen(ruidoFake.to(device))
            real = real.to(device)

            prediccionRealDisc = disc(real.to(device))
            prediccionFalsaDisc = disc(fake.detach().to(device))

            perdidasFalsasDisc = criterion(prediccionFalsaDisc, torch.zeros_like(prediccionFalsaDisc))
            perdidasDiscriminadorFake.append(perdidasFalsasDisc.item())

            perdidasRealesDisc = criterion(prediccionRealDisc, torch.ones_like(prediccionRealDisc))
            perdidasDiscriminadorReal.append(perdidasRealesDisc.item())

            perdidasDiscriminadorAvg.append((perdidasRealesDisc.item() + perdidasFalsasDisc.item()) / 2)

            perdidas = (perdidasFalsasDisc + perdidasRealesDisc)/2
            perdidas.backward(retain_graph = True)
            disc_opt.step()

            perdidasGenerador.append(perdidasGenerador[-1])

        elif perdidasGenerador[-1] > dif + perdidasDiscriminadorAvg[-1]:
            # Entrenar generador
            ruidoFake = StyleComponents.get_ruido_truncado(Constants.EJEMPLOSTEST, Constants.Z_DIM, Constants.TRUNCATION).to(device)

            gen_opt.zero_grad()

            fake = gen(ruidoFake.to(device))

            prediccionFalsaDisc = disc(fake.detach().to(device))

            perdidasFalsasDisc = criterion(prediccionFalsaDisc, torch.zeros_like(prediccionFalsaDisc))
            perdidasDiscriminadorFake.append(perdidasFalsasDisc.item())

            perdidasDiscriminadorReal.append(perdidasDiscriminadorReal[-1])

            perdidasDiscriminadorAvg.append((perdidasDiscriminadorReal[-1] + perdidasFalsasDisc.item()) / 2) 

            prediccionFalsa = disc(fake)
            perdidaGen = criterion(prediccionFalsa, torch.ones_like(prediccionFalsa))
            perdidaGen.backward(retain_graph=True)
            gen_opt.step()

            perdidasGenerador.append(perdidaGen.item())

        else:
            # Entrenar los dos
            disc_opt.zero_grad()
            ruidoFake = StyleComponents.get_ruido_truncado(Constants.EJEMPLOSTEST, Constants.Z_DIM, Constants.TRUNCATION).to(device)

            fake = gen(ruidoFake.to(device))
            real = real.to(device)

            prediccionRealDisc = disc(real.to(device))
            prediccionFalsaDisc = disc(fake.detach().to(device))

            perdidasFalsasDisc = criterion(prediccionFalsaDisc, torch.zeros_like(prediccionFalsaDisc))
            perdidasDiscriminadorFake.append(perdidasFalsasDisc.item())

            perdidasRealesDisc = criterion(prediccionRealDisc, torch.ones_like(prediccionRealDisc))
            perdidasDiscriminadorReal.append(perdidasRealesDisc.item())

            perdidasDiscriminadorAvg.append((perdidasRealesDisc.item() + perdidasFalsasDisc.item()) / 2)

            perdidas = (perdidasFalsasDisc + perdidasRealesDisc)/2
            perdidas.backward(retain_graph = True)
            disc_opt.step()

            gen_opt.zero_grad()

            prediccionFalsa = disc(fake)
            perdidaGen = criterion(prediccionFalsa, torch.ones_like(prediccionFalsa))
            perdidaGen.backward()
            gen_opt.step()

            perdidasGenerador.append(perdidaGen.item())
        
        if i % increase_alpha_step == 0 and i > 0:
            gen.increaseAlfa(alfa_step)
            disc.increaseAlfa(alfa_step)

        if i % display_step == 0 and i > 0 :
            promedioGen = sum(perdidasGenerador[-display_step :]) / display_step
            promedioDiscFake = sum(perdidasDiscriminadorFake[-display_step :]) / display_step
            promedioDiscReal = sum(perdidasDiscriminadorReal[-display_step :]) / display_step
            promedioDiscAvg = sum(perdidasDiscriminadorAvg[-display_step :]) / display_step

            perdidasGeneradorPlot.append(promedioGen)
            perdidasDiscriminadorFakePlot.append(promedioDiscFake)
            perdidasDiscriminadorRealPlot.append(promedioDiscReal)
            perdidasDiscriminadorAvgPlot.append(promedioDiscAvg)

            ejeX.append(i)

            s = "Step " + str(i) + " : GenLoss: " + str(promedioGen) + " , DiscLoss: " + str(promedioDiscAvg)

            print(s)

            ImageFunctions.tensor_as_image(fake, i, "fake", dir, save = True, show = show)
            ImageFunctions.tensor_as_image(real, i, "real", dir, save = True, show = show)

            ar_plot = [(perdidasGeneradorPlot, "perdida generador"), (perdidasDiscriminadorAvgPlot, "perdida discriminador avg"), (perdidasDiscriminadorFakePlot,"perdida discriminador fakes"), 
            (perdidasDiscriminadorRealPlot, "perdida discriminador reales")]

            graph_GANS_losses(ar_plot, ejeX, s, dir, save, show)
            

        i += 1


def train_discriminator(disc, disc_opt, dataset, n_epochs, device, criterion, display_step, increase_alfa_step, alfa_step, dir, save = True, show = False, prog = True):

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

                print("Perdida del disc en el paso " + str(i) + " = " + str(perdidas_plot[-1]))

                plt.plot(np.array(ejex), np.array(perdidas_plot))
                plt.title("Bloque " + str(disc.bloque_act) + " con alfa " + str(disc.bloques[disc.bloque_act].getAlfa()))
                plt.legend()

                if save :
                    os.chdir(dir)
                    plt.savefig(datetime.now().strftime("%H-%M-%S-%f %d-%m-%y")+' iter ' + str(i) + '.png')
                    os.chdir('..')
                if show :
                    plt.show()

                plt.clf()

            if prog:
                if i % increase_alfa_step == 0 and i > 0 and disc.alfa < 1 :

                    disc.increaseAlfa(alfa_step)

            disc_opt.step()

            i = i+1

def trainGenerator(gen, pggen, gen_opt, disc, pgdisc, disc_opt, dataloader, n_epochs, device, criterion, display_step, 
increase_alpha_step, alfa_step, dir, save = True, show = False):

    perdidasGenerador = []
    perdidasGeneradorPlot = []
    perdidasDiscriminadorReal = []
    perdidasDiscriminadorRealPlot = []
    perdidasDiscriminadorFake = []
    perdidasDiscriminadorFakePlot = []
    perdidasDiscriminadorAvg = []
    perdidasDiscriminadorAvgPlot = []
    ejeX = []
    i = 0

    for epoch in range(n_epochs):
        for real in tqdm(dataloader):

            disc_opt.zero_grad()
            ruidoFake = torch.randn(Constants.EJEMPLOSTEST, 1024)

            fake = gen(ruidoFake.to(device))

            print(fake.shape)

            real = real.to(device)

            prediccionRealDisc = disc(real.to(device))
            prediccionFalsaDisc = disc(fake.detach().to(device))

            perdidasFalsasDisc = criterion(prediccionFalsaDisc, torch.zeros_like(prediccionFalsaDisc))
            perdidasDiscriminadorFake.append(perdidasFalsasDisc.item())

            perdidasRealesDisc = criterion(prediccionRealDisc, torch.ones_like(prediccionRealDisc))
            perdidasDiscriminadorReal.append(perdidasRealesDisc.item())

            perdidasDiscriminadorAvg.append((perdidasRealesDisc.item() + perdidasFalsasDisc.item()) / 2)

            gen_opt.zero_grad()

            ###
            # Cambiado fake.detach
            ###
            prediccionFalsa = disc(fake)
            ##
            # Detatch pendiente de poner para no alterar el discriminador
            ## 

            perdidaGen = criterion(prediccionFalsa, torch.ones_like(prediccionFalsa))
            perdidaGen.backward()
            gen_opt.step()

            perdidasGenerador.append(perdidaGen.item())

            if i % increase_alpha_step == 0 and i > 0 and pggen:
                gen.increaseAlfa(alfa_step)

            if i % increase_alpha_step == 0 and i > 0 and pgdisc:
                disc.increaseAlfa(alfa_step)

            if i % display_step == 0 and i > 0 :
                promedioGen = sum(perdidasGenerador[-display_step :]) / display_step
                promedioDiscFake = sum(perdidasDiscriminadorFake[-display_step :]) / display_step
                promedioDiscReal = sum(perdidasDiscriminadorReal[-display_step :]) / display_step
                promedioDiscAvg = sum(perdidasDiscriminadorAvg[-display_step :]) / display_step

                perdidasGeneradorPlot.append(promedioGen)
                perdidasDiscriminadorFakePlot.append(promedioDiscFake)
                perdidasDiscriminadorRealPlot.append(promedioDiscReal)
                perdidasDiscriminadorAvgPlot.append(promedioDiscAvg)

                ejeX.append(i)

                s = "Step " + str(i) + " : GenLoss: " + str(promedioGen) + " , DiscLoss: " + str(promedioDiscAvg)

                print(s)

                ImageFunctions.tensor_as_image(fake, i, "fake", dir, save = True, show = show)
                ImageFunctions.tensor_as_image(real, i, "real", dir, save = True, show = show)

                ar_plot = [(perdidasGeneradorPlot, "perdida generador"), (perdidasDiscriminadorAvgPlot, "perdida discriminador avg"), (perdidasDiscriminadorFakePlot,"perdida discriminador fakes"), 
                (perdidasDiscriminadorRealPlot, "perdida discriminador reales")]

                graph_GANS_losses(ar_plot, ejeX, s, dir, save, show)
                
            
            i += 1

