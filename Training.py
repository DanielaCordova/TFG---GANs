
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
import torch
import torch.nn as nn

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

def saveCheckpoint(dir, gen, disc, gen_opt, disc_opt, gen_loss, disc_loss, epoch, iter):
    os.chdir(dir)

    g_s = 'gen_' + str(iter) + '.tar'
    d_s = 'disc_' + str(iter) + '.tar'

    torch.save({
       'epoch' : epoch ,
       'iter' : iter,
       'model_state_dict' : gen.state_dict(), 
       'optimizer_state_dict' : gen_opt.state_dict(),
       'loss' : gen_loss
    }, g_s)

    torch.save({
        'epoch' : epoch,
        'iter' : iter,
        'model_state_dict' : disc.state_dict(),
        'optimizer_state_dict' : disc_opt.state_dict(),
        'loss' : disc_loss
    }, d_s)

    os.chdir('..')

def train_styleGAN(gen, pggen, gen_opt, disc, pgdisc, disc_opt, dataloader, n_epochs, device, criterion, display_step, 
increase_alpha_step, alfa_step, dir, checkpoint_step, save = True, show = False):

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
            
            if i % checkpoint_step == 0 and i > 0:
                saveCheckpoint(dir,gen,disc,gen_opt,disc_opt, perdidasGenerador[-1], perdidasDiscriminadorAvg[-1], epoch, i)
            
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

class Style_Trainer:
    def __init__(self, dataloader, generator, discriminator, criterion, dir, log_step, verb, prog, alfa_step, increase_alfa_step, device, 
    checksave = False, save_step = None, load = False, load_dir = None, gen_load = None, disc_load = None):
        
        self.disc = discriminator
        self.gen  = generator
        self.criterion = criterion

        self.device = device

        self.dataloader = dataloader

        self.resdir = dir
        self.log_step = log_step

        self.iter = 0

        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=Constants.LR)
        self.disc_opt = torch.optim.Adam(self.disc.parameters(), lr=Constants.LR)
        self.downsampler = nn.AvgPool2d(4,2,1)

        self.verb = verb
        self.gen_loss = []
        self.gen_loss_plot = []
        self.disc_loss = []
        self.disc_loss_plot = []
        if self.verb :
            self.disc_fake_loss = []
            self.disc_fake_loss_plot = []
            self.disc_real_loss = []
            self.disc_real_loss_plot = []
        self.ejeX = []

        if load :
            os.chdir(load_dir)

            c_g = torch.load(gen_load)
            c_d = torch.load(disc_load)

            self.disc.load_state_dict(c_d['model_state_dict'])
            self.disc_opt.load_state_dict(c_d['optimizer_state_dict'])
            self.disc_loss.append( c_d['loss'] )

            self.gen.load_state_dict(c_g['model_state_dict'])
            self.gen_opt.load_state_dict(c_g['optimizer_state_dict'])
            self.gen_loss.append( c_g['loss'] )

            os.chdir('..')

        if checksave:
            self.save_step = save_step

        if prog :
            self.alfa_step = alfa_step
            self.increase_alfa_step = increase_alfa_step
            self.prog = prog
        else:
            self.prog = False
            self.alfa_step = 0
            self.increase_alfa_step = 0


    def enable_training(self, model, flag):
        for p in model.parameters():
            p.requires_grad = flag

    def check_training_params(self, model, flag):
        for p in model.parameters():
            assert(p.requires_grad == flag)

    def train_gen(self):

        self.enable_training(self.gen, True)
        self.enable_training(self.disc,False)

        noise = torch.randn(Constants.BATCH_SIZE, 512).to(self.device)

        fake = self.gen(noise)
        fake_pred = self.disc(fake)

        loss = self.criterion(fake_pred, torch.zeros_like(fake_pred))

        self.check_training_params(self.gen, True)
        self.check_training_params(self.disc, False)

        self.gen_opt.zero_grad()
        loss.backward(retain_graph = True)
        self.gen_opt.step()
        
        self.check_training_params(self.gen, True)
        self.check_training_params(self.disc, False)

        return (loss.item(), fake)


    def train_disc(self, real_data):
        
        real_data = real_data.to(self.device)

        in_size = self.disc.getinSize()
        while real_data.shape[2] != in_size:
            real_data = self.downsampler(real_data)

        self.enable_training(self.gen, False)
        self.enable_training(self.disc,True)

        noise = torch.randn(Constants.BATCH_SIZE, 512).to('cuda')

        fake = self.gen(noise)
        fake_pred = self.disc(fake)

        real_pred = self.disc(real_data)

        perdidasFalsas = self.criterion(fake_pred, torch.zeros_like(fake_pred))
        perdidasReales = self.criterion(real_pred, torch.ones_like(real_pred))
        perdidasTotales = (perdidasFalsas + perdidasReales) / 2

        self.disc_opt.zero_grad()
        perdidasTotales.backward()
        self.disc_opt.step()

        return (perdidasTotales.item(), perdidasFalsas.item(), perdidasReales.item())

    def epoch(self):

        for real in tqdm(self.dataloader):

            g_loss, generated = self.train_gen()
            t_loss, f_loss, r_loss = self.train_disc(real)

            self.gen_loss.append(g_loss)
            self.disc_loss.append(t_loss)

            if self.verb:
                self.disc_fake_loss.append(f_loss)
                self.disc_real_loss.append(r_loss)

            self.ejeX.append(self.iter)

            if self.iter % self.log_step == 0 and self.iter > 0 :
                self.plot_losses()
                self.save_results(real, generated)
                
            
            if self.prog and self.iter % self.increase_alfa_step and self.iter > 0 :
                self.gen.increaseAlfa(self.alfa_step)
                self.disc.increaseAlfa(self.alfa_step)
            
            self.iter = self.iter + 1

            if self.iter % self.save_step == 0:
                self.saveCheckpoint(self.resdir, self.gen, self.disc, self.gen_opt, self.disc_opt, self.gen_loss[-1], self.disc_loss[-1], 0, self.iter)

    
    def gen_epoch(self):

        for _ in range(self.dataloader.__len__()):
            g_loss, fake = self.train_gen()

            self.gen_loss.append(g_loss)

            if self.iter % self.log_step == 0 and self.iter > 0:

                prom = sum(self.gen_loss[-self.log_step: ]) / self.log_step
                self.gen_loss_plot.append( prom )

                self.ejeX.append(self.iter)
                
                print("ITER " + str(self.iter) + " GENLOSS " + str(self.gen_loss_plot[-1]))

                os.chdir(self.resdir)
                
                plt.plot(np.array(self.ejeX), np.array(self.gen_loss_plot), label = "Gen Loss")
                plt.title("Gen Loss: " + str(self.gen_loss_plot[-1]))
                plt.legend()
                plt.savefig("Gen " + datetime.now().strftime("%d-%m") + " iter " + str(self.iter) + '.svg')
                plt.clf()

                os.chdir('..')
                
            if self.prog and self.iter % self.increase_alfa_step == 0 and self.iter > 0:
                self.gen.increaseAlfa(self.alfa_step)

            self.iter = self.iter + 1

    def disc_epoch(self):

        for real in tqdm(self.dataloader):
            d_loss, f_loss, r_loss = self.train_disc(real)

            self.disc_loss.append(d_loss)
            
            if self.verb :
                self.disc_fake_loss.append(f_loss)
                self.disc_real_loss.append(r_loss)
            
            if self.prog and self.iter % self.increase_alfa_step == 0 and self.iter > 0:
                self.disc.increaseAlfa(self.alfa_step)

            if self.iter % self.log_step == 0 and self.iter > 0:
                self.disc_loss_plot.append(sum(self.disc_loss[-self.log_step: ])/self.log_step)

                if self.verb :
                    self.disc_fake_loss_plot.append(sum(self.disc_fake_loss[-self.log_step:])/self.log_step)
                    self.disc_real_loss_plot.append(sum(self.disc_real_loss[-self.log_step:])/self.log_step)
                
                os.chdir(self.resdir)

                plt.plot(np.array(self.ejeX), np.array(self.disc_loss_plot), label = "Disc avg loss")
                if self.verb:
                    plt.plot(np.array(self.ejeX), np.array(self.disc_fake_loss_plot), label = "Disc fake loss")
                    plt.plot(np.array(self.ejeX), np.array(self.disc_real_loss_plot), label = "Disc real loss")
                
                plt.legend()
                plt.savefig("Disc " + datetime.now().strftime("%d-%m") + " iter " + str(self.iter) + ".svg")
                plt.clf()

                os.chdir("..")

            self.iter = self.iter + 1

    def train_for_epochs(self, n_epochs):

        for _ in range(n_epochs):
            self.epoch()

    def train_generator_for_epochs(self, n_epochs):
        print("Empieza el entrenamiento")
        for _ in range(n_epochs):
            self.gen_epoch()

    def train_discriminator_for_epochs(self, n_epochs):
        for _ in range(n_epochs):
            self.disc_epoch()


    def plot_losses(self):

        self.disc_loss_plot.append(sum(self.disc_loss[-self.log_step:])/self.log_step)
        self.gen_loss_plot.append(sum(self.gen_loss[-self.log_step:])/self.log_step)

        if self.verb :
            self.disc_fake_loss_plot.append(sum(self.disc_fake_loss[-self.log_step:])/self.log_step)
            self.disc_real_loss_plot.append(sum(self.disc_real_loss[-self.log_step:])/self.log_step)

        title = "Gen Loss: " + str(self.gen_loss_plot[-1]) + " Disc Loss: " + str(self.disc_loss_plot[-1])
        print(title)
        
        arEjeX = np.array(self.ejeX)
        plt.plot(arEjeX, np.array(self.gen_loss_plot), label = "Gen Loss")
        plt.plot(arEjeX, np.array(self.disc_loss_plot), label = "Disc Loss")

        if self.verb :
            plt.plot(arEjeX, np.array(self.disc_fake_loss_plot), label = "Disc Fake Loss")
            plt.plot(arEjeX, np.array(self.disc_real_loss_plot), label = "Disc Real Loss")

        plt.title(title)
        plt.legend()

        os.chdir(self.resdir)
        plt.savefig(datetime.now().strftime("%d-%m")+" iter " + str(self.iter) + '.svg')
        os.chdir('..')

        plt.clf()

    def save_results(self, real, generated):
        ImageFunctions.tensor_as_image(generated, self.iter, "fake", self.resdir, save = True, show = False)
        ImageFunctions.tensor_as_image(real, self.iter, "real", self.resdir, save = True, show = False)            

    def saveCheckpoint(self, dir, gen, disc, gen_opt, disc_opt, gen_loss, disc_loss, epoch, iter):
        os.chdir(dir)

        g_s = 'gen_' + str(iter) + '.tar'
        d_s = 'disc_' + str(iter) + '.tar'

        torch.save({
        'epoch' : epoch ,
        'iter' : iter,
        'model_state_dict' : gen.state_dict(), 
        'optimizer_state_dict' : gen_opt.state_dict(),
        'loss' : gen_loss
        }, g_s)

        torch.save({
            'epoch' : epoch,
            'iter' : iter,
            'model_state_dict' : disc.state_dict(),
            'optimizer_state_dict' : disc_opt.state_dict(),
            'loss' : disc_loss
        }, d_s)

        os.chdir('..')
        


