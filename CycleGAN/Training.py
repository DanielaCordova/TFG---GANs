import ImageFunctions
import matplotlib.pyplot as plt
import os
from datetime import datetime
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
    
    
    
class Cycle_Trainer():
    def __init__(self, dataloader1, dataloader2, gen_AB, gen_BA, gen_opt, disc_A, disc_A_opt, disc_B, disc_B_opt, adv_criterion, 
                 recon_criterion, log_step, log_dir, target_shape, device='cuda', checksave=False, save_step=None, load=False, 
                 load_dir=None, gen_disc_load=None, time_steps=False, time_epochs=False):
        
        
        self.device        = device
        self.adv_criterion = adv_criterion
        self.recon_criterion = recon_criterion
        self.log_step      = log_step
        self.log_dir       = log_dir
        self.checksave     = checksave
        self.time_epochs   = time_epochs
        self.time_steps    = time_steps
        self.target_shape  = target_shape
        
        if checksave :
            self.save_step = save_step
           
        self.load          = load
        
        if load:
            self.load_dir  = load_dir
            self.gen_disc_load  = gen_disc_load
            
        self.act           = 0
        self.iter          = 0
        
        if time_steps :
            self.step_times    = []
            self.num_steps     = []
        if time_epochs :
            self.epoch_times   = []
            self.num_epochs    = []
        
        
        
        
        self.dataloader1 = dataloader1
        self.dataloader2 = dataloader2
            
        self.gen_AB = gen_AB
        self.gen_BA = gen_BA
        self.gen_opt = gen_opt
        self.disc_A = disc_A
        self.disc_A_opt = disc_A_opt
        self.disc_B = disc_B
        self.disc_B_opt = disc_B_opt
        self.gen_loss = []
        self.dis_loss = []

        if load:
            self.load_checkpoint()
        else:
            gen_AB = gen_AB.apply(self.weights_init)
            gen_BA = gen_BA.apply(self.weights_init)
            disc_A = disc_A.apply(self.weights_init)
            disc_B = disc_B.apply(self.weights_init)

    def get_disc_loss(self, real_X, fake_X, disc_X, adv_criterion):
        '''
        Return the loss of the discriminator given inputs.
        Parameters:
            real_X: the real images from pile X
            fake_X: the generated images of class X
            disc_X: the discriminator for class X; takes images and returns real/fake class X
                prediction matrices
            adv_criterion: the adversarial loss function; takes the discriminator 
                predictions and the target labels and returns a adversarial 
                loss (which you aim to minimize)
        '''
        #### START CODE HERE ####
        disc_fake_X_hat = disc_X(fake_X.detach()) # Detach generator
        disc_fake_X_loss = adv_criterion(disc_fake_X_hat, torch.zeros_like(disc_fake_X_hat))
        disc_real_X_hat = disc_X(real_X)
        disc_real_X_loss = adv_criterion(disc_real_X_hat, torch.ones_like(disc_real_X_hat))
        disc_loss = (disc_fake_X_loss + disc_real_X_loss) / 2    
        #### END CODE HERE ####
        return disc_loss
    
    
        
    def get_gen_adversarial_loss(self, real_X, disc_Y, gen_XY, adv_criterion):
        '''
        Return the adversarial loss of the generator given inputs
        (and the generated images for testing purposes).
        Parameters:
            real_X: the real images from pile X
            disc_Y: the discriminator for class Y; takes images and returns real/fake class Y
                prediction matrices
            gen_XY: the generator for class X to Y; takes images and returns the images 
                transformed to class Y
            adv_criterion: the adversarial loss function; takes the discriminator 
                      predictions and the target labels and returns a adversarial 
                      loss (which you aim to minimize)
        '''
        #### START CODE HERE ####
        fake_Y = gen_XY(real_X)
        disc_fake_Y_hat = disc_Y(fake_Y)
        adversarial_loss = adv_criterion(disc_fake_Y_hat, torch.ones_like(disc_fake_Y_hat))    
        #### END CODE HERE ####
        return adversarial_loss, fake_Y
    
    def get_identity_loss(self, real_X, gen_YX, identity_criterion):
        '''
        Return the identity loss of the generator given inputs
        (and the generated images for testing purposes).
        Parameters:
            real_X: the real images from pile X
            gen_YX: the generator for class Y to X; takes images and returns the images 
                transformed to class X
            identity_criterion: the identity loss function; takes the real images from X and
                            those images put through a Y->X generator and returns the identity 
                            loss (which you aim to minimize)
        '''
        #### START CODE HERE ####
        identity_X = gen_YX(real_X)
        identity_loss = identity_criterion(identity_X, real_X)
        #### END CODE HERE ####
        return identity_loss, identity_X
    
    def get_cycle_consistency_loss(self, real_X, fake_Y, gen_YX, cycle_criterion):
        '''
        Return the cycle consistency loss of the generator given inputs
        (and the generated images for testing purposes).
        Parameters:
            real_X: the real images from pile X
            fake_Y: the generated images of class Y
            gen_YX: the generator for class Y to X; takes images and returns the images 
                transformed to class X
            cycle_criterion: the cycle consistency loss function; takes the real images from X and
                            those images put through a X->Y generator and then Y->X generator
                            and returns the cycle consistency loss (which you aim to minimize)
        '''
        #### START CODE HERE ####
        cycle_X = gen_YX(fake_Y)
        cycle_loss = cycle_criterion(cycle_X, real_X) 
        #### END CODE HERE ####
        return cycle_loss, cycle_X
    
    def get_gen_loss(self, real_A, real_B, gen_AB, gen_BA, disc_A, disc_B, adv_criterion, identity_criterion, cycle_criterion, lambda_identity=0.1, lambda_cycle=10):
        '''
        Return the loss of the generator given inputs.
        Parameters:
            real_A: the real images from pile A
            real_B: the real images from pile B
            gen_AB: the generator for class A to B; takes images and returns the images 
                transformed to class B
            gen_BA: the generator for class B to A; takes images and returns the images 
                transformed to class A
            disc_A: the discriminator for class A; takes images and returns real/fake class A
                prediction matrices
            disc_B: the discriminator for class B; takes images and returns real/fake class B
                prediction matrices
            adv_criterion: the adversarial loss function; takes the discriminator 
                predictions and the true labels and returns a adversarial 
                loss (which you aim to minimize)
            identity_criterion: the reconstruction loss function used for identity loss
                and cycle consistency loss; takes two sets of images and returns
                their pixel differences (which you aim to minimize)
            cycle_criterion: the cycle consistency loss function; takes the real images from X and
                those images put through a X->Y generator and then Y->X generator
                and returns the cycle consistency loss (which you aim to minimize).
                Note that in practice, cycle_criterion == identity_criterion == L1 loss
            lambda_identity: the weight of the identity loss
            lambda_cycle: the weight of the cycle-consistency loss
        '''
        # Hint 1: Make sure you include both directions - you can think of the generators as collaborating
        # Hint 2: Don't forget to use the lambdas for the identity loss and cycle loss!
        #### START CODE HERE ####
        # Adversarial Loss -- get_gen_adversarial_loss(real_X, disc_Y, gen_XY, adv_criterion)
        adv_loss_BA, fake_A = self.get_gen_adversarial_loss(real_B, disc_A, gen_BA, adv_criterion)
        adv_loss_AB, fake_B = self.get_gen_adversarial_loss(real_A, disc_B, gen_AB, adv_criterion)
        gen_adversarial_loss = adv_loss_BA + adv_loss_AB

        # Identity Loss -- get_identity_loss(real_X, gen_YX, identity_criterion)
        identity_loss_A, identity_A = self.get_identity_loss(real_A, gen_BA, identity_criterion)
        identity_loss_B, identity_B = self.get_identity_loss(real_B, gen_AB, identity_criterion)
        gen_identity_loss = identity_loss_A + identity_loss_B

        # Cycle-consistency Loss -- get_cycle_consistency_loss(real_X, fake_Y, gen_YX, cycle_criterion)
        cycle_loss_BA, cycle_A = self.get_cycle_consistency_loss(real_A, fake_B, gen_BA, cycle_criterion)
        cycle_loss_AB, cycle_B = self.get_cycle_consistency_loss(real_B, fake_A, gen_AB, cycle_criterion)
        gen_cycle_loss = cycle_loss_BA + cycle_loss_AB

        # Total loss
        gen_loss = lambda_identity * gen_identity_loss + lambda_cycle * gen_cycle_loss + gen_adversarial_loss
        #### END CODE HERE ####
        return gen_loss, fake_A, fake_B
       
    
    def weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)
            

    def load_checkpoint(self):
        os.chdir(self.load_dir)
        
        pre_dict = torch.load(self.gen_disc_load)
        
        self.gen_AB.load_state_dict(pre_dict['gen_AB'])
        self.gen_BA.load_state_dict(pre_dict['gen_BA'])
        self.gen_opt.load_state_dict(pre_dict['gen_opt'])
        
        self.disc_A.load_state_dict(pre_dict['disc_A'])
        self.disc_A_opt.load_state_dict(pre_dict['disc_A_opt'])
        self.disc_B.load_state_dict(pre_dict['disc_B'])
        self.disc_B_opt.load_state_dict(pre_dict['disc_B_opt'])

        self.act = pre_dict['epoch']
    
        self.dis_loss = pre_dict['disc_loss']

        self.gen_loss = pre_dict['gen_loss']

        os.chdir('..')

    def saveCheckpoint(self, epoch):
        os.chdir(self.log_dir)
        
        torch.save({
            'gen_AB': self.gen_AB.state_dict(),
            'gen_BA': self.gen_BA.state_dict(),
            'gen_opt': self.gen_opt.state_dict(),
            'disc_A': self.disc_A.state_dict(),
            'disc_A_opt': self.disc_A_opt.state_dict(),
            'disc_B': self.disc_B.state_dict(),
            'disc_B_opt': self.disc_B_opt.state_dict(),
            'epoch': epoch,
            'gen_loss': self.gen_loss,
            'disc_loss': self.dis_loss
        }, 'cycleGAN_' + str(epoch) + '.pth')

        os.chdir('..')

    def check_per_batch(self, real, generated, it):
        return

    def get_InputVector_paraEtiquetar(self, etiquetas, numClases):
        return F.one_hot(etiquetas, numClases)

    def combinarVectores(self, x, y):
        return torch.cat((x.float(), y.float()), 1)

    def train_gen(self, real_A, real_B):
        self.enable_training(self.gen_BA, True)
        self.enable_training(self.gen_AB, True)
        self.enable_training(self.disc_A, False)
        self.enable_training(self.disc_B, False)

        ### Update generator ###
        self.gen_opt.zero_grad()
        gen_loss, fake_A, fake_B = self.get_gen_loss(
            real_A, real_B, self.gen_AB, self.gen_BA, self.disc_A, self.disc_B, self.adv_criterion, self.recon_criterion, self.recon_criterion
        )
        gen_loss.backward() # Update gradients
        self.gen_opt.step() # Update optimizer

        return (gen_loss, fake_A, fake_B)

    def train_disc(self, real_A, real_B):
        self.enable_training(self.gen_BA, False)
        self.enable_training(self.gen_AB, False)
        self.enable_training(self.disc_A, True)
        self.enable_training(self.disc_B, True)

        ### Update discriminator A ###
        self.disc_A_opt.zero_grad() # Zero out the gradient before backpropagation
        with torch.no_grad():
            fake_A = self.gen_BA(real_B)
        disc_A_loss =self.get_disc_loss(real_A, fake_A, self.disc_A, self.adv_criterion)
        disc_A_loss.backward(retain_graph=True) # Update gradients
        self.disc_A_opt.step() # Update optimizer

        ### Update discriminator B ###
        self.disc_B_opt.zero_grad() # Zero out the gradient before backpropagation
        with torch.no_grad():
            fake_B = self.gen_AB(real_A)
        disc_B_loss = self.get_disc_loss(real_B, fake_B, self.disc_B, self.adv_criterion)
        disc_B_loss.backward(retain_graph=True) # Update gradients
        self.disc_B_opt.step() # Update optimizer
        
        return disc_A_loss, disc_B_loss

    def train_for_epochs(self, n_epochs):
        self.initial__time = time.time()
        self.mean_generator_loss = 0
        self.mean_discriminator_loss = 0
        for e in range(0, n_epochs):
            self.epoch(e + self.act)
            self.act += 1


    def epoch(self, ep):
        it = 0
        for (real_A, real_B) in tqdm(zip(self.dataloader1, self.dataloader2)):
            real_A = nn.functional.interpolate(real_A, size=self.target_shape)
            real_B = nn.functional.interpolate(real_B, size=self.target_shape)
            real_A = real_A.to(self.device)
            real_B = real_B.to(self.device)

            
            generator_loss, fake_A, fake_B = self.train_gen(real_A, real_B)
            disc_A_loss, disc_B_loss = self.train_disc(real_A, real_B)
            

            # Keep track of the average discriminator loss
            self.mean_discriminator_loss += disc_A_loss.item() / display_step
            # Keep track of the average generator loss
            self.mean_generator_loss += generator_loss.item() / display_step

            # Keep track of the generator losses
            self.gen_loss.append(generator_loss.item())
            # Keep track of the average discriminator loss
            self.dis_loss.append(disc_A_loss.item())
            
            self.iter = self.iter + 1
            it = it + 1
            
            ### Visualization code ###
            if self.iter % self.log_step == 0 and self.iter > 0:
                self.plot_losses()
                self.save_results(torch.cat([real_A, real_B]), torch.cat([fake_B, fake_A]))

                if self.time_steps:
                    self.plot_step_time()

            if self.checksave and self.iter % self.save_step == 0:
                self.saveCheckpoint(ep)

        if self.time_epochs:
            self.plot_epoch_time()

        self.check_per_batch(torch.cat([real_A, real_B]), torch.cat([fake_B, fake_A]), it)

    def save_results(self, real, generated):
        ImageFunctions.tensor_as_image(generated, self.iter, "fake", self.log_dir, save = True, show = False)
        ImageFunctions.tensor_as_image(real, self.iter, "real", self.log_dir, save = True, show = False)

    def enable_training(self, model, flag):
        for p in model.parameters():
            p.requires_grad = flag

    def check_training_params(self, model, flag):
        for p in model.parameters():
            assert(p.requires_grad == flag)
        
    def plot_epoch_time(self):
        epoch_time = (time.time() - self.initial__time)
        self.epoch_times.append(epoch_time)
        self.num_epochs.append(self.act)

        title = str(self.epoch_times[-1]) + " seconds taken"
        print(title)

        plt.plot(self.num_epochs,self.epoch_times, label="Epoch Time")
        
        plt.title(title)
        plt.legend()

        os.chdir(self.log_dir)
        plt.savefig(datetime.now().strftime("%d-%m")+" iter " + str(self.act) + " epoch_times" + '.pdf')
        os.chdir('..')

        plt.clf()
        
    def plot_step_time(self):
        
        step_time = (time.time() - self.initial__time)
        self.step_times.append(step_time)
        self.num_steps.append(self.iter)

        title = str(self.step_times[-1]) + " seconds to take " + str(self.iter) + " steps"
        print(title)
        
        #Time visualization

        plt.plot(self.num_steps, self.step_times, label="Step Time")
        
        plt.title(title)
        plt.legend()

        os.chdir(self.log_dir)
        plt.savefig(datetime.now().strftime("%d-%m")+" iter " + str(self.iter) + " step_times" + '.pdf')
        os.chdir('..')

        plt.clf()


    def plot_losses(self):
        
        self.mean_generator_loss = 0
        self.mean_discriminator_loss = 0
        step_bins = 20
        x_axis = sorted([i * step_bins for i in range(len(self.gen_loss) // step_bins)] * step_bins)
        num_examples = (len(self.gen_loss) // step_bins) * step_bins
        print(self.gen_loss)
        plt.plot(
            range(num_examples // step_bins), 
            torch.Tensor(self.gen_loss[:num_examples]).view(-1, step_bins).mean(1),
            label="Generator Loss"
        )
        plt.plot(
            range(num_examples // step_bins), 
            torch.Tensor(self.dis_loss[:num_examples]).view(-1, step_bins).mean(1),
            label="Discriminator Loss"
        )
        plt.legend()
        
        os.chdir(self.log_dir)
        plt.savefig(datetime.now().strftime("%d-%m") + " iter " + str(self.iter) +'.pdf')
        os.chdir('..')
        plt.clf()



    def generate_samples(self, n_samples = 1):
        i = 0
        for (real_A, real_B) in tqdm(zip(self.dataloader1, self.dataloader2)):
            i += 1
            real_A = nn.functional.interpolate(real_A, size=self.target_shape)
            real_B = nn.functional.interpolate(real_B, size=self.target_shape)
            real_A = real_A.to(self.device)
            real_B = real_B.to(self.device)
            gen_loss, fake_A, fake_B = self.get_gen_loss(
                real_A, real_B, self.gen_AB, self.gen_BA, self.disc_A, self.disc_B, self.adv_criterion,
                self.recon_criterion, self.recon_criterion
            )
            self.save_results(torch.cat([real_A, real_B]), torch.cat([fake_B, fake_A]))

            if i > n_samples :
                break
