import sys, os

import Training
from StyleGan.Components import make_dataset, make_logger
import StyleGenerador
import StyleDiscriminator

curentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(curentdir)
sys.path.append(parentdir)



output_dir ='/data/hzh/checkpoints/StyleGAN.pytorch/prob'
img_dir="C:/Users/Daniela/Documents/TFG/fruits-360_dataset/fruits-360\Training"
logger = make_logger("project", output_dir, 'log')


if __name__ == '__main__':
    dataset = make_dataset(resolution=128, ##La resolucion la cargamos como si fuesen imagenes 128x128 para evitar problemas
                           folder=True,
                           img_dir=img_dir,
                           conditional=False)


    epochs = [4, 4, 4, 4, 8, 16, 32, 64, 64]

    batch_sizes = [8, 8, 8, 8, 8, 4, 2, 1, 1]

    trainer = Training.Style_Prog_Trainer(
                                        generator=StyleGenerador.Generator,
                                        discriminator=StyleDiscriminator.Discriminator,
                                        conditional=False,
                                         n_classes=131,
                                         resolution=128,
                                         num_channels=3,
                                         latent_size=512,
                                         loss="logistic",
                                         drift=0.001,
                                         d_repeats=1,
                                         use_ema=True,
                                         ema_decay=0.999,
                                         device='cuda',
                                         checksave = False,
                                          load = False,
                                          load_dir= None,
                                          gen_load= None,
                                          disc_load = None,
                                          time_steps = True,
                                          time_epochs= True)
    trainer.train_for_epochs(dataset=dataset,
                    num_workers=1,
                    epochs=epochs,
                    batch_sizes=batch_sizes,
                    logger=logger,
                    output=output_dir,
                    num_samples=36,
                    start_depth=0,
                    feedback_factor=10,
                    checkpoint_factor=10)


    # # init the network
    # style_gan = StyleGAN(conditional=False,
    #                      n_classes=131,
    #                      resolution=128,
    #                      num_channels=3,
    #                      latent_size=512,
    #                      loss="logistic",
    #                      drift=0.001,
    #                      d_repeats=1,
    #                      use_ema=True,
    #                      ema_decay=0.999,
    #                      device='cuda')
    #
    # epochs = [4, 4, 4, 4, 8, 16, 32, 64, 64]
    #
    # batch_sizes = [8, 8, 8, 8, 8, 4, 2, 1, 1]
    #
    # # train the network
    # style_gan.train(dataset=dataset,
    #                 num_workers=1,
    #                 epochs=epochs,
    #                 batch_sizes=batch_sizes,
    #                 logger=logger,
    #                 output=output_dir,
    #                 num_samples=36,
    #                 start_depth=0,
    #                 feedback_factor=10,
    #                 checkpoint_factor=10)
