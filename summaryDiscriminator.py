import GAN as gan
from torchsummary import summary

dis = gan.Discriminator(64)
summary(dis, (3,4,4))