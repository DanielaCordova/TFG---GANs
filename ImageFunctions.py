from __future__ import print_function
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime


def mostrar_tensor_imagenes(image_tensor, num_images=16, size=(3, 64, 64), nrow=3):
    '''
    Funcion para ver imagenes dado un tensor, el numero,
    el tamaño y las imagines por fila.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu().clamp_(0, 1)
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow, padding=0)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.axis('off')
    plt.show()

def image_loader(image_name, device):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = os.loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def ver_imagen(img, dataset, label):
    print('Label: ', dataset.classes[label], "("+str(label)+")")
    plt.imshow(img.permute(1, 2, 0))

def getDatasets(data_dir):
    transform = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor()
    ##transforms.Normalize((0.5,), (0.5,)),
    ])

    dataset = ImageFolder(data_dir + '/Training', transform=transform) ##El transform anterior oscurece las imagenes
    print('Tamaño del dataset de entrenamiento :', len(dataset))
    test = ImageFolder(data_dir + '/Test', transform=transform)
    print('Tamaño del dataset para test :', len(test))

    return (dataset,test)

def mostrar_tensor_imagenes(image_tensor, num_images=16, size=(3, 64, 64), nrow=3):
    '''
    Funcion para ver imagenes dado un tensor, el numero,
    el tamaño y las imagines por fila.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu().clamp_(0, 1)
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow, padding=0)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.axis('off')
    plt.show()

def ver_tensor_img(img_tensor, num_images=25, size=(3, 64, 64), nrow=5, show=True): ##para ver las imagenes en una grid
    img_tensor = (img_tensor + 1) / 2
    image_unflat = img_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    if show:
        plt.show()

def tensor_as_image(img_tensor, tag = None, dir = None, num_images=25, size=(3,64,64), nrow=5, save=True ,show=True):
    img_tensor = (img_tensor + 1) / 2
    image_unflat = img_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow)
    plt.imshow(image_grid.permute(1,2,0).squeeze())
    if show:
        plt.show()
    if save:
        os.chdir(dir)
        plt.savefig(datetime.now().strftime("%H-%M-%S-%f %d-%m-%y") + tag + '.png')
        os.chdir('..')