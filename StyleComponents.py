import torch
import torch.nn as nn
from scipy.stats import truncnorm

def  get_ruido_truncado(n_samples, z_dim, truncation):
    '''
    Function para crear vectores de ruido truncado
    Dado las dimensiones(n_samples, z_dim) y un valor de truncacción, crea el tensor de ese tamaño
    ocupado por valores aleatorios de la distribucion normal truncada.
    Parametros:
        n_samples(escalar): el número de ejemplos a generar
        z_dim(escalar): la dimensión del vector de ruido
        truncation(escalar no negativo): el valor de truncacion
    '''
    truncated_noise = truncnorm.rvs(-1*truncation, truncation, size=(n_samples, z_dim))
    return torch.Tensor(truncated_noise)

def get_ruido_truncado_cuadrado(n_samples, z_dim, truncation):
    truncated_noise = truncnorm.rvs(-1*truncation, truncation, size=(n_samples, 512, z_dim, z_dim))
    return torch.Tensor(truncated_noise)

class CapasMapeadoras(nn.Module):
    '''
    Mapping Layers Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        hidden_dim: the inner dimension, a scalar
        w_dim: the dimension of the intermediate noise vector, a scalar
    '''
 
    def __init__(self, z_dim, hidden_dim, w_dim):
        super().__init__()
        self.mapping = nn.Sequential(
            nn.Linear(z_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, w_dim)

        )

    def forward(self, noise):
        '''
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        return self.mapping(noise)
    
 
    def get_mapping(self):
        return self.mapping


class InyecciondeRuido(nn.Module):
    '''
    Inject Noise Class
    Values:
        channels: the number of channels the image has, a scalar
    '''
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter( # You use nn.Parameter so that these weights can be optimized
            # Initiate the weights for the channels from a random normal distribution

            torch.randn(channels)[None, :, None, None] #torch.randn((1,channels,1,1))

        )

    def forward(self, image):
        '''
        Function for completing a forward pass of InjectNoise: Given an image, 
        returns the image with random noise added.
        Parameters:
            image: the feature map of shape (n_samples, channels, width, height)
        '''
        # Set the appropriate shape for the noise!
        

        noise_shape = (image.shape[0], 1, image.shape[2], image.shape[3])

        
        noise = torch.randn(noise_shape, device=image.device) # Creates the random noise
        return image + self.weight * noise # Applies to image after multiplying by the weight for each channel
    
    def get_weight(self):
        return self.weight
    
    def get_self(self):
        return self

class AdaIN(nn.Module):
    '''
    AdaIN Class
    Values:
        channels: the number of channels the image has, a scalar
        w_dim: the dimension of the intermediate noise vector, a scalar
    '''

    def __init__(self, channels, w_dim):
        super().__init__()

        # Normalize the input per-dimension
        self.instance_norm = nn.InstanceNorm2d(channels)

        self.style_scale_transform = nn.Linear(w_dim, channels)
        self.style_shift_transform = nn.Linear(w_dim, channels)

    def forward(self, image, w):
        '''
        Function for completing a forward pass of AdaIN: Given an image and intermediate noise vector w, 
        returns the normalized image that has been scaled and shifted by the style.
        Parameters:
            image: the feature map of shape (n_samples, channels, width, height)
            w: the intermediate noise vector
        '''
        normalized_image = self.instance_norm(image)
        style_scale = self.style_scale_transform(w)[:, :, None, None]
        style_shift = self.style_shift_transform(w)[:, :, None, None]
        
        # Calculate the transformed image

        transformed_image = style_scale * normalized_image + style_shift

        return transformed_image
    

    def get_style_scale_transform(self):
        return self.style_scale_transform
    

    def get_style_shift_transform(self):
        return self.style_shift_transform
    

    def get_self(self):
        return self 