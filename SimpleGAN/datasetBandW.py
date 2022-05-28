
import os
from PIL import Image
from torch.utils.data import Dataset

class DataSetCarpetaBlackandWhite(Dataset):

  def __init__(self, dir, t, tag):
    super().__init__()
    self.dir = dir
    self.transform = t
    self.imagenes = os.listdir(dir)
    self.tag = tag

  def __len__(self):
    return len(self.imagenes)

  def __getitem__(self, index):
    img = os.path.join(self.dir, self.imagenes[index])
    imagen = Image.open(img).convert()
    tensor = self.transform(imagen)
    return (tensor, self.tag)