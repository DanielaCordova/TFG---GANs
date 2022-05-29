
import os
import numpy as np

from torch.utils.data import Dataset


class DirectorioDatasetSinCarpetas(Dataset):

    def __setup_files(self):
        file_names = os.listdir(self.data_dir)
        files = []  # initialize to empty list

        for file_name in file_names:
            pFile = os.path.join(self.data_dir, file_name)
            if os.path.isfile(pFile):
                files.append(pFile)

        return files

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform ##Como cargaremos
        self.files = self.__setup_files()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        from PIL import Image

        img_file = self.files[idx]

        if img_file[-4:] == ".npy":
            # files are in .npy format
            img = np.load(img_file)
            img = Image.fromarray(img.squeeze(0).transpose(1, 2, 0))

        else:
            # read the image:
            img = Image.open(self.files[idx]).convert('RGB')

        # apply the transforms on the image
        if self.transform is not None:
            img = self.transform(img)

        if img.shape[0] >= 4:
            img = img[:3, :, :]

        # return the image:
        return img


class DirectorioDatasetConCarpetas(Dataset):

    def __setup_files(self):
        dir_names = os.listdir(self.data_dir)
        files = []  # initialize to empty list
        for dir_name in dir_names:
            file_path = os.path.join(self.data_dir, dir_name)
            file_names = os.listdir(file_path)
            for file_name in file_names:
                pFile = os.path.join(file_path, file_name)
                if os.path.isfile(pFile):
                    files.append(pFile)

        # return the files list
        return files

    def __init__(self, data_dir, transform=None):
        # define the state of the object
        self.data_dir = data_dir
        self.transform = transform

        # setup the files for reading
        self.files = self.__setup_files()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        from PIL import Image

        # read the image:
        img_name = self.files[idx]
        if img_name[-4:] == ".npy":
            img = np.load(img_name)
            img = Image.fromarray(img.squeeze(0).transpose(1, 2, 0))
        else:
            img = Image.open(img_name).convert('RGB')

        # apply the transforms on the image
        if self.transform is not None:
            img = self.transform(img)

        if img.shape[0] >= 4:
            # ignore the alpha channel
            # in the image if it exists
            img = img[:3, :, :]

        # return the image:
        return img
