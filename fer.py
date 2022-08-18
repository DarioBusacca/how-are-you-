import matplotlib as plt
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import h5py
from PIL import Image

if(torch.cuda.is_available):
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('Using PyTorch version:', torch.__version__, ' Device:', device)

class FER2013(data.Dataset):
    
    def __init__(self, split = 'Training', transform = None):
        self.transform = transform
        self.split = split  #training set or test set
        self.data = h5py.File('./data/data.h5', 'r', driver = 'core')
        
        # load numpy arrays
        if self.split == 'Training':
            self.train_data = self.data['Training_pixel']
            self.train_label = self.data['Training_label']
            self.traina_data = np.asarray(self.train_data)
            self.train_data = self.train_data.reshape(28709, 48, 48)
            
        elif self.split == 'PublicTest':
            self.PublicTest_data = self.data['PublicTest_pixel']
            self.PublicTest_label = self.data['PublicTest_label']
            self.PublicTest_data = np.asarray(self.PublicTest_data)
            self.PublicTest_data = self.PublicTest_data.reshape(3589, 48, 48)
        
        elif self.split == 'PrivateTest':
            self.PrivateTest_data = self.data['PrivateTest_pixel']
            self.PrivateTest_label = self.data['PrivateTest_label']
            self.PrivateTest_data = np.asarray(self.PrivateTest_data)
            self.PrivateTest_data = self.PrivateTest_data.reshape(3589, 48, 48)
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split == 'Training':
            img, target = self.train_data[index], self.train_label[index]
        elif self.split == 'PublicTest':
            img, target = self.PublicTest_data[index], self.PublicTest_label[index]
        elif self.split ==  'PrivateTest':
            img, target = self.PrivateTest_data[index], self.PrivateTest_label[index]
            
        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis = 2)
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target
    
    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data)
        elif self.split == 'PublicTest':
            return len(self.PublicTest_data)
        elif self.split == 'PrivateTest':
            return len(self.PrivateTest_data)
        



