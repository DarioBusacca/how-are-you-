"""
plot confusion_matrix of PublicTest and PrivateTest
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
from fer import FER2013
from torch.autograd import Variable
import torchvision
import transforms as transforms
from sklearn.metrics import confusion_matrix
from models import *

parser = argparse.ArgumentParser(description="Pytorch FER2013 CNN Training")
parser.add_argument('--model', type = str, default = 'VGG19', help = 'CNN architecture')
parser.add_argument('--dataset', type = str, default = 'FER2013', help='CNN architecture')
parser.add_argument('--split', type=str, default='PrivateTest', help='split')
opt = parser.parse_args()

cut_size = 44

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
])


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """