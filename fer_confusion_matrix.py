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
    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    print(cm)
    
    plt.imshow(cm, interpolation='nearest', cmap = cmap)
    plt.title(title, fontsize = 16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh =cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment = "center", color = "white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label', fontsize = 18)
    plt.xlabel('Predicted label', fontsize = 18)
    plt.tight_layout()
    

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


# model
net = VGG('VGG19')
path = os.path.join(opt.dataset + '_' + opt.model)
checkpoint = torch.load(os.path.join(path, opt.split + '_model.t7'))
