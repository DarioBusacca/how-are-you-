import cv2  
from turtle import width
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable

import transforms as transforms
from skimage import io
from skimage.transform import resize
from models import *

def getFirstFrame(videofile):
    vidcap = cv2.VideoCapture(videofile)
    success, image = vidcap.read()
    if(success):
        cv2.imwrite("first_frame.jpg", image)
        
def getLastFrame(videofile):
    vidcap = cv2.VideoCapture(videofile)
    last_frame_num = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, last_frame_num - 0.5)
    success, image = vidcap.read()
    if(success):
        cv2.imwrite("last_frame.jpg", image)
    
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    
# similar to run.py file
def emotionDetection(imageFile):
    cut_size = 44

    transform_test = transforms.Compose([
        transforms.TenCrop(cut_size),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    ])
    
    raw_img = io.imread(imageFile)
    gray = rgb2gray(raw_img)
    gray = resize(gray, (48, 48), mode = 'symmetric').astype(np.uint8)
    
    img = gray[:, :, np.newaxis]

    img = np.concatenate((img, img, img), axis=2)
    img = Image.fromarray(img)
    inputs = transform_test(img)
    
    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    net = VGG('VGG16')
    checkpoint = torch.load(os.path.join('FER2013_VGG19', 'PrivateTest_model.t7'))
    net.load_state_dict(checkpoint['net'])
    net.cuda()
    net.eval()

    ncrops, c, h, w = np.shape(inputs)

    inputs = inputs.view(-1, c, h, w)
    inputs = inputs.cuda()
    inputs = Variable(inputs, volatile = True)
    outputs = net(inputs)
    
    outputs_avg = outputs.view(ncrops, -1).mean(0)  #avg crops
    
    score = F.softmax(outputs_avg)
    _, predicted = torch.max(outputs_avg.data, 0)
    
    return str(class_names[int(predicted.cpu().numpy())])


# Load the cascade  
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  
  
# To capture video from existing video.
#INSERIRE NOME VIDEO QUI
cap = cv2.VideoCapture('video/NOMEFILE.mp4')
getFirstFrame('video/NOMEFILE.mp4')
getLastFrame('video/NOMEFILE.mp4')

first_emotion = ''

while(first_emotion == ''):
    first_emotion = emotionDetection('first_frame.jpg')

last_emotion = emotionDetection('last_frame.jpg')
print(first_emotion)
print(last_emotion)

if(cap.isOpened() == False):
    print("Error opening video for stream")
  
while(cap.isOpened()):
    # Read the frame        
    _, img = cap.read()
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
  
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, (first_emotion+'===>'+ last_emotion), (int(x+(w/4)), y+h+40), cv2.FONT_HERSHEY_TRIPLEX, 1, (9, 255, 0), 4)
  
    # Display
    cv2.imshow('Video', img)
  
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break  
          
# Release the VideoCapture object
cap.release()
