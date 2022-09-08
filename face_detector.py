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
    
    
def emotionDetection(imageFile):
    #TODO
    return


# Load the cascade  
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  
  
# To capture video from existing video.   
cap = cv2.VideoCapture('test.mp4')
getFirstFrame('test.mp4')
getLastFrame('test.mp4')
    

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
  
    # Display
    cv2.imshow('Video', img)
  
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break  
          
# Release the VideoCapture object
cap.release()
