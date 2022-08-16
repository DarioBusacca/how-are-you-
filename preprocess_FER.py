import numpy as np
import csv
import h5py
import os

file = './data/fer2013.csv'

#create list to score data and label information
Training_x = []
Training_y = []
PublicTest_x = []
PublicTest_y = []
PrivateTest_x = []
PrivateTest_y = []

datapath = os.path.join('data', 'data.h5')
if not os.path.exists(os.path.dirname(datapath)):
    os.makedirs(os.path.dirname(datapath))
    
with open(file, 'r') as csvin:
    data = csv.reader(csvin)
    for row in data:
        if row[-1] == 'Training':
            tmp_list = []
            for pixel in row[1].split( ):
                tmp_list.append(int(pixel))
            I = np.asarray(tmp_list)
            Training_y.append(int(row[0]))
            Training_x.append(I.tolist())
            
        if row[-1] == 'PublicTest':
            tmp_list = []
            for pixel in row[1].split( ):
                tmp_list.append(int(pixel))
            I = np.asarray(tmp_list)
            PublicTest_y.append(int(row[0]))
            PublicTest_x.append(I.tolist())
            
        if row[-1] == 'PrivateTest':
            tmp_list = []
            for pixel in row[1].split( ):
                tmp_list.append(int(pixel))
            I = np.asarray(tmp_list)
            PrivateTest_y.append(int(row[0]))
            PrivateTest_x.append(I.tolist())
            
print(np.shape(Training_x))
print(np.shape(PublicTest_x))
print(np.shape(PrivateTest_x))

datafile = h5py.File(datapath, 'w')
datafile.create_dataset('Training_pixel', dtype = 'uint8', data = Training_x)
datafile.create_dataset('Training_label', dtype = 'int64', data = Training_y)
datafile.create_dataset('PublicTest_pixel', dtype = 'uint8', data = PublicTest_x)
datafile.create_dataset('PublicTest_label', dtype = 'int64', data = PublicTest_y)
datafile.create_dataset('PrivateTest_pixel', dtype = 'uint8', data = PrivateTest_x)
datafile.create_dataset('PrivateTest_label', dtype = 'int64', data = PrivateTest_y)
datafile.close()

print("Save data finish!")
                