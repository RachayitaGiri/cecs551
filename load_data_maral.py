from __future__ import print_function

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:08:06 2019

@author: surface
"""
## ImageClassification
# COCO Dataset spring 2019
# Last Updated 4/22/2019
from matplotlib import image
import csv
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop, Adam
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras.layers import Activation
from keras.utils import np_utils
import random
import glob
import os, sys
import progressbar as pb
from numpy import genfromtxt
import json

X_train, y_train= loadtraindata()
np.savetxt("train_dataX.csv", X_train, delimiter= ",")
np.savetxt("train_datay.csv", y_train, delimiter= ",")


def loadtraindata():
    labels=genfromtxt('/home/datasets/annotations/train_labels.csv',delimiter=',')
    y_train= labels[:,1:]
    
    #Image_ids= labels[:,0]
    #N= np.shape(Image_ids)[0]
    N=5
    
    json_file='/home/datasets/annotations/instances_train2014.json'
    with open(json_file, 'r') as COCO:
        js= json.loads(COCO.read())
    image_names= [None]*N
    for i in range(N):
        for j in range(len(js['images'])):
            if labels[i,0]== js['images'][j]['id']:
                image_names[i]= js['images'][j]['file_name']
                
    images= np.zeros(224,224,3,N)            
    for i in range(N):
        images[:,:,:,i]= image.imread('/home/datasets/%s/',image_names[i])
    
    X_train= images
    
    return X_train, y_train
        
        

    
    
