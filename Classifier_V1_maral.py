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
from sklearn.metrics import f1_score
## MyCode 

## Import labesl

y_train1= np.load('train_datay.npy')
X_train1= np.load('train_dataX.npy')





y_train= y_train1[0:4000,:]
y_val= y_train1[4000:5000,:]

X_train=np.reshape(X_train1[:,:,:,0:4000], [4000,224,224,3])
X_val= np.reshape(X_train1[:,:,:,4000:], [1000,224,224,3])


print(type(X_train) , 'is type and the shape of the images is:', np.shape(X_train))
print(type(y_train), 'is type and the shape of the labels is:' ,np.shape(y_train))

## Params
classes= 91

batch_size= 128
epochs= 1
input_shape= 224,224,3


## model


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), strides= 1 ,padding= 'same', activation='relu', input_shape=input_shape))
model.add(Dropout(0.4))

model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(Dropout(0.4))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(Dropout(0.4))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(batch_size, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(classes, activation='sigmoid'))
model.summary()


### Try optimizers
'''
beta1= random.uniform(0.85, 0.95)
beta2= random.uniform(0.95, 0,9999)
decay= 10 ** random.uniform(-6,-2)
'''
for i in range(1):
    power= random.uniform(-6,-2)
    lr_rate= 0.01
    #lr_rate= 10 ** power
    print('Lr rate is :', lr_rate)
    
    model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=Adam(lr=lr_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0001, amsgrad=False),
              metrics=['accuracy'])
    ## Training & fit
    history= model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_val, y_val))
    pred = model.predict(X_val)
    pred[pred >= 0.5]=1
    pred[pred<0.5]=0
    
    score= sklearn.metrics.f1_score(y_val, pred)
    loss =keras.losses.binary_crossentropy(y_val, pred)
    
    print('Test loss:', loss)
    print('Test accuracy:', score)
'''

def plot_loss_accuracy(history):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(history.history["loss"],'r-x', label="Train Loss")
    ax.plot(history.history["val_loss"],'b-x', label="Validation Loss")
    ax.legend()
    ax.set_title('cross_entropy loss')
    ax.grid(True)


    ax = fig.add_subplot(1, 2, 2)
    ax.plot(history.history["acc"],'r-x', label="Train Accuracy")
    ax.plot(history.history["val_acc"],'b-x', label="Validation Accuracy")
    ax.legend()
    ax.set_title('accuracy')
    ax.grid(True)
    


plot_loss_accuracy(history)
'''