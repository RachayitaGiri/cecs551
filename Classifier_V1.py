from __future__ import print_function

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:08:06 2019

@author: surface
"""
## ImageClassification
# COCO Dataset spring 2019
# Last Updated 4/22/2019

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
import PIL
from PIL import Image
import progressbar as pb


## MyCode 

## Import Images
image_list=[]

allPictures = glob.glob('/home/datasets/%s/*.jpg' % sys.argv[0])
##allPictures = glob.glob('/home/datasets/train2014/*.jpeg')

for file in allPictures:
    im= Image.open(file)
    image_list.append(im)
    
    


print('Image list length is :' , len(image_list))

## Params
classes= 91
power= random.uniform(-6,-2)
lr_rate= 10 ** power
batch_size= 128
epochs= 5
input_shape= 224, 224, 3
print('Lr rate is :', lr_rate)

## model


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), strides= 1 ,padding= 'same', activation='relu', input_shape=input_shape))
model.add(Dropout(0.4))

model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
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
model.add(Dense(classes, activation='softmax'))
model.summary()

'''
### Try optimizers

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=RMSprop(lr=lr_rate),
              metrics=['accuracy'])

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=SGD(lr=lr_rate, momentum=0.9),
              metrics=['accuracy'])

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=Adam(lr=lr_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0001, amsgrad=False),
              metrics=['accuracy'])

## Training & fit
history= model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))




score = model.evaluate(x_test, y_test, verbose=1)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

 
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