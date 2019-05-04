from __future__ import print_function

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:08:06 2019

@author: surface
"""
## ImageClassification
# COCO Dataset spring 2019
# Last Updated 4/22/2019
import matplotlib

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.layers import Activation
from keras.utils import np_utils
import random
from sklearn.metrics import f1_score
import tensorflow as tf
from numpy import genfromtxt
## MyCode 

## Import labesl



from scripts.coco_dataset import *
'''
loc_label= '/home/cecs551/annotations/val_labels.csv'
loc_image='/home/cecs551/img2array.csv'
loc_image2= '/home/cecs551/img2array_pt2.csv'
y_train1= genfromtxt(loc_label, delimiter=',')
X_train1= genfromtxt(loc_image, delimiter=',')
#X_train2= genfromtxt(loc_image2, delimiter=',')

print(type(X_train1) , 'is type and the shape of the images is:', np.shape(X_train1))
#print(type(X_train2) , 'is type and the shape of the images is:', np.shape(X_train2))
print(type(y_train1), 'is type and the shape of the labels is:' ,np.shape(y_train1))
'''





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


### Hyper parameter tuning


def Tune_optimizer_param():
    for i in range(1):
        beta1= random.uniform(0.85, 0.95)
        beta2= random.uniform(0.95, 0,9999)
        decay= 10 ** random.uniform(-6,-2)
        lr_rate= 0.0001
        model.compile(loss=keras.losses.binary_crossentropy,
                      optimizer=Adam(lr=lr_rate, beta_1=beta1, beta_2=beta2, epsilon=None, decay=decay, amsgrad=False),
                      metrics=['accuracy'])
        history= model.fit(X_train, y_train,
                           batch_size=batch_size,
                           epochs=epochs,
                           verbose=1,
                           validation_data=(X_val, y_val))
        
        pred = model.predict(X_val)
        pred[pred >= 0.5]=1
        pred[pred<0.5]=0
        print('beta1, beta2, and decay are', beta1 , beta2, decay)
        score= f1_score(y_val, pred, average= 'samples')
        print('Test accuracy:', score)
        pred = tf.convert_to_tensor(pred, np.float64)
        loss =keras.losses.binary_crossentropy(y_val, pred)
        print('Test loss:', loss)
        return history
        
    
'''

def Tune_Learning_rate():
    epochs=1
    for i in range(100):
        k= random.uniform(1,9)
        lr_rate= k* 10 ** (-4)
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
        
        score= f1_score(y_val, pred, average= 'samples')
        print('Test accuracy:', score)
        pred = tf.convert_to_tensor(pred, np.float64)
        loss =keras.losses.binary_crossentropy(y_val, pred)
        print('Test loss:', loss)
        return history
        
    
    


def plot_loss_accuracy(history):
    matplotlib.use('GTK')
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
    matplotlib.pylot.savfig("ax.png")
    


plot_loss_accuracy(history)
'''