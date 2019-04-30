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
import sys
from numpy import genfromtxt
import json
import matplotlib.image as mpimg



def loadtraindata():
    labels=genfromtxt('/home/datasets/annotations/train_labels.csv',delimiter=',')
    y_train= labels[:,1:]
    
    
    N= np.shape(labels)[0]
    
    
    json_file='/home/datasets/annotations/instances_train2014.json'
    with open(json_file, 'r') as COCO:
        js= json.loads(COCO.read())
    N=2    
    image_names= [None]*N
    for i in range(N):
        for j in range(len(js['images'])):
            if labels[i,0]== js['images'][j]['id']:
                image_names[i]= js['images'][j]['file_name']
                
    images= np.full((224,224,3,N),0)            
    for i in range(3):
        name=image_names[i]
        images[:,:,:,i]= mpimg.imread('/home/datasets/%s/%s'  % (sys.argv[1], name) )
    
    X_train= images
    
    return X_train, y_train
        
'''

def loadvaldata():
    labels=genfromtxt('/home/datasets/annotations/val_labels.csv',delimiter=',')
    y_val= labels[:,1:]
    
    
    N= np.shape(labels)[0]
    
    
    json_file='/home/datasets/annotations/instances_val2014.json'
    with open(json_file, 'r') as COCO:
        js= json.loads(COCO.read())
    image_names= [None]*N
    for i in range(N):
        for j in range(len(js['images'])):
            if labels[i,0]== js['images'][j]['id']:
                image_names[i]= js['images'][j]['file_name']
                
    images= np.full((224,224,3,N),0)            
    for i in range(3):
        name=image_names[i]
        images[:,:,:,i]= mpimg.imread('/home/datasets/%s/%s'  % (sys.argv[2], name) )
    
    X_val= images
    
    return X_val, y_val      
'''
X_train, y_train= loadtraindata()



print(type(X_train) , 'is type and the shape of the train images is:', np.shape(y_train))
print(type(y_train), 'is type and the shape of the train labels is:' ,np.shape(X_train))
'''
X_val, y_val= loadvaldata()

print(type(X_val) , 'is type and the shape of the val images is:', np.shape(y_val))
print(type(y_val), 'is type and the shape of the val labels is:' ,np.shape(X_val))


'''
'''
np.save("train_dataX", X_train)
np.save("train_datay", y_train)
np.save("val_datay", y_val)
np.save("val_dataX", X_val)
''' 
    