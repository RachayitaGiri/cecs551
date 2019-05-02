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


def Imidmaker():
    labels=genfromtxt('/home/datasets/annotations/val_labels.csv',delimiter=',')
    N= np.shape(labels)[0]
    print(N)
    
    json_file='/home/datasets/annotations/instances_val2014.json'
    with open(json_file, 'r') as COCO:
        js= json.loads(COCO.read())
    image_names= [None]*N
    for i in range(N):
        print(i)
        for j in range(len(js['images'])):
            if labels[i,0]== js['images'][j]['id']:
                image_names[i]= js['images'][j]['file_name']
    return image_names

Image_names= Imidmaker()

with open('imageID.txt', 'w') as f:
    for item in Image_names:
        f.write("%s\n" % item)