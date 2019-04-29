import json
import numpy as np
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 17:48:44 2019

@author: surface
"""

def create_labels(json_file):
    with open(json_file, 'r') as COCO:
        js= json.loads(COCO.read())
        image_labels=np.zeros([len(js['images']),92])
        for i in range(len(js['images'])):
            image_labels[i,0]=js['images'][i]['id']
            for anot in js['annotations']:
                if anot['image_id']==js['images'][i]['id']:
                    image_labels[i,anot['category_id']]=1
    
    np.savetxt("val_labels.csv", image_labels, delimiter=",")

create_labels('instances_val2014.json')