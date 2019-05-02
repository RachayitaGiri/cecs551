import json
import numpy as np
# -*- coding: utf-8 -*-

def create_labels(json_file):
    with open(json_file, 'r') as COCO:
        js= json.loads(COCO.read())

        annotationMapping = {}

        counter = 0
        for img in annotation:
            if counter < 10:
                if img['image_id'] in annotationMapping:
                    annotationMapping[img['image_id']].append(img['category_id'])
                else:
                    annotationMapping[img['image_id']] = []
                    annotationMapping[img['image_id']].append(img['category_id'])

            counter = counter + 1

        print(str(annotationMapping))

        """ image_labels=np.zeros([len(js['images']),92])
        for i in range(len(js['images'])):
            if i < 100:
                image_labels[i,0]=js['images'][i]['id']
                for anot in js['annotations']:
                    if anot['image_id']==js['images'][i]['id']:
                        image_labels[i,anot['category_id']]=1
            else:
                break 
        """
    
    np.savetxt("val_labels_new.csv", image_labels, delimiter=",")

create_labels('instances_val2014.json')