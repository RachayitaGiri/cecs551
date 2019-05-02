import json
import numpy as np
# -*- coding: utf-8 -*-

def create_labels(json_file):
    with open(json_file, 'r') as COCO:
        js= json.loads(COCO.read())

        annotationMapping = {}

        counter = 0
        for img in js['annotations']:
            if counter < 100:
                if img['image_id'] in annotationMapping:
                    annotationMapping[img['image_id']].append(img['category_id'])
                else:
                    annotationMapping[img['image_id']] = []
                    annotationMapping[img['image_id']].append(img['category_id'])
            else:
                break

            counter = counter + 1

        print(str(annotationMapping))

        image_labels=np.zeros([len(js['images']),92])
        for i in range(len(js['images'])):
            if i < 100:
                image_labels[i,0]=js['images'][i]['id']
                try:
                    image_labels[i,1]=annotationMapping[js['images'][i]['id']]
                except:
                    pass
            else:
                break 
    
    np.savetxt("val_labels_new.csv", image_labels, delimiter=",")

create_labels('instances_val2014.json')