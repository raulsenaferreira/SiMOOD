import os, os.path
import numpy as np
import math
from PIL import Image
import albumentations as AUG
import shutil
import pystache, yaml
from distutils.dir_util import copy_tree
import corruptions


def generate_yml(num_individual):
    input = """
    {{#data_paths}}
    path: temp  # dataset root dir
    train: '{{.}}/images/train2017'  # train images (relative to 'path') 128 images
    val: '{{.}}/images/train2017'  # val images (relative to 'path') 128 images
    test:  # test images (optional)

    # Classes
    nc: 80  # number of classes
    names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush']  # class names
    {{/data_paths}}
    """
    result = pystache.render(input, {"data_paths": num_individual})
    
    return result


def transform_images(data_path, labels_path, num_individual, technique, lvl):
    prefix_yaml = 'transformed_coco128.yaml'
    folder_to_save = os.path.join('temp', '{}/images/train2017'.format(str(num_individual)))
    isExist = os.path.exists(folder_to_save)

    if not isExist:
        # Create a new directory because it does not exist 
        os.makedirs(folder_to_save)
    # else:
    #     # remove temp datasets and create new ones for new generated data
    #     shutil.rmtree(folder_to_save)
    #     os.makedirs(folder_to_save)

    for f in os.listdir(data_path):
        file_path = os.path.join(data_path,f)
        extension = os.path.splitext(file_path)[1]

        #if extension == '.jpg':
        try:
            if lvl!=0:
                img = Image.open(file_path)
                img = np.asarray(img)
                img = img / 255
                img = np.array(img, dtype=np.float32)
                new_img = corruptions.apply_threats(img, technique, lvl)
                new_img = Image.fromarray(np.uint8(new_img))
                #new_img = Image.fromarray(new_img.astype(np.uint8))
                new_img.save(os.path.join(folder_to_save, f))
            else:
                img = Image.open(file_path)
                img.save(os.path.join(folder_to_save, f))

        except Exception as e:
            print('file {} not processed due to the error: {}'.format( file_path, e))
        #else:
            #print('image extension not allowed: ', extension)

    #copy labels from COCO folder to the new generated one
    labels = os.path.join(os.path.join('temp', '{}'.format(str(num_individual)),'labels', 'train2017'))
    isExist = os.path.exists(labels)

    if not isExist:
        # Create a new directory because it does not exist 
        os.makedirs(labels)
    copy_tree(labels_path, labels)

    result_yaml = generate_yml(num_individual)
    #np.savetxt('./temp/{}.yaml'.format(num_individual), result_yaml)
        
    with open('temp/{}/{}'.format(num_individual, prefix_yaml), 'w') as file:
        file.write(result_yaml)
        #file.close()

    return folder_to_save