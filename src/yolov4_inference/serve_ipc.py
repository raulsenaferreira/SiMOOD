import Yolov4_KL_demo.models as models
from Yolov4_KL_demo.tool.utils import load_class_names, plot_boxes_cv2, plot_boxes_cv2_kl
from Yolov4_KL_demo.tool.torch_utils import do_detect

import torch
import numpy as np
import cv2
from typing import *
import argparse
import multiprocessing
import os

from easydict import EasyDict
from cfg import Cfg

from safety_monitors import OOB


def detect_objects(model, image: np.ndarray, kl=True, model_input_size = (608, 608), use_cuda=True, class_names=None) -> List[Dict[str, Union[List[int], str, float]]]:
    width, height = image.shape[1], image.shape[0]
    sized = cv2.resize(image, (model_input_size[1], model_input_size[0]))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    boxes = do_detect(model, sized, 0.2, 0.4, use_cuda)
    boxes = boxes[0]

    detections = []

    for i in range(len(boxes)):
        box = boxes[i]
        box[0] *=width
        box[1] *= height
        box[2] *= width
        box[3] *= height

        if not(class_names is None):
            label = class_names[box[5]]
        else:
            label = box[5]

        detection =  {
            'bounding_box': box[:4],
            'label': label,
            'score': box[4]
        }

        if kl:
            sigma_x1 = int(box[6] * width)
            sigma_y1 = int(box[7] * height)
            sigma_x2 = int(box[8] * width)
            sigma_y2 = int(box[9] * height)
            detection['stds'] = [sigma_x1, sigma_y1, sigma_x2, sigma_y2]

        detections.append(detection)

    return detections


def main(**Cfg):
    cfg = EasyDict(Cfg)
    model_input_size = cfg.model_input_size
    namesfile = cfg.namesfile
    class_names = load_class_names(os.path.join(os.path.dirname(__file__), namesfile))
    num_classes = len(class_names)
    kl = cfg.kl
    weightfile = os.path.join(os.path.dirname(__file__), cfg.weightfile)

    # Create and load model
    if not kl:
        model = models.Yolov4(yolov4conv137weight=None, n_classes=num_classes, inference=True)
    else:
        model = models.Yolov4_KL(yolov4conv137weight=None, n_classes=num_classes, inference=True, get_vars=True)

    pretrained_dict = torch.load(weightfile, map_location=torch.device('cuda'))
    model.load_state_dict(pretrained_dict)

    use_cuda = True
    if use_cuda:
        model.cuda()

    # Create a listener for new connections
    address = ('localhost', 6000)
    listener = multiprocessing.connection.Listener(address, authkey=b'password')


    activation = {}
    relu_activ = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output[0].detach().cpu().numpy() 
        return hook
    
    def hook_registers_for_all_layers(image):
        for name, layer in model.named_modules():
            #print('name', name)
            layer.register_forward_hook(get_activation(name))

    # Accept new connections, get images sent via these connections, and return the resulting bounding boxes
    while True:
        try:
            conn = listener.accept()
            image = conn.recv()
            detection = detect_objects(model, image, kl=kl, model_input_size=model_input_size, use_cuda=use_cuda, class_names=class_names)

            ##### for OOB safety monitor
            key_layer = model.head.conv17.conv[2]
            layer_name = 'head.conv17.conv.2'

            key_layer.register_forward_hook(get_activation(layer_name))

            #for key in activation:
            #    print('key', key)
            
            OOB(activation, layer_name)
            ###############

            conn.send(detection)
            #print('detection ================= ',detection)
            conn.close()
        except (EOFError, ConnectionResetError):
            # Just listen for a new connection
            pass


if __name__ == '__main__':
    main(**Cfg)
