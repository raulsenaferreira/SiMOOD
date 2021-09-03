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


    #'''
    def experiment_lol(image):
        # https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/4
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                #print('output', output)
                activation[name] = output[0].detach()
                #print('activation[name]', name, np.shape(activation[name]))
            return hook


        for name, layer in model.named_modules():
            #print('name', name)
            layer.register_forward_hook(get_activation(name))

        #x = torch.randn(1, 10)
        #model_input_size = (608, 608)
        #x = cv2.resize(image, (model_input_size[1], model_input_size[0]))
        #x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        #input_tensor = torch.tensor(image)
        #output = model(input_tensor)
        
        #for key in activation:
            #print('key', key)
            #print('activation[key]', activation[key])


        #print('activation[key]', activation['head.conv17.conv'])
    #'''
    activation = {}
    relu_activ = {}
    key_layer = 'head.conv17.conv.2'
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output[0].detach().cpu().numpy()
            #if name == key_layer:
                #print('activation[name]', name, np.shape(activation[name]))
                #print(name, activation[name][0])
        return hook

    # Accept new connections, get images sent via these connections, and return the resulting bounding boxes
    while True:
        try:
            conn = listener.accept()
            image = conn.recv()
            detection = detect_objects(model, image, kl=kl, model_input_size=model_input_size, use_cuda=use_cuda, class_names=class_names)

            #experiment_lol(image)
            
            #print('MODEL =============== ', model)
            #print('MODEL last leaky relu =============== ', list(model.head.conv17.conv.parameters()))
            #print('MODEL last leaky relu =============== ', dir(model.head.conv17.conv[2]))
            
            #'''
            # https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/4
            #model_input_size = (608, 608)
            #sized = cv2.resize(image, (model_input_size[1], model_input_size[0]))
            #sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
            #model.head.conv17.conv[2].register_forward_hook(get_activation('head.conv17.conv.2'))
            #model.neek.register_forward_hook(get_activation('neek'))
            #input_tensor = torch.tensor(sized)
            #output = model(input_tensor)

            #for name, layer in model.named_modules():
                #print('name', name)
            #    layer.register_forward_hook(get_activation(name))

            model.head.conv17.conv[2].register_forward_hook(get_activation('head.conv17.conv.2'))

            #for key in activation:
            #    print('key', key)
            try:
                print('TEST', np.shape(activation['head.conv17.conv.2']), activation['head.conv17.conv.2'])
            except:
                pass
            #'''

            conn.send(detection)
            #print('detection ================= ',detection)
            conn.close()
        except (EOFError, ConnectionResetError):
            # Just listen for a new connection
            pass


if __name__ == '__main__':
    main(**Cfg)
