import torch
from yolov5 import YOLOv5
from facebook import DETR as detr
import numpy as np
from PIL import Image

class ML_object:

    def __init__(self, object_detector_model_type, use_expert_model=False, threat_type=None):

        self.object_detector_model_type = object_detector_model_type

        self.YOLO_WEIGHTS_PATH = "yolov5/weights/yolov5s.pt" # default

        if use_expert_model and threat_type!=None:
            print('using a yolo model trained with data transformed with {} ...'.format(threat_type))
            self.YOLO_WEIGHTS_PATH = 'runs/train/{}/weights/best.pt'.format(threat_type)

        self.object_detector = None


    def load_model(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device = 'cuda' if torch.cuda.is_available() else 'cpu' ...YOLOv5(model_path, device)

        if self.object_detector_model_type == 'yolo':
            model_path = self.YOLO_WEIGHTS_PATH 
            self.object_detector = YOLOv5(model_path, device)

        elif self.object_detector_model_type == 'detr':
            self.object_detector = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
            #self.object_detector = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50_dc5', pretrained=True)
            #self.object_detector = torch.hub.load('facebookresearch/detr:main', 'detr_resnet101', pretrained=True)
            #self.object_detector = torch.hub.load('facebookresearch/detr:main', 'detr_resnet101_dc5', pretrained=True)
            self.object_detector = self.object_detector.to(device)
            self.object_detector.eval();


    def make_predictions(self, incoming_image):
        if self.object_detector_model_type == 'yolo':                                                
            return self.object_detector.predict(incoming_image)                                   
                                                                                                
        elif self.object_detector_model_type == 'detr':                                              
            img = Image.fromarray((incoming_image * 255).astype(np.uint8))                      
            scores, boxes = detr.detect(img, self.object_detector, device)                           
            return [boxes, scores, detr.CLASSES] 