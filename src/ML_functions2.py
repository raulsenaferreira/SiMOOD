import os
import torch
from yolov5 import YOLOv5
from facebook import DETR as detr
import numpy as np
from PIL import Image
from utils.augmentations import letterbox
import argparse
#import os
import sys
from pathlib import Path

import cv2
#import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync



class ML_object:

    def __init__(self, object_detector_model_type):
        self.object_detector_model_type = object_detector_model_type
        self.YOLO_WEIGHTS_PATH = os.path.join("yolov5", "weights", 'yolov5s.pt')
        self.object_detector = None


    def load_model(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device = 'cuda' if torch.cuda.is_available() else 'cpu' ...YOLOv5(model_path, device)

        if self.object_detector_model_type == 'yolo':
            model_path = self.YOLO_WEIGHTS_PATH # it automatically downloads yolov5s model to given path
            self.object_detector = YOLOv5(model_path, device)

        elif self.object_detector_model_type == 'detr':
            self.object_detector = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
            #self.object_detector = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50_dc5', pretrained=True)
            #self.object_detector = torch.hub.load('facebookresearch/detr:main', 'detr_resnet101', pretrained=True)
            #self.object_detector = torch.hub.load('facebookresearch/detr:main', 'detr_resnet101_dc5', pretrained=True)
            self.object_detector = self.object_detector.to(device)
            self.object_detector.eval();


    def make_predictions(self, incoming_image):
        model = self.object_detector
        classes=None  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False  # class-agnostic NMS
        imgsz=640
        conf_thres=0.95  # confidence threshold
        iou_thres=0.45  # NMS IOU threshold
        max_det=10  # maximum detections per image
        stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        imgsz = check_img_size(imgsz, s=stride)

        if self.object_detector_model_type == 'yolo':                                                
            #return self.object_detector.predict(incoming_image)

            # Padded resize
            img = letterbox(np.array(incoming_image), imgsz, stride=stride, auto=True)[0]
            # Convert
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)
            

            img = torch.from_numpy(img).to(device)
            #im = im.permute(2, 0, 1)
            

            

            #im = im.half() if half else im.float()  # uint8 to fp16/32
            #im /= 255  # 0 - 255 to 0.0 - 1.0

            
            #print('imgsz', imgsz)
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim
            #print('im.shape',im.shape)

            

            pred = self.object_detector(img)
            
            # NMS
            predictions = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            predictions = predictions[0].cpu().data.numpy()
            #print('predictions', predictions)

            # bboxes = predictions[:, :4] # x1, x2, y1, y2
            # scores = predictions[:, 4]
            # categories = predictions[:, 5]

            # print('bboxes', bboxes)
            # print('scores', scores)
            # print('categories', categories)

            results = {'names':names, 'predictions':predictions}

            return  results                         
                                                                                                
        elif self.object_detector_model_type == 'detr':                                              
            img = Image.fromarray((incoming_image * 255).astype(np.uint8))                      
            scores, boxes = detr.detect(img, self.object_detector, device)                           
            return [boxes, scores, detr.CLASSES]  


    def load_expert_model(self, threat_type):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device = 'cuda' if torch.cuda.is_available() else 'cpu' ...YOLOv5(model_path, device)
        
        # self.object_detector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=False) #, pretrained=True
        # self.YOLO_WEIGHTS_PATH = os.path.join('yolov5', 'weights', 'yolov5s.pt') #
        
        # ckp = torch.load(self.YOLO_WEIGHTS_PATH)
        # #print('ckp',ckp)
        # self.object_detector.load_state_dict(ckp, strict=False)#
        # self.object_detector = self.object_detector.to(device)
        # self.object_detector.eval()

        # Load model
        self.YOLO_WEIGHTS_PATH = os.path.join('yolov5', 'weights', 'yolov5s.pt')
        device = select_device(device)
        model = DetectMultiBackend(self.YOLO_WEIGHTS_PATH, device=device, dnn=False)
        self.object_detector = model