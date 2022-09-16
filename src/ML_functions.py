import torch
from yolov5 import YOLOv5
from facebook import DETR as detr
import numpy as np
from PIL import Image
from models.yolo import Model
import yaml
from utils.general import intersect_dicts, check_img_size
import cv2 



class ML_object:

    def __init__(self, object_detector_model_type, use_expert_model=False, threat_type=None, augmented_data_percentage='no'):

        self.object_detector_model_type = object_detector_model_type
        self.device = None
        self.YOLO_WEIGHTS_PATH = "yolov5/weights/yolov5s.pt" # default
        self.augmented_data_percentage = augmented_data_percentage

        if use_expert_model==True and threat_type!=None:
            print('using a yolo model trained with data transformed with {} ...'.format(threat_type))
            self.YOLO_WEIGHTS_PATH = 'runs/train/{}/weights/best.pt'.format(threat_type)

        elif self.augmented_data_percentage!='no': # 1, 10, 25, 50, 100
            print('using a yolo model trained with {}% of augmented data ...'.format(self.augmented_data_percentage))
            self.YOLO_WEIGHTS_PATH = 'src/evolutionary_step/runs/train/{}_augmentation/weights/best.pt'.format(self.augmented_data_percentage)

        self.object_detector = None


    def load_model(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device = 'cuda' if torch.cuda.is_available() else 'cpu' ...YOLOv5(model_path, device)

        model_path = self.YOLO_WEIGHTS_PATH

        if self.object_detector_model_type == 'custom_yolo':
            #print("\n\nusing custom yolo")
             
            if self.augmented_data_percentage!='no':

                from models.common import DetectMultiBackend

                # nc = 80
                # cfg = 'src/evolutionary_step/models/yolov5s.yaml'.format(self.augmented_data_percentage)
                # hyp = 'src/evolutionary_step/runs/train/{}_augmentation/hyp.yaml'.format(self.augmented_data_percentage)
                # if isinstance(hyp, str):
                #     with open(hyp, errors='ignore') as f:
                #         hyp = yaml.safe_load(f)  # load hyps dict
                # resume = False
                # ckpt = torch.load(model_path, map_location=self.device)  # load checkpoint
                # model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(self.device)  # create
                # exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
                # csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
                # csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
                # model.load_state_dict(csd, strict=False)  # load
                dnn=False
                model = DetectMultiBackend(model_path, device=self.device, dnn=dnn)
                #stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
                #imgsz=640
                #imgsz = check_img_size(imgsz, s=stride)

                self.object_detector = model
        elif self.object_detector_model_type == 'yolo':
            #print("\n\nusing normal yolo")
            self.object_detector = YOLOv5(model_path, self.device)

        elif self.object_detector_model_type == 'detr':
            self.object_detector = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
            #self.object_detector = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50_dc5', pretrained=True)
            #self.object_detector = torch.hub.load('facebookresearch/detr:main', 'detr_resnet101', pretrained=True)
            #self.object_detector = torch.hub.load('facebookresearch/detr:main', 'detr_resnet101_dc5', pretrained=True)
            self.object_detector = self.object_detector.to(self.device)
            self.object_detector.eval();


    def make_predictions(self, incoming_image):
        #print('object_detector_model_type',self.object_detector_model_type)
        if self.object_detector_model_type == 'yolo': 
            #print(np.shape(incoming_image))  # (720,1280,3)                                             
            return self.object_detector.predict(incoming_image)

        elif self.object_detector_model_type == 'custom_yolo': 
            #print('np.shape(incoming_image)',np.shape(incoming_image))
            
            # pytorch shape format
            img_size = 640
            h0, w0 = incoming_image.shape[:2]  # orig hw
            #print('h0, w0',h0, w0)
            r = img_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                incoming_image = cv2.resize(incoming_image, (img_size, img_size),
                                interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
                                #cv2.resize(incoming_image, (int(w0 * r), int(h0 * r)),
                                #interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
            #print('np.shape(incoming_image) after resize', np.shape(incoming_image))
            
            incoming_image = incoming_image.reshape(3, incoming_image.shape[0], incoming_image.shape[1])
            #print('np.shape(incoming_image) after resize and reshape', np.shape(incoming_image))

            im = torch.from_numpy(incoming_image).to(self.device)
            #print('np.shape(im) pytorch from numpy', np.shape(im))

            im = im.float()#im.half() if half else im.float()  # uint8 to fp16/32
            #im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim 
            #print('np.shape(im) after adding a new dimension', np.shape(im))     
            
            prediction_tensor = self.object_detector(im, augment=False, visualize=False)  
            prediction = prediction_tensor #prediction_tensor.cpu().detach().numpy()[0]
            
            #print('np.shape(prediction)',np.shape(prediction))         
            
            return (prediction,self.object_detector.names)

        elif self.object_detector_model_type == 'detr':                                              
            img = Image.fromarray((incoming_image * 255).astype(np.uint8))                      
            scores, boxes = detr.detect(img, self.object_detector, self.device)                           
            return [boxes, scores, detr.CLASSES] 