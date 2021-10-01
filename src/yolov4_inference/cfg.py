from easydict import EasyDict

Cfg = EasyDict()

Cfg.gpu = '0'
Cfg.use_cuda = False
Cfg.model_input_size = (416, 416)
Cfg.kl = False
Cfg.namesfile = 'coco.names'
Cfg.weightfile = 'yolov4.pth'

