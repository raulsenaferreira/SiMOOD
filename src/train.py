from yolov5 import train, val, detect, export
import argparse


argparser = argparse.ArgumentParser(description='CARLA Scenario Generator')

argparser.add_argument(
    '-t', '--threat_type',
   type=str,
   default='none',
   choices=['no_learning', 'sun_flare', 'channel_dropout', 'channel_shuffle', 'heavy_smoke', 'heavy_gaussian_noise', 'heavy_gaussian_blur', 'grid_dropout', 'coarse_dropout', 'snow'],
   help='If and how much fog should be present')

args = argparser.parse_args()

train.run(imgsz=640, data='data/{}.yaml'.format(args.threat_type), workers=16)
#val.run(imgsz=640, data='data/test.yaml', weights='../yolov5/weights/yolov5s.pt')
#detect.run(imgsz=640)
#export.run(imgsz=640, weights='best_weights/yolov5s.pt')