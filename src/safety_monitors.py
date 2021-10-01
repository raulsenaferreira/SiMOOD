import numpy as np
from PIL import Image
import cv2 as cv

from FFA2 import LAAS as ffa

'''class Safety_monitor:
    def __init__(self, monitor_type, data):
        self.monitor_type = monitor_type
        self.data = data'''

# detection methods
def foggy(img):
    dangerous_fog = False

    def slow_horizontal_variance(im):
        '''Return average variance of horizontal lines of a grayscale image'''
        #width, height = np.shape(im)[1], np.shape(im)[0]
        width, height = im.size
        if not width or not height: return 0
        vars = []
        pix = im.load()
        for y in range(height):
            row = [pix[x,y] for x in range(width)]
            mean = sum(row)/width
            variance = sum([(x-mean)**2 for x in row])/width
            vars.append(variance)
        return sum(vars)/height

    FOG_THRESHOLD = 1580
    PIL_image = Image.fromarray(np.uint8(img)).convert('RGB')
    PIL_image = Image.fromarray(img.astype('uint8'), 'RGB')
    im = PIL_image.convert('L')
    var = slow_horizontal_variance(im)

    dangerous_fog = var < FOG_THRESHOLD
    #print('FOGGY ???',fog)

    return dangerous_fog


# reaction methods
def image_dehazing(img):
    img = ffa.run('ots', img)
    return img