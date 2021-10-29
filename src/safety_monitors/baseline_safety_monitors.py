### A set of baseline monitors

import numpy as np
from PIL import Image
import cv2 as cv
import pygame
import safe_regions
#from FFA2 import img_dehazing as ffa
#from PSD import img_dehazing as psd
from AOD import main as aod #img dehazing



########### helpers ###################

def is_rect_overlap(bbox,R2):
    R1 = pygame.Rect(bbox)

    if R1.colliderect(R2):
        return True
    else:
        return False

######################### detection methods #########################
def detect_dangerous_fog(img, react):
    
    is_dangerous_fog = False

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

    is_dangerous_fog = var < FOG_THRESHOLD
    #print('FOGGY ???',fog)

    if react and is_dangerous_fog:
        dehazed_img = image_dehazing(img)
        #dehazed_img = img # do nothing, for debug purposes
        return dehazed_img, is_dangerous_fog
    else:
        return img, is_dangerous_fog

def detect_if_different_ODD(detection, react):
    # verifies if ML prediction makes sense according to the context 
    # (ex: a class donut probably will not cross the street)
    different_ODD = True

    return different_ODD

def detect_dangerous_approximation(data_obj, react=False):
    # verifies if detected object has suddenly disapeared when it's actualy in a dangerous zone
    # Ex: pedestrian walking towards to the street suddenly disapears
    frame = 0
    array_data = data_obj[0]
    surface = data_obj[1]

    has_temporal_incoherence = False

    warning_region = safe_regions.WARNING_AREA
    warning_region_rect = pygame.draw.polygon(surface, 0, warning_region) #{'x':515, 'y':942, 'w':525, 'h':954}

    for array_data_per_frame in array_data:
        frame+=1
        closest_rect_horizontal = 1000
        closest_rect_vertical = 1000

        for obj in array_data_per_frame:
            bbox = obj.bbox #id, bbox, label, dist_eagle_vehicle
            id_obj = obj.id
            
            rect = pygame.Rect(bbox)
            horizontal_distance = min(rect.x + rect.w - warning_region_rect.x, warning_region_rect.x + warning_region_rect.w - rect.x)                      
            
            closest_rect_horizontal = horizontal_distance if horizontal_distance < closest_rect_horizontal else closest_rect_horizontal
            #print('horizontal_distance for obj', id_obj, horizontal_distance*-1)

            vertical_distance = min(rect.y + rect.h - warning_region_rect.y, warning_region_rect.y + warning_region_rect.h - rect.y) 
                                        
            closest_rect_vertical = vertical_distance if vertical_distance < closest_rect_vertical else closest_rect_vertical
            #print('vertical_distance for obj', id_obj, vertical_distance*-1)

        print('closest_rect_horizontal in frame {}: {}'.format(frame, closest_rect_horizontal*-1))
        print('closest_rect_vertical in frame {}: {}'.format(frame, closest_rect_vertical*-1))

    #dx = closest_rect_tn - closest_rect_t0
    #dy = closest_rect_tn_2 - closest_rect_t0_2
    print('====================')

    corrector_factor = -40 # correction factor for the warning rectangle 
    # print('(dx, dy)', dx, dy)
    # #if np.abs(dx) > pixel_threshold:
    # if dx < 0:
    #     print('Pedestrian approaching. Closest bbox value:', dx)
    #     if dy < 0:
    #         print('Low the speed!!!')

    # ensure there is significant movement in the x-direction
    # if np.abs(dx) > pixel_threshold:
    #     direction = "East" if np.sign(dx) == 1 else "West"

    # # ensure there is significant movement in the y-direction
    # if np.abs(dy) > pixel_threshold:
    #     dirY = "North" if np.sign(dy) == 1 else "South"

    # # handle when both directions are non-empty
    # if dirX != "" and dirY != "":
    #     direction = "{}-{}".format(dirY, dirX)

    # # otherwise, only one direction is non-empty
    # else:
    #     direction = dirX if dirX != "" else dirY

    #print('direction:', direction)

    # rect1 = {'x':515, 'y':942, 'w':525, 'h':954}
    # rect2 = {'x':382, 'y':938, 'w':508, 'h':960}
    # test = min(rect1['x']+rect1['w']-rect2['x'],rect2['x']+rect2['w']-rect1['x'])
    #print('test', test)

    return None, has_temporal_incoherence


def detect_probable_unseen_object(image, react):
    # apply the concept of introspection
    # the goal is decrease false negatives in the task of detecting an object
    is_there_an_obj = False

    return is_there_an_obj

def detect_temporal_incoherence(data_obj, react):
    # verifies if detected target object just teleported from a region to another, or
    # if detected target object touchs two regions at the same time 
    # as a result, this method delete the object due to a possible temporal incoherence
    # the goal is to decrease false positives in the task of detecting a target class
    
    has_temporal_incoherence = False

    target_class = 'person'
    
    has_intersection_previous_region = False
    has_intersection_next_region = has_intersection_previous_region

    array_data = data_obj[0]
    surface = data_obj[1]

    previous_region = safe_regions.WARNING_AREA
    previous_region_rect = pygame.draw.polygon(surface, 0, previous_region)
    next_region = safe_regions.DANGER_AREA
    next_region_rect = pygame.draw.polygon(surface, 0, next_region)

    for i in range(len(array_data)):
        detected_objects_per_frame = array_data[i]
        
        for j in range(len(detected_objects_per_frame)):
            det_object = detected_objects_per_frame[j]

            bbox = det_object[0]
            score = det_object[1]
            label = det_object[2]
            
            if is_rect_overlap(bbox, previous_region_rect):
                if label == target_class:
                    has_intersection_previous_region = True
            if is_rect_overlap(bbox, next_region_rect):
                if label == target_class:
                    has_intersection_next_region = True

            if (has_intersection_previous_region == False and has_intersection_next_region) or (has_intersection_previous_region and has_intersection_next_region):
                array_data[i][j] = None
                has_temporal_incoherence = True

    return array_data, has_temporal_incoherence


######################### reaction methods #########################
def image_dehazing(img):
    #img = ffa.run('its', img*255)
    #img = psd.run(img)
    img = aod.unfog_image(img)
    
    resized_img = np.reshape(img[0], (480,640,3))#np.transpose(img[0])
    #print('np.shape(resized_img)', np.shape(resized_img))
    return resized_img



#mapping
THREATS = {'smoke': detect_dangerous_fog, 
           'fog': detect_dangerous_fog,
           'temporal_incoherence': detect_temporal_incoherence
    }



class Safety_monitor:
    def __init__(self, data, verification, react=False):
        self.data = data
        self.verification = verification
        self.react = react

    def run(self):
        try:
            if self.verification == 'pre':
                #verifying threats on the image (it should be parallelized)
                #smoke/fog/haze
                monitor = THREATS['smoke']

            elif self.verification == 'post':
                #verifying threats after the object detection task on the image (it should be parallelized)
                monitor = THREATS['temporal_incoherence']

            image, has_reacted = monitor(self.data, self.react)

            return image, has_reacted

        except Exception as e:
            with open("src/log/log_safety_monitor.log", "a") as myfile:
                myfile.write(str(e))