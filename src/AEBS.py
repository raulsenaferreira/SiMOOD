import safe_regions
from shapely.geometry import Polygon
import pygame
from typing import *
import numpy as np
import time


def get_distance_by_camera(bbox):
    ## Distance Measurement for each bounding box
    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    ## item() is used to retrieve the value from the tensor
    distance = (2 * 3.14 * 180) / (w.item()+ h.item() * 360) * 1000 + 3 ### Distance measuring in Inch 
    #feedback = ("{}".format(detection['label'])+ " " +"is"+" at {} ".format(round(distance/39.37, 2))+"Meters")

    return "{} meters".format(round(distance, 3)) # meters   /39.37


def draw_safety_margin(pygame, surface, color, lines, thickness):
    # pygame.draw.line(surface, color, lines[0], lines[1], thickness)
    # pygame.draw.line(surface, color, lines[1], lines[2], thickness)
    # pygame.draw.line(surface, color, lines[2], lines[3], thickness)

    #pygame.draw.line(bb_surface, color, lines[0], lines[3], thickness)
    ## left, top, width, height
    rect = pygame.draw.polygon(surface, color, lines, thickness)
    return rect


def is_rect_overlap(bbox,R2):
    R1 = pygame.Rect(bbox)

    if R1.colliderect(R2):
        return True
    else:
        return False


def emergency_braking(world, 
                carla,
                display,
                safety_surface,
                WARNING_AREA,
                DANGER_AREA,
                detected_objects_per_frame,
                args, 
                individual):
    
    AEBS_activated = False

    #print('bboxes', bboxes)
    #print('scores', scores)
    #print('labels', labels)

    player = None
    for actor in world.world.get_actors():
        try:
            if actor.attributes.get('role_name') == 'hero':
                player = actor
                break
        except:
            pass
    ctrl = carla.VehicleControl()

    label_dawing_operations = []
    
    for det_object in detected_objects_per_frame:
        #if det_object is not None:
        bbox = det_object[0]
        score = det_object[1]
        label = det_object[2]

        #Verify if an object enters in the warning/danger areas
        if is_rect_overlap(bbox, WARNING_AREA):
            draw_safety_margin(pygame, safety_surface, "yellow", safe_regions.WARNING_AREA, 5)

            if label == 'person':
                draw_safety_margin(pygame, safety_surface, "red", safe_regions.DANGER_AREA, 5)
                AEBS_activated = True
                #ctrl.steer = -0.7
                ctrl.brake = 0.99
                ctrl.hand_brake=True
                #carla.Vehicle.set_autopilot(False)
                carla.Vehicle.apply_control(player, ctrl)
                #time.sleep(2)
                #player.set_simulate_physics(False)

        elif is_rect_overlap(bbox, DANGER_AREA):
            draw_safety_margin(pygame, safety_surface, "red", safe_regions.DANGER_AREA, 5)

            if label == 'person':
                AEBS_activated = True
                #ctrl.steer = -0.7
                ctrl.brake = 0.99
                ctrl.hand_brake=True
                #carla.Vehicle.set_autopilot(False)
                carla.Vehicle.apply_control(player, ctrl)
                #time.sleep(2)
                #player.set_simulate_physics(False)

        colhist = world.collision_sensor.get_collision_history()

        if len(colhist) > 0:
            threat,intensity = individual[0]
            # save info about the frame and the params that led to the hazard
            with open("src/results/hazards/{0}_{1}.csv".format(str(threat),str(intensity)), "a") as myfile:
                myfile.write(('ML MODEL: {}; THREAT: {}; LEVEL: {}; DETAILS: {} \n').format(str(args.object_detector_model_type),
                 str(threat), str(intensity), str(colhist)))

            #terminates the program if there is a hazard
            quit()

        if AEBS_activated:
            return AEBS_activated

    display.blit(safety_surface, (0, 0))

    for ldo in label_dawing_operations:
        ldo()

    return AEBS_activated