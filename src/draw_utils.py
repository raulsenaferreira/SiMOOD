import safe_regions
from shapely.geometry import LineString
from shapely.geometry import Polygon
import pygame
from typing import *
import numpy as np


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


def bbox_conversion(box, width, height):
    # normalizing data    
    box[0] *=width
    box[1] *= height
    box[2] *= width
    box[3] *= height

    # correcting bounding box offset related to the center of the image
    box[0] = box[0] - box[2]/2
    box[1] = box[1] - box[3]/2

    return [int(box[0]), int(box[1]), int(box[2]), int(box[3])]


def is_rect_overlap(bbox,R2):
    R1 = pygame.Rect(bbox)

    if R1.colliderect(R2):
        return True
    else:
        return False


def draw_bboxes(world, 
                carla,
                camera_manager,
                display,
                view_width,
                view_height,
                results,
                args,
                show_distance= False):

    bboxes = None
    scores = None
    categories = None

    # parse results
    if args.object_detector_type == 'yolo':
        #predictions = results.pred[0]
        predictions = results.xywhn[0] #xywh normalized
        bboxes = predictions[:, :4] # x1, x2, y1, y2
        scores = predictions[:, 4]
        categories = predictions[:, 5]

    elif args.object_detector_type == 'detr':
        bboxes = results[0]
        scores = results[1]
        categories = results[2]

    #print('bboxes', bboxes)
    #print('scores', scores)
    #print('categories', categories)

    player = None
    for actor in world.world.get_actors():
        if actor.attributes.get('role_name') == 'hero':
            player = actor
            break
    ctrl = carla.VehicleControl()

    # Create surface for the bounding boxes
    bb_surface = pygame.Surface((view_width, view_height))
    bb_surface.set_colorkey((0, 0, 0))

    # Create a surface for the confidences
    conf_surface = pygame.Surface((view_width, view_height))
    conf_surface.set_colorkey((0, 0, 0))
    conf_surface.set_alpha(80)

    # Create a surface for the labels
    label_surface = pygame.Surface((view_width, view_height))
    label_surface.fill((128, 128, 128))
    label_surface.set_colorkey((128, 128, 128))

    # Create a surface for the safety margins
    safety_surface = pygame.Surface((view_width, view_height))
    safety_surface.set_colorkey((0, 0, 0))

    # Initialize font if not done already
    if not hasattr(camera_manager, 'bb_font'):
        camera_manager.bb_font = pygame.font.SysFont('Monospace', 25)
        camera_manager.bb_font.set_bold(True)

    WARNING_AREA = draw_safety_margin(pygame, safety_surface, "green", safe_regions.WARNING_AREA, 5)
    DANGER_AREA = draw_safety_margin(pygame, safety_surface, "green", safe_regions.DANGER_AREA, 5)

    label_dawing_operations = []
    
    for tensor_bbox, score, category in zip(bboxes, scores, categories):
    #for tensor_bbox, score in zip(bboxes, scores):
        is_entered_warning_area = False
        is_entered_danger_area = False

        if args.object_detector_type == 'yolo':
            label = results.names[int(category.item())]
        elif args.object_detector_type == 'detr':
            label = categories[score.argmax()]
        
        bbox = tensor_bbox.cpu().numpy()
        if args.object_detector_type == 'detr':
            bbox = bbox.astype(dtype=np.int16, copy=False)

        np.random.seed(hash(label) % (2 ** 32 - 1))
        color = np.random.uniform(low=0, high=255, size=(3))

        #print('{} detected \n bounding box {} \n score {} \n'.format(label, bbox, score[score.argmax()]))

        if args.object_detector_type == 'yolo':
            # converting yolov5 bbox to acceptable format for pygame rect
            bbox = bbox_conversion(bbox, view_width, view_height)
        #elif args.object_detector_type == 'detr':
        #    pass

        # Draw the bounding box
        pygame.draw.rect(bb_surface, color, pygame.Rect(bbox), 3)

        # Draw the label
        pygame.draw.rect(bb_surface, color, pygame.Rect(bbox[0], bbox[1] - 30, 25 * len(label), 30))

        # label_dawing_operations.append(
        #    lambda: self.display.blit(self.bb_font.render(detection['label'], True, (0, 0, 0)), (bbox[0] + 5, bbox[1] - 30)))
        label_surface.blit(camera_manager.bb_font.render(label, True, (0, 0, 0)), (bbox[0] + 5, bbox[1] - 30))

        #label_surface.blit(camera_manager.bb_font.render(str(round(detection['score'], 3)), True, (0, 0, 0)),
        #                   (bbox[2] + 5, bbox[3] - 30))

        if show_distance:
            #Draw the distance
            label_surface.blit(camera_manager.bb_font.render(get_distance_by_camera(bbox), True, (0, 0, 0)), (int(bbox[2]) + 15, int(bbox[3]) - 30))

        # Draw the confidence
        # pygame.draw.rect(conf_surface, color, pygame.Rect(int(bbox[0]), int(bbox[3]), int(bbox[2]) - int(bbox[0]), (int(bbox[1]) - int(bbox[3])) * score.item()))

        if not args.no_intervention:

            #Verify if an object enters in the warning/danger areas
            if is_rect_overlap(bbox, WARNING_AREA):
                if label != 'person':
                    draw_safety_margin(pygame, safety_surface, "yellow", safe_regions.WARNING_AREA, 5)
                else:
                    draw_safety_margin(pygame, safety_surface, "red", safe_regions.DANGER_AREA, 5)
                    #ctrl.steer = -0.7
                    #ctrl.brake = 0.99
                    ctrl.hand_brake=True
                    carla.Vehicle.apply_control(player, ctrl)

            if is_rect_overlap(bbox, DANGER_AREA):
                draw_safety_margin(pygame, safety_surface, "red", safe_regions.DANGER_AREA, 5)

            colhist = world.collision_sensor.get_collision_history()
            if len(colhist) > 0:
                # print('Collision at frame: ', colhist)
                # #ctrl.brake = 0.99
                # ctrl.hand_brake=True
                # carla.Vehicle.apply_control(player, ctrl)

                # save info about the frame and the params that led to the hazard
                with open("src/hazards/{0}.csv".format(str(args.fault_type)), "a") as myfile:
                    myfile.write(('ML MODEL: {}; SEVERITY: {}; DAY_TIME: {}; DETAILS: {} \n').format(str(args.object_detector_type), str(args.severity), str(args.time_of_day), str(colhist)))

                #terminates the program if there is a hazard
                quit()

            #collision = [colhist[x + world.frame - 200] for x in range(0, 200)]
            #max_col = max(1.0, max(collision))
            #collision = [x / max_col for x in collision]
            #print('#### {}'.format(collision))

    display.blit(bb_surface, (0, 0))
    display.blit(conf_surface, (0, 0))
    display.blit(label_surface, (0, 0))
    display.blit(safety_surface, (0, 0))

    for ldo in label_dawing_operations:
        ldo()