import pygame
import bbox_oracle



def is_rect_overlap(R1,R2):
        #R1 = pygame.Rect(bbox)

        if R1.colliderect(R2):
            return True
        else:
            return False


def intersection_area(bbox,rectangle):
    from shapely.geometry import Polygon

    polygon = Polygon(pygame.Rect(bbox))
    other_polygon = Polygon(rectangle)

    #polygon = Polygon([(3, 3), (5, 3), (5, 5), (3, 5)])
    #other_polygon = Polygon([(1, 1), (4, 1), (4, 3.5), (1, 3.5)])
    intersection = polygon.intersection(other_polygon)
    #print('intersection.area',intersection.area)
    return intersection.area


def area(a, b):  # returns None if rectangles don't intersect
    
    # dx = min(a[0], b[0]) - max(a[2], b[2])
    # dy = min(a[1], b[1]) - max(a[3], b[3])

    # if (dx>=0) and (dy>=0):
    #     area_overleap = dx*dy
    #     return area_overleap

    area_a = a.height * a.width
    area_b = b.height * b.width

    area_overleap = (max(a.left,b.left)-min(a.right,b.right))*(max(a.top,b.top)-min(a.bottom,b.bottom)) 

    return area_overleap, area_overleap/area_a, area_overleap/area_b, area_overleap/(area_a+area_b-area_overleap)

            
def evaluate_ML(world, display, image, detected_objects_per_frame, view_width, view_height, show_ground_truth_bbox_screen, important_region):
    
    # ('walker' ==> Carla) == ('person' ==> COCO dataset)
    target_object_label = 'person'
    carla_obj_equivalent_label = 'walker'

    true_pos_ML_per_frame, false_pos_ML_per_frame, false_neg_ML_per_frame = 0, 0, 0
    true_pos_ML_imp_region_per_frame, false_pos_ML_imp_region_per_frame, false_neg_ML_imp_region_per_frame = 0, 0, 0 

    for actor in world.world.get_actors():
        threshold_overlap = 0.85
        real_bbox = bbox_oracle.get_bounding_box_actor(display, actor, world.camera_manager.sensor, image, view_width, view_height, show_ground_truth_bbox_screen)

        for det_object in detected_objects_per_frame:

            #if det_object is not None:   
            predicted_bbox = det_object[0]
            predicted_class = det_object[2] 

            if real_bbox is not None:

                # it will not be accurate if we have several pedestrians entering and leaving the important region     
                if is_rect_overlap(pygame.Rect(predicted_bbox), pygame.Rect(real_bbox)):

                    # [area overlap, % within the 1st rect, % within the 2nd rect, % from either rectangle]
                    areas = area(pygame.Rect(predicted_bbox), pygame.Rect(real_bbox)) 

                    if areas[1]>threshold_overlap or areas[2]>threshold_overlap: 
                        #print('areeeeeas',areas)
                        
                        # (There is a pedestrian in the screen) Potential true positives and false negatives for the ML
                        if actor.type_id.startswith(carla_obj_equivalent_label):
                            #print('has pedestrians')
                            if target_object_label == predicted_class:
                                ### is there an object of interest that was correctly detected by the ML?
                                ### (true positive in the entire image) 
                                true_pos_ML_per_frame+=1
                                #print('pedestrian detected')

                                ### is there an object of interest INSIDE of a region of interest that was correctly detected by the ML? 
                                ### (true positive in the specific region)
                                if is_rect_overlap(pygame.Rect(important_region), pygame.Rect(predicted_bbox)): 
                                   true_pos_ML_imp_region_per_frame+=1
                                   #print('pedestrian detected in the region')

                            else: 
                                ### there is a real pedestrian and the ML detected it as another object 
                                ### (false negative in the entire image)
                                false_neg_ML_per_frame+=1

                                ### there is a real pedestrian and the ML detected it as another object inside a specific region 
                                ### (false negative in the specific region)
                                if is_rect_overlap(pygame.Rect(important_region), pygame.Rect(predicted_bbox)): 
                                   false_neg_ML_imp_region_per_frame+=1

                        # (There is NO pedestrian in the screen) Potential false positives for the ML
                        else:
                            #if actor.type_id.startswith('vehicle') and predicted_class == 'car':
                                #print('car correctly detected')
                            
                            ### there is NO real pedestrian and the ML detected it anyway 
                            ### (false positive in the entire image)
                            if target_object_label == predicted_class:
                                false_pos_ML_per_frame+=1

                                ### there is NO real pedestrian in the specific region and the ML detected it anyway 
                                ### (false positive in the specific region of an image)
                                if is_rect_overlap(pygame.Rect(important_region), pygame.Rect(predicted_bbox)):
                                    false_pos_ML_imp_region_per_frame+=1
                
    return true_pos_ML_per_frame, false_pos_ML_per_frame, false_neg_ML_per_frame,\
     true_pos_ML_imp_region_per_frame, false_pos_ML_imp_region_per_frame, false_neg_ML_imp_region_per_frame


def evaluate_SM(false_ML_detections, true_ML_detections, threat_detection_per_frame):
    true_pos_SM_per_frame, true_neg_SM_per_frame, false_pos_SM_per_frame, false_neg_SM_per_frame = 0,0,0,0

    if false_ML_detections > 0 and threat_detection_per_frame == 0:
        false_neg_SM_per_frame+=1
    elif false_ML_detections > 0 and threat_detection_per_frame > 0:
        true_pos_SM_per_frame+=1
    elif true_ML_detections > 0 and threat_detection_per_frame == 0:
        true_neg_SM_per_frame+=1
    elif true_ML_detections > 0 and threat_detection_per_frame > 0:
        false_pos_SM_per_frame+=1

    return true_pos_SM_per_frame, true_neg_SM_per_frame, false_pos_SM_per_frame, false_neg_SM_per_frame