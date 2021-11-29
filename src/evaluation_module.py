import pygame
import bbox_oracle



def is_rect_overlap(bbox,R2):
        R1 = pygame.Rect(bbox)

        if R1.colliderect(R2):
            return True
        else:
            return False

            
def evaluate(world, display, image, detected_objects_per_frame, view_width, view_height, show_ground_truth_bbox_screen, important_region):

    true_pos_SUT_per_frame, true_neg_SUT_per_frame, false_pos_SUT_per_frame, false_neg_SUT_per_frame = 0, 0, 0, 0

    for actor in world.world.get_actors():
        real_bbox = None
        
        # verifying just real bounding boxes of pedestrians (e.g., ('walker' inside Carla) == ('person' inside COCO dataset))
        if actor.type_id.startswith('walker'):
            target_object_label = 'person'
            real_bbox = bbox_oracle.get_bounding_box_actor(display, actor, world.camera_manager.sensor, image, view_width, view_height, show_ground_truth_bbox_screen)

            #################### starting evaluating the predicted bbox regarding the real bbox ####################
            # We are not interested in the exact accuracy of the bounding boxes but if the ML model was able to correctly find the object in an image
            if real_bbox is not None:

                if important_region is not None:
                    # real pedestrian entered in the important region
                    if is_rect_overlap(important_region, pygame.Rect(real_bbox)):
                        
                        for det_object in detected_objects_per_frame:
                            predicted_bbox = det_object[0]
                            predicted_class = det_object[2]

                            # search for a predicted pedestrian that intersects with the real pedestrian that entered in the important region
                            if is_rect_overlap(important_region, predicted_bbox):

                                #there is a real pedestrian and the ML detected it (true positive)
                                if target_object_label == predicted_class: 
                                    true_pos_SUT_per_frame+=1

                                #there is a real pedestrian and the ML detected it as another object (false negative)
                                else: 
                                    false_neg_SUT_per_frame+=1

                    # NO real pedestrian entered in the important region
                    else:

                        for det_object in detected_objects_per_frame:
                            if det_object is not None:
                                #print('det_object', len(det_object))
                                predicted_bbox = det_object[0]
                                predicted_class = det_object[2]

                                # search for a predicted pedestrian that entered in the important region (when actually there is no real pedestrian on that region)
                                if is_rect_overlap(important_region, predicted_bbox):

                                    #there is NO real pedestrian and the ML detected it anyway (false positive)
                                    if target_object_label == predicted_class: 
                                        false_pos_SUT_per_frame+=1

                                #there is NO real pedestrian in the important region and the ML correctly did not detected anything (true negative)
                                else: 
                                    true_neg_SUT_per_frame+=1

            #################### ending evaluating the predicted bbox regarding the real bbox ####################

    return true_pos_SUT_per_frame, true_neg_SUT_per_frame, false_pos_SUT_per_frame, false_neg_SUT_per_frame