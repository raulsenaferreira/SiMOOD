import pygame
import numpy as np
from bounding_box_extractor import get_2d_bounding_box, get_class_from_actor_type



def calculate_camera_calibration(image_width, image_height, fov):
    calibration = np.identity(3)
    calibration[0, 2] = image_width / 2.0
    calibration[1, 2] = image_height / 2.0
    calibration[0, 0] = calibration[1, 1] = image_width / (2.0 * np.tan(float(fov) * np.pi / 360.0))

    return calibration


def get_bounding_box_actor(display, actor, sensor, image, view_width, view_height, show_bbox_screen=False):
    # Create surface for the bounding boxes
    bb_surface = pygame.Surface((view_width, view_height))
    bb_surface.set_colorkey((0, 0, 0))
    color = np.random.uniform(low=0, high=255, size=(3))

    # IKS: Iterate over all actors and save their position, orientation, 3d bounding box, ...
    # TODO: Only supports dynamic actors right now
    #if world.camera_manager.surface is not None:
    
    try:
        type_id = actor.type_id
        bounding_box = actor.bounding_box # 3D bbox
        transform =  actor.get_transform()
        actor_id = actor.id
        actor_name = actor.attributes['role_name']

        sensor_transform = sensor.get_transform()
        camera_calibration = calculate_camera_calibration(
                              int(sensor.attributes['image_size_x']),
                              int(sensor.attributes['image_size_y']),
                              float(sensor.attributes['fov'])
                              )
        # 2D bbox
        bbox = get_2d_bounding_box(transform, bounding_box.extent, bounding_box.location, sensor_transform, type_id, camera_calibration, image)
        
        # Draw the bounding box
        if bbox is not None:
            #print('bbox', bbox)
            x,y,xx,yy = bbox
            xx-=x
            yy-=y
            bbox = [x,y,xx,yy]

            if show_bbox_screen:
                pygame.draw.rect(bb_surface, color, pygame.Rect(bbox), 3)

                display.blit(bb_surface, (0, 0))

            return bbox

    except Exception as e:
        print('Exception:', e)

