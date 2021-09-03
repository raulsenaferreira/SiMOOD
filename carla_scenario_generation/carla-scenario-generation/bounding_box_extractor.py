import carla
import numpy as np

SEMANTIC_SEG_COLOR_DEFS = {
    'pedestrian': (60, 20, 220),
    'bike': (142, 0, 0),
    'motorcycle': (142, 0, 0),
    'car': (142, 0, 0)
}


def get_2d_bounding_box(actor_transform,
                        actor_3d_bbox_extent,
                        actor_3d_bbox_location,
                        sensor_transform,
                        actor_type,
                        camera_calibration,
                        semantic_seg_image):
    """Get a 2d bounding box for an actor combining a projection of the ground truth 3d bounding box with the semantic
    segmentation ground truth

    :return: The bounding box as a tuple (x_left, y_top, x_right, y_bottom) or None, if the bounding box is not
        visible on the screen (NOTE: Not visible here means only behind the camera)
    """
    # Get the non-optimized 2D bounding boxes by projecting the 3d bounding box to the camera screen space
    non_optimal_bounding_box = get_projected_2d_bounding_box(actor_3d_bbox_extent, actor_3d_bbox_location,
                                                             actor_transform, sensor_transform, camera_calibration)

    if non_optimal_bounding_box is None:
        return None
    else:
        # Optimize the bounding box by refining with the semantic segmentation map
        # NOTE: Can be incorrect, if two bounding boxes of actors belonging to the same class overlap
        enhanced_bounding_box = enhance_2d_bounding_box_with_semantic_segmentation_map(
            non_optimal_bounding_box,
            semantic_seg_image,
            SEMANTIC_SEG_COLOR_DEFS[get_class_from_actor_type(actor_type)]
        )

        if enhanced_bounding_box is None:
            return None
        else:
            min_x = int(enhanced_bounding_box[3, 0])
            min_y = int(enhanced_bounding_box[3, 1])
            max_x = int(enhanced_bounding_box[1, 0])
            max_y = int(enhanced_bounding_box[1, 1])

            return min_x, min_y, max_x, max_y


def enhance_2d_bounding_box_with_semantic_segmentation_map(bounding_box, semantic_seg_image, semantic_seg_color,
                                                           min_pixel_ratio=0.4):
    """Enhances bounding boxes for a single class

    Returns the improved bounding boxes given the semantic segmentation image and the non optimial bounding boxes, for a single class

    INPUTS:
        original_bounding_boxes: a list of bounding boxes of elements from the same class (only vehicle, walker, ... etc) with shape 4x2 (each row is a point in the image) and the form:

            [np.array(...), np.array(...), ...]

        array_semseg: a semantic segmented image (a np.array() image) with each pixel color corresponding to a different class

        semseg_color: a dictionary with the pixel color of each class on the semantic segmentation image

    OUTPUTS:

        A list of np.array(), each with shape 4x2, representing the enhanced bounding boxes, with possibly less elements than original_bounding_boxes.

    """
    # Minimum amount of pixels of a certain class for a bounding box of that class be considered
    PIXELS_THRESHOLD = 50

    # Padding around the object
    PADDING = 2

    # Convert semantic segmentation image to array
    semantic_seg_array = carla_image_to_array(semantic_seg_image)

    # Maximum and minimum valid x and y
    max_valid_x = semantic_seg_array.shape[1] - 1
    max_valid_y = semantic_seg_array.shape[0] - 1

    # Looking only at one color channel, for performance measures
    channel = 0  # blue channel
    semantic_seg_array_blue_channel = semantic_seg_array[:, :, channel]
    channel_color = semantic_seg_color[channel]  # only look for the blue component

    minx = bounding_box[3, 0]
    miny = bounding_box[3, 1]
    maxx = bounding_box[1, 0]
    maxy = bounding_box[1, 1]

    # test if object is visible
    if (minx < 0 and maxx < 0) or (miny < 0 and maxy < 0):
        return None

    # # test object size
    # MIN_SIZE = 15
    # if abs(maxx - minx) < MIN_SIZE and abs(maxy - miny) < MIN_SIZE:
    #     return None

    minx = max(minx, 0)
    miny = max(miny, 0)

    # Apply inefficient method in case the bounding box is small
    section = semantic_seg_array_blue_channel[miny: maxy, minx: maxx]
    ys, xs = np.where(section == channel_color)

    if len(ys) >= PIXELS_THRESHOLD and len(xs) >= PIXELS_THRESHOLD:
        enhanced_minx = max(int(min(xs) + minx) - PADDING, 0)
        enhanced_miny = max(int(min(ys) + miny) - PADDING, 0)
        enhanced_maxx = min(int(max(xs) + minx) + PADDING, max_valid_x)
        enhanced_maxy = min(int(max(ys) + miny) + PADDING, max_valid_y)

        enhanced_bounding_box = np.zeros([4, 2], dtype=np.int16)
        enhanced_bounding_box[0, :] = [enhanced_maxx, enhanced_miny]
        enhanced_bounding_box[1, :] = [enhanced_maxx, enhanced_maxy]
        enhanced_bounding_box[2, :] = [enhanced_minx, enhanced_maxy]
        enhanced_bounding_box[3, :] = [enhanced_minx, enhanced_miny]

        return enhanced_bounding_box
    else:
        return None


def carla_image_to_array(carla_image):
    array = np.frombuffer(carla_image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (carla_image.height, carla_image.width, 4))
    array = array[:, :, :3]

    return array


def get_projected_2d_bounding_box(bounding_box_extent, bounding_box_location, actor_transform, sensor_transform,
                                  camera_calibration):
    """Projects a 3d bounding box to the camera screen space

    :return: The screen space coordinates (x_left, y_top, x_right, y_bottom) of the projected 2d bounding box or None, if not visible (NOTE: Not visible only relates to the bbox being behind the camera
    """
    # Get 3D projected bounding boxes
    bbox_camera_screen_space = get_3d_bounding_box_camera_screen_space(bounding_box_extent, bounding_box_location,
                                                                       actor_transform, sensor_transform,
                                                                       camera_calibration)

    if bbox_camera_screen_space is None:
        return None
    else:
        # transform to 2D bounding boxes
        points = [(int(bbox_camera_screen_space[n, 0]), int(bbox_camera_screen_space[n, 1])) for n in range(8)]
        E1 = points[0]
        E2 = points[1]
        E3 = points[2]
        E4 = points[3]
        E5 = points[4]
        E6 = points[5]
        E7 = points[6]
        E8 = points[7]

        E1_E4_all_x = (E1[0], E2[0], E3[0], E4[0])
        E1_E4_all_y = (E1[1], E2[1], E3[1], E4[1])
        E5_E8_all_x = (E5[0], E6[0], E7[0], E8[0])
        E5_E8_all_y = (E5[1], E6[1], E7[1], E8[1])

        E1_E4_max_x = np.max(E1_E4_all_x)
        E1_E4_min_x = np.min(E1_E4_all_x)
        E1_E4_max_y = np.max(E1_E4_all_y)
        E1_E4_min_y = np.min(E1_E4_all_y)

        E5_E8_max_x = np.max(E5_E8_all_x)
        E5_E8_min_x = np.min(E5_E8_all_x)
        E5_E8_max_y = np.max(E5_E8_all_y)
        E5_E8_min_y = np.min(E5_E8_all_y)

        E1_E8_max_x = np.max((E1_E4_max_x, E5_E8_max_x))
        E1_E8_min_x = np.min((E1_E4_min_x, E5_E8_min_x))
        E1_E8_max_y = np.max((E1_E4_max_y, E5_E8_max_y))
        E1_E8_min_y = np.min((E1_E4_min_y, E5_E8_min_y))

        E1_2D = (E1_E8_max_x, E1_E8_min_y)
        E2_2D = (E1_E8_max_x, E1_E8_max_y)
        E3_2D = (E1_E8_min_x, E1_E8_max_y)
        E4_2D = (E1_E8_min_x, E1_E8_min_y)

        # Vetex order: top-right, bottom-right, bottom-left, top-left
        bbox_2d = np.array([E1_2D, E2_2D, E3_2D, E4_2D])

        return bbox_2d


def get_3d_bounding_box_camera_screen_space(bounding_box_extent, bounding_box_location, actor_transform,
                                            sensor_transform, camera_calibration):
    """Returns the 3d bounding box coordinates in camera screen space
    """
    bbox_actor_space = get_3d_bounding_box_actor_space(bounding_box_extent)

    # 3x8 matrix, where each column is a [x,y,z] vector in space
    # x+:foward, y+:right, z+: up
    bbox_sensor_space = transform_3d_bounding_box_from_actor_to_sensor_space(bbox_actor_space, bounding_box_location,
                                                                             actor_transform, sensor_transform)[:3, :]

    # 3x8 matrix, each column is a [x,y,z]
    # x+:right, y+:down, z+forward:  (usual 2D image projection convention)
    cords_y_minus_z_x = np.concatenate([bbox_sensor_space[1, :], -bbox_sensor_space[2, :], bbox_sensor_space[0, :]])

    # Get the position of the 2D bounding box coordinates on the image frame [u,v], multiplied by scalar S
    # 8x3 matrix, each row is a [u*S, v*S, S]
    # u+:right, v+:down (usual 2D image convention)
    bbox = np.transpose(np.dot(camera_calibration, cords_y_minus_z_x))

    # Divide by S and get positions [u,v,1]
    bbox_camera_screen_space = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)

    # Return None, if the bounding box is behind the camera
    if all(bbox_camera_screen_space[:, 2] > 0.2):
        return bbox_camera_screen_space
    else:
        return None


# _vehicle_to_sensor
def transform_3d_bounding_box_from_actor_to_sensor_space(bbox_actor_space, bbox_location, actor_transform,
                                                         sensor_transform):
    """
    Transforms coordinates of a vehicle bounding box to sensor space.
    """

    # Actor space -> world space
    bbox_world_space = transform_3d_bounding_box_from_actor_to_world_space(bbox_actor_space, bbox_location,
                                                                           actor_transform)

    # World space -> camera screen space
    bbox_sensor_space = transform_3d_bounding_box_from_world_to_sensor_space(bbox_world_space, sensor_transform)

    return bbox_sensor_space


# _vehicle_to_world
def transform_3d_bounding_box_from_actor_to_world_space(bbox_actor_space, bbox_location, actor_transform):
    """Transform a 3d bounding box from actor to world space

    :bbox_location: The bounding box location as returned by actor.bounding_box.location
    "actor_transform: The location and rotation of the actor as return by actor.get_transform()
    """

    # Center of bounding box center with respect to the actor location
    bbox_transform = carla.Transform(bbox_location)
    bbox_actor_matrix = carla_transform_to_matrix(bbox_transform)

    # Actor location with respect to the world frame
    actor_world_matrix = carla_transform_to_matrix(actor_transform)

    # Center of bounding box with respect to the world frame, matrix
    bbox_world_matrix = np.dot(actor_world_matrix, bbox_actor_matrix)
    bbox_world_space = np.dot(bbox_world_matrix, np.transpose(bbox_actor_space))

    return bbox_world_space


# _world_to_sensor
def transform_3d_bounding_box_from_world_to_sensor_space(bbox_world_space, sensor_transform):
    """Transforms world space to sensor space
    """
    # Sensor -> world coords
    sensor_world_matrix = carla_transform_to_matrix(sensor_transform)

    # World -> Sensor coords
    world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
    bbox_sensor_space = np.dot(world_sensor_matrix, bbox_world_space)

    return bbox_sensor_space


def carla_transform_to_matrix(transform):
    """Creates matrix from carla transform.
    """
    rotation = transform.rotation
    location = transform.location
    c_y = np.cos(np.radians(rotation.yaw))
    s_y = np.sin(np.radians(rotation.yaw))
    c_r = np.cos(np.radians(rotation.roll))
    s_r = np.sin(np.radians(rotation.roll))
    c_p = np.cos(np.radians(rotation.pitch))
    s_p = np.sin(np.radians(rotation.pitch))
    matrix = np.matrix(np.identity(4))
    matrix[0, 3] = location.x
    matrix[1, 3] = location.y
    matrix[2, 3] = location.z
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r

    return matrix


def get_3d_bounding_box_actor_space(bounding_box_extent):
    """Returns the 3d bounding box coordinates in actor space
    """
    cords = np.zeros((8, 4))

    cords[0, :] = np.array([bounding_box_extent.x, bounding_box_extent.y, -bounding_box_extent.z, 1])  # E1
    cords[1, :] = np.array([-bounding_box_extent.x, bounding_box_extent.y, -bounding_box_extent.z, 1])  # E2
    cords[2, :] = np.array([-bounding_box_extent.x, -bounding_box_extent.y, -bounding_box_extent.z, 1])  # E3
    cords[3, :] = np.array([bounding_box_extent.x, -bounding_box_extent.y, -bounding_box_extent.z, 1])  # E4
    cords[4, :] = np.array([bounding_box_extent.x, bounding_box_extent.y, bounding_box_extent.z, 1])  # E5
    cords[5, :] = np.array([-bounding_box_extent.x, bounding_box_extent.y, bounding_box_extent.z, 1])  # E6
    cords[6, :] = np.array([-bounding_box_extent.x, -bounding_box_extent.y, bounding_box_extent.z, 1])  # E7
    cords[7, :] = np.array([bounding_box_extent.x, -bounding_box_extent.y, bounding_box_extent.z, 1])  # E8

    return cords


def get_class_from_actor_type(actor_type: str) -> str:
    """Returns the class, e.g. car, pedestrian, ..., from the blueprint id of an actor, e.g. vehicle.ford.mustang

    :param actor_type: CARLA blueprint library id of the actor, e.g. vehicle.ford.mustang
    :return: Corresponding class to the actor type, e.g. car
    """
    if actor_type.startswith('walker'):
        return 'pedestrian'
    elif actor_type.startswith('vehicle'):
        if actor_type.endswith('bh.crossbike'):
            return 'bike'
        elif actor_type.endswith(('harley-davidson.low_rider',
                                  'yamaha.yzf',
                                  'kawasaki.ninja',
                                  'diamondback.century',
                                  'gazelle.omafiets')):
            return 'motorcycle'
        else:
            return 'car'
