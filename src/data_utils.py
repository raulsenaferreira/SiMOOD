import numpy as np
from numba import jit


class Object:
    """docstring for Object"""
    def __init__(self, id, bbox, class_name, dist_eagle_vehicle):
        super(Object, self).__init__()
        self.id = id
        self.bbox = bbox
        self.class_name = class_name
        self.dist_eagle_vehicle = dist_eagle_vehicle
        
@jit(nopython=True)
def get_distance_by_camera(bbox):
    ## Distance Measurement for each bounding box
    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    ## item() is used to retrieve the value from the tensor
    distance = (2 * 3.14 * 180) / (w.item()+ h.item() * 360) * 1000 + 3 ### Distance measuring in Inch 
    #feedback = ("{}".format(detection['label'])+ " " +"is"+" at {} ".format(round(distance/39.37, 2))+"Meters")

    return "{} meters".format(round(distance, 3)) # meters   /39.37

@jit(nopython=True)
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


def compose_data(results, args, array_data, target_class, view_width,
                view_height, get_distance = False):
    data = []
    
    bboxes = None
    scores = None
    categories = None

    # parse results
    if args.object_detector_model_type == 'yolo':
        #predictions = results.pred[0]
        predictions = results.xywhn[0] #xywh normalized
        bboxes = predictions[:, :4] # x1, x2, y1, y2
        scores = predictions[:, 4]
        categories = predictions[:, 5]

    elif args.object_detector_model_type == 'detr':
        bboxes = results[0]
        scores = results[1]
        categories = results[2]

    #print('bboxes', bboxes)
    #print('scores', scores)
    #print('categories', categories)

    id = 0
    dist_eagle_vehicle = None
    
    for tensor_bbox, score, category in zip(bboxes, scores, categories):
        
        if args.object_detector_model_type == 'yolo':
            label = results.names[int(category.item())]
        elif args.object_detector_model_type == 'detr':
            label = categories[score.argmax()]

        if label == target_class:
            id+=1
            bbox = tensor_bbox.cpu().numpy()
            if args.object_detector_model_type == 'detr':
                bbox = bbox.astype(dtype=np.int16, copy=False)

            #print('{} detected \n bounding box {} \n score {} \n'.format(label, bbox, score[score.argmax()]))

            if args.object_detector_model_type == 'yolo':
                # converting yolov5 bbox to acceptable format for pygame rect
                bbox = bbox_conversion(bbox, view_width, view_height)

            if get_distance:
                dist_eagle_vehicle = get_distance_by_camera(bbox)

            obj = Object(id, bbox, label, dist_eagle_vehicle)

            data.append(obj)

    return data