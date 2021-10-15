import torch
import torchvision.transforms as T

torch.set_grad_enabled(False);

# COCO classes
CLASSES = [
	'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
	'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
	'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
	'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
	'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
	'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
	'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
	'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
	'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
	'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
	'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
	'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
	'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
	'toothbrush'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
		  [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# standard PyTorch mean-std input image normalization
transform = T.Compose([T.Resize(800),T.ToTensor(),T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# begin detectron helpers
# for output bounding box post-processing
def box_cxcywh_to_xyxy(x, img_w, img_h):
	x_c, y_c, w, h = x.unbind(1)

	# correcting bounding box offset related to the center of the image
	x_c = x_c - w/2
	y_c = y_c - h/2

	b = [(x_c), (y_c),
		 (w), (h)]
	
	return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size, device):
	img_w, img_h = size
	a = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(device)
	b = box_cxcywh_to_xyxy(out_bbox, img_w, img_h)
	b *= a
	return b

def detect(im, model, device):
	# mean-std normalize the input image (batch-size: 1)
	img = transform(im).unsqueeze(0)

	img = img.to(device)
	# propagate through the model
	outputs = model(img)

	# keep only predictions with 0.7+ confidence
	probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
	keep = probas.max(-1).values > 0.5

	# convert boxes from [0; 1] to image scales
	bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size, device)
	#bboxes_scaled = outputs['pred_boxes'][0, keep]
	return probas[keep], bboxes_scaled
# end detectron helpers