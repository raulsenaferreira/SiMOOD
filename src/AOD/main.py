import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import random
from AOD import net
import numpy as np
from torchvision import transforms as tfs
from PIL import Image
import glob


def unfog_image(data_foggy):
	#print('np.shape(data_foggy)', np.shape(data_foggy))

	data_foggy = np.resize(data_foggy, (480, 640, 3))#, Image.ANTIALIAS)
	
	data_foggy = (np.asarray(data_foggy)/255.0) 

	data_foggy = torch.from_numpy(data_foggy).float()
	# tfs.Compose([
	#        tfs.ToTensor(),
	#        #tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])
	#        tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	#    ])(data_foggy)[None,::]
	
	data_foggy = data_foggy.permute(2, 0, 1)
	data_foggy = data_foggy.cuda().unsqueeze(0)

	unfog_net = net.unfog_net().cuda()
	unfog_net.load_state_dict(torch.load('src/AOD/snapshots/net.pth'))
	
	clean_image = unfog_net(data_foggy)
	
	torchvision.utils.save_image(torch.cat((data_foggy, clean_image),0), "src/AOD/results/imgtest{}.png".format(random.randint(0, 999)))
	
	return clean_image.cpu().detach().numpy()*255
	