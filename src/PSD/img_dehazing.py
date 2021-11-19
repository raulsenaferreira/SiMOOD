import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torchvision.transforms as tfs 
#from datasets.pretrain_datasets import TrainData, ValData, TestData, TestData2, TestData_GCA, TestData_FFA
#from models.GCA import GCANet
from PSD.FFA import FFANet
#from models.MSBDN import MSBDNNet
from PSD.PSD_utils import to_psnr, print_log, validation, adjust_learning_rate
import numpy as np
import os
from PIL import Image


def run(img):
    
    #img = np.array([np.transpose(img)])
    #print('np.shape(img)', np.shape(img))
    epoch = 14

    #test_data_dir = '/data/nnice1216/Dehazing/unlabeled/'
        
    device_ids = [Id for Id in range(torch.cuda.device_count())]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # net = GCANet(in_c=4, out_c=3, only_residual=True).to(device)
    net = FFANet(3, 19)
    # net = MSBDNNet()

    net = nn.DataParallel(net, device_ids=device_ids)

    # net.load_state_dict(torch.load('PSD-GCANET'))
    net.load_state_dict(torch.load('src/PSD/PSD-FFANET'))
    # net.load_state_dict(torch.load('PSB-MSBDN'))

    net.eval()

    # test_data_loader = DataLoader(TestData_GCA(test_data_dir), batch_size=1, shuffle=False, num_workers=8) # For GCA
    #test_data_loader = DataLoader(TestData_FFA(test_data_dir), batch_size=1, shuffle=False, num_workers=8) # For FFA and MSBDN


    output_dir = 'src/PSD/output/base_JEPG8/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with torch.no_grad():
        #haze_no=tfs.ToTensor()(img[0])[None,::]
        #print('np.shape(haze_no)',np.shape(haze_no))
        #haze_no=haze_no.reshape(1, 3, 1280, 720)
        #print('np.shape(haze_no) MODIFIED',np.shape(haze_no))

        #for batch_id, val_data in enumerate(test_data_loader):
        #if batch_id > 150:
        #    break
        # haze, name = val_data # For GCA
        #haze, haze_A, name = val_data # For FFA and MSBDN

        # haze= tfs.Compose([
        #     tfs.ToTensor(),
        #     #tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])
        #     tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])(img[0])[None,::]

        #print('np.shape(img)',np.shape(img))
        #img=img.reshape(np.shape(img)[0], np.shape(img)[3], 1, 2)

        haze1= tfs.Compose([
            tfs.ToTensor(),
            #tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])
            tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])(img)[None,::]
        
        haze_no=tfs.ToTensor()(img)[None,::]
        #haze1=haze1.resize(1, 3, 256, 256)
        with torch.no_grad():
            pred = net(haze_no)

        print('!!!!!!!!!')
        ts=torch.squeeze(pred.clamp(0,1).cpu())
        ts2=torch.squeeze(haze_no)
        print('!!!!!!!!!!!!!')
        ### GCA ###
        # dehaze = pred.float().round().clamp(0, 255)
        # out_img = Image.fromarray(dehaze[0].cpu().numpy().astype(np.uint8).transpose(1, 2, 0))
        # out_img.save(output_dir + name[0].split('.')[0] + '_MyModel_{}.png'.format(batch_id))
        ###########
        batch_id = 99
        ### FFA & MSBDN ###
        #ts = torch.squeeze(pred.clamp(0, 1).cpu())
        vutils.save_image(ts, output_dir + 'image' + '_MyModel_{}.png'.format(batch_id))
        ###################