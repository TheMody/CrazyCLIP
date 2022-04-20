#for setup use:
# !git clone https://github.com/openai/CLIP
# !pip install -e ./CLIP
# !pip install einops ninja


import sys
#sys.path.append('./stylegan_xl')

import io
import os, time, glob
import pickle
import shutil
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import requests
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import  DataLoader
import unicodedata
import re
from tqdm.notebook import tqdm
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import matplotlib.pyplot as plt

from loss import changeloss,prompts_dist_loss, spherical_dist_loss
from postprocess import process_mask
from patching import patch_img
from data import PascalVOCDataset
from Clip import CLIP
from clip_segmenter import Segmenter

def test():
    device = torch.device('cuda:0')
    print('Using device:', device, file=sys.stderr)
    
    def load_img(path):
        image = Image.open(path).convert('RGB')
        return TF.to_tensor(image).to(device).unsqueeze(0).requires_grad_()
    
    def transform(img):
        return TF.to_tensor(img).to(device).unsqueeze(0).requires_grad_()#,size = 224)
    
    def IOU(id,img, pred_img):
#         img = img[0]
#         pred_img = pred_img[0]
        tp = 0
        fp = 0
        tn = 0
        fn = 0
#         tmat = img == pred_img
#         tpmat = tmat
        
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
             #   if img[x,y] == id or pred_img[x,y]==id or img[x,y] == 0 or pred_img[x,y]==0:
                if img[x,y] == pred_img[x,y]:
                    if img[x,y]  == id:
                        tp += 1.0
                    else:
                        tn += 1.0
                else:
                    if img[x,y] == id:
                        fn += 1.0
                    else:
                        fp += 1.0
        return tp/(tp+fp+fn)
    dataset = PascalVOCDataset(data_type = "val", transform = transform, mode = "VOC")
    segmenter = Segmenter()
#     
#     input_image = load_img(path = "Images/kid.jpg") 
#     input_image = TF.resize(input_image,size = 224)
#     print(input_image.shape)
#     plt.imshow(TF.to_pil_image(input_image[0]))
#     plt.show()
#     pred_mask = segmenter.Segment_image(input_image,"kids arm")
    ious = {}
    for key in dataset.get_labels():
        ious[key] =[]
    
    dl = DataLoader(dataset, batch_size=1)
    for image,mask in dl:
       # print(image.shape)
#         plt.imshow(TF.to_pil_image(image[0]))
#         plt.show()
        for id in np.unique(mask):
            if id != 255 and id != 0:
                pred_mask = segmenter.Segment_image(image,dataset.get_labels()[id], id).to("cpu").detach().numpy()
    #             plt.imshow(TF.to_pil_image(mask[0]))
    #             plt.show()
    #             plt.imshow(TF.to_pil_image(pred_mask*5))
    #             plt.show()
                iou = IOU(id,mask[0],pred_mask)
                print(dataset.get_labels()[id],iou)
                ious[id].append(iou)
    miou = [(dataset.get_labels()[key],np.mean(ious[key])) for key in ious]
    print("mIOU", miou)
    Mmiou = np.mean([a[1] for a in miou])
    print("Mmiou", Mmiou)
#         plt.imshow(mask)
#         plt.show()
#         plt.imshow(pred_mask)
#         plt.show()
        



        


     

        
 

       
       
  

