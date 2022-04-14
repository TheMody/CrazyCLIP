#for setup use:
# !git clone https://github.com/openai/CLIP
# !pip install -e ./CLIP
# !pip install einops ninja
# !pip install timm


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
import unicodedata
import re
from tqdm.notebook import tqdm
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import matplotlib.pyplot as plt

from loss import changeloss,prompts_dist_loss, spherical_dist_loss
from postprocess import process_mask
from patching import patch_img#
#from pascal_voc import PASCALVOCTrain
from Clip import CLIP


def train():
    device = torch.device('cuda:0')
    print('Using device:', device, file=sys.stderr)
    

    def load_img(path):
        image = Image.open(path).convert('RGB')
        return TF.to_tensor(image).to(device).unsqueeze(0).requires_grad_()
    

    

        
    texts = "a realistic picture of a kid on a black background"#@param {type:"string"}
    steps = 500#@param {type:"number"}
    seed = -1#@param {type:"number"}
    init_steps = 10
    n_patches = 50 

     
    if seed == -1:
        seed = np.random.randint(0,9e7)
        print(f"Your random seed is: {seed}")#
        
        
    clip_model = CLIP()
    texts = [frase.strip() for frase in texts.split("|") if frase]
    targets = [clip_model.embed_text(text) for text in texts]
     
    input_image = load_img(path = "Images/kid.jpg") 
    input_image = TF.resize(input_image,size = 224)
     

    def Segment_image(timestring,input_image):
        torch.manual_seed(seed)
        
        patches = patch_img(input_image.cpu().detach(), k = n_patches)
        patches = torch.Tensor(patches).to(device)
        patches = patches + 1
        
        minloss = 10000
        min_patches =[]
        random_patches = []
        
        def mask_img_patches(input_img,patch_img ,patches):
            zero_img = torch.zeros(input_image.size()).to(device)
            for patch in patches:
                patch_img = zero_img.where(patch_img == patch, patch_img)
            image = input_image.where(patch_img == 0, zero_img)
            return image
        
        for i in range(init_steps):
            random_patches = (np.random.randint(50, size = 3)+1 ).tolist()
            mask_img = mask_img_patches(input_image,patches,random_patches )
            embed = clip_model.embed_image(mask_img.add(1).div(2))
            loss = prompts_dist_loss(embed, targets, spherical_dist_loss).mean()
            
            if loss < minloss:
                minloss = loss
                min_patches = random_patches
#             plt.imshow(TF.to_pil_image(mask_img[0].add(1).div(2).clamp(0,1)))
#             plt.show()
        temperature = 1.0
        loop = tqdm(range(steps))
        prev_loss = minloss
        random_patches = min_patches
        final_patches = []
        for i in loop:
            temperature = temperature *0.99
            prev_random_patches = random_patches.copy()
            if np.random.random() < 0.5:
                choicelist = list(range(1,51)) 
                for patch in random_patches:
                    choicelist.remove(patch)
                new_patch = np.random.choice(choicelist)
                random_patches.append(new_patch)
            #    print("added element")
            else:
                if len(random_patches)>1:
                    random_patches.pop(np.random.randint(len(random_patches)) )
              #      print("removed element")
            print(random_patches)
            mask_img = mask_img_patches(input_image,patches,random_patches )
            embed = clip_model.embed_image(mask_img.add(1).div(2))
            loss = prompts_dist_loss(embed, targets, spherical_dist_loss).mean()
            if loss > prev_loss:
                if np.random.random() < 0.5 + (0.5- temperature*0.5):
                    random_patches = prev_random_patches
                else:
                    prev_loss = loss
            else:
                if np.random.random() > 0.5 + (0.5- temperature*0.5):
                    random_patches = prev_random_patches
                else:
                    prev_loss = loss
#            if i % 200 == 0:
#                 with torch.no_grad():
#                      plt.imshow(TF.to_pil_image(mask_img[0].clamp(0,1)))
#                      plt.show()
#                 plt.imshow(TF.to_pil_image(tf(image)[0]))
#                 plt.show() 
#                 pil_image = TF.to_pil_image(image[0].add(1).div(2).clamp(0,1))
#                 os.makedirs(f'samples/{timestring}', exist_ok=True)
#                 pil_image.save(f'samples/{timestring}/{i:04}.jpg')
            print(f"Image {i}/{steps} | Current loss: {prev_loss}", "temp", temperature)
            final_patches = random_patches
        mask_img = mask_img_patches(input_image,patches,final_patches )
        plt.imshow(TF.to_pil_image(mask_img[0].clamp(0,1)))
        plt.show()
        return

    def generate_dream(timestring):


        torch.manual_seed(seed)
        
        def mask_img(input_img, mask):

           # image = torch.empty(input_image.size(), dtype = torch.float32, device = device,requires_grad=False) #hier scheint das problem zu seien
           # image = input_image.where(mask[0] > 0.5, torch.zeros(input_image.size()).to(device))
            image = input_image * mask#[0]
            return image
#         with torch.no_grad():
#             masknograd = torch.unsqueeze(torch.rand(input_image.size()[2:4],requires_grad=True, device = device),0).expand(2,-1,-1)
#             masknograd = masknograd.clone()
        mask = torch.rand(input_image.size()[2:4],requires_grad=True, device = device)
    #    mask = masknograd.requires_grad_()
        print(mask.size())
        soft = torch.nn.Softmax(dim = 0)
        opt = torch.optim.AdamW([mask], lr=0.05, betas=(0.5,0.999))
  #      l2loss = torch.nn.MSELoss()
        l2loss = torch.nn.L1Loss()
        l2_target = torch.zeros(mask.size(), device = device)
       # change_loss = 
        loop = tqdm(range(steps))
        for i in loop:
            opt.zero_grad()
           # softmask = soft(mask)
            clampedmask = torch.clamp(mask, min = 0, max = 1)
            image = mask_img(input_image, clampedmask)
            embed = embed_image(image.add(1).div(2))
            loss1 = prompts_dist_loss(embed, targets, spherical_dist_loss).mean()
            loss2 = l2loss(mask,l2_target)
            loss3 = changeloss(torch.unsqueeze(torch.unsqueeze(clampedmask,0),0))
            loss = loss1 + loss2 +  loss3
            loss.backward()
            opt.step()
            if i % 10 == 0:
                with torch.no_grad():
                    process_mask(torch.clamp(mask, min = 0, max = 1).cpu().numpy())
                #    plt.imshow(mask[1].cpu())
                plt.imshow(TF.to_pil_image(image[0]))
                plt.show() 
                pil_image = TF.to_pil_image(image[0].add(1).div(2).clamp(0,1))
                os.makedirs(f'samples/{timestring}', exist_ok=True)
                pil_image.save(f'samples/{timestring}/{i:04}.jpg')
            print(f"Image {i}/{steps} | Current loss: {loss}")
         
         
    try:
      timestring = time.strftime('%Y%m%d%H%M%S')
      Segment_image(timestring,input_image)
    except KeyboardInterrupt:
      pass
       
       
  

