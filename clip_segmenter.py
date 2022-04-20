
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
from patching import patch_img
from data import PascalVOCDataset
from Clip import CLIP
import torchvision.transforms.functional as TF

class Segmenter():
    
    def __init__(self, steps = 500):
        self.clip_model = CLIP()
        self.steps = steps

    def Segment_image(self,input_image, input_prompt, prompt_id):
        device = torch.device('cuda:0')
        texts = ["a realistic picture of a " + input_prompt + " on a black background"]
 #       input_image = TF.resize(input_image,size = 224)
 
 
        targets = self.clip_model.embed_text(texts)

        seed = 42
        init_steps = 20
        n_patches = 50
        torch.manual_seed(seed)
        np.random.seed(seed)
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
            return image#
        
        def mask_img_patches_id(input_img,patch_img ,patches,id):
            zero_img = torch.zeros(input_image.size()).to(device)
            for patch in patches:
                patch_img = zero_img.where(patch_img == patch, patch_img)
        #    print(input_image.size())
            id_image = torch.empty(input_image.size()).fill_(id).to(device)
            image = id_image.where(patch_img == 0, zero_img)
            return image[0,0]
        
        for i in range(init_steps):
            random_patches = (np.random.randint(50, size = np.random.randint(4)+2)+1 ).tolist()
            mask_img = mask_img_patches(input_image,patches,random_patches )
            embed = self.clip_model.embed_image(mask_img.add(1).div(2))
            loss = prompts_dist_loss(embed, targets, spherical_dist_loss).mean()
            
            if loss < minloss:
                minloss = loss
                min_patches = random_patches


        temperature = 0.5
        prev_loss = minloss
        random_patches = min_patches
        final_patches = []
        for i in range(self.steps):
            temperature = temperature *0.99
            prev_random_patches = random_patches.copy()
            if np.random.random() < 0.5:
                choicelist = list(range(1,51)) 
                for patch in random_patches:
                    try:
                        choicelist.remove(patch)
                    except:
                        print("patch not in list", patch , choicelist, random_patches)
                new_patch = np.random.choice(choicelist)
                random_patches.append(new_patch)
            #    print("added element")
            else:
                if len(random_patches)>1:
                    random_patches.pop(np.random.randint(len(random_patches)) )
              #      print("removed element")
          #  print(random_patches)
            mask_img = mask_img_patches(input_image,patches,random_patches )
            embed = self.clip_model.embed_image(mask_img.add(1).div(2))
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
#             if i % 100 == 0:
#                 with torch.no_grad():
#                       plt.imshow(TF.to_pil_image(mask_img[0].clamp(0,1)))
#                       plt.show()
#                 plt.imshow(TF.to_pil_image(input_image[0]))
#                 plt.show() 
#                 pil_image = TF.to_pil_image(image[0].add(1).div(2).clamp(0,1))
#                 os.makedirs(f'samples/{timestring}', exist_ok=True)
#                 pil_image.save(f'samples/{timestring}/{i:04}.jpg')
       #     print(f"Image {i}/{steps} | Current loss: {prev_loss}", "temp", temperature)
            final_patches = random_patches
#         mask_img = mask_img_patches(input_image,patches,final_patches )
#         plt.imshow(TF.to_pil_image(mask_img[0].clamp(0,1)))
#         plt.show()
        return mask_img_patches_id(input_image,patches,final_patches, id = prompt_id)
    
    
    
    def generate_dream(self,timestring):


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
#                 os.makedirs(f'samples/{timestring}', exist_ok=True)
#                 pil_image.save(f'samples/{timestring}/{i:04}.jpg')
#             print(f"Image {i}/{steps} | Current loss: {loss}")
         
         
