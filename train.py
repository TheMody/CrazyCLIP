

# !git clone https://github.com/openai/CLIP
# !pip install -e ./CLIP
# !pip install einops ninja
# !pip install timm

## I'll probably have to trim stuff here

import sys
sys.path.append('./CLIP')
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
import clip
import unicodedata
import re
from tqdm.notebook import tqdm
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import matplotlib.pyplot as plt
#from IPython.display import display
from einops import rearrange
from loss import changeloss
from postprocess import process_mask
from patching import patch_img
#from google.colab import files
# import dnnlib
# import legacy


def train():
    device = torch.device('cuda:0')
    print('Using device:', device, file=sys.stderr)
    
    # Functions (many must be trimmed too)
    
    def fetch(url_or_path):
        if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
            r = requests.get(url_or_path)
            r.raise_for_status()
            fd = io.BytesIO()
            fd.write(r.content)
            fd.seek(0)
            return fd
        return open(url_or_path, 'rb')
    
#     def fetch_model(url_or_path):
#         !wget -c '{url_or_path}'
    
    def slugify(value, allow_unicode=False):
        """
        Taken from https://github.com/django/django/blob/master/django/utils/text.py
        Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
        dashes to single dashes. Remove characters that aren't alphanumerics,
        underscores, or hyphens. Convert to lowercase. Also strip leading and
        trailing whitespace, dashes, and underscores.
        """
        value = str(value)
        if allow_unicode:
            value = unicodedata.normalize('NFKC', value)
        else:
            value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
        value = re.sub(r'[^\w\s-]', '', value.lower())
        return re.sub(r'[-\s]+', '-', value).strip('-_')
    
    def norm1(prompt):
        "Normalize to the unit sphere."
        return prompt / prompt.square().sum(dim=-1,keepdim=True).sqrt()
    
    def spherical_dist_loss(x, y):
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)
    
    def prompts_dist_loss(x, targets, loss):
        if len(targets) == 1: # Keeps consitent results vs previous method for single objective guidance 
          return loss(x, targets[0])
        distances = [loss(x, target) for target in targets]
        return torch.stack(distances, dim=-1).sum(dim=-1)  
    
    class MakeCutouts(torch.nn.Module):
        def __init__(self, cut_size, cutn, cut_pow=1.):
            super().__init__()
            self.cut_size = cut_size
            self.cutn = cutn
            self.cut_pow = cut_pow
    
        def forward(self, input):
            sideY, sideX = input.shape[2:4]
            max_size = min(sideX, sideY)
            min_size = min(sideX, sideY, self.cut_size)
            cutouts = []
            for _ in range(self.cutn):
                size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
                cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
            return torch.cat(cutouts)
    
    make_cutouts = MakeCutouts(224, 32, 0.5)
    
    def embed_image(image):
      n = image.shape[0]
      cutouts = make_cutouts(image)
      embeds = clip_model.embed_cutout(cutouts)
      embeds = rearrange(embeds, '(cc n) c -> cc n c', n=n)
      return embeds
    
    def embed_url(url):
      image = Image.open(fetch(url)).convert('RGB')
      return embed_image(TF.to_tensor(image).to(device).unsqueeze(0)).mean(0).squeeze(0)
    
    class CLIP(object):
      def __init__(self):
        clip_model = "ViT-B/32"
        self.model, _ = clip.load(clip_model)
        self.model = self.model.requires_grad_(False)
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                              std=[0.26862954, 0.26130258, 0.27577711])
    
      @torch.no_grad()
      def embed_text(self, prompt):
          "Normalized clip text embedding."
          return norm1(self.model.encode_text(clip.tokenize(prompt).to(device)).float())
    
      def embed_cutout(self, image):
          "Normalized clip image embedding."
          return norm1(self.model.encode_image(self.normalize(image)))
      
    clip_model = CLIP()

    def load_img(path):
        image = Image.open(path).convert('RGB')
        return TF.to_tensor(image).to(device).unsqueeze(0).requires_grad_()
    
    def mask_img(input_img, mask):

       # image = torch.empty(input_image.size(), dtype = torch.float32, device = device,requires_grad=False) #hier scheint das problem zu seien
       # image = input_image.where(mask[0] > 0.5, torch.zeros(input_image.size()).to(device))
        image = input_image * mask#[0]
        return image
    
    def mask_img_patches(input_img,patch_img ,patches):
        zero_img = torch.zeros(input_image.size()).to(device)
        for patch in patches:
            patch_img = zero_img.where(patch_img == patch, patch_img)
        image = input_image.where(patch_img == 0, zero_img)
        return image
        
    texts = "brown hair on a black background"#@param {type:"string"}
    steps = 10001#@param {type:"number"}
    seed = -1#@param {type:"number"}
     
    tf = Compose([
      Resize(224),
      lambda x: torch.clamp((x+1)/2,min=0,max=1),
    ])
     
    if seed == -1:
        seed = np.random.randint(0,9e9)
        print(f"Your random seed is: {seed}")
     
    texts = [frase.strip() for frase in texts.split("|") if frase]
    init_steps = 10
    targets = [clip_model.embed_text(text) for text in texts]
     
    input_image = load_img(path = "Images/kid.jpg") 
    input_image = TF.resize(input_image,size = 224)
    n_patches = 50
    patches = patch_img(input_image.cpu().detach(), k = n_patches)
    patches = torch.Tensor(patches).to(device)
    patches = patches + 1
  #  mask = torch.rand(input_image.size()[2:4], dtype = torch.float16, device = device,requires_grad=True)
#     plt.imshow(TF.to_pil_image(tf(input_image)[0]))
#     plt.show()
#     print(mask.size())
#     print(input_image.size())
#     image = mask_img(input_image, mask > 0)
#     plt.imshow(TF.to_pil_image(tf(image)[0]))
#     plt.show() 
     

    def run_withcluster(timestring):
        torch.manual_seed(seed)
        minloss = 10000
        min_patches =[]
        random_patches = []
        for i in range(init_steps):
            random_patches = np.random.randint(50, size = 3)+1 
            mask_img = mask_img_patches(input_image,patches,random_patches )
            embed = embed_image(mask_img.add(1).div(2))
            loss = prompts_dist_loss(embed, targets, spherical_dist_loss).mean()
            
            if loss < minloss:
                minloss = loss
                min_patches = random_patches
#             plt.imshow(TF.to_pil_image(mask_img[0].add(1).div(2).clamp(0,1)))
#             plt.show()
        
        loop = tqdm(range(steps))
        prev_loss = minloss
        for i in loop:
            prev_random_patches = random_patches
            if np.random.random() < 0.5:
               new_patch = np.random.choice(list(range(50))+1 not in random_patches)
               random_patches.append(new_patch)
            else:
               np.delete(random_patches,np.random.choice(range(len(random_patches)))) 
            mask_img = mask_img_patches(input_image,patches,random_patches )
            embed = embed_image(mask_img.add(1).div(2))
            loss = prompts_dist_loss(embed, targets, spherical_dist_loss).mean()
            if loss < prev_loss:
                random_patches = random_patches
            else:
                random_patches = prev_random_patches
            
            if i % 10 == 0:
                with torch.no_grad():
                     plt.imshow(TF.to_pil_image(mask_img[0].add(1).div(2).clamp(0,1)))
                     plt.show()
#                 plt.imshow(TF.to_pil_image(tf(image)[0]))
#                 plt.show() 
#                 pil_image = TF.to_pil_image(image[0].add(1).div(2).clamp(0,1))
#                 os.makedirs(f'samples/{timestring}', exist_ok=True)
#                 pil_image.save(f'samples/{timestring}/{i:04}.jpg')
            print(f"Image {i}/{steps} | Current loss: {loss}")
        return

    def run(timestring):


        torch.manual_seed(seed)
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
                plt.imshow(TF.to_pil_image(tf(image)[0]))
                plt.show() 
                pil_image = TF.to_pil_image(image[0].add(1).div(2).clamp(0,1))
                os.makedirs(f'samples/{timestring}', exist_ok=True)
                pil_image.save(f'samples/{timestring}/{i:04}.jpg')
            print(f"Image {i}/{steps} | Current loss: {loss}")
         
         
    try:
      timestring = time.strftime('%Y%m%d%H%M%S')
      run_withcluster(timestring)
    except KeyboardInterrupt:
      pass
       
       
  

