

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
        image = input_image * mask
        return image
    texts = "a dog jumping on a trampoline in the snow"#@param {type:"string"}
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
     
    targets = [clip_model.embed_text(text) for text in texts]
     
    input_image = load_img(path = "Images/dog.jpg") 
  #  mask = torch.rand(input_image.size()[2:4], dtype = torch.float16, device = device,requires_grad=True)
#     plt.imshow(TF.to_pil_image(tf(input_image)[0]))
#     plt.show()
#     print(mask.size())
#     print(input_image.size())
#     image = mask_img(input_image, mask > 0)
#     plt.imshow(TF.to_pil_image(tf(image)[0]))
#     plt.show() 
     

     
#     initial_batch=4 #actually that will be multiplied by initial_image_steps
#     initial_image_steps=8


    def run(timestring):

      #with torch.no_grad():
#         qs = []
#         losses = []
#         for _ in range(initial_image_steps):
# #           a = torch.randn([initial_batch, 512], device=device)*0.6 + w_stds*0.4
# #           q = ((a-w_all_classes_avg)/w_stds)
# #           images = G.synthesis((q * w_stds + w_all_classes_avg).unsqueeze(1).repeat([1, G.num_ws, 1]))
#             
#             images = input_images.masked_fill(masks, [1,1,1])
#             embeds = embed_image(images.add(1).div(2))
#             loss = prompts_dist_loss(embeds, targets, spherical_dist_loss).mean(0)
#             i = torch.argmin(loss)
#             qs.append(q[i])
#             losses.append(loss[i])
#         qs = torch.stack(qs)
#         losses = torch.stack(losses)
#         # print(losses)
#         # print(losses.shape, qs.shape)
#         i = torch.argmin(losses)
#         q = qs[i].unsqueeze(0).requires_grad_()
#      
#      
#       # Sampling loop
#       q_ema = q
#       print (q.shape)
        torch.manual_seed(seed)
        
        mask = torch.rand(input_image.size()[1:4],requires_grad=True, device = device)
    #    mask = mask.requires_grad_()
    #    with torch.no_grad():
    #    mask = mask.to(device)
        opt = torch.optim.AdamW([mask], lr=0.05, betas=(0.5,0.999))
        
        loop = tqdm(range(steps))
        for i in loop:
          opt.zero_grad()
          #w = q * w_stds
          # image = G.synthesis((q * w_stds + w_all_classes_avg).unsqueeze(1).repeat([1, G.num_ws, 1]), noise_mode='const')
          image = mask_img(input_image, torch.clamp(mask, min = 0, max = 1))
          embed = embed_image(image.add(1).div(2))
          loss = prompts_dist_loss(embed, targets, spherical_dist_loss).mean()
          loss.backward()
          opt.step()
        #         loop.set_postfix(loss=loss.item(), q_magnitude=q.std().item())
        #         
        #         q_ema = q_ema * 0.9 + q * 0.1
        #         image = G.synthesis((q_ema * w_stds + w_all_classes_avg).unsqueeze(1).repeat([1, G.num_ws, 1]), noise_mode='const')
          print(mask)
          if i % 100 == 0:
               plt.imshow(TF.to_pil_image(tf(image)[0]))
               plt.show() 
               pil_image = TF.to_pil_image(image[0].add(1).div(2).clamp(0,1))
               os.makedirs(f'samples/{timestring}', exist_ok=True)
               pil_image.save(f'samples/{timestring}/{i:04}.jpg')
          print(f"Image {i}/{steps} | Current loss: {loss}")
         
         
    try:
      timestring = time.strftime('%Y%m%d%H%M%S')
      run(timestring)
    except KeyboardInterrupt:
      pass
       
       
  

