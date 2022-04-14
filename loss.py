import torch
import torch.nn.functional as F
def changeloss(image):
    conv = torch.nn.Conv2d(1, 1, 5, bias = False)

  #  conv.weight = torch.nn.Parameter(torch.Tensor([[[0,1,0],[1,0,1],[0,1,0]], [[0,1,0],[1,0,1],[0,1,0]] , [[0,1,0],[1,0,1],[0,1,0]]]))
    conv.weight = torch.nn.Parameter(torch.cuda.FloatTensor([[[[ 1,1,1,1,1],[ 1,1,1,1,1],
          [ 1,1,-15,1,1],
          [ 1,1,1,1,1],[ 1,1,1,1,1]]]]))
    loss = torch.abs(conv(image))/15.0
    return torch.mean(loss)

def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

def prompts_dist_loss(x, targets, loss):
    if len(targets) == 1: # Keeps consitent results vs previous method for single objective guidance 
      return loss(x, targets[0])
    distances = [loss(x, target) for target in targets]
    return torch.stack(distances, dim=-1).sum(dim=-1)  

def norm1(prompt):
    "Normalize to the unit sphere."
    return prompt / prompt.square().sum(dim=-1,keepdim=True).sqrt()