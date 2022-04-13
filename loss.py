import torch

def changeloss(image):
    conv = torch.nn.Conv2d(1, 1, 5, bias = False)

  #  conv.weight = torch.nn.Parameter(torch.Tensor([[[0,1,0],[1,0,1],[0,1,0]], [[0,1,0],[1,0,1],[0,1,0]] , [[0,1,0],[1,0,1],[0,1,0]]]))
    conv.weight = torch.nn.Parameter(torch.cuda.FloatTensor([[[[ 1,1,1,1,1],[ 1,1,1,1,1],
          [ 1,1,-15,1,1],
          [ 1,1,1,1,1],[ 1,1,1,1,1]]]]))
    loss = torch.abs(conv(image))/15.0
    return torch.mean(loss)