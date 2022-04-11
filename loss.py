import torch

def changeloss(image):
    conv = torch.nn.Conv2d(3, 1, 3, bias = False)
    print(conv.weight.size())
    conv.weight = torch.nn.Parameter(torch.Tensor([[[0,1,0],[1,0,1],[0,1,0]], [[0,1,0],[1,0,1],[0,1,0]] , [[0,1,0],[1,0,1],[0,1,0]]]))
    loss = conv(image)
    return torch.mean(loss)