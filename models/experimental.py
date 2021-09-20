import torch
import torch.nn as nn
from models.common import Conv
from models.yolo import Detect, Model


class Ensemble(nn.ModuleList):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, augment=False, profile=False, visualize=False):
        y = []
        for module in self:
            y.append(module(x, augment, profile, visualize)[0])
        y = torch.cat(y, 1)  
        return y, None  


def attempt_load(weights, map_location=None, inplace=True, fuse=True):
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(w, map_location=map_location)

        if fuse:
            model.append(ckpt['model'].float().fuse().eval()) 

    for m in model.modules():
        m.inplace = inplace
        if type(m) in [nn.SiLU, Detect, Model]:
            m.inplace = inplace 
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set() 

    return model[-1] 

