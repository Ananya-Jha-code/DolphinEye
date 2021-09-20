import yaml 
from copy import deepcopy
from pathlib import Path
from models.common import *
from models.experimental import *
from utils.utils import *

class Detect(nn.Module):
    stride = None

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True): 
        super().__init__()
        
        self.no = nc + 5  
        self.nl = len(anchors) 
        self.na = len(anchors[0]) // 2  
        self.grid = [torch.zeros(1)] * self.nl  
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  
        self.inplace = inplace 

    def forward(self, x):
        z = [] 
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  
            bs, _, ny, nx = x[i].shape  
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

            y = x[i].sigmoid()
            if self.inplace:
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  
            z.append(y.view(bs, -1, self.no))
        return (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, cfg='models/yolov5s.yaml', ch=3): 
        super().__init__()
        
        self.yaml_file = Path(cfg).name
        with open(cfg) as f:
            self.yaml = yaml.safe_load(f)  

        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  
        self.model= parse_model(deepcopy(self.yaml), ch=[ch]) 
        self.names = [str(i) for i in range(self.yaml['nc'])]  
        self.inplace = self.yaml.get('inplace', True)

        m = self.model[-1] 
        if isinstance(m, Detect):
            s = 256  
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))]) 
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
    
    def forward(self, x):
        y= [] 
        for m in self.model:
            if m.f != -1:  
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  
            y.append(x if m.i in self.save else None)  
        return x

    def fuse(self):
        for m in self.model.modules():
            if isinstance(m, (Conv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn) 
                m.forward = m.forward_fuse 
        return self



def parse_model(d, ch): 
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  
    no = na * (nc + 5) 
    layers,c2 = [],ch[-1]  
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']): 
        m = eval(m) if isinstance(m, str) else m  
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n 
        if m in [Conv, Bottleneck,  SPP, Focus, BottleneckCSP, C3]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  
                c2 = math.ceil(c2 * gw / 8) * 8

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3]:
                args.insert(2, n) 
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  
                args[1] = [list(range(args[1] * 2))] * len(f)
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
        np = sum([x.numel() for x in m_.parameters()]) 
        m_.i, m_.f, m_.np = i, f, np  
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers)
