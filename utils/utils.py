import numpy as np
import torch
import torchvision
from torch import nn

def xyxy2xywh(x):
    '''
    Converts x0,y0,x1,y1 to x,y,width,height
    '''
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2 
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  
    y[:, 2] = x[:, 2] - x[:, 0]  
    y[:, 3] = x[:, 3] - x[:, 1]  
    return y

def xywh2xyxy(x):
    '''
    Converts x,y,width,height to x0,y0,x1,y1
    '''
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  
    y[:, 1] = x[:, 1] - x[:, 3] / 2  
    y[:, 2] = x[:, 0] + x[:, 2] / 2 
    y[:, 3] = x[:, 1] + x[:, 3] / 2  
    return y

def scale_coords(img1_shape, coords, img0_shape):
    '''
    Resizes predicted img coords to original img coords
    '''
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    
    coords[:, [0, 2]] -= pad[0]  
    coords[:, [1, 3]] -= pad[1]  
    coords[:, :4] /= gain 
    coords[:, [0, 2]] = coords[:, [0, 2]].clip(0, img0_shape[1])  
    coords[:, [1, 3]] = coords[:, [1, 3]].clip(0, img0_shape[0])  

    return coords

def nms(prediction, conf_thres=0.25, iou_thres=0.45,labels=(), max_det=300):
    '''
    Non Max supression
    '''
    nc = prediction.shape[2] - 5  
    xc = prediction[..., 4] > conf_thres  
    max_wh = 4096  
    max_nms = 30000  
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  
        x = x[xc[xi]]  
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5] 
            v[:, 4] = 1.0 
            v[range(len(l)), l[:, 0].long() + 5] = 1.0 
            x = torch.cat((x, v), 0)
        if not x.shape[0]:
            continue
        x[:, 5:] *= x[:, 4:5]  
        box = xywh2xyxy(x[:, :4])
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        n = x.shape[0] 
        if not n: 
            continue
        elif n > max_nms: 
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  
        
        c = x[:, 5:6] * (max_wh) 
        boxes, scores = x[:, :4] + c, x[:, 4]  
        i = torchvision.ops.nms(boxes, scores, iou_thres) 
        if i.shape[0] > max_det:  
            i = i[:max_det]
        
        output[xi] = x[i]

    return output

def dist_calc(bboxes):
    '''
    Calculates distance from bounding boxes
    '''
    x, y, w, h = bboxes[0], bboxes[1], bboxes[2], bboxes[3]
    distance = (2 * 3.14 * 180) / (w + h * 360) * 1000 + 3 
    return distance


def fuse_conv_and_bn(conv, bn):
    '''
    Fuses conv and bn layer
    '''
    fusedconv = nn.Conv2d(conv.in_channels, conv.out_channels, kernel_size=conv.kernel_size, stride=conv.stride, padding=conv.padding, groups=conv.groups,bias=True).requires_grad_(False).to(conv.weight.device)
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


