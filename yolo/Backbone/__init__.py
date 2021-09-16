import torch
from torch import nn 

from nb.torch.blocks.bottleneck_blocks import SimBottleneck, SimBottleneckCSP
from nb.torch.blocks.trans_blocks import Focus
from nb.torch.blocks.head_blocks import SPP
from nb.torch.blocks.conv_blocks import ConvBase
from nb.torch.utils import device
from torch._C import CudaByteStorageBase

class YoloV5(nn.Module):
    def __init__(self,num_cls=80,ch=3,anchors=None):
        super (YoloV5,self).__init__()
        assert anchors !=None,'anchor must be provided'

        cd = 2
        wd = 3

        self.focus = Focus(ch,64//cd)
        self.conv1 = ConvBase(64//cd,128//cd,3,2)
        self.csp1 = SimBottleneckCSP(128//cd,128//cd ,n=3//wd)
        self.conv2 = ConvBase(128//cd,256//cd,3,2)
        self.csp2 = SimBottleneckCSP(256//cd,256//cd,n=9//wd)
        self.conv3 = ConvBase(256//cd,512//cd,3,2)
        self.csp3 = SimBottleneckCSP(512//cd,512//cd,n=9//wd)
        self.conv4 = ConvBase(512//cd,1024//cd,3,2)
        self.spp = SPP(1024//cd,1024//cd)
        self.csp4 = SimBottleneckCSP(1024//cd,1024//cd,n=3//wd,shortcut=False)

        #PANet
        self.conv5 = ConvBase(1024//cd,512//cd)
        self.up1 = nn.Upsample(scale_factor=2)
        self.csp5 = SimBottleneckCSP(1024//cd,512//cd,n=3//wd,shortcut=False)

        self.conv6 = ConvBase(512//cd,256//cd)
        self.up2 = nn.Upsample(scale_factor=2)
        self.csp6 = SimBottleneckCSP(512//cd,256//cd,n=3//wd, shortcut=False)

        self.conv7 = ConvBase(256//cd,256//cd,3,2)
        self.csp7 = SimBottleneckCSP(512//cd,512//cd,n=3//wd,shortcut=False)

        self.conv8 = ConvBase(512//cd,512//cd,3,2)
        self.csp8 = SimBottleneckCSP(512//cd,1024//cd,n=3//wd,shortcut=False)

    def _build_backbone(self,x):
        x = self.focus(x)
        x = self.conv1(x)
        x = self.csp1(x)
        x_p3 = self.conv2(x)
        x = self.csp2(x_p3)
        x_p4 = self.conv3(x)
        x = self.csp3(x_p4)
        x_p5 = self.conv4(x)
        x = self.spp(x_p5)
        x = self.csp4(x)
        return x_p3,x_p4,x_p5,x
    






