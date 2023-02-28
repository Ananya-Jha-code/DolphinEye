import cv2
import torch
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.streamUtils import *
from utils.utils import *
from utils.annotateUtils import *

device = 'cpu'

@torch.no_grad()
def inference(weights='weights/yolov5s.pt', source=0, imgsz=640, hide_labels=False, hide_conf=True):

    stride, names = 64, ["class{}".format(i for i in range(1000))] 
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max()) 
    names = model.module.names if hasattr(model, 'module') else model.names  
    cudnn.benchmark = True 
    stream = vidLoader(source, img_size=imgsz, stride=stride)
    seen = 0
    for path, img, im0s  in stream:
        img = torch.from_numpy(img).to(device)
        img = img.float() 
        img = img / 255.0  
        if len(img.shape) == 3:
            img = img[None]  
        pred = model(img)[0]
        pred = nms(pred, conf_thres=0.25, iou_thres=0.45, max_det=500)
        for i, det in enumerate(pred): 
            seen += 1
            p, im0 = path[i], im0s[i].copy()
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  
            pen = MagicPen(im0)
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    d=dist_calc(xywh)
                    label = label+". Distance = "+str(d)
                    color=Colors()
                    color=color(c, True)
                    if color == (56, 56, 255):
                        winsound.Beep(5000, 1000)
                    pen.box_label(xyxy, label, color)
            
            ######### Showing Result ###########
            im0 = pen.result()
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)  



if __name__ == "__main__":
    inference()
    
