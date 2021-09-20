import cv2
import numpy as np

class Colors:
    '''
    Random color generator for different bounding boxes
    '''
    def __init__(self):
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c
    @staticmethod
    def hex2rgb(h): 
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

class MagicPen:
    '''
    Draws bounding box on frames
    '''
    def __init__(self, im):
        self.im = im
        self.lw = 2 

    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(self.im, c1, c2, color=color, thickness=self.lw, lineType=cv2.LINE_AA)
        if label:
            w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=1)[0]
            c2 = c1[0] + w, c1[1] - h - 3
            cv2.rectangle(self.im, c1, c2, color, -1, cv2.LINE_AA)  
            cv2.putText(self.im, label, (c1[0], c1[1] - 2), 0, self.lw / 3, txt_color, thickness=1, lineType=cv2.LINE_AA)
            
    def result(self):
        return np.asarray(self.im)

