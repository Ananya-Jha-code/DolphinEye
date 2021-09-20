from threading import Thread
import cv2
import numpy as np

class vidLoader:  
    '''
    Load in video stream
    '''
    def __init__(self, sources='0', img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride
        self.imgs, self.fps, self.frames, self.threads = [None], [0], [0], [None] 
        self.sources = '0'
        src=[sources]
        for i, s in enumerate(src):  
            s = '0'
            cap = cv2.VideoCapture(int(s))
            assert cap.isOpened(), f'Failed to open {s}'
            self.fps[i] = max(cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 30.0  
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  
            _, self.imgs[i] = cap.read() 
            self.threads[i] = Thread(target=self.update, args=([i, cap]), daemon=True)
            self.threads[i].start()
        s = np.stack([resize_img(x, self.img_size, stride=self.stride).shape for x in self.imgs])
        self.rect = np.unique(s, axis=0).shape[0] == 1 

    def update(self, i, cap):
        n, f, read = 0, self.frames[i], 1 
        while cap.isOpened() and n < f:
            n += 1
            cap.grab()
            if n % read == 0:
                success, im = cap.retrieve()
                self.imgs[i] = im if success else self.imgs[i] * 0

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration
        img0 = self.imgs.copy()
        img = [resize_img(x, self.img_size, stride=self.stride) for x in img0]

        img = np.stack(img, 0)
        img = img[..., ::-1].transpose((0, 3, 1, 2)) 
        img = np.ascontiguousarray(img)
        return self.sources, img, img0

    def __len__(self):
        return len(self.sources)  


def resize_img(im, new_shape=(640, 640), color=(114, 114, 114), stride=32):
    '''
    Resizes image with unchanged aspect ratio using padding
    '''
    shape = im.shape[:2] 
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    r = min(r, 1.0)
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  
    dw, dh = np.mod(dw, stride), np.mod(dh, stride) 
    dw /= 2  
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im

