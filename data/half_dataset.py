import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import PIL
from pdb import set_trace as st
import random
import matplotlib.pyplot as plt
 
class HalfDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir = os.path.join(opt.dataroot, opt.phase)
        self.paths = make_dataset(self.dir)
        self.paths = sorted(self.paths)
        self.size = len(self.paths)
        self.fineSize = opt.fineSize #size images should be cropped into
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        path = self.paths[index % self.size]
        B_img = Image.open(path).convert('RGB')

        #flips randomly the image if it is training 
        if self.opt.isTrain and not self.opt.no_flip:
            if random.random() > 0.5:
                B_img = B_img.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                B_img = B_img
                
        w, h = B_img.size
        #print(w, h)
        random_w = random.randint(0, w - self.fineSize)
        random_h = random.randint(0, h - self.fineSize)
        #print('Left, upper coordinates: ', random_w, random_h)
        
        #RESIZING THE ORIGINAL IMAGE TO 128x128
        B_img = B_img.crop((random_w, random_h, random_w + self.fineSize, random_h + self.fineSize))  #a 4-tuple defining the left, upper, right, and lower pixel coordinate.
        w, h = B_img.size
        random_w = random.randint(0, int(w/2))
        random_h = random.randint(0, int(h/2))

        #CROPPED IMAGE TO 64x64
        A_img = B_img.crop((random_w, random_h, int(random_w + w/2), int(random_h + h/2)))
        #print(random_w, random_h, int(random_w + w/2), int(random_h + h/2))
        
        A_img = self.transform(A_img)
        B_img = self.transform(B_img)
        

        return {'A': A_img, 'B': B_img,
                'A_paths': path, 'B_paths': path,
                'A_start_point':[(random_w, random_h)]}

    def __len__(self):
        return self.size

    def name(self):
        return 'HalfDataset'

