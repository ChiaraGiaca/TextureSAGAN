import numpy as np
import os
import ntpath
import time
from . import util
from . import html
from PIL import Image
import matplotlib.pyplot as plt

class Visualizer():
    def __init__(self, opt):
        self.use_html = opt.isTrain 
        self.win_size = 256
        self.name = opt.name
        self.fineSize = 256

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            #print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, A_start_point, epoch, image_label):
    
        batchsize = len(visuals['real_A'])
        #print(batchsize)
        newIm = Image.new("RGB", (self.fineSize*3, self.fineSize * batchsize))
        #print('NewIm generated')
        i = 0
        for label, image_numpy in visuals.items():
            img_path_dir = os.path.join(self.img_dir, image_label)
            img_path = img_path_dir + '/epoch%.3d_%s.png' % (epoch, label)
            for col in range(batchsize):
                image = image_numpy[col].astype(np.uint8)
                img = Image.fromarray(image)
                w, h = img.size
                #print(w,h)
                newIm.paste(img, (int(self.fineSize/3)*(i+1)*2, col * int(self.fineSize/2)))
            util.mkdir(img_path_dir)
            if epoch == 50:
              img.save(img_path)
            else: 
              if 'fake' in img_path: #for computation purposes
                img.save(img_path)
            i = i + 1
        newIm.save(os.path.join(self.img_dir, 'epoch%.3d_all.png' % (epoch)))

    # plot image
    def save_images(self, webpage, visuals, image_path):
        label, image_numpy = visuals[0], visuals[1]
        #print(label)
        img = list(label.items())[1][1]

        real_img = list(label.items())[0][1]
        #print(np.squeeze(real_img).shape)
        #print(real_img)
        #f, axarr = plt.subplots(1, 2, figsize=(9, 3))
        #axarr[0].imshow(np.squeeze(real_img).astype(np.uint8)) 
        #axarr[1].imshow(np.squeeze(img).astype(np.uint8)) 
        #plt.imshow(np.squeeze(img))
        return real_img, img 
        #plt.show()

