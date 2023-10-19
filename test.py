import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
import matplotlib.pyplot as plt
import numpy as np

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
imgs = []
for j in range(5):
  for i, data in enumerate(dataset):
      model.set_input(data)
      model.test()
      visuals = model.get_current_visuals()
      img_path = model.get_image_paths()
      print('process image... %s' % img_path)
      imgs.append(visualizer.save_images(webpage, visuals, img_path))
f, axarr = plt.subplots(1, 6, figsize=(40, 20))
axarr[0].imshow(np.squeeze(imgs[0][0]).astype(np.uint8)) 
axarr[0].text(0.5,-0.1, "Smaller texture", size=12, ha="center", 
         transform=axarr[0].transAxes)
for i in range(1, 6):
  axarr[i].imshow(np.squeeze(imgs[i-1][1]).astype(np.uint8)) 
  axarr[i].text(0.5,-0.1, "Extended texture " + str(i), size=12, ha="center", 
         transform=axarr[i].transAxes)
plt.show()
