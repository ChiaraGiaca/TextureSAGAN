import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
import copy
from test_function import test_func
import os
import torch
import tqdm

opt = TrainOptions().parse()
#print(opt.dataset_mode)
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#Training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)
total_steps = 0

opt.results_dir = os.path.dirname(opt.checkpoints_dir) + 'checkpoints'
print('Images are stored into: ', opt.results_dir)
image_label = opt.dataroot.split('/')[-1]

for epoch in tqdm.tqdm(range(1, opt.niter + 1)):
    epoch_start_time = time.time()
    epoch_iter = 0
    for i, data in enumerate(dataset):
        if epoch == 1:
            save_data = data
        else:
            save_data = save_data

        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        
       
        model.set_input(data)
        model.optimize_parameters()
        
        if total_steps % opt.display_freq == 0:
            #print("Saving images......")
            model.set_input(save_data)
            model.forward()
          
            visuals, start_points = model.get_current_visuals()
            visualizer.display_current_results(visuals, start_points, epoch, image_label)

    if epoch % opt.niter == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)
        #test_func(opt, webpage, epoch=str(epoch))

