import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import torch.nn as nn
import util.util as util
from .base_model import BaseModel
from . import networks
from .vgg import VGG, GramMatrix, GramMSELoss
import torchvision
from math import exp
from util.image_pool import ImagePool

class SelfAttentionGANModel(BaseModel):
    def name(self):
        return 'Self Attention GAN Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        
        # define tensors
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                  int(opt.fineSize / 2), int(opt.fineSize / 2))
        #real texture
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                  opt.fineSize, opt.fineSize)

        #residual layers
        self.style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
        self.loss_layers = self.style_layers
    
        #feature extraction
        self.vgg = VGG()
        self.vgg.load_state_dict(torch.load(os.getcwd() + '/vgg19.pth'))
        
        for param in self.vgg.parameters():
            param.requires_grad = False
        if torch.cuda.is_available():
            self.vgg.cuda()

        self.style_weights = [1e3 / n ** 2 for n in [64, 128, 256, 512, 512]]
        self.weights = self.style_weights

        # load/define networks
        
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, not opt.no_dropout, self.gpu_ids)
        if self.isTrain:
            use_sigmoid = True
            self.netD = networks.define_D(opt.output_nc, opt.ndf, use_sigmoid, self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=False, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

    def set_input(self, input):
        #from 128x128 to 64x64
        input_A = input['A']
        input_B = input['B']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths']
        self.start_points= input['A_start_point']

    def forward(self):
        #I get the tensors by packing them with Variable
        self.real_A = Variable(self.input_A)
        #I put A as input to the generator and get the fake_B
        self.fake_B = self.netG.forward(self.real_A)
        #Test image
        self.real_B = Variable(self.input_B)

    # no backprop gradients
    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG.forward(self.real_A)
        self.real_B = Variable(self.input_B, volatile=True)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        # here we use real image to create fake_AB
        fake_AB = self.fake_AB_pool.query(self.fake_B.clone())
        self.pred_fake = self.netD.forward(fake_AB.detach())
        #adversarial loss
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)

        # Real
        real_AB = self.real_B.clone()
        self.pred_real = self.netD.forward(real_AB)
        self.loss_D_real = self.criterionGAN(self.pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        
        self.fake_B = self.fake_B.cuda()
        out = self.vgg(self.fake_B, self.loss_layers)
  
        # First, G(A) should fake the discriminator
        fake_AB = self.fake_B.clone()
        pred_fake = self.netD.forward(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        #print(self.fake_B.shape, self.real_B.shape)
        fake_B_resized = self.fake_B.detach().clone()
        fake_B_resized= fake_B_resized.resize_(self.real_B.shape)

        #L1 loss
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * 10.0 #10.0 is the weight for cycle loss A -> B -> A

        #diversity regularization
        regularization_strength= 0.15
        self.diversity_reg_G = self.diversity_regularization(fake_B_resized, regularization_strength)
        #print('Diversity: ', self.diversity_reg_G)

        self.loss_G = self.loss_G_GAN + self.loss_G_L1 - self.diversity_reg_G
        #print('Diversity loss: ', self.loss_G)
        #print('Non diversity loss: ', self.loss_G_GAN + self.loss_G_L1)
      
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)]), self.start_points

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

    def diversity_regularization(self, fake, strength):
        """
        The diversity regularization applied in the feature space of the generator
        """
        loss = torch.nn.L1Loss()
        ans = 0
        
        #I compare the tensors of the fake image channel per channel and compute the loss: 
        #in the one-shot image synthesis setting, the perceptual distance of the generated images
        #should not be dependent on the distance between their latent codes

        for i in range(len(fake)): #1
            for k in range(fake[i].shape[0]): 
                for m in range(k + 1, fake[i].shape[0]):
                    ans += -loss(fake[i][k], fake[i][m])
        
        return ans *  strength/len(fake) 

