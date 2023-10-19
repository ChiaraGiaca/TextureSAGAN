import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.nn.modules.utils import _pair, _triple
import torch.nn.functional as F
from torch.nn.modules import conv, Linear
import numpy as np


###############################################################################
# Functions
###############################################################################
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#function to initialize the weights of the NN layers depending on the type of layer 
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

#Batch normalization layer
def get_norm_layer():
    norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    return norm_layer

#function to define the generator of the NN
def define_G(input_nc, output_nc, ngf, use_dropout=False, gpu_ids=[], padding_type='reflect'):
    netG = None
    norm_layer = get_norm_layer()
    netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9,
                            gpu_ids=gpu_ids).to(device)
    netG.apply(weights_init)
    return netG

#function to define the discriminator of the NN
def define_D(input_nc, ndf, use_sigmoid=False, gpu_ids=[]):
    netD = None
    norm_layer = get_norm_layer()
    netD = NLayerDiscriminator(input_nc, ndf, n_layers=4, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                gpu_ids=gpu_ids).to(device)
    netD.apply(weights_init)
    return netD


##############################################################################
# Network classes
##############################################################################

#Self attention Layer: original code from https://github.com/heykeetae/Self-Attention-GAN
class Self_Attn(nn.Module):
    
    def __init__(self, in_channels):
        super(Self_Attn, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels//8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels//8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1)) #learnable scalar
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        #
        #    inputs:
        #        x: input feature maps (B x C x W x H)
        #    returns:
        #        out: self attention value + input feature 
        #        attention: B x N x N (N is width * height)
        
        B, C, W, H = x.size() 
        
        proj_query = self.query_conv(x).view(B, -1, W * H).permute(0, 2, 1) # B x N x C
        proj_key = self.key_conv(x).view(B, -1, W * H) # B X C x N

        torch.cuda.empty_cache() #for computational purposes

        energy = torch.bmm(proj_query, proj_key)  #dot product transpose check
        attention = self.softmax(energy) # B x N x N
        proj_value = self.value_conv(x).view(B, -1, W * H) # B x C x N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)) # B x C x N
        out = out.view(B, C, W, H)
        out = self.gamma * out + x
        return out
        
# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input --> for texture synthesis
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
  def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 gpu_ids=[], padding_type='reflect'):
      assert (n_blocks >= 0)
      super(ResnetGenerator, self).__init__()
      self.input_nc = input_nc #number of input channels
      self.output_nc = output_nc #number of output channels 
      self.ngf = ngf 
      self.gpu_ids = gpu_ids
      if type(norm_layer) == functools.partial:
          use_bias = norm_layer.func == nn.InstanceNorm2d
      else:
          use_bias = norm_layer == nn.InstanceNorm2d

      #upsampling
      model = [nn.ReflectionPad2d(3),
                nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                          bias=use_bias),
                norm_layer(ngf),
                nn.ReLU(True)]

      #ADDED SELF ATTENTION
      model += [Self_Attn(ngf)]
      
      n_downsampling = 2
      for i in range(n_downsampling):
          mult = 2 ** i
          model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                              stride=2, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.ReLU(True)]

      mult = 2 ** n_downsampling
      for i in range(n_blocks):
          model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                use_bias=use_bias)]

      for i in range(n_downsampling):
          mult = 2 ** (n_downsampling - i)
          model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1,
                                        bias=use_bias),
                    norm_layer(int(ngf * mult / 2)),
                    nn.ReLU(True)]
  
      model += [nn.ReflectionPad2d(3)]
      model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
      model += [nn.Tanh()]

      self.model = nn.Sequential(*model)
    
  def forward(self, input):
    if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
        return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
    else:
        return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        conv_block += [nn.ReplicationPad2d(1)] #padding type = replicate

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        #ADDED SELF ATTENTION
        sequence += [Self_Attn(ndf)]

        nf_mult = 1
        nf_mult_prev = 1
        #downsampling
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) > 1 and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)
            
 