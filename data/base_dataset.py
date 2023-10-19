import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def get_transform(opt):
    transform_list = []
    osize = [opt.fineSize, opt.fineSize]
    transform_list.append(transforms.Resize(osize, Image.BICUBIC))
    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


