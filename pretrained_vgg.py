import torch.nn as nn
import torch.nn.functional as F


class CustomVGG19(nn.Module):
    def __init__(self, pool='MaxPool'):
        super(CustomVGG19, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3_4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4_4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5_4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        if pool == 'MaxPool':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'AvgPool':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, output_feature_list):
        out = dict()
        out['conv1_1'] = F.relu(self.conv1_1(x))
        out['conv1_2'] = F.relu(self.conv1_2(out['conv1_1']))
        out['pool1'] = self.pool1(out['conv1_2'])
        out['conv2_1'] = F.relu(self.conv2_1(out['pool1']))
        out['conv2_2'] = F.relu(self.conv2_2(out['conv2_1']))
        out['pool2'] = self.pool2(out['conv2_2'])
        out['conv3_1'] = F.relu(self.conv3_1(out['pool2']))
        out['conv3_2'] = F.relu(self.conv3_2(out['conv3_1']))
        out['conv3_3'] = F.relu(self.conv3_3(out['conv3_2']))
        out['conv3_4'] = F.relu(self.conv3_4(out['conv3_3']))
        out['pool3'] = self.pool3(out['conv3_4'])
        out['conv4_1'] = F.relu(self.conv4_1(out['pool3']))
        out['conv4_2'] = F.relu(self.conv4_2(out['conv4_1']))
        out['conv4_3'] = F.relu(self.conv4_3(out['conv4_2']))
        out['conv4_4'] = F.relu(self.conv4_4(out['conv4_3']))
        out['pool4'] = self.pool4(out['conv4_4'])
        out['conv5_1'] = F.relu(self.conv5_1(out['pool4']))
        out['conv5_2'] = F.relu(self.conv5_2(out['conv5_1']))
        out['conv5_3'] = F.relu(self.conv5_3(out['conv5_2']))
        out['conv5_4'] = F.relu(self.conv5_4(out['conv5_3']))
        out['pool5'] = self.pool5(out['conv5_4'])
        return [out[k] for k in output_feature_list]


def export_official_vgg19_weight():
    from collections import OrderedDict
    from torchvision.models import VGG19_Weights, vgg19
    import torch

    custom_vgg19 = CustomVGG19()
    pytorch_vgg19 = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)

    custom_vgg19_state_dict = custom_vgg19.state_dict()
    pytorch_vgg19_state_dict = pytorch_vgg19.state_dict()

    custom_vgg19_key_list = [k for k in custom_vgg19_state_dict.keys()]
    pytorch_vgg19_key_list = [k for k in pytorch_vgg19_state_dict.keys() if k.startswith("features.")]

    key_map = OrderedDict()
    for k, v in zip(pytorch_vgg19_key_list, custom_vgg19_key_list):
        key_map[k] = v
    new_state_dict = OrderedDict()
    for k, v in key_map.items():
        print("replace key '{}' with '{}'".format(k, v))
        new_state_dict[v] = pytorch_vgg19_state_dict[k]

    print("exported state dict:")
    for k, v in new_state_dict.items():
        print("{} {}".format(k, v.shape))

    torch.save(new_state_dict, "./vgg19.pth")


if __name__ == '__main__':
    export_official_vgg19_weight()
