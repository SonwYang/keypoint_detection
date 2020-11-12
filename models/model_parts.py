import torch.nn.functional as F
import torch.nn as nn
import torch
from .Attention.CBAM import CBAM
from .modules.deform_conv_v2 import DeformConv2d


class CombinationModule(nn.Module):
    def __init__(self, c_low, c_up, batch_norm=False, group_norm=False, instance_norm=False):
        super(CombinationModule, self).__init__()
        if batch_norm:
            self.up =  nn.Sequential(nn.Conv2d(c_low, c_up, kernel_size=3, padding=1, stride=1),
                                     nn.BatchNorm2d(c_up),
                                     nn.ReLU(inplace=True))
            self.cat_conv =  nn.Sequential(nn.Conv2d(c_up*2, c_up, kernel_size=1, stride=1),
                                           nn.BatchNorm2d(c_up),
                                           nn.ReLU(inplace=True))
        elif group_norm:
            self.up = nn.Sequential(nn.Conv2d(c_low, c_up, kernel_size=3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=32, num_channels=c_up),
                                    nn.ReLU(inplace=True))
            self.cat_conv = nn.Sequential(nn.Conv2d(c_up * 2, c_up, kernel_size=1, stride=1),
                                          nn.GroupNorm(num_groups=32, num_channels=c_up),
                                          nn.ReLU(inplace=True))
        elif instance_norm:
            self.up = nn.Sequential(nn.Conv2d(c_low, c_up, kernel_size=3, padding=1, stride=1),
                                    nn.InstanceNorm2d(num_features=c_up),
                                    nn.ReLU(inplace=True))
            self.cat_conv = nn.Sequential(nn.Conv2d(c_up * 2, c_up, kernel_size=1, stride=1),
                                          nn.InstanceNorm2d(num_features=c_up),
                                          nn.ReLU(inplace=True))
        else:
            self.up =  nn.Sequential(nn.Conv2d(c_low, c_up, kernel_size=3, padding=1, stride=1),
                                     nn.ReLU(inplace=True))
            self.cat_conv = nn.Sequential(nn.Conv2d(c_up*2, c_up, kernel_size=1, stride=1),
                                           nn.ReLU(inplace=True))
        self.cbam = CBAM(channel=c_up)

    def forward(self, x_low, x_up):
        x_low = self.up(F.interpolate(x_low, x_up.shape[2:], mode='bilinear', align_corners=False))
        x_low = self.cbam(x_low)
        return self.cat_conv(torch.cat((x_up, x_low), 1))


class CombinationModuleDCN(nn.Module):
    def __init__(self, c_low, c_up):
        super(CombinationModuleDCN, self).__init__()
        self.up = nn.Sequential(DeformConv2d(c_low, c_up, kernel_size=3, padding=1, stride=1),
                                nn.BatchNorm2d(c_up),
                                nn.ReLU(inplace=True))
        self.cat_conv = nn.Sequential(DeformConv2d(c_up * 2, c_up, kernel_size=1, stride=1),
                                      nn.BatchNorm2d(c_up),
                                      nn.ReLU(inplace=True))
        self.cbam = CBAM(channel=c_up)

    def forward(self, x_low, x_up):
        x_low = self.up(F.interpolate(x_low, x_up.shape[2:], mode='bilinear', align_corners=False))
        x_low = self.cbam(x_low)
        return self.cat_conv(torch.cat((x_up, x_low), 1))