import torch.nn as nn
import numpy as np
import torch
from .backbone import get_encoder
from .FPN.bi import BiFpn
from .Attention.CBAM import CBAM
from functools import partial

nonlinear=partial(nn.SELU, inplace=True)

class FReLU(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv_frelu = nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch)
        self.bn_frelu = nn.BatchNorm2d(in_ch)

    def forward(self, x):
        x1 = self.conv_frelu(x)
        x1 = self.bn_frelu(x1)
        x = torch.max(x, x1)
        return x


class CTRBOX(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super(CTRBOX, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))
        self.base_network = get_encoder('efficientnet-b5')
        encoder_channels = self.base_network.out_shapes
        self.bi = BiFpn([encoder_channels[3], encoder_channels[2], encoder_channels[1], encoder_channels[0]],
                        encoder_channels[3], 4, 2)
        self.heads = heads

        for head in self.heads:
            classes = self.heads[head]
            fc = nn.Sequential(CBAM(encoder_channels[3]),
                                   nn.Conv2d(encoder_channels[3], head_conv, kernel_size=3, padding=1, bias=True),
                                   FReLU(head_conv),
                                   nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(fc)

            self.__setattr__(head, fc)

    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base_network(x)
        b = self.bi([x[3], x[2], x[1], x[0]])

        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(b[0])
            if 'hm' in head or 'cls' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        return dec_dict