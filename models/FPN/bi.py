import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
import math
from torch.nn.init import _calculate_correct_fan


def siren_uniform_(tensor: torch.Tensor, mode: str = 'fan_in', c: float = 6):
    r"""Fills the input `Tensor` with values according to the method
    described in ` Implicit Neural Representations with Periodic Activation
    Functions.` - Sitzmann, Martel et al. (2020), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where
    .. math::
        \text{bound} = \sqrt{\frac{6}{\text{fan\_mode}}}
    Also known as Siren initialization.

    :param tensor: an n-dimensional `torch.Tensor`
    :type tensor: torch.Tensor
    :param mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing
        ``'fan_in'`` preserves the magnitude of the variance of the weights in
        the forward pass. Choosing ``'fan_out'`` preserves the magnitudes in
        the backwards pass.s
    :type mode: str, optional
    :param c: value used to compute the bound. defaults to 6
    :type c: float, optional
    """
    fan = _calculate_correct_fan(tensor, mode)
    std = 1 / math.sqrt(fan)
    bound = math.sqrt(c) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)

class Sine(nn.Module):
    def __init__(self, w0: float = 1.0):
        """Sine activation function with w0 scaling support.
        """
        super(Sine, self).__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        return torch.sin(self.w0 * x)

    @staticmethod
    def _check_input(x):
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                'input to forward() must be torch.xTensor')

nonlinear = partial(nn.SELU, inplace=True)

class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel = 3,strides = 1,padding=1,
                 bias = True,act = True,bn = True):
        super(ConvBlock,self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                              kernel_size=kernel,stride=strides,padding=padding,
                              bias=bias)
        siren_uniform_(self.conv.weight)
        self.act = nonlinear() if act else None
        self.bn = nn.BatchNorm2d(in_channels) if bn else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.act:
            x = self.act(x)
        return x


class BiBlock(nn.Module):

    def __init__(self,feature_size):
        super(BiBlock,self).__init__()


        self.maxpooling = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.p7_to_p6 = ConvBlock(feature_size,feature_size)
        self.p6_to_p5 = ConvBlock(feature_size,feature_size)
        self.p5_to_p4 = ConvBlock(feature_size,feature_size)

        self.p3 = ConvBlock(feature_size,feature_size)
        self.p4 = ConvBlock(feature_size,feature_size)
        self.p5 = ConvBlock(feature_size,feature_size)
        self.p6 = ConvBlock(feature_size,feature_size)
        self.p7 = ConvBlock(feature_size,feature_size)



    def forward(self, pyramids):
        p3,p4,p5,p6,p7 = pyramids

        p7_to_p6 = F.upsample(p7,size=p6.shape[-2:])
        # p7_to_p6 = F.upsample(p7,scale_factor=2)
        p7_to_p6 = self.p7_to_p6(p7_to_p6 + p6)

        p6_to_p5 = F.upsample(p7_to_p6,p5.shape[-2:])
        # p6_to_p5 = F.upsample(p7_to_p6,scale_factor=2)
        p6_to_p5 = self.p6_to_p5(p6_to_p5 + p5)

        p5_to_p4 = F.upsample(p6_to_p5,size=p4.shape[-2:])
        # p5_to_p4 = F.upsample(p6_to_p5,scale_factor=2)
        p5_to_p4 = self.p5_to_p4(p5_to_p4 + p4)

        p4_to_p3 = F.upsample(p5_to_p4,size=p3.shape[-2:])
        # p4_to_p3 = F.upsample(p5_to_p4,scale_factor=2)
        p3 = self.p3(p4_to_p3 + p3)

        p3_to_p4 = self.maxpooling(p3)
        p4 = self.p4(p3_to_p4 + p5_to_p4 + p4)

        p4_to_p5 = self.maxpooling(p4)
        p5 = self.p5(p4_to_p5 + p6_to_p5 + p5)

        p5_to_p6 = self.maxpooling(p5)
        p5_to_p6 = F.upsample(p5_to_p6,size=p6.shape[-2:])
        p6 = self.p6(p5_to_p6 + p7_to_p6 + p6)

        p6_to_p7 = self.maxpooling(p6)
        p6_to_p7 = F.upsample(p6_to_p7,size=p7.shape[-2:])
        p7 = self.p7(p6_to_p7 + p7)

        return p3,p4,p5,p6,p7


class BiFpn(nn.Module):

    def __init__(self, in_channels, out_channels, len_input,bi = 3):
        super(BiFpn,self).__init__()
        assert len_input <= 5
        self.len_input = len_input
        self.bi = bi
        self.default = 5 - len_input
        for i in range(len_input):
            setattr(self, 'p{}'.format(str(i)), ConvBlock(in_channels=in_channels[i], out_channels=out_channels,
                                                          kernel=1, strides=1, padding=0, act=False, bn=False))
        if self.default > 0:
            for i in range(self.default):
                setattr(self,'make_pyramid{}'.format(str(i)),ConvBlock(in_channels=in_channels[-1] if i == 0 else out_channels,out_channels=out_channels,kernel=3,strides=2,
                                                                       padding=1,act=False,bn=False))
        for i in range(bi):
            setattr(self, 'biblock{}'.format(str(i)), BiBlock(out_channels))

    def forward(self, inputs):
        pyramids = []
        for i in range(self.len_input):
            pyramids.append(getattr(self,'p{}'.format(str(i)))(inputs[i]))

        if self.default > 0:
            x = inputs[-1]
            for i in range(self.default):
                x = getattr(self,'make_pyramid{}'.format(str(i)))(x)
                pyramids.append(x)

        for i in range(self.bi):
            pyramids = getattr(self,'biblock{}'.format(str(i)))(pyramids)

        return pyramids


if __name__ == '__main__':
    p2 = torch.randn(2,64,128,128)
    p3 = torch.randn(2,128,64,64)
    p4 = torch.randn(2,256,32,32)
    p5 = torch.randn(2,512,16,16)

    b = BiFpn([64, 128, 256, 512], 256, 4, 3)
    py = b([p2, p3,p4,p5])
    for i in py:
        print(i.shape)

