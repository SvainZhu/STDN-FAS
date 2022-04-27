from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import rgb_to_yuv, rgb_to_hsv, ConvBNAct, Upsample, Downsample
import numpy as np

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    '''
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)
    '''
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    '''
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)
    '''

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class residual_gradient_conv(nn.Module):
    def __init__(self,
                 input_c: int,
                 output_c: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 norm: bool = True,
                 act: bool = True,
                 apply_dropout: bool = False
                 ):
        super(residual_gradient_conv, self).__init__()
        self.stem_conv = nn.Conv2d(in_channels=input_c, out_channels=output_c, kernel_size=kernel_size, stride=stride, padding=1)
        self.conv = nn.Conv2d(in_channels=input_c, out_channels=output_c, kernel_size=3, stride=stride)
        
        self.out_channels = output_c
        self.stride = stride
        self.padding = 1
        self.norm = norm
        self.act = act
        self.apply_dropout = apply_dropout

        self.sobel_plane = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
        if norm:
            self.bn = nn.BatchNorm2d(output_c)
        if act:
            self.ac = nn.PReLU()
        if apply_dropout:
            self.dropout = nn.Dropout(p=0.3)

    def forward(self, input):
        x = self.stem_conv(input)
        [co, ci, h, w] = self.stem_conv.weight.shape
        sobel_kernel = torch.FloatTensor(self.sobel_plane).expand(ci, ci, 3, 3).cuda()
        weight = nn.Parameter(data=sobel_kernel, requires_grad=False)
        gradient_kernel = F.conv2d(input=self.stem_conv.weight, weight=weight, stride=1, padding=self.padding)
        spatial_gradient = F.conv2d(input=input, weight=gradient_kernel, stride=self.stride, padding=self.padding)

        out = x + spatial_gradient
        if self.norm:
            out = self.bn(out)
        if self.act:
            out = self.ac(out)
        if self.apply_dropout:
            self.dropout = nn.Dropout(p=0.3)

        return out

    

class Generator(nn.Module):
    def __init__(self,
                 in_c: int,
                 out_c: int
                 ):
        super(Generator, self).__init__()

        channels = [16, 64, 94, 128]

        self.stem_conv = nn.Sequential(
          ConvBNAct(input_c=in_c * 2, output_c=channels[1]),
          ConvBNAct(input_c=channels[1], output_c=channels[2]),
        )

        # Block 1
        self.Block1 = nn.Sequential(
          ConvBNAct(input_c=channels[2], output_c=channels[3]),
          ConvBNAct(input_c=channels[3], output_c=channels[2]),
          Downsample(input_c=channels[2], output_c=channels[2]),
        )

        # Block 2
        self.Block2 = nn.Sequential(
          ConvBNAct(input_c=channels[2], output_c=channels[3]),
          ConvBNAct(input_c=channels[3], output_c=channels[2]),
          Downsample(input_c=channels[2], output_c=channels[2]),
        )

        # Block 3
        self.Block3 = nn.Sequential(
          ConvBNAct(input_c=channels[2], output_c=channels[3]),
          ConvBNAct(input_c=channels[3], output_c=channels[2]),
          Downsample(input_c=channels[2], output_c=channels[2]),
        )

        # Decoder
        self.decoder1 = Upsample(input_c=channels[2], output_c=channels[1])
        self.decoder2 = Upsample(input_c=channels[1]+channels[2], output_c=channels[1])
        self.decoder3 = Upsample(input_c=channels[1]+channels[2], output_c=channels[1])
        self.ac1 = nn.Sequential(
          ConvBNAct(input_c=channels[1], output_c=channels[0]),
          ConvBNAct(input_c=channels[0], output_c=6, act=False, norm=False),
          nn.Tanh(),
        )
        self.ac2 = nn.Sequential(
          ConvBNAct(input_c=channels[1], output_c=channels[0]),
          ConvBNAct(input_c=channels[0], output_c=out_c, act=False, norm=False),
          nn.Tanh(),
        )
        self.ac3 = nn.Sequential(
          ConvBNAct(input_c=channels[1], output_c=channels[0]),
          ConvBNAct(input_c=channels[0], output_c=out_c, act=False, norm=False),
          nn.Tanh(),
        )

        # ESR
        self.ESR = nn.Sequential(
          residual_gradient_conv(input_c=channels[2] * 3, output_c=channels[2], apply_dropout=True),
          residual_gradient_conv(input_c=channels[2], output_c=channels[1], apply_dropout=True),
          Downsample(input_c=channels[1], output_c=channels[1]),
          residual_gradient_conv(input_c=channels[1], output_c=1, act=False, norm=False),
        )

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = torch.cat((x, rgb_to_yuv(x)), dim=1)
        out = self.stem_conv(x)

        # encoder
        out1 = self.Block1(out)
        out2 = self.Block2(out1)
        out3 = self.Block3(out2)

        # decoder
        u1 = self.decoder1(out3)
        u2 = self.decoder2(torch.cat((u1, out2), dim=1))
        u3 = self.decoder3(torch.cat((u2, out1), dim=1))
        n1 = self.ac1(u1)
        n2 = self.ac2(u2)
        n3 = self.ac2(u3)

        # disentangling the traces
        s = torch.mean(n1[:, 3:6, :, :], dim=[2, 3], keepdim=True)
        b = torch.mean(n1[:, :3, :, :], dim=[2, 3], keepdim=True)
        C = self.avg_pool(n2)
        T = n3

        map1 = F.interpolate(out1, [32, 32])
        map2 = F.interpolate(out2, [32, 32])
        map3 = F.interpolate(out3, [32, 32])
        map = torch.cat((map1, map2, map3), dim=1)
        map_out = self.ESR(map)

        return map_out, s, b, C, T


class Discrinator(nn.Module):
    def __init__(self,
                 in_c: int,
                 out_c: int
                 ):
        super(Discrinator, self).__init__()

        channels = [16, 32, 64, 96]

        self.stem_conv = ConvBNAct(input_c=in_c*2, output_c=channels[1])

        # Block 1
        self.Block1 = nn.Sequential(
            ConvBNAct(input_c=channels[1], output_c=channels[1]),
            Downsample(input_c=channels[1], output_c=channels[2]),
        )

        # Block 2
        self.Block2 = nn.Sequential(
            ConvBNAct(input_c=channels[2], output_c=channels[2]),
            Downsample(input_c=channels[2], output_c=channels[3]),
        )

        # Block 3
        self.Block3 = nn.Sequential(
            ConvBNAct(input_c=channels[3], output_c=channels[3]),
            Downsample(input_c=channels[3], output_c=channels[3]),
        )

        # Block 4
        self.Block4 = ConvBNAct(input_c=channels[3], output_c=channels[3])
        self.Block4l = ConvBNAct(input_c=channels[3], output_c=out_c // 2, act=False, norm=False)
        self.Block4s = ConvBNAct(input_c=channels[3], output_c=out_c // 2, act=False, norm=False)


    def forward(self, x):
        x = torch.cat((x, rgb_to_yuv(x)), dim=1)
        out = self.stem_conv(x)

        out1 = self.Block1(out)
        out2 = self.Block2(out1)
        out3 = self.Block3(out2)
        out4 = self.Block4(out3)

        return self.Block4l(out4), self.Block4s(out4)


class Discrinator_s(nn.Module):
    def __init__(self,
                 in_c: int,
                 out_c: int
                 ):
        super(Discrinator_s, self).__init__()

        channels = [16, 32, 64, 96]

        self.stem_conv = ConvBNAct(input_c=in_c*2, output_c=channels[1])

        # Block 1
        self.Block1 = Downsample(input_c=channels[1], output_c=channels[2])

        # Block 2
        self.Block2 = Downsample(input_c=channels[2], output_c=channels[3])

        # Block 3
        self.Block3 = Downsample(input_c=channels[3], output_c=channels[3])

        # Block 4
        self.Block4 = ConvBNAct(input_c=channels[3], output_c=channels[3])
        self.Block4l = ConvBNAct(input_c=channels[3], output_c=out_c // 2, act=False, norm=False)
        self.Block4s = ConvBNAct (input_c=channels[3], output_c=out_c // 2, act=False, norm=False)

    def forward(self, x):
        x = torch.cat((x, rgb_to_yuv(x)), dim=1)
        out = self.stem_conv(x)

        out1 = self.Block1(out)
        out2 = self.Block2(out1)
        out3 = self.Block3(out2)
        out4 = self.Block4(out3)

        return self.Block4l(out4), self.Block4s(out4)



