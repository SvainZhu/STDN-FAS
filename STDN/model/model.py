from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import rgb_to_yuv, rgb_to_hsv, ConvBNAct, Upsample, Downsample
import cv2
import numpy as np

class Generator(nn.Module):
    def __init__(self,
                 in_c: int,
                 out_c: int
                 ):
        super(Generator, self).__init__()

        channels = [16, 64, 94, 128]

        self.stem_conv = nn.Sequential(
          ConvBNAct(input_c=in_c * 3, output_c=channels[1]),
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
          ConvBNAct(input_c=channels[2] * 3, output_c=channels[2], apply_dropout=True),
          ConvBNAct(input_c=channels[2], output_c=channels[1], apply_dropout=True),
          # Downsample(input_c=channels[1], output_c=channels[1]),
          ConvBNAct(input_c=channels[1], output_c=1, act=False, norm=False),
        )

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # x = torch.cat((x, rgb_to_yuv(x)), dim=1)
        x = decompose_image(x)
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

        self.stem_conv = ConvBNAct(input_c=in_c*3, output_c=channels[1])

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
        # x = torch.cat((x, rgb_to_yuv(x)), dim=1)

        x = decompose_image(x)
        x = self.stem_conv(x)
        out1 = self.Block1(x)
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

        self.stem_conv = ConvBNAct(input_c=in_c*3, output_c=channels[1])

        # Block 1
        self.Block1 = Downsample(input_c=channels[1], output_c=channels[2])

        # Block 2
        self.Block2 = Downsample(input_c=channels[2], output_c=channels[3])

        # Block 3
        self.Block3 = Downsample(input_c=channels[3], output_c=channels[3])

        # Block 4
        self.Block4 = ConvBNAct(input_c=channels[3], output_c=channels[3])
        self.Block4l = ConvBNAct(input_c=channels[3], output_c=out_c // 2, act=False, norm=False)
        self.Block4s = ConvBNAct(input_c=channels[3], output_c=out_c // 2, act=False, norm=False)

    def forward(self, x):
        # x = torch.cat((x, rgb_to_yuv(x)), dim=1)

        x = decompose_image(x)

        x = self.stem_conv(x)
        out1 = self.Block1(x)
        out2 = self.Block2(out1)
        out3 = self.Block3(out2)
        out4 = self.Block4(out3)

        return self.Block4l(out4), self.Block4s(out4)


def decompose_image(x):
    b, c, h, w = x.size()
    images = []
    for i in range(b):
        image = x[i,...].permute(1, 2, 0).data.cpu().numpy()
        image_B = cv2.blur(image, (8, 8))
        image_C = cv2.blur(image, (2, 2)) - image_B
        image_T = image - cv2.blur(image, (2, 2))
        image = np.concatenate((image_B, image_C * 15, image_T * 30), axis=2)
        images += [np.expand_dims(image, axis=0)]
    images = np.concatenate(images, axis=0)
    images = torch.from_numpy(images).float().permute(0, 3, 1, 2).cuda()
    return images

