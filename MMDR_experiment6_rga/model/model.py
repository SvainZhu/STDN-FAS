import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .CBAM import CBAM


torch.set_printoptions(threshold=np.inf)

class RGC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):

        super(RGC, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out_normal = self.conv(x)
        kernel_diff = self.conv.weight.sum(2).sum(2)
        kernel_diff = kernel_diff[:, :, None, None]
        out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0,
                            groups=self.conv.groups)

        return self.bn(out_normal - out_diff)

class GRL(nn.Module):

    def __init__(self, max_iter):
        super(GRL, self).__init__()
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = max_iter

    def forward(self, input):
        self.iter_num += 1
        return input * 1.0

    def backward(self, gradOutput):
        coeff = np.float(2.0 * (self.high - self.low) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter))
                         - (self.high - self.low) + self.low)
        return -coeff * gradOutput


class adaIN(nn.Module):

    def __init__(self, eps=1e-5):
        super(adaIN, self).__init__()
        self.eps = eps

    def forward(self, input, gamma, beta):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        out = out_in
        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)
        return out


class ResnetAdaINBlock(nn.Module):

    def __init__(self, dim):
        super(ResnetAdaINBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = adaIN()
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = adaIN()

    def forward(self, x, gamma, beta):
        out = self.conv1(x)
        out = self.norm1(x, gamma, beta)
        out = self.relu1(x)
        out = self.conv2(x)
        out = self.norm2(x, gamma, beta)
        return x+out


# class Discriminator(nn.Module):
#     def __init__(self, max_iter):
#         super(Discriminator, self).__init__()
#         self.ad_net = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.AvgPool2d(4, 4)
#         )
#         self.grl_layer = GRL(max_iter)
#         self.fc = nn.Linear(512, 3)
#
#     def forward(self, feature):
#         adversarial_out = self.grl_layer(feature)
#         adversarial_out = self.ad_net(adversarial_out).reshape(adversarial_out.shape[0], -1)
#         adversarial_out = self.fc(adversarial_out)
#         return adversarial_out

class Discriminator(nn.Module):
    def __init__(self, max_iter):
        super(Discriminator, self).__init__()
        self.ad_net = nn.Sequential(
            RGC(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            RGC(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(4, 4)
        )
        self.grl_layer = GRL(max_iter)
        self.fc = nn.Linear(512, 3)

    def forward(self, feature):
        adversarial_out = self.grl_layer(feature)
        adversarial_out = self.ad_net(adversarial_out).reshape(adversarial_out.shape[0], -1)
        adversarial_out = self.fc(adversarial_out)
        return adversarial_out


# class SSAN_M(nn.Module):
#     def __init__(self, ada_num=2, max_iter=4000):
#         super(SSAN_M, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#         )
#
#         self.Block1 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 196, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(196),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(196, 128, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#         )
#
#         self.Block2 = nn.Sequential(
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 196, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(196),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(196, 128, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#         )
#
#         self.Block3 = nn.Sequential(
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 196, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(196),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(196, 128, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#         )
#
#         self.layer4 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True)
#         )
#
#         self.cbam1 = CBAM(gate_channels=128, kernel_size=7)
#         self.cbam2 = CBAM(gate_channels=128, kernel_size=5)
#         self.cbam3 = CBAM(gate_channels=128, kernel_size=3)
#
#         self.adaIN_layers = nn.ModuleList([ResnetAdaINBlock(256) for i in range(ada_num)])
#
#         self.conv_final = nn.Sequential(
#             nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(512)
#         )
#
#         self.gamma = nn.Linear(256, 256, bias=False)
#         self.beta = nn.Linear(256, 256, bias=False)
#
#         self.FC = nn.Sequential(
#             nn.Linear(256, 256, bias=False),
#             nn.ReLU(inplace=True)
#         )
#
#         self.ada_conv1 = nn.Sequential(
#             nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.InstanceNorm2d(128),
#             nn.ReLU(inplace=True)
#         )
#
#         self.ada_conv2 = nn.Sequential(
#             nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.InstanceNorm2d(128),
#             nn.ReLU(inplace=True)
#         )
#
#         self.ada_conv3 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.InstanceNorm2d(256)
#         )
#         self.dis = Discriminator(max_iter)
#
#         self.decoder = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear'),
#             nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#
#             nn.Upsample(scale_factor=2, mode='bilinear'),
#             nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#
#             nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.ReLU(inplace=True)
#         )
#
#     def cal_gamma_beta(self, x1):
#         x1 = self.conv1(x1)
#         x1_1 = self.Block1(x1)
#         # x1_1 = self.cbam1(x1_1)
#         x1_2 = self.Block2(x1_1)
#         # x1_2 = self.cbam2(x1_2)
#         x1_3 = self.Block3(x1_2)
#         # x1_3 = self.cbam3(x1_3)
#
#         x1_4 = self.layer4(x1_3)
#
#         x1_add = x1_1
#         x1_add = self.ada_conv1(x1_add) + x1_2
#         x1_add = self.ada_conv2(x1_add) + x1_3
#         x1_add = self.ada_conv3(x1_add)
#
#         gmp = torch.nn.functional.adaptive_max_pool2d(x1_add, 1)
#         gmp_ = self.FC(gmp.view(gmp.shape[0], -1))
#         gamma, beta = self.gamma(gmp_), self.beta(gmp_)
#
#         domain_invariant = x1_4
#         return x1_4, gamma, beta, domain_invariant
#
#     def forward(self, input1, input2):
#         x1, gamma1, beta1, domain_invariant = self.cal_gamma_beta(input1)
#         x2, gamma2, beta2, _ = self.cal_gamma_beta(input2)
#
#         fea_x1_x1 = x1
#         for i in range(len(self.adaIN_layers)):
#             fea_x1_x1 = self.adaIN_layers[i](fea_x1_x1, gamma1, beta1)
#         fea_x1_x1 = self.conv_final(fea_x1_x1)
#         cls_x1_x1 = self.decoder(fea_x1_x1)
#
#         fea_x1_x1 = torch.nn.functional.adaptive_avg_pool2d(fea_x1_x1, 1)
#         fea_x1_x1 = fea_x1_x1.reshape(fea_x1_x1.shape[0], -1)
#
#         fea_x1_x2 = x1
#         for i in range(len(self.adaIN_layers)):
#             fea_x1_x2 = self.adaIN_layers[i](fea_x1_x2, gamma2, beta2)
#         fea_x1_x2 = self.conv_final(fea_x1_x2)
#         fea_x1_x2 = torch.nn.functional.adaptive_avg_pool2d(fea_x1_x2, 1)
#         fea_x1_x2 = fea_x1_x2.reshape(fea_x1_x2.shape[0], -1)
#
#         dis_invariant = self.dis(domain_invariant).reshape(domain_invariant.shape[0], -1)
#         return cls_x1_x1[:, 0, :, :], fea_x1_x1, fea_x1_x2, dis_invariant


class SSAN_R(nn.Module):
    def __init__(self, ada_num=2, max_iter=4000):
        super(SSAN_R, self).__init__()
        self.conv1 = nn.Sequential(
            RGC(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.Block1 = nn.Sequential(
            RGC(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            RGC(128, 196, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(196),
            nn.ReLU(inplace=True),
            RGC(196, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.Block2 = nn.Sequential(
            RGC(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            RGC(128, 196, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(196),
            nn.ReLU(inplace=True),
            RGC(196, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.Block3 = nn.Sequential(
            RGC(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            RGC(128, 196, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(196),
            nn.ReLU(inplace=True),
            RGC(196, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.adaIN_layers = nn.ModuleList([ResnetAdaINBlock(256) for i in range(ada_num)])

        self.conv_final = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512)
        )

        self.gamma = nn.Linear(256, 256, bias=False)
        self.beta = nn.Linear(256, 256, bias=False)

        self.FC = nn.Sequential(
            nn.Linear(256, 256, bias=False),
            nn.ReLU(inplace=True)
        )

        self.ada_conv1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.ada_conv2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.ada_conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256)
        )
        self.dis = Discriminator(max_iter)

        self.cls = nn.Linear(512, 2, bias=True)

    def cal_gamma_beta(self, x1):
        x1 = self.conv1(x1)
        x1_1 = self.Block1(x1)
        x1_2 = self.Block2(x1_1)
        x1_3 = self.Block3(x1_2)

        x1_4 = self.layer4(x1_3)

        x1_add = x1_1
        x1_add = self.ada_conv1(x1_add) + x1_2
        x1_add = self.ada_conv2(x1_add) + x1_3
        x1_add = self.ada_conv3(x1_add)

        gmp = torch.nn.functional.adaptive_max_pool2d(x1_add, 1)
        gmp_ = self.FC(gmp.view(gmp.shape[0], -1))
        gamma, beta = self.gamma(gmp_), self.beta(gmp_)

        domain_invariant = x1_4
        return x1_4, gamma, beta, domain_invariant

    def forward(self, input1, input2):
        x1, gamma1, beta1, domain_invariant = self.cal_gamma_beta(input1)
        x2, gamma2, beta2, _ = self.cal_gamma_beta(input2)

        fea_x1_x1 = x1
        for i in range(len(self.adaIN_layers)):
            fea_x1_x1 = self.adaIN_layers[i](fea_x1_x1, gamma1, beta1)
        fea_x1_x1 = self.conv_final(fea_x1_x1)
        cls_x1_x1 = self.cls(fea_x1_x1)

        fea_x1_x1 = torch.nn.functional.adaptive_avg_pool2d(fea_x1_x1, 1)
        fea_x1_x1 = fea_x1_x1.reshape(fea_x1_x1.shape[0], -1)

        fea_x1_x2 = x1
        for i in range(len(self.adaIN_layers)):
            fea_x1_x2 = self.adaIN_layers[i](fea_x1_x2, gamma2, beta2)
        fea_x1_x2 = self.conv_final(fea_x1_x2)
        fea_x1_x2 = torch.nn.functional.adaptive_avg_pool2d(fea_x1_x2, 1)
        fea_x1_x2 = fea_x1_x2.reshape(fea_x1_x2.shape[0], -1)

        dis_invariant = self.dis(domain_invariant).reshape(domain_invariant.shape[0], -1)
        return cls_x1_x1, fea_x1_x1, fea_x1_x2, dis_invariant

class DME(nn.Module):
    def __init__(self,
                 input_c,
                 output_c,
                 n_layer=3):
        super(DME, self).__init__()
        self.n_layer = n_layer

        kernel_szs = [7, 5, 3]
        gate_channels = [512, 128, 64, output_c]
        self.cbam = []
        self.depth_map = []
        for i in range(self.n_layer):
            self.cbam.append(CBAM(gate_channels=gate_channels[i], kernel_size=kernel_szs[i]))
            if i < 2:
                self.depth_map.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear'),
                    RGC(gate_channels[i], gate_channels[i+1], kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(gate_channels[i+1]),
                    nn.ReLU(inplace=True),
                    )
                )
            else:
                self.depth_map.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear'),
                    RGC(gate_channels[i], gate_channels[i + 1], kernel_size=3, stride=2, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    )
                )

    def forward(self, x):
        out = x
        for i in range(self.n_layer):
            out_cbam = self.cbam[i](out)
            out = self.depth_map[i](out_cbam)
        return out


class SSAN_M(nn.Module):
    def __init__(self, ada_num=2, max_iter=4000):
        super(SSAN_M, self).__init__()
        self.conv1 = nn.Sequential(
            RGC(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.Block1 = nn.Sequential(
            RGC(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            RGC(128, 196, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(196),
            nn.ReLU(inplace=True),
            RGC(196, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.Block2 = nn.Sequential(
            RGC(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            RGC(128, 196, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(196),
            nn.ReLU(inplace=True),
            RGC(196, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.Block3 = nn.Sequential(
            RGC(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            RGC(128, 196, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(196),
            nn.ReLU(inplace=True),
            RGC(196, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.adaIN_layers = nn.ModuleList([ResnetAdaINBlock(256) for i in range(ada_num)])

        self.conv_final = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512)
        )

        self.gamma = nn.Linear(256, 256, bias=False)
        self.beta = nn.Linear(256, 256, bias=False)

        self.FC = nn.Sequential(
            nn.Linear(256, 256, bias=False),
            nn.ReLU(inplace=True)
        )

        self.ada_conv1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.ada_conv2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.ada_conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256)
        )
        self.dis = Discriminator(max_iter)
        self.map = DME(512, 1, 3)

    def cal_gamma_beta(self, x1):
        x1 = self.conv1(x1)
        x1_1 = self.Block1(x1)
        x1_2 = self.Block2(x1_1)
        x1_3 = self.Block3(x1_2)

        x1_4 = self.layer4(x1_3)

        x1_add = x1_1
        x1_add = self.ada_conv1(x1_add) + x1_2
        x1_add = self.ada_conv2(x1_add) + x1_3
        x1_add = self.ada_conv3(x1_add)

        gmp = torch.nn.functional.adaptive_max_pool2d(x1_add, 1)
        gmp_ = self.FC(gmp.view(gmp.shape[0], -1))
        gamma, beta = self.gamma(gmp_), self.beta(gmp_)

        domain_invariant = x1_4
        return x1_4, gamma, beta, domain_invariant

    def forward(self, input1, input2):
        x1, gamma1, beta1, domain_invariant = self.cal_gamma_beta(input1)   # domain: 16*16*256, x1: 16*16*256
        x2, gamma2, beta2, _ = self.cal_gamma_beta(input2)

        fea_x1_x1 = x1
        for i in range(len(self.adaIN_layers)):
            fea_x1_x1 = self.adaIN_layers[i](fea_x1_x1, gamma1, beta1)
        fea_x1_x1 = self.conv_final(fea_x1_x1)  # 8*8*512
        depth_map = self.map(fea_x1_x1).squeeze(1)   # 32*32*1

        fea_x1_x1 = torch.nn.functional.adaptive_avg_pool2d(fea_x1_x1, 1)       # 1*1*512
        fea_x1_x1 = fea_x1_x1.reshape(fea_x1_x1.shape[0], -1)                   # 512

        fea_x1_x2 = x1
        for i in range(len(self.adaIN_layers)):
            fea_x1_x2 = self.adaIN_layers[i](fea_x1_x2, gamma2, beta2)
        fea_x1_x2 = self.conv_final(fea_x1_x2)

        fea_x1_x2 = torch.nn.functional.adaptive_avg_pool2d(fea_x1_x2, 1)
        fea_x1_x2 = fea_x1_x2.reshape(fea_x1_x2.shape[0], -1)

        dis_invariant = self.dis(domain_invariant).reshape(domain_invariant.shape[0], -1)
        return depth_map, fea_x1_x1, fea_x1_x2, dis_invariant

def get_model(name, max_iter):
    model = SSAN_M(max_iter=max_iter)
    return model