import numpy as np
from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
try:
    from itertools import izip as zip
except ImportError:     # detect the 3.x series
    pass


##################################################################################
# Generator
##################################################################################

class Generator(nn.Module):
    # content-style feature generator architecture
    def __init__(self,
                 input_c,
                 output_c,
                 feature_c=64,
                 n_downsample=3,
                 n_blocks=3,
                 norm='bn',
                 act='prelu',
                 pad_type='zero'):
        super(Generator, self).__init__()

        # content-style encoder
        self.enc = Encoder(input_c=input_c, output_c=feature_c, n_downsample=n_downsample, n_blocks=n_blocks, norm=norm, act=act, pad_type=pad_type)

        # content-style decoder
        self.dec = Decoder(input_c=feature_c, output_c=output_c, n_upsample=n_downsample, n_blocks= n_blocks, norm=norm, act=act, pad_type=pad_type)

    def encode(self, images):
        return self.enc(images)

    def decode(self, content_map, style_maps):
        return self.dec(content_map, style_maps)

    def forward(self, images):
        # reconstruct an image
        content_map, style_maps = self.enc(images)
        images_recon = self.decode(content_map, style_maps)
        return images_recon

##################################################################################
# Discriminator
##################################################################################

class MultiScaleDis(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, input_c, output_c, num_scales=3, num_layer=3, norm='bn', act='prelu', pad_type='zero'):
        super(MultiScaleDis, self).__init__()
        self.input_c = input_c
        self.output_c = output_c
        self.num_scales = num_scales
        self.num_layer = num_layer
        self.norm = norm
        self.act = act
        self.pad_type = pad_type
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.cnns = nn.ModuleList()
        self.channels = [32, 64, 96, 96]
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())

    def _make_net(self, last_net=False):
        cnn_x = [ConvNormAct(self.input_c, self.channels[0], kernel_size=3, stride=1, padding=1, norm=self.norm, act=self.act, pad_type=self.pad_type)]
        for i in range(self.num_layer):
            cnn_x += [ConvNormAct(self.channels[i], self.channels[i], kernel_size=3, stride=1, padding=1, norm=self.norm, act=self.act, pad_type=self.pad_type),
                      ConvNormAct(self.channels[i], self.channels[i+1], kernel_size=3, stride=2, padding=1, norm=self.norm, act=self.act, pad_type=self.pad_type)]
        cnn_x += [ConvNormAct(self.channels[-1], self.output_c, kernel_size=3, stride=1, padding=0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs


##################################################################################
# Feature Map Estimator
##################################################################################

class FeatureEstimator(nn.Module):
    def __init__(self,
                 input_c,
                 output_c,
                 n_layer=3,
                 norm='bn',
                 act='prelu',
                 pad_type='zero'):
        super(FeatureEstimator, self).__init__()
        self.norm = norm
        self.act = act
        self.pad_type = pad_type

        channels = [input_c, 64, output_c]
        self.est = nn.Sequential(
          ConvNormAct(input_c=channels[0]*3, output_c=channels[0], norm=norm, act=act, pad_type=pad_type, apply_dropout=True),
          ConvNormAct(input_c=channels[0], output_c=channels[1], norm=norm, act=act, pad_type=pad_type, apply_dropout=True),
          ConvNormAct(input_c=channels[1], output_c=channels[1], stride=2, norm=norm, act=act, pad_type=pad_type),
          ConvNormAct(input_c=channels[1], output_c=channels[2], norm='none', act='none', pad_type=pad_type),
        )

    def forward(self, x):
        maps = []
        for i in range(3):
            maps += [F.interpolate(x[i], [32, 32])]
        map = torch.cat(maps, dim=1)
        return self.est(map)

    def _make_block(self, output_cs, n_layer):
        block_x = []
        for i in range(n_layer):
            if self.layer_type == 'conv':
                block_x += [ConvNormAct(output_cs[i], output_cs[i+1], kernel_size=3, stride=1, padding=1, norm=self.norm,
                                  act=self.act, pad_type=self.pad_type)]
            elif self.layer_type == 'cdconv':
                block_x += [CDConvNormAct(output_cs[i], output_cs[i+1], kernel_size=3, stride=1, padding=1, norm=self.norm,
                                  act=self.act, pad_type=self.pad_type)]
            elif self.layer_type == 'rgconv':
                block_x += [RGConvNormAct(output_cs[i], output_cs[i+1], kernel_size=3, stride=1, padding=1, norm=self.norm,
                                  act=self.act, pad_type=self.pad_type)]
            else:
                assert 0, "Unsupported Estimator type: {}".format(self.layer_type)
        if n_layer > 1:
            block_x += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        block_x = nn.Sequential(*block_x)
        return block_x.cuda()

##################################################################################
# Encoder and Decoders
##################################################################################

class Encoder(nn.Module):
    def __init__(self,
                 input_c,
                 output_c,
                 n_downsample=3,
                 n_blocks=3,
                 norm='bn',
                 act='prelu',
                 pad_type='zero'):
        super(Encoder, self).__init__()
        self.output_c = output_c
        self.norm = norm
        self.act = act
        self.pad_type = pad_type
        self.n_downsample = n_downsample
        channels = [output_c, int(output_c * 1.5), output_c * 2]

        # content encoder
        self.content_encoder = []
        output_cs = [input_c*2, channels[0], channels[1], channels[0]]
        for i in range(n_blocks):
            self.content_encoder += [ConvNormAct(output_cs[i], output_cs[i + 1], 3, 1, 1, norm=norm, act=act, pad_type=pad_type)]
        self.content_encoder = nn.Sequential(*self.content_encoder)

        # style encoders
        self.style_encoders = [ConvNormAct(channels[0], channels[1], 3, 1, 1, norm=norm, act=act, pad_type=pad_type).cuda()]
        layer_channels = [channels[1], channels[2], channels[1]]
        for i in range(n_downsample):
            self.style_encoders += [self._make_block(layer_channels, 2, 'downsample')]

    def forward(self, x):
        out = torch.cat((x, rgb_to_yuv(x)), dim=1)

        # content encode
        out = self.content_encoder(out)
        content_map = out

        # style encode
        out = self.style_encoders[0](out)
        style_maps = []
        for i in range(1, self.n_downsample+1):
            out = self.style_encoders[i](out)
            style_maps += [out]
        return content_map, style_maps

    def _make_block(self, output_cs, n_layer, last_layer='downsample'):
        block_x = []
        if last_layer == 'act':
            for i in range(n_layer-1):
                block_x += [ConvNormAct(output_cs[i], output_cs[i + 1], kernel_size=3, padding=1, norm=self.norm,
                                act=self.act, pad_type=self.pad_type)]
            block_x += [ConvNormAct(output_cs[-2], output_cs[-1], kernel_size=3, stride=1, padding=1, norm='none',
                                    act='none', pad_type=self.pad_type)]
            block_x += nn.Tanh()
        else:
            for i in range(n_layer):
                block_x += [
                    ConvNormAct(output_cs[i], output_cs[i + 1], kernel_size=3, stride=1, padding=1, norm=self.norm,
                                act=self.act, pad_type=self.pad_type)]
            if last_layer == 'downsample':
                block_x += [ConvNormAct(output_cs[-1], output_cs[-1], kernel_size=3, stride=2, padding=1, norm=self.norm,
                                    act=self.act, pad_type=self.pad_type)]

        block_x = nn.Sequential(*block_x)
        return block_x.cuda()

class Decoder(nn.Module):
    def __init__(self,
                 input_c,
                 output_c,
                 n_upsample,
                 n_blocks,
                 norm='bn',
                 act='prelu',
                 pad_type='zero'):
        super(Decoder, self).__init__()
        self.n_downsample = n_upsample
        self.norm = norm
        self.act = act
        self.pad_type = pad_type

        # style decoder
        self.style_decoder = []
        channels = [input_c // 4, input_c, int(input_c * 1.5)]
        layer_channels = [channels[2], channels[1], channels[1] + channels[2], channels[1], channels[1] + channels[2],
                          channels[1]]
        for i in range(n_upsample):
            self.style_decoder += [Upsample(layer_channels[2 * i], layer_channels[2 * i + 1], stride=2, norm=norm, act=act,
                                     pad_type=pad_type)]

        # # style act
        # self.style_act = []
        # layer_channels = [channels[1], channels[0], output_c]
        # for i in range(n_upsample):
        #     self.style_act += [self._make_block(layer_channels, 2, 'act')]
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # content decoder
        self.content_decoder = []
        # blocks
        for i in range(n_blocks):
            self.content_decoder += [ConvNormAct(input_c=input_c, output_c=input_c // 2, kernel_size=3, stride=1, padding=1, norm=norm, act=act, pad_type=pad_type)]
            input_c //= 2
        self.content_decoder += [ConvNormAct(input_c=input_c, output_c=output_c, kernel_size=3, stride=1, padding=1, norm=norm, act=act, pad_type=pad_type)]
        self.content_decoder = nn.Sequential(*self.content_decoder)

    def _make_block(self, output_cs, n_layer, last_layer='downsample'):
        block_x = []
        if last_layer == 'act':
            for i in range(n_layer-1):
                block_x += [ConvNormAct(output_cs[i], output_cs[i + 1], kernel_size=3, padding=1, norm=self.norm,
                                act=self.act, pad_type=self.pad_type)]
            block_x += [ConvNormAct(output_cs[-2], output_cs[-1], kernel_size=3, stride=1, padding=1, norm='none',
                                    act='none', pad_type=self.pad_type)]
            block_x += [nn.Tanh()]
        else:
            for i in range(n_layer):
                block_x += [
                    ConvNormAct(output_cs[i], output_cs[i + 1], kernel_size=3, stride=1, padding=1, norm=self.norm,
                                act=self.act, pad_type=self.pad_type)]
            if last_layer == 'downsample':
                block_x += [ConvNormAct(output_cs[-1], output_cs[-1], kernel_size=3, stride=2, padding=1, norm=self.norm,
                                    act=self.act, pad_type=self.pad_type)]

        block_x = nn.Sequential(*block_x)
        return block_x.cuda()

    def forward(self, content_x, style_x):
        # style decode
        style_decode = []
        for i in range(self.n_downsample):
            style_in = style_x[self.n_downsample - i - 1] if i == 0 else torch.cat(
                (out, style_x[self.n_downsample - i - 1]), dim=1)
            out = self.style_decoder[i](style_in)
            style_decode += [out]

        # # style act
        # style_act = []
        # for i in range(self.n_downsample):
        #     style_act += [self.style_act[i](style_decode[i])]

        # style feature
        style_out = content_x * torch.mean(style_decode[0], dim=[2, 3], keepdim=True)
        style_out += F.interpolate(self.avg_pool(style_decode[1]), (256, 256))
        style_out += style_decode[2]

        return self.content_decoder(content_x+style_out)

##################################################################################
# Sequential Models
##################################################################################
class ResBlocks(nn.Module):
    def __init__(self,
                 channels,
                 num_blocks,
                 norm='in',
                 act='relu',
                 pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(channels=channels, norm=norm, act=act, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class MLP(nn.Module):
    def __init__(self,
                 input_c,
                 output_c,
                 dim,
                 n_block,
                 norm='none',
                 act='relu'):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_c, dim, norm=norm, act=act)]
        for i in range(n_block - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, act=act)]
        self.model += [LinearBlock(dim, output_c, norm='none', act='none')] # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

##################################################################################
# Basic Blocks
##################################################################################
class ResBlock(nn.Module):
    def __init__(self,
                 channels,
                 norm='in',
                 act='relu',
                 pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [ConvNormAct(input_c=channels, output_c=channels, kernel_size=3, stride=1, padding=1, norm=norm, act=act, pad_type=pad_type)]
        model += [ConvNormAct(input_c=channels, output_c=channels, kernel_size=3, stride=1, padding=1, norm=norm, act='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class ConvNormAct(nn.Module):
    def __init__(self,
                 input_c,
                 output_c,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 norm='none',
                 act='relu',
                 pad_type='zero',
                 apply_dropout=False):
        super(ConvNormAct, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_c = output_c
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_c)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_c)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_c)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_c)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'lrelu':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        elif act == 'prelu':
            self.act = nn.PReLU()
        elif act == 'selu':
            self.act = nn.SELU(inplace=True)
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'none':
            self.act = None
        else:
            assert 0, "Unsupported activation: {}".format(act)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_c, output_c, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(in_channels=input_c, out_channels=output_c, kernel_size=kernel_size, stride=stride, bias=self.use_bias)

        if apply_dropout:
            self.dropout = nn.Dropout(p=0.3)
        else:
            self.dropout = None

    def forward(self, x):
        out = self.conv(self.pad(x))
        if self.norm:
            out = self.norm(out)
        if self.act:
            out = self.act(out)
        if self.dropout:
            out = self.dropout(out)
        return out

class LinearBlock(nn.Module):
    def __init__(self,
                 input_c,
                 output_c,
                 norm='none',
                 act='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        if norm == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_c, output_c, bias=use_bias))
        else:
            self.fc = nn.Linear(input_c, output_c, bias=use_bias)

        # initialize normalization
        norm_dim = output_c
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'lrelu':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        elif act == 'prelu':
            self.act = nn.PReLU()
        elif act == 'selu':
            self.act = nn.SELU(inplace=True)
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'none':
            self.act = None
        else:
            assert 0, "Unsupported activation: {}".format(act)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.act:
            out = self.act(out)
        return out


class RGConvNormAct(nn.Module):
    def __init__(self,
                 input_c,
                 output_c,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 norm='none',
                 act='relu',
                 pad_type='zero',
                 apply_dropout=False):
        super(RGConvNormAct, self).__init__()

        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_c = output_c
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_c)
        elif norm == 'in':
            # self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_c)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_c)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_c)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'lrelu':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        elif act == 'prelu':
            self.act = nn.PReLU()
        elif act == 'selu':
            self.act = nn.SELU(inplace=True)
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'none':
            self.act = None
        else:
            assert 0, "Unsupported activation: {}".format(act)

        # initialize convolution
        if norm == 'sn':
            self.stem_conv = SpectralNorm(nn.Conv2d(input_c, output_c, kernel_size, stride, padding, bias=self.use_bias))
            self.conv = SpectralNorm(nn.Conv2d(input_c, output_c, kernel_size, stride, padding, bias=self.use_bias))
        else:
            self.stem_conv = nn.Conv2d(input_c, output_c, kernel_size, stride, padding, bias=self.use_bias)
            self.conv = nn.Conv2d(input_c, output_c, 3, 1, padding, bias=self.use_bias)

        if apply_dropout:
            self.dropout = nn.Dropout(p=0.3)

        self.output_c = output_c
        self.stride = stride
        self.padding = padding
        self.grad_kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])

    def forward(self, x):
        out = self.stem_conv(x)
        [co, ci, h, w] = self.stem_conv.weight.shape
        sobel_kernel = torch.FloatTensor(self.grad_kernel).expand(ci, ci, 3, 3).cuda()
        weight = nn.Parameter(data=sobel_kernel, requires_grad=False)
        gradient_kernel = F.conv2d(input=self.stem_conv.weight, weight=weight, stride=1, padding=self.padding)
        spatial_gradient = F.conv2d(input=x, weight=gradient_kernel, stride=self.stride, padding=self.padding)
        out = out + spatial_gradient

        if self.norm:
            out = self.norm(out)
        if self.act:
            out = self.act(out)
        if self.dropout:
            out = self.dropout(out)
        return out

class CDConvNormAct(nn.Module):
    def __init__(self,
                 input_c,
                 output_c,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 norm='none',
                 act='relu',
                 pad_type='zero',
                 apply_dropout=False):
        super(CDConvNormAct, self).__init__()

        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_c = output_c
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_c)
        elif norm == 'in':
            # self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_c)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_c)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_c)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'lrelu':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        elif act == 'prelu':
            self.act = nn.PReLU()
        elif act == 'selu':
            self.act = nn.SELU(inplace=True)
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'none':
            self.act = None
        else:
            assert 0, "Unsupported activation: {}".format(act)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_c, output_c, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_c, output_c, kernel_size, stride, padding=1, bias=self.use_bias)

        if apply_dropout:
            self.dropout = nn.Dropout(p=0.3)


    def forward(self, x):
        out = self.conv(x)
        kernel_diff = self.conv.weight.sum(2).sum(2)
        kernel_diff = kernel_diff[:, :, None, None]
        out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0,
                            groups=self.conv.groups)
        out = out - out_diff
        if self.norm:
            out = self.norm(out)
        if self.act:
            out = self.act(out)
        if self.dropout:
            out = self.dropout(out)

        return out

class Upsample(nn.Module):
    def __init__(self,
                 input_c: int,
                 output_c: int,
                 kernel_size: int = 3,
                 stride: int = 2,
                 norm: bool = True,
                 act: bool = True,
                 pad_type='zero'
                 ):
        super(Upsample, self).__init__()

        padding = (kernel_size - 1) // 2

        # initialize convolution
        self.conv = nn.ConvTranspose2d(in_channels=input_c, out_channels=output_c, kernel_size=kernel_size,
                                       stride=stride, padding=padding, output_padding=1, bias=True).cuda()


        # initialize normalization
        norm_c = output_c
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_c).cuda()
        elif norm == 'in':
            # self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_c).cuda()
        elif norm == 'ln':
            self.norm = LayerNorm(norm_c).cuda()
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_c).cuda()
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if act == 'relu':
            self.act = nn.ReLU(inplace=True).cuda()
        elif act == 'lrelu':
            self.act = nn.LeakyReLU(0.2, inplace=True).cuda()
        elif act == 'prelu':
            self.act = nn.PReLU().cuda()
        elif act == 'selu':
            self.act = nn.SELU(inplace=True).cuda()
        elif act == 'tanh':
            self.act = nn.Tanh().cuda()
        elif act == 'none':
            self.act = None
        else:
            assert 0, "Unsupported activation: {}".format(act)

    def forward(self, x):
        out = self.conv(x)
        if self.norm:
            out = self.norm(out)
        if self.act:
            out = self.act(out)
        return out

##################################################################################
# Normalization layers
##################################################################################
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

def rgb_to_yuv(input: torch.Tensor, consts='yuv'):
    """Converts one or more images from RGB to YUV.
    Outputs a tensor of the same shape as the `input` image tensor, containing the YUV
    value of the pixels.
    The output is only well defined if the value in images are in [0,1].
    Y′CbCr is often confused with the YUV color space, and typically the terms YCbCr
    and YUV are used interchangeably, leading to some confusion. The main difference
    is that YUV is analog and YCbCr is digital: https://en.wikipedia.org/wiki/YCbCr
    Args:
      input: 2-D or higher rank. Image data to convert. Last dimension must be
        size 3. (Could add additional channels, ie, AlphaRGB = AlphaYUV)
      consts: YUV constant parameters to use. BT.601 or BT.709. Could add YCbCr
        https://en.wikipedia.org/wiki/YUV
    Returns:
      images: images tensor with the same shape as `input`.
    """

    # channels = input.shape[0]

    if consts == 'BT.709':  # HDTV YUV
        Wr = 0.2126
        Wb = 0.0722
        Wg = 1 - Wr - Wb  # 0.7152
        Uc = 0.539
        Vc = 0.635
        delta: float = 0.5  # 128 if image range in [0,255]
    elif consts == 'ycbcr':  # Alt. BT.601 from Kornia YCbCr values, from JPEG conversion
        Wr = 0.299
        Wb = 0.114
        Wg = 1 - Wr - Wb  # 0.587
        Uc = 0.564  # (b-y) #cb
        Vc = 0.713  # (r-y) #cr
        delta: float = .5  # 128 if image range in [0,255]
    elif consts == 'yuvK':  # Alt. yuv from Kornia YUV values: https://github.com/kornia/kornia/blob/master/kornia/color/yuv.py
        Wr = 0.299
        Wb = 0.114
        Wg = 1 - Wr - Wb  # 0.587
        Ur = -0.147
        Ug = -0.289
        Ub = 0.436
        Vr = 0.615
        Vg = -0.515
        Vb = -0.100
        # delta: float = 0.0
    elif consts == 'y':  # returns only Y channel, same as rgb_to_grayscale()
        # Note: torchvision uses ITU-R 601-2: Wr = 0.2989, Wg = 0.5870, Wb = 0.1140
        Wr = 0.299
        Wb = 0.114
        Wg = 1 - Wr - Wb  # 0.587
    else:  # Default to 'BT.601', SDTV YUV
        Wr = 0.299
        Wb = 0.114
        Wg = 1 - Wr - Wb  # 0.587
        Uc = 0.493  # 0.492
        Vc = 0.877
        delta: float = 0.5  # 128 if image range in [0,255]

    r: torch.Tensor = input[..., 0, :, :]
    g: torch.Tensor = input[..., 1, :, :]
    b: torch.Tensor = input[..., 2, :, :]

    if consts == 'y':
        y: torch.Tensor = Wr * r + Wg * g + Wb * b
        # (0.2989 * input[0] + 0.5870 * input[1] + 0.1140 * input[2]).to(img.dtype)
        return y
    elif consts == 'yuvK':
        y: torch.Tensor = Wr * r + Wg * g + Wb * b
        u: torch.Tensor = Ur * r + Ug * g + Ub * b
        v: torch.Tensor = Vr * r + Vg * g + Vb * b
    else:  # if consts == 'ycbcr' or consts == 'yuv' or consts == 'BT.709':
        y: torch.Tensor = Wr * r + Wg * g + Wb * b
        u: torch.Tensor = (b - y) * Uc + delta  # cb
        v: torch.Tensor = (r - y) * Vc + delta  # cr

    if consts == 'uv':  # returns only UV channels
        return torch.stack((u, v), -3)
    else:
        return torch.stack((y, u, v), -3)