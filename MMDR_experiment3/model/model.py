from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
import numpy as np
from .CBAM import CBAM

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


##################################################################################
# Encode
##################################################################################

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
        channels = [64, 128, 196]
        self.stem_conv = nn.Sequential(
            RGConvNormAct(input_c, channels[0], kernel_size=3, stride=2, padding=1, norm=norm, act=act, pad_type=pad_type, apply_dropout=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        )

        self.Block1 = nn.Sequential(
            RGConvNormAct(channels[0], channels[1], kernel_size=3, stride=1, padding=1, norm=norm, act=act, pad_type=pad_type, apply_dropout=True),
            RGConvNormAct(channels[1], channels[2], kernel_size=3, stride=1, padding=1, norm=norm, act=act, pad_type=pad_type, apply_dropout=True),
            RGConvNormAct(channels[2], channels[1], kernel_size=3, stride=1, padding=1, norm=norm, act=act,
                          pad_type=pad_type, apply_dropout=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.Block2 = nn.Sequential(
            RGConvNormAct(channels[1], channels[1], kernel_size=3, stride=1, padding=1, norm=norm, act=act,
                          pad_type=pad_type, apply_dropout=True),
            RGConvNormAct(channels[1], channels[2], kernel_size=3, stride=1, padding=1, norm=norm, act=act,
                          pad_type=pad_type, apply_dropout=True),
            RGConvNormAct(channels[2], channels[1], kernel_size=3, stride=1, padding=1, norm=norm, act=act,
                          pad_type=pad_type, apply_dropout=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.Block3 = nn.Sequential(
            RGConvNormAct(channels[1], channels[1], kernel_size=3, stride=1, padding=1, norm=norm, act=act,
                          pad_type=pad_type, apply_dropout=True),
            RGConvNormAct(channels[1], channels[2], kernel_size=3, stride=1, padding=1, norm=norm, act=act,
                          pad_type=pad_type, apply_dropout=True),
            RGConvNormAct(channels[2], channels[1], kernel_size=3, stride=1, padding=1, norm=norm, act=act,
                          pad_type=pad_type, apply_dropout=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.ConvGRU = ConvGRU()

        output_channels = [128 * 3, 128, 64, 1]
        self.Feature_Block = nn.Sequential(
            RGConvNormAct(channels[0], channels[1], kernel_size=3, stride=1, padding=1, norm=norm, act=act,
                          pad_type=pad_type, apply_dropout=True),
            RGConvNormAct(channels[1], channels[2], kernel_size=3, stride=1, padding=1, norm=norm, act=act,
                          pad_type=pad_type, apply_dropout=True),
            RGConvNormAct(channels[2], channels[3], kernel_size=3, stride=1, padding=1, norm=norm, act=act,
                          pad_type=pad_type, apply_dropout=True),
        )

        self.cbam1 = CBAM(gate_channels=128, kernel_size=7)
        self.cbam2 = CBAM(gate_channels=128, kernel_size=5)
        self.cbam3 = CBAM(gate_channels=128, kernel_size=3)
        self.downsample = nn.Upsample(size=(32, 32), mode='bilinear', align_corners=True)

    def forward(self, x):
        # content encode
        x = decompose_image(x).detach()
        out = self.content_encoder(x)
        content_map = out

        # style encode
        out = self.style_encoders[0](out)
        style_maps = []
        for i in range(1, self.n_downsample+1):
            out = self.style_encoders[i](out)
            style_maps += [out]
        return content_map, style_maps

    def _make_block(self, output_cs, n_layer):
        block_x = []
        for i in range(n_layer):
            block_x += [
                ConvNormAct(output_cs[i], output_cs[i + 1], kernel_size=3, stride=1, padding=1, norm=self.norm,
                            act=self.act, pad_type=self.pad_type)]
        block_x += [ConvNormAct(output_cs[-1], output_cs[-1], kernel_size=3, stride=2, padding=1, norm=self.norm,
                                act=self.act, pad_type=self.pad_type)]
        block_x = nn.Sequential(*block_x)
        return block_x.cuda()


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
        self.n_layer = n_layer

        kernel_szs = [7, 5, 3]
        gate_channels = [64, 96, 128]
        self.cbam = []
        for i in range(self.n_layer):
            self.cbam += [CBAM(gate_channels=gate_channels[i], kernel_size=kernel_szs[i]).cuda()]
        self.est = nn.Sequential(
          RGConvNormAct(input_c=input_c * 9, output_c=input_c * 4, padding=1, norm=norm, act=act, pad_type=pad_type, apply_dropout=True),
          RGConvNormAct(input_c=input_c * 4, output_c=input_c, padding=1, norm=norm, act=act, pad_type=pad_type, apply_dropout=True),
          RGConvNormAct(input_c=input_c, output_c=output_c, padding=1, norm='none', act='none', pad_type=pad_type),
        )


    def forward(self, x):
        maps = []
        for i in range(self.n_layer):
            # x[i] = x[i].to("cuda:0")
            out_cbam = self.cbam[i](x[i].cuda(0))
            maps += [F.interpolate(out_cbam, [32, 32])]
        map = torch.cat(maps, dim=1)
        return self.est(map)



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
        else:
            self.dropout = None

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

class ConvGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

        init.orthogonal(self.reset_gate.weight)
        init.orthogonal(self.update_gate.weight)
        init.orthogonal(self.out_gate.weight)
        init.constant(self.reset_gate.bias, 0.)
        init.constant(self.update_gate.bias, 0.)
        init.constant(self.out_gate.bias, 0.)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if torch.cuda.is_available():
                prev_state = Variable(torch.zeros(state_size)).cuda()
            else:
                prev_state = Variable(torch.zeros(state_size))

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = F.sigmoid(self.update_gate(stacked_inputs))
        reset = F.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = F.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state

class ConvGRU(nn.Module):

    def __init__(self, input_size, hidden_sizes, kernel_sizes, n_layers):
        '''
        Generates a multi-layer convolutional GRU.
        Preserves spatial dimensions across cells, only altering depth.
        Parameters
        ----------
        input_size : integer. depth dimension of input tensors.
        hidden_sizes : integer or list. depth dimensions of hidden state.
            if integer, the same hidden size is used for all cells.
        kernel_sizes : integer or list. sizes of Conv2d gate kernels.
            if integer, the same kernel size is used for all cells.
        n_layers : integer. number of chained `ConvGRUCell`.
        '''

        super(ConvGRU, self).__init__()

        self.input_size = input_size

        if type(hidden_sizes) != list:
            self.hidden_sizes = [hidden_sizes]*n_layers
        else:
            assert len(hidden_sizes) == n_layers, '`hidden_sizes` must have the same length as n_layers'
            self.hidden_sizes = hidden_sizes
        if type(kernel_sizes) != list:
            self.kernel_sizes = [kernel_sizes]*n_layers
        else:
            assert len(kernel_sizes) == n_layers, '`kernel_sizes` must have the same length as n_layers'
            self.kernel_sizes = kernel_sizes

        self.n_layers = n_layers

        cells = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i-1]

            cell = ConvGRUCell(input_dim, self.hidden_sizes[i], self.kernel_sizes[i])
            name = 'ConvGRUCell_' + str(i).zfill(2)

            setattr(self, name, cell)
            cells.append(getattr(self, name))

        self.cells = cells


    def forward(self, x, hidden=None):
        '''
        Parameters
        ----------
        x : 4D input tensor. (batch, channels, height, width).
        hidden : list of 4D hidden state representations. (batch, channels, height, width).
        Returns
        -------
        upd_hidden : 5D hidden representation. (layer, batch, channels, height, width).
        '''
        if not hidden:
            hidden = [None]*self.n_layers

        input_ = x

        upd_hidden = []

        for layer_idx in range(self.n_layers):
            cell = self.cells[layer_idx]
            cell_hidden = hidden[layer_idx]

            # pass through layer
            upd_cell_hidden = cell(input_, cell_hidden)
            upd_hidden.append(upd_cell_hidden)
            # update input_ to the last updated hidden layer for next pass
            input_ = upd_cell_hidden

        # retain tensors in list to allow different hidden sizes
        return upd_hidden

class Upsample(nn.Module):
    def __init__(self,
                 input_c: int,
                 output_c: int,
                 kernel_size: int = 3,
                 stride: int = 2,
                 output_padding: int = 1,
                 norm: bool = True,
                 act: bool = True,
                 pad_type='zero'
                 ):
        super(Upsample, self).__init__()

        padding = (kernel_size - 1) // 2

        # initialize convolution
        self.conv = nn.ConvTranspose2d(in_channels=input_c, out_channels=output_c, kernel_size=kernel_size,
                                       stride=stride, padding=padding, output_padding=output_padding, bias=True).cuda()


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

