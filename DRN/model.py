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

class AdaINGen(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self,
                 input_c,
                 output_c,
                 style_c,
                 n_downsample,
                 n_resblock,
                 mlp_c,
                 act='prelu',
                 pad_type='zero'):
        super(AdaINGen, self).__init__()

        # style encoder
        self.enc_style = StyleEncoder(input_c=input_c, output_c=output_c, style_dim=style_c, n_downsample=4 , norm='none', act=act, pad_type=pad_type)


        # content encoder
        self.enc_content = ContentEncoder(input_c=input_c, output_c=output_c, n_res=n_resblock, n_downsample=n_downsample, norm='in', act=act, pad_type=pad_type)
        self.dec = Decoder(input_c=self.enc_content.output_c, output_c=input_c, n_upsample=n_downsample, n_res= n_resblock, res_norm='adain', act=act, pad_type=pad_type)

        # MLP to generate AdaIN parameters
        self.mlp = MLP(input_c=style_c, output_c=self.get_num_adain_params(self.dec), dim=mlp_c, n_block=3, norm='none', act=act)

    def forward(self, images):
        # reconstruct an image
        content, style = self.encode(images)
        images_recon = self.decode(content, style)
        return images_recon

    def encode(self, images):
        # encode an image to its content and style codes
        style_feature = self.enc_style(images)
        content_feature = self.enc_content(images)
        return content_feature, style_feature

    def decode(self, content, style):
        # decode content and style codes to an image
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, self.dec)
        images = self.dec(content)
        return images

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params


class VAEGen(nn.Module):
    # VAE architecture
    def __init__(self,
                 input_c,
                 output_c,
                 n_downsample,
                 n_resblock,
                 act,
                 pad_type):
        super(VAEGen, self).__init__()

        # content encoder

        self.enc = ContentEncoder(input_c=input_c, output_c=output_c, n_res=n_resblock, n_downsample=n_downsample, norm='in',
                       act=act, pad_type=pad_type)
        self.dec = Decoder(input_c=input_c, output_c=self.enc.output_c, n_upsample=n_downsample, n_res= n_resblock, res_norm='adain', act=act, pad_type=pad_type)

    def forward(self, images):
        # This is a reduced VAE implementation where we assume the outputs are multivariate Gaussian distribution with mean = hiddens and std_dev = all ones.
        hiddens = self.encode(images)
        if self.training == True:
            noise = Variable(torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))
            images_recon = self.decode(hiddens + noise)
        else:
            images_recon = self.decode(hiddens)
        return images_recon, hiddens

    def encode(self, images):
        hiddens = self.enc(images)
        noise = Variable(torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))
        return hiddens, noise

    def decode(self, hiddens):
        images = self.dec(hiddens)
        return images


##################################################################################
# Discriminator
##################################################################################

class MultiScaleDis(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, input_c, output_c, num_scales, n_layer, gan_type, norm, act, pad_type):
        super(MultiScaleDis, self).__init__()
        self.input_c = input_c
        self.output_c = output_c
        self.num_scales = num_scales
        self.n_layer = n_layer
        self.gan_type = gan_type
        self.norm = norm
        self.act = act
        self.pad_type = pad_type
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cnns = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())

    def _make_net(self):
        output_c = self.output_c
        cnn_x = []
        cnn_x += [ConvNormAct(self.input_c, output_c, kernel_size=4, stride=2, padding=1, norm='none', act=self.act, pad_type=self.pad_type)]
        for i in range(self.n_layer - 1):
            cnn_x += [ConvNormAct(output_c, output_c * 2, kernel_size=4, stride=2, padding=1, norm=self.norm, act=self.act, pad_type=self.pad_type)]
            output_c *= 2
        cnn_x += [nn.Conv2d(in_channels=output_c, out_channels=1, kernel_size=1, stride=1, padding=0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        outs_fake = self.forward(input_fake)
        outs_real = self.forward(input_real)
        loss = 0

        for it, (out_fake, out_real) in enumerate(zip(outs_fake, outs_real)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out_fake - 0)**2) + torch.mean((out_real - 1)**2)
            elif self.gan_type == 'nsgan':
                all_fake = Variable(torch.zeros_like(out_fake.data).cuda(), requires_grad=False)
                all_real = Variable(torch.ones_like(out_real.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out_fake), all_fake) +
                                   F.binary_cross_entropy(F.sigmoid(out_real), all_real))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        outs_fake = self.forward(input_fake)
        loss = 0
        for it, (out_fake) in enumerate(outs_fake):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out_fake - 1)**2) # LSGAN
            elif self.gan_type == 'nsgan':
                all_fake = Variable(torch.ones_like(out_fake.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out_fake), all_fake))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss


##################################################################################
# Depth Map Estimator
##################################################################################

class DepthEstimator(nn.Module):
    def __init__(self,
                 input_c,
                 output_c,
                 n_layer=3,
                 layer_type='conv',
                 norm='none',
                 act='relu',
                 pad_type='zero'):
        super(DepthEstimator, self).__init__()
        self.layer_type = layer_type
        self.norm = norm
        self.act = act
        self.pad_type = pad_type

        self.stem_conv = self._make_block([input_c, 128], 1)

        output_cs = [128, 128, 196, 128]
        self.blocks = []
        for i in range(3):
            self.blocks += [self._make_block(output_cs, n_layer)]
        self.downsample = nn.Upsample(size=(32, 32), mode='bilinear', align_corners=True)

        last_output_cs = [[128*3, 128], [128, 64], [64, output_c]]
        self.last_blocks = []
        for i in range(3):
            self.last_blocks += [self._make_block(last_output_cs[i], 1)]

    def forward(self, x):
        outs = self.stem_conv(x)
        outs_ds = []
        for i in range(3):
            outs = self.blocks[i](outs)
            outs_ds += [self.downsample(outs)]
        maps = torch.cat(outs_ds, dim=1)

        for i in range(3):
            maps = self.last_blocks[i](maps)

        return maps

    def calc_map_loss(self, maps_est, maps_gt):
        # calculate the loss

        maps_est = maps_est.squeeze(0)
        loss = 0
        for it, (map_est, map_gt) in enumerate(zip(maps_est, maps_gt)):

            if self.layer_type == 'cdconv':
                contrast_est = self.contrast_depth_conv(map_est.unsqueeze(0))
                contrast_gt = self.contrast_depth_conv(map_gt.unsqueeze(0))
                loss = loss + F.mse_loss(contrast_est, contrast_gt)
            elif self.layer_type == 'rgconv':
                adjacent_est = self.adjacent_depth_conv(map_est)
                adjacent_gt = self.adjacent_depth_conv(map_gt)
                loss = loss + F.mse_loss(adjacent_est, adjacent_gt)
            elif self.layer_type == 'conv':
                l2_loss = nn.MSELoss()
                loss = loss + l2_loss(map_est, map_gt)
        return loss

    def contrast_depth_conv(self, input):
        ''' compute contrast depth in both of (out, label) '''
        '''
            input  32x32
            output 8x32x32
        '''

        kernel_filter_list = [
            [[1, 0, 0], [0, -1, 0], [0, 0, 0]], [[0, 1, 0], [0, -1, 0], [0, 0, 0]], [[0, 0, 1], [0, -1, 0], [0, 0, 0]],
            [[0, 0, 0], [1, -1, 0], [0, 0, 0]], [[0, 0, 0], [0, -1, 1], [0, 0, 0]],
            [[0, 0, 0], [0, -1, 0], [1, 0, 0]], [[0, 0, 0], [0, -1, 0], [0, 1, 0]], [[0, 0, 0], [0, -1, 0], [0, 0, 1]]
        ]

        kernel_filter = np.array(kernel_filter_list, np.float32)

        kernel_filter = torch.from_numpy(kernel_filter.astype(np.float32)).float().cuda()
        # weights (in_channel, out_channel, kernel, kernel)
        kernel_filter = kernel_filter.unsqueeze(dim=1)

        input = input.unsqueeze(dim=1).expand(input.shape[0], 8, input.shape[1], input.shape[2])

        contrast_depth = F.conv2d(input, weight=kernel_filter, groups=8)  # depthwise conv

        return contrast_depth

    def adjacent_depth_conv(self, input):
        ''' compute adjacent depth in both of (out, label) '''
        '''
            input  32x32
            output 8x32x32
        '''

        kernel_filter_list = [[0.5, 1, 0.5], [1, -6, 1], [0.5, 1, 0.5]]

        kernel_filter = np.array(kernel_filter_list, np.float32)

        kernel_filter = torch.from_numpy(kernel_filter.astype(np.float32)).float().cuda()
        # weights (in_channel, out_channel, kernel, kernel)
        kernel_filter = kernel_filter.unsqueeze(dim=0).unsqueeze(dim=0)

        input = input.unsqueeze(dim=1).expand(input.shape[0], 1, input.shape[1], input.shape[2])

        adjacent_depth = F.conv2d(input, weight=kernel_filter, groups=1)  # depthwise conv

        return adjacent_depth

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

class StyleEncoder(nn.Module):
    def __init__(self,
                 input_c,
                 output_c,
                 n_downsample,
                 style_dim,
                 norm,
                 act,
                 pad_type):
        super(StyleEncoder, self).__init__()
        self.model = []
        self.model += [ConvNormAct(input_c, output_c, 7, 1, 3, norm=norm, act=act, pad_type=pad_type)]
        for i in range(2):
            self.model += [ConvNormAct(output_c, 2 * output_c, 4, 2, 1, norm=norm, act=act, pad_type=pad_type)]
            output_c *= 2
        for i in range(n_downsample - 2):
            self.model += [ConvNormAct(output_c, output_c, 4, 2, 1, norm=norm, act=act, pad_type=pad_type)]
        self.model += [nn.AdaptiveAvgPool2d(1)] # global average pooling
        self.model += [nn.Conv2d(output_c, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.output_c = output_c

    def forward(self, x):
        return self.model(x)

class ContentEncoder(nn.Module):
    def __init__(self,
                 input_c,
                 output_c,
                 n_downsample,
                 n_res,
                 norm,
                 act,
                 pad_type):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [ConvNormAct(input_c, output_c, 7, 1, 3, norm=norm, act=act, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [ConvNormAct(output_c, 2 * output_c, 4, 2, 1, norm=norm, act=act, pad_type=pad_type)]
            output_c *= 2
        # residual blocks
        self.model += [ResBlocks(channels=output_c, num_blocks=n_res, norm=norm, act=act, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_c = output_c

    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self,
                 input_c,
                 output_c,
                 n_upsample,
                 n_res,
                 res_norm='adain',
                 act='relu',
                 pad_type='zero'):
        super(Decoder, self).__init__()

        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(channels=input_c, num_blocks=n_res, norm=res_norm, act=act, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2, mode='bilinear'),
                           ConvNormAct(input_c=input_c, output_c=input_c // 2, kernel_size=5, stride=1, padding=2, norm='ln', act=act, pad_type=pad_type)]
            input_c //= 2
        # use reflection padding in the last conv layer
        self.model += [ConvNormAct(input_c=input_c, output_c=output_c, kernel_size=7, stride=1, padding=3, norm='none', act='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)



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
                 pad_type='zero'):
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

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x

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
                 pad_type='zero'):
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
                 pad_type='zero'):
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