import torch
import torch.nn as nn
import torch.nn.functional as F


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

def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    t = t.float()
    t_min = t_min.float()
    t_max = t_max.float()

    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result


def plotResults(result_list):
    column = []
    for fig in result_list:
        shape = fig.shape
        fig = clip_by_tensor(fig, 0.0, 1.0)
        row = []
        if fig.shape[3] == 1:
            fig = torch.cat([fig, fig, fig], dim=1)
        else:
            r, g, b = torch.split(fig, 3, 3)
            fig = torch.cat([b,g,r], 3)
        fig = F.interpolate(fig, [256, 256], mode='bilinear')
        row = torch.split(fig, shape[0])
        row = torch.cat(row, dim=3)
        column.append(row[0,:,:,:])

    column = torch.cat(column, dim=0)
    img = torch.IntTensor(column * 255)
    return img

class Error:
    def __init__(self):
        self.losses = {}

    def __call__(self, update, val=0):
        loss_name   = update[0]
        loss_update = update[1]
        if loss_name not in self.losses.keys():
            self.losses[loss_name] = {'value':0, 'step': 0, 'value_val':0, 'step_val':0}
        if val == 1:
            if loss_update is not None:
                self.losses[loss_name]['value_val'] += loss_update
                self.losses[loss_name]['step_val'] += 1
            smooth_loss = str(round(self.losses[loss_name]['value_val'] / (self.losses[loss_name]['step_val']+1e-5),3))
            return loss_name +':'+smooth_loss+','
        else:
            if loss_update is not None:
                self.losses[loss_name]['value'] = self.losses[loss_name]['value'] * 0.9 + loss_update*0.1
                self.losses[loss_name]['step'] += 1
            if self.losses[loss_name]['step'] == 1:
                self.losses[loss_name]['value'] = loss_update
            smooth_loss = str(round(self.losses[loss_name]['value'], 3))
            return loss_name +':'+smooth_loss+','

    def reset(self):
        self.losses = {}

class FC(nn.Module):
    def __init__(self,
                 input_c: int,
                 output_c: int,
                 act: bool = True,
                 norm: bool = True,
                 apply_dropout: bool = False):
        super(FC, self).__init__()

        self.fc = nn.Linear(in_features=input_c, out_features=output_c, bias=True)
        self.act = act
        self.norm = norm
        self.apply_dropout = apply_dropout
        if norm:
            self.bn = nn.BatchNorm2d(eps=1e-5, momentum=0.01)
        if act:
            self.ac = nn.PReLU()
        if apply_dropout:
            self.dropout = nn.Dropout(p=0.3)
    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.bn(out)
        if self.act:
            out = self.ac(out)
        if self.apply_dropout:
            out = self.dropout(out)

        return out

class ConvBNAct(nn.Module):
    def __init__(self,
                 input_c: int,
                 output_c: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 norm: bool = True,
                 act: bool = True,
                 apply_dropout: bool = False
                 ):
        super(ConvBNAct, self).__init__()

        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels=input_c, out_channels=output_c, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.norm = norm
        self.act = act
        self.apply_dropout = apply_dropout
        if norm:
            self.bn = nn.BatchNorm2d(output_c, eps=1e-5, momentum=0.01)
        if act:
            self.ac = nn.PReLU()
        if apply_dropout:
            self.dropout = nn.Dropout(p=0.3)
    def forward(self, x):
        out = self.conv(x)
        if self.norm:
            out = self.bn(out)
        if self.act:
            out = self.ac(out)
        if self.apply_dropout:
            out = self.dropout(out)

        return out

class Downsample(nn.Module):
    def __init__(self,
                 input_c: int,
                 output_c: int,
                 kernel_size: int = 3,
                 stride: int = 2,
                 norm: bool = True,
                 act: bool = True,
                 apply_dropout: bool = False
                 ):
        super(Downsample, self).__init__()

        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels=input_c, out_channels=output_c, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.norm = norm
        self.act = act
        self.apply_dropout = apply_dropout
        if norm:
            self.bn = nn.BatchNorm2d(output_c, eps=1e-5, momentum=0.01)
        if act:
            self.ac = nn.PReLU()
        if apply_dropout:
            self.dropout = nn.Dropout(p=0.3)
    def forward(self, x):
        out = self.conv(x)
        if self.norm:
            out = self.bn(out)
        if self.act:
            out = self.ac(out)
        if self.apply_dropout:
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
                 apply_dropout: bool = False
                 ):
        super(Upsample, self).__init__()

        padding = (kernel_size - 1) // 2

        self.conv = nn.ConvTranspose2d(in_channels=input_c, out_channels=output_c, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=1, bias=True)
        self.norm = norm
        self.act = act
        self.apply_dropout = apply_dropout
        if norm:
            self.bn = nn.BatchNorm2d(output_c, eps=1e-5, momentum=0.01)
        if act:
            self.ac = nn.PReLU()

    def forward(self, x):
        out = self.conv(x)
        if self.norm:
            out = self.bn(out)
        if self.act:
            out = self.ac(out)
        return out
