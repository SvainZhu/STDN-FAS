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

def rgb_to_hsv(input, epsilon=1e-10):
    assert(input.shape[1] == 3)

    r, g, b = input[:, 0], input[:, 1], input[:, 2]
    max_rgb, argmax_rgb = input.max(1)
    min_rgb, argmin_rgb = input.min(1)

    max_min = max_rgb - min_rgb + epsilon

    h1 = 60.0 * (g - r) / max_min + 60.0
    h2 = 60.0 * (b - g) / max_min + 180.0
    h3 = 60.0 * (r - b) / max_min + 300.0

    h = torch.stack((h2, h3, h1), dim=0).gather(dim=0, index=argmin_rgb.unsqueeze(0)).squeeze(0)
    s = max_min / (max_rgb + epsilon)
    v = max_rgb

    return torch.stack((h, s, v), dim=1)


def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    t = t.float()

    result = (t >= t_min).to(torch.float32).cuda() * t + (t < t_min).to(torch.float32).cuda() * t_min
    result = (result <= t_max).to(torch.float32).cuda() * result + (result > t_max).to(torch.float32).cuda() * t_max
    return result


def plotResults(result_list):
    column = []
    for fig in result_list:
        shape = fig.shape
        fig = clip_by_tensor(fig, 0.0, 1.0)
        if shape[1] == 1:
            fig = torch.cat((fig, fig, fig), dim=1)
        # else:
        #     r, g, b = torch.split(fig, shape[1] // 3, 1)
        #     fig = torch.cat((b, g, r), dim=1)
        fig = F.interpolate(fig, [256, 256])
        row = torch.split(fig, 1)
        row = torch.cat(row, dim=3)
        column.append(row[0, :, :, :])

    column = torch.cat(column, dim=1).data.cpu()
    column = column * 255
    img = column.permute(1, 2, 0)
    return img


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
            self.bn = nn.BatchNorm2d(output_c, eps=1e-5, momentum=0.1)
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
        if norm:
            self.bn = nn.BatchNorm2d(output_c, eps=1e-5, momentum=0.1)
        if act:
            self.ac = nn.PReLU()

    def forward(self, x):
        out = self.conv(x)
        if self.norm:
            out = self.bn(out)
        if self.act:
            out = self.ac(out)
        return out
