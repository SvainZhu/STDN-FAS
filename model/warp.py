import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.tri as mtri


def flatten(a):
    """Flatten tensor"""
    return a.contiguous().view(-1)

def repeat(a, repeats, axis=0):
    a = torch.unsqueeze(a, -1)
    a = a.repeat(1, repeats)
    return a.reshape(-1)

def repeat_2d(a, repeats):
    a = torch.unsqueeze(a, 0)
    a = a.repeat(repeats, 1, 1)
    return a


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


# def warping(x, offsets, imsize):
#     bsize = x.shape[0]
#     xsize = x.shape[2]
#     offsets = offsets * imsize
#     offsets = torch.reshape(offsets[:, 0:2, :, :], (bsize, 2, -1)) # do not need z information
#     offsets = offsets.cuda()
# 
#     # first build the grid for target face coordinates
#     t_coords = torch.meshgrid(torch.arange(xsize), torch.arange(xsize))
#     t_coords = torch.stack(t_coords, dim=0)
#     t_coords = t_coords.to(torch.float32)
#     t_coords = t_coords.reshape(2, -1)
#     t_coords = repeat_2d(t_coords, bsize)
#     t_coords = t_coords.cuda()
# 
#     # find the coordinates in the source image to copy pixels
#     s_coords = t_coords + offsets
#     s_coords = clip_by_tensor(s_coords, 0, xsize-1)
# 
#     n_coords = s_coords.shape[2]
# 
#     def _gather_pixel(_x, coords):
#         coords = coords.to(torch.int32).permute(0, 2, 1)
# 
#         coords_chunked = coords.chunk(2, 2)
#         masked = _x[torch.arange(_x.shape[0]).view(_x.shape[0], 1).to(torch.long), :, coords_chunked[0].squeeze().to(torch.long), coords_chunked[1].squeeze().to(torch.long)]
#         _y = masked.expand(1, *masked.shape).permute(1, 3, 2, 0).view(*_x.shape)
# 
#         _y = torch.reshape(_y, (bsize, -1, n_coords))
#         return _y
# 
#     # solve fractional coordinates via bilinear interpolation
#     s_coords_lu = torch.floor(s_coords)     # floor the coordinate values
#     s_coords_rb = torch.ceil(s_coords)      # ceil the coordinate values
# 
#     s_coords_lb = torch.stack((s_coords_lu[:, 0, :], s_coords_rb[:, 1, :]), dim=1)       # lu[0] + rb [1] = lb
#     s_coords_ru = torch.stack((s_coords_rb[:, 0, :], s_coords_lu[:, 1, :]), dim=1)      # rb[0] + lu[1] = ru
#     _x_lu = _gather_pixel(x, s_coords_lu)
#     _x_rb = _gather_pixel(x, s_coords_rb)
#     _x_lb = _gather_pixel(x, s_coords_lb)
#     _x_ru = _gather_pixel(x, s_coords_ru)
#     # bilinear interpolation
#     s_coords_fraction = s_coords.cuda() - s_coords_lu.cuda()
#     s_coords_fraction_x = s_coords_fraction[:, 0, :]
#     s_coords_fraction_y = s_coords_fraction[:, 1, :]
#     _xs, _ys = s_coords_fraction_x.shape
#     s_coords_fraction_x = torch.reshape(s_coords_fraction_x, (_xs, 1, _ys))
#     s_coords_fraction_y = torch.reshape(s_coords_fraction_y, (_xs, 1, _ys))
#     _x_u = _x_lu + (_x_ru - _x_lu) * s_coords_fraction_x
#     _x_b = _x_lb + (_x_rb - _x_lb) * s_coords_fraction_x
#     warped_x = _x_u + (_x_b - _x_u) * s_coords_fraction_y
#     warped_x = torch.reshape(warped_x, (bsize, -1, xsize, xsize))
# 
#     return warped_x

def warping(x, offsets, imsize):
    bsize = x.shape[0]
    xsize = x.shape[2]
    offsets = offsets * imsize
    offsets = torch.reshape(offsets[:, 0:2, :, :], (bsize, 2, xsize, xsize)) # do not need z information
    offsets = offsets.cuda()

    # first build the grid for target face coordinates
    xx = torch.arange(0, xsize).view(1, -1).repeat(xsize, 1)
    yy = torch.arange(0, xsize).view(-1, 1).repeat(1, xsize)
    xx = xx.view(1, 1, xsize, xsize).repeat(bsize, 1, 1, 1)
    yy = yy.view(1, 1, xsize, xsize).repeat(bsize, 1, 1, 1)
    t_coords = torch.cat((xx, yy), dim=1).float().cuda()

    # find the coordinates in the source image to copy pixels
    s_coords = t_coords + offsets
    # scale grid to [-1, 1]
    s_coords[:, 0, :, :] = 2.0 * s_coords[:, 0, :, :].clone() / max(xsize - 1, 1) - 1.0
    s_coords[:, 1, :, :] = 2.0 * s_coords[:, 1, :, :].clone() / max(xsize - 1, 1) - 1.0
    s_coords = s_coords.permute(0, 2, 3, 1)
    warp_x = F.grid_sample(x, s_coords, mode='bilinear', padding_mode='zeros')
    
    mask = torch.ones(x.size()).cuda()
    mask = F.grid_sample(mask, s_coords)
    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1
    
    return warp_x * mask


def generate_offset_map(source, target):
    anchor_pts = [[0,0],[0,256],[256,0],[256,256],
                  [0,128],[128,0],[256,128],[128,256],
                  [0,64],[0,192],[256,64],[256,192],
                  [64,0],[192,0],[64,256],[192,256]]
    anchor_pts = np.asarray(anchor_pts)/ 256
    xi, yi = np.meshgrid(np.linspace(0, 1, 256), np.linspace(0, 1, 256))
    _source = np.concatenate([source, anchor_pts], axis=0).astype(np.float32)
    _target = np.concatenate([target, anchor_pts], axis=0).astype(np.float32)
    _offset = _source - _target

    # interp2d
    _triang  = mtri.Triangulation(_target[:,0], _target[:,1])
    _interpx = mtri.LinearTriInterpolator(_triang, _offset[:,0])
    _interpy = mtri.LinearTriInterpolator(_triang, _offset[:,1])
    _offsetmapx = _interpx(xi, yi)
    _offsetmapy = _interpy(xi, yi)

    offsetmap = np.stack([_offsetmapy, _offsetmapx, _offsetmapx*0], axis=2)
    return offsetmap
