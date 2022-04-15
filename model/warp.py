import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.tri as mtri


def flatten(a):
    """Flatten tensor"""
    return a.contiguous().view(-1)

def repeat(a, repeats, axis=0):
    assert len(a.get_shape()) == 1
    a = torch.unsqueeze(a, -1)
    a = a.repeat(1, repeats)
    return a.reshape(-1)

def repeat_2d(a, repeats):
    assert len(a.get_shape()) == 2
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
    t_min = t_min.float()
    t_max = t_max.float()

    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def torch_gather_nd(x, coords):
    x = x.contiguous()
    inds = coords.mv(torch.LongTensor(x.stride()))
    x_gather = torch.index_select(flatten(x), 0, inds)
    return x_gather


def warping(x, offsets, imsize):
    bsize = x.shape[0]
    xsize = x.shape[2]
    offsets = offsets * imsize
    offsets = torch.reshape(offsets[:, 0:2, :, :], (bsize, 2, -1)) # do not need z information

    # first build the grid for target face coordinates
    t_coords = torch.meshgrid(torch.arange(xsize), torch.arange(xsize))
    t_coords = torch.stack(t_coords, dim=0)
    t_coords = torch.FloatTensor(t_coords)
    t_coords = t_coords.reshape(2, -1)
    t_coords = repeat_2d(t_coords, bsize)

    # find the coordinates in the source image to copy pixels
    s_coords = t_coords + offsets
    s_coords = clip_by_tensor(s_coords, 0, xsize-1)

    n_coords = s_coords.shape[2]
    idx = repeat(torch.arange(bsize), n_coords)

    def _gather_pixel(_x, coords):
        coords = torch.IntTensor(coords)
        xcoords = torch.reshape(coords[..., 0], (-1))
        ycoords = torch.reshape(coords[..., 1], (-1))
        ind = torch.stack([idx, xcoords, ycoords], dim=-1)

        _y = torch_gather_nd(_x, ind)
        _y = torch.reshape(_y, (bsize, _x.shape[3], n_coords))
        return _y

    # solve fractional coordinates via bilinear interpolation
    s_coords_lu = torch.floor(s_coords)     # floor the coordinate values
    s_coords_rb = torch.ceil(s_coords)      # ceil the coordinate values

    s_coords_lb = torch.stack([s_coords_lu[..., 0], s_coords_rb[..., 1]], dim=-1)       # lu[0] + rb [1] = lb
    s_coords_ru = torch.stack([s_coords_rb[..., 0], s_coords_lu[..., 1]], dim=-1)      # rb[0] + lu[1] = ru
    _x_lu = _gather_pixel(x, s_coords_lu)
    _x_rb = _gather_pixel(x, s_coords_rb)
    _x_lb = _gather_pixel(x, s_coords_lb)
    _x_ru = _gather_pixel(x, s_coords_ru)
    # bilinear interpolation
    s_coords_fraction = s_coords - torch.FloatTensor(s_coords_lu)
    s_coords_fraction_x = s_coords_fraction[..., 0]
    s_coords_fraction_y = s_coords_fraction[..., 1]
    _xs, _ys = s_coords_fraction_x.shape
    s_coords_fraction_x = torch.reshape(s_coords_fraction_x, (1, _xs, _ys))
    s_coords_fraction_y = torch.reshape(s_coords_fraction_y, (1, _xs, _ys))
    _x_u = _x_lu + (_x_ru - _x_lu) * s_coords_fraction_x
    _x_b = _x_lb + (_x_rb - _x_lb) * s_coords_fraction_x
    warped_x = _x_u + (_x_b - _x_u) * s_coords_fraction_y
    warped_x = torch.reshape(warped_x, (bsize, xsize, xsize, -1))

    return warped_x

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
