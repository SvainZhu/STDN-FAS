import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.tri as mtri
from model.utils import clip_by_tensor

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
