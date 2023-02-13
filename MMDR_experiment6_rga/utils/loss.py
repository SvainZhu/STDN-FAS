import torch.nn.functional as F
import torch.nn as nn
import torch

def l1_loss(x, y, mask=None):
    xshape = x.shape
    if mask is not None:
        loss = torch.mean(torch.reshape(torch.abs(x-y), (xshape[0], -1)), dim=1, keepdim=True)
        loss = torch.sum(loss * mask) / (torch.sum(mask) + 1e-8)
    else:
        loss = torch.mean(torch.abs(x-y))
    return loss

def l2_loss(x, y, mask=None):
    xshape = x.shape
    if mask is not None:
        loss = torch.mean(torch.reshape(torch.square(x-y), (xshape[0], -1)), dim=1, keepdim=True)
        loss = torch.sum(loss * mask) / (torch.sum(mask) + 1e-8)
    else:
        loss = torch.mean(torch.square(x-y))
    return loss

class ContrastLoss(nn.Module):

    def __init__(self):
        super(ContrastLoss, self).__init__()
        pass

    def forward(self, anchor_fea, reassembly_fea, contrast_label):
        contrast_label = contrast_label.float()
        anchor_fea = anchor_fea.detach()
        loss = -(F.cosine_similarity(anchor_fea, reassembly_fea, dim=-1))
        loss = loss * contrast_label
        return loss.mean()