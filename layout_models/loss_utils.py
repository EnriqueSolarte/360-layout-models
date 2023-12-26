from geometry_perception_utils.spherical_utils import phi_coords2xyz
from shapely.geometry import Polygon
from geometry_perception_utils.vispy_utils import plot_color_plc
import logging
import numpy as np
import torch.nn.functional as F
import torch

def compute_L1_loss(y_est, y_ref):
    return F.l1_loss(y_est, y_ref)


def compute_weighted_L1(y_est, y_ref, std, eps=1E-6):
    return F.l1_loss(y_est / (std + eps)**2, y_ref / (std + eps)**2)


def compute_norm_weighted_L1(y_est, y_ref, std, eps=1E-6):
    std = std / std.max(dim=1, keepdim=True)[0]
    return F.l1_loss(y_est / (std + eps)**2, y_ref / (std + eps)**2)


def compute_batch_norm_weighted_L1(y_est, y_ref, std, eps=1E-6):
    std = std / std.max()
    return F.l1_loss(y_est / (std + eps)**2, y_ref / (std + eps)**2)


def compute_bev_weighted_L1(y_est, y_ref, std, kappa=1, in_phi=0.5, eps=1E-6):
    # std_phi = torch.tan(torch.abs(y_ref)) * std
    # kappa = 10
    # power = 2
    # sigma = (std_phi**power + eps)
    # w = kappa / (sigma)
    # # return F.smooth_l1_loss(y_est * w, y_ref * w)
    std_phi = torch.sin(torch.abs(y_ref)) * std 
    
    eps=1E-4
    phi_norm = torch.abs(y_ref*2/np.pi)
    w_cam = torch.exp(-kappa * (phi_norm-in_phi))
    w =  w_cam / (std_phi**2 + eps)
    return F.l1_loss(y_est * w, y_ref * w)
