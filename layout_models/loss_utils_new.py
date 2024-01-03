from geometry_perception_utils.spherical_utils import phi_coords2xyz
from shapely.geometry import Polygon
from geometry_perception_utils.vispy_utils import plot_color_plc
import logging
import numpy as np
import torch.nn.functional as F
import torch


def compute_weighted_L1(y_est, y_ref, eps=1E-6):
    y_bound_ref = y_ref[0].to(y_est.device)
    std = y_ref[1].to(y_est.device)
    return F.l1_loss(y_est / (std + eps)**2, y_bound_ref / (std + eps)**2)
