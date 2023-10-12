from geometry_perception_utils.spherical_utils import phi_coords2xyz
from shapely.geometry import Polygon
import torch.nn.functional as F
from geometry_perception_utils.vispy_utils import plot_color_plc
import logging
import numpy as np


def compute_L1_loss(y_est, y_ref):
    return F.l1_loss(y_est, y_ref)


def compute_weighted_L1(y_est, y_ref, std, min_std=1E-2):
    return F.l1_loss(y_est / (std + min_std)**2, y_ref / (std + min_std)**2)
