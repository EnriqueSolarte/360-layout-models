from layout_models import LY_MODELS_CFG
from geometry_perception_utils.config_utils import read_omega_cfg, merge_cfg
from geometry_perception_utils.vispy_utils import plot_color_plc
from vslab_360_datasets import HM3D_MVL
from layout_models import load_layout_model
import os
import numpy as np


def main(cfg):
    # ! Loading dataset
    dt = HM3D_MVL(cfg.datasets.hm3d_mvl)
    list_ly = dt.get_list_ly()

    #! Loading HorizonNet
    hn = load_layout_model(cfg)

    #! Inference within list_ly
    hn.estimate_within_list_ly(list_ly)

    #! Estimated Floor boundary
    pcl_est = np.hstack([ly.boundary_floor for ly in list_ly])
    plot_color_plc(pcl_est.T)

    # ! GT Floor boundary
    [ly.set_gt_phi_coords_as_default() for ly in list_ly]
    pcl_gt = np.hstack([ly.boundary_floor for ly in list_ly])

    plot_color_plc(pcl_gt.T)


if __name__ == '__main__':
    # ! Loading HorizonNet cfg
    cfg_hn = os.path.join(LY_MODELS_CFG, 'horizon_net.yaml')
    cfg_hn = read_omega_cfg(cfg_hn)

    # ! Loading HM3D-MVL cfg
    cfg_dt = os.path.join(LY_MODELS_CFG, 'datasets.yaml')
    cfg_dt = read_omega_cfg(cfg_dt)

    cfg = merge_cfg([cfg_hn, cfg_dt])
    main(cfg)
