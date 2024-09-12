from layout_models import load_layout_model
from multiview_datasets import HM3D_MVL
from layout_models import LY_MODELS_CFG
from geometry_perception_utils.config_utils import read_cfg
from geometry_perception_utils.io_utils import get_abs_path, create_directory
import os
import logging
from omegaconf import OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
from layout_models.horizon_net_wrapper.wrapper_horizon_net_new import estimate_within_list_ly
from geometry_perception_utils.vispy_utils import plot_list_pcl
from imageio.v2 import imwrite


@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    # ! Loading HN model
    logging.info(f"Loading HN model")
    net = load_layout_model(cfg.model)

    vis_dir = create_directory(f"{cfg.log_dir}/vis")
    
    # ! Loading dataset
    dataset = HM3D_MVL(cfg.datasets.hm3d_mvl)
    for scene in dataset.list_scenes:
        logging.info(f"Scene: {scene}")
        list_ly = dataset.get_list_ly(scene_name=scene)
        
        # Estimate layout within the passed list_ly given a model
        estimate_within_list_ly(model=net, list_ly=list_ly)
        
        canvas = plot_list_pcl(list_pcl=[ly.boundary_floor for ly in list_ly], return_canvas=True,
                      scale_factor=None)
        img = canvas.render()
        fn = vis_dir + f"/{scene}.png"
        imwrite(fn, img)
        
    logging.info("done")

    
if __name__ == '__main__':    
    main()