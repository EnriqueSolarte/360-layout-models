import hydra
from geometry_perception_utils.io_utils import get_abs_path, create_directory
from geometry_perception_utils.config_utils import save_cfg
from layout_models import horizon_net_v2 as hn
from layout_models.utils import load_module
from multiview_datasets import HM3D_MVL, ZInD_mvl, MP3D_FPE_MVL
from tqdm import tqdm
from geometry_perception_utils.vispy_utils import plot_list_pcl
from imageio.v2 import imwrite


@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg")
def main(cfg):
    # Save CFG and this script
    save_cfg(cfg, cfg_file=f"{cfg.log_dir}/cfg.yaml",
             save_list_scripts=[__file__])

    model = hn.WrapperHorizonNetV2(cfg.model)

    # load pre-trained model
    hn.load_model(cfg.model.ckpt, model)

    # Load list_ly from cfg.test.data
    mvl_dataset = HM3D_MVL(cfg.mvl_dataset)

    vis_dir = create_directory(
        f"{cfg.log_dir}/vis", delete_prev=True, ignore_request=True)
    for scene_name in mvl_dataset.list_scenes:
        list_ly = mvl_dataset.get_list_ly(scene_name=scene_name)

        # inference within list_ly
        hn.estimate_within_list_ly(list_ly=list_ly, model=model)
        canvas = plot_list_pcl(
            [ly.boundary_floor for ly in list_ly], return_canvas=True)
        img = canvas.render(alpha=True, bgcolor=(1, 1, 1, 0),)
        fn = f"{vis_dir}/{scene_name}.png"

        imwrite(fn, img)
        print("Saved", fn)
        input("Press Enter to continue...")


if __name__ == "__main__":
    main()
