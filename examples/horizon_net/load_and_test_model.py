import hydra
from geometry_perception_utils.io_utils import get_abs_path
from geometry_perception_utils.config_utils import save_cfg
from layout_models import horizon_net_v2 as hn
from layout_models.utils import load_module


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

    test_loader = hn.DataLoader(
        hn.SimpleDataloader(cfg.test.data),
        batch_size=cfg.test.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.test.num_workers,
        pin_memory=True,
        worker_init_fn=lambda x: cfg.test.seed,
    )

    test_eval = hn.test_loop(model, test_loader)
    hn.print_data_eval(test_eval)


if __name__ == "__main__":
    main()
