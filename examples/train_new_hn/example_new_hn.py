from mlc_pp import MLC_PP_CFG_DIR
from layout_models import load_layout_model
import geometry_perception_utils.config_utils as cfg_utils
from geometry_perception_utils.io_utils import get_abs_path
import hydra
import wandb
import logging
from layout_models.dataloaders import get_dataloader
from layout_models.dataloaders.mvl_dataloader import MVLDataLoader
import layout_models.horizon_net_wrapper.wrapper_horizon_net_new as ly_estimator
from layout_models.loss_utils_new import compute_weighted_L1


@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    logging.info("Starting training...")
    cfg_utils.save_cfg(cfg.copy(), __file__)

    for trial in range(cfg.get("trials", 1)):
        # Loading based on the config
        model = load_layout_model(cfg.model)
        train_dataloader = get_dataloader(dataset_class=MVLDataLoader, cfg=model.cfg_train)

        test_dataloader = get_dataloader(dataset_class=MVLDataLoader, cfg=model.cfg_test)
        logging.info(f"Starting the training Trial: {trial}")
        for epoch in range(model.cfg_train.epochs):
            logging.info(f"Init epoch: {epoch}")
            logging.info(f"Experiment name: {cfg.experiment_name}")
            ly_estimator.train_loop(model=model, dataloader=train_dataloader, loss_func=compute_weighted_L1)    
            ly_estimator.test_loop(model=model, dataloader=test_dataloader)
        wandb.finish()


if __name__ == '__main__':
    main()
