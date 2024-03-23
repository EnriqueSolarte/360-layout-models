import torch
from typing import Callable, Optional, List
from layout_models.models.HorizonNet.misc import utils as hn_utils
from layout_models.models.HorizonNet.model import HorizonNet
from torch.utils.data import DataLoader
from layout_models.dataloaders.image_idx_dataloader import ImageIdxDataloader
import logging
from torch import nn
import os 
from torch import optim
import numpy as np
from tqdm import tqdm
from vslab_360_datasets.data_structure.layout import Layout
from tqdm import tqdm, trange
import hydra
from geometry_perception_utils.io_utils import get_abs_path
from geometry_perception_utils.config_utils import save_cfg
            
            
class WrapperHorizonNetV2:
    net: Callable = nn.Identity()
    optimizer: Optional[optim.Optimizer] = None
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    
    def __init__(self, cfg):
        # Set parameters in the class
        [setattr(self, key, val) for key, val in cfg.items()]
        # ! Setting cuda-device
        self.device = torch.device(f"{cfg.device}" if torch.cuda.is_available() else "cpu")        
        logging.info("HorizonNet Wrapper Successfully initialized")


def load_model(ckpt, model: WrapperHorizonNetV2):
    """
    Loads pre-trained model weights from the checkpoint file specified in the config file
    Args:
        ckpt: saved check point 
        model (WrapperHorizonNetV2): model instance
    """
    assert os.path.isfile(ckpt), f"Not found {ckpt}"
    logging.info("Loading HorizonNet...")
    model.net = hn_utils.load_trained_model(HorizonNet, ckpt).to(model.device)
    logging.info(f"ckpt: {ckpt}")
    logging.info("HorizonNet Wrapper Successfully initialized")


def set_default_optimizer(model: WrapperHorizonNetV2):
    # Setting optimizer
    logging.info(f"Setting Optimizer: {model.train.optimizer.name}")
    if model.train.optimizer.name == "SGD":
        model.optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.net.parameters()),
            lr=model.train.optimizer.lr,
            momentum=model.train.optimizer.beta1,
            weight_decay=model.train.optimizer.weight_decay,
        )
    elif model.train.optimizer.name == "Adam":
        model.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.net.parameters()),
            lr=model.train.optimizer.lr,
            betas=(model.train.optimizer.beta1, 0.999),
            weight_decay=model.train.optimizer.weight_decay
        )
    else:
        raise NotImplementedError(
            f"Optimizer {model.train.optimizer.name} not implemented")

        
def set_default_scheduler(model: WrapperHorizonNetV2):
    assert hasattr(model, 'optimizer'), "Optimizer not set"
    
    logging.info(f"Setting scheduler: {model.train.scheduler.name}")
    # Setting scheduler
    if model.train.scheduler.name == "ExponentialLR":   
        decayRate = model.train.scheduler.lr_decay_rate
        model.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=model.optimizer, gamma=decayRate)
    else:
        raise NotImplementedError(
            f"Scheduler {model.cfg_train.scheduler.name} not implemented")
        
        
def set_for_training(model: WrapperHorizonNetV2, optimizer=None, scheduler=None):
    # ! Freezing some layer. This is based on the original implementation
    if model.train.freeze_earlier_blocks != -1:
        b0, b1, b2, b3, b4 = model.net.feature_extractor.list_blocks()
        blocks = [b0, b1, b2, b3, b4]
        for i in range(model.train.freeze_earlier_blocks + 1):
            logging.warning('Freeze block %d' % i)
            for m in blocks[i]:
                for param in m.parameters():
                    param.requires_grad = False

    if model.train.bn_momentum != 0:
        for m in model.net.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.momentum = model.cfg_train.bn_momentum
    
    if optimizer is not None:
        model.optimizer = optimizer
    else:
        set_default_optimizer(model)
    
    if scheduler is not None:
        model.lr_scheduler = scheduler
    else:
        set_default_scheduler(model)
    logging.info("HorizonNet ready for training")    
    
        
def train_loop(model: WrapperHorizonNetV2, dataloader, loss_func, logger=None):
    # Setting optimizer and scheduler if not set
    if not hasattr(model, 'optimizer'):
        set_default_optimizer(model)
    if not hasattr(model, 'scheduler'):
        set_default_scheduler(model)

    if not hasattr(model, 'ready_for_training'):
        # Set specific details for training HN.
        set_for_training(model)
        
    model.net.train()
    
    logging.info(f"Loss function: {loss_func.__module__}.{loss_func.__name__}")
        
    iterator_train = iter(dataloader)
    for _ in trange(
            len(dataloader),
            desc=f"Training HorizonNet..."
    ):

        # * dataloader returns (x, y_bon_ref, std, cam_dist)
        iter_data = next(iterator_train)
        
        y_bon_est, _ = model.net(iter_data['x'].to(model.device))

        if y_bon_est is np.nan:
            raise ValueError("Nan value")

        loss = loss_func(y_bon_est.to(model.device),
                        iter_data['y'])
        
        # back-prop
        model.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.net.parameters(),
                                3.0,
                                norm_type="inf")
        model.optimizer.step()
    model.lr_scheduler.step()


def test_loop(model: WrapperHorizonNetV2, dataloader, logger=None):
    pass


def estimate_within_list_ly(model: WrapperHorizonNetV2, list_ly: List[Layout]):
    """
    Estimates phi_coords (layout boundaries) for all ly defined in list_ly using the passed model instance
    """
    layout_dataloader = DataLoader(
        ImageIdxDataloader([(ly.img_fn, ly.idx) for ly in list_ly]),
        batch_size=model.cfg_inference.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=model.cfg_inference.num_workers,
        pin_memory=True if model.device != "cpu" else False,
        worker_init_fn=lambda x: model.seed,
    )
    model.net.eval()
    evaluated_data = {}
    for x in tqdm(layout_dataloader, desc=f"Estimating layouts..."):
        with torch.no_grad():
            y_bon_, y_cor_ = model.net(x["images"].to(model.device))
        
        # y_bon_ Bx2xHxW, y_cor_ Bx1xHxW
        data = torch.cat([y_bon_, y_cor_], dim=1)
        # data Bx3xHxW <- idx[B] 
        local_eval = {idx: phi_coords.numpy() for phi_coords, idx in zip(data.cpu(), x["idx"])}
        evaluated_data = {**evaluated_data, **local_eval}
        
    [ly.set_phi_coords(phi_coords=evaluated_data[ly.idx])
        for ly in list_ly
    ]


def save_model(model, cfg):
    pass


        
@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="horizon_net_v2")   
def main(cfg):
    # Save CFG and this script
    save_cfg(cfg, cfg_file=f"{cfg.log_dir}/horizon_net_v2.yaml", save_list_scripts=[__file__])
    
    model = WrapperHorizonNetV2(cfg.model)
    
    # load pre-trained model
    load_model(cfg.model.ckpt, model)
    
    # prepare model for training
    set_for_training(model)
    

if __name__ == "__main__":
    main()