
import importlib
import logging
import numpy as np


def load_layout_model(cfg):
    """
    Load a layout model estimator and returns an instance of it
    """

    if cfg.model.ly_model == "HorizonNet":
        from layout_models.horizon_net_wrapper.wrapper_horizon_net import WrapperHorizonNet
        model = WrapperHorizonNet(cfg)
    # if cfg.ly_model == "HorizonNetV2":
    #     from layout_models.horizon_net_wrapper.wrapper_horizon_net_new import WrapperHorizonNet as WrapperHorizonNetV2
    #     model = WrapperHorizonNetV2(cfg)
    else:
        raise NotImplementedError("")

    return model


def print_data_eval(data_dict, logger=logging, func=np.mean):
    """
    This is a generic function that prints in the active logger the data_dict.
    The data_dic is a dictionary of metrics as the keys with a list of evaluated results
    e.g.:
    {
        'loss': [0.1, 0.2, 0.3, ..., 0.3 ],
        'accuracy': [0.9, 0.8, 0.7, ... 0,9]
    }
    """
    [logging.info(f"* {k}:{func(v)}") for k, v in data_dict.items()]


def load_module(module_name):
    """
    Load a module and return the instance. The module path 
    is given by standard python module path, e.g. layout_model.horizon_net_wrapper.wrapper_horizon_net_v2
    """
    path_module = module_name.split(".")
    module = importlib.import_module(".".join(path_module[:-1]))
    instance = getattr(module, path_module[-1])
    return instance
