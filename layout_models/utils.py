from layout_models.models.horizon_net_wrapper import WrapperHorizonNet

def load_layout_model(cfg):
    """
    Load a layout model estimator and returns an instance of it
    """
    if cfg.model.ly_model == "HorizonNet":
        # ! loading HorizonNet
        model = WrapperHorizonNet(cfg)
    else:
        raise NotImplementedError("")

    return model
