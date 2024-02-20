
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
