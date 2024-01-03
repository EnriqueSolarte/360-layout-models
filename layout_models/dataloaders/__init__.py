
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def get_dataloader(dataset_class: Dataset, cfg: dict):
    """
    Returns a dataloader based on the dataset and the config.
    """
    loader = DataLoader(
        dataset_class(cfg),
        batch_size=cfg.batch_size,
        shuffle=cfg.get("shuffle", True),
        drop_last=cfg.get("drop_last", False),
        num_workers=cfg.num_workers,
        pin_memory=True if cfg.device != "cpu" else False,
        worker_init_fn=lambda x: cfg.seed,
    )
    return loader
    