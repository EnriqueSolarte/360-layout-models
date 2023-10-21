import os
import numpy as np
import pathlib
import torch.utils.data as data
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm import trange
from PIL import Image
import json
import torch
import logging
from copy import deepcopy
from geometry_perception_utils.io_utils import check_file_exist


class MVLSimpleDataLoader(data.Dataset):
    '''
    Dataloader that handles MLC dataset format.
    '''
    def __init__(self, cfg):
        self.cfg = cfg

        # ! List of scenes defined in a list file
        if cfg.data.get('scene_list', '') == '':
            # ! Reading from available labels data
            self.raw_data = os.listdir(
                os.path.join(self.cfg.data.labels_dir, self.cfg.label))
            self.list_frames = self.raw_data
            self.list_rooms = None
        else:
            scene_list = self.cfg.data.scene_list
            assert os.path.exists(scene_list), f"No found {scene_list}"
            raw_data = json.load(open(scene_list))
            self.list_rooms = list(raw_data.keys())
            self.list_frames = [raw_data[room] for room in self.list_rooms]
            self.list_frames = [
                item for sublist in self.list_frames for item in sublist
            ]
            
        seed = cfg.get('seed', 1000)
        logging.info(f"Seed: {seed}")
        np.random.seed(seed)
        if cfg.get('size', -1) < 0:
            np.random.shuffle(self.list_frames)
            self.data = self.list_frames
        elif cfg.size < 1:
            np.random.shuffle(self.list_frames)
            self.data = self.list_frames[:int(cfg.size *
                                              self.list_frames.__len__())]
        else:
            np.random.shuffle(self.list_frames)
            self.data = self.list_frames[:cfg.size]
        # ! By default this dataloader iterates by frames

        # ! Pre compute list of files to speed-up iterations
        self.pre_compute_list_files()
        # if cfg.data.check:
        #     self.check_data()

    def pre_compute_list_files(self):
        self.list_imgs = []
        self.list_labels = []
        self.img_dir = self.cfg.data.img_dir
        self.labels_dir = self.cfg.data.labels_dir
        [(self.list_imgs.append(os.path.join(self.img_dir, f"{scene}")),
          self.list_labels.append(os.path.join(self.labels_dir, f"{scene}")))
         for scene in self.data]
        logging.info(
            f"Simple MLC dataloader initialized with: {self.cfg.data.img_dir}")
        logging.info(f"Total data in this dataloader: {self.data.__len__()}")
        # logging.info(f"Used scene list: {self.cfg.data.get('scene_list', 'None')}")
        logging.info(f"Labels dir: {self.cfg.data.labels_dir}")

    def check_data(self):
        logging.info(f"Checking data img & labels")
        check_imgs = check_file_exist(self.list_imgs, ext='jpg')
        if np.sum(check_imgs) != self.list_imgs.__len__():
            logging.error(f"Missing images in {self.cfg.data.img_dir}")
            raise ValueError(f"Missing images in {self.cfg.data.img_dir}")

        check_labels = check_file_exist(self.list_labels, ext='npy')
        if np.sum(check_labels) != self.list_labels.__len__():
            check_labels = check_file_exist(self.list_labels, ext='npz')
            if np.sum(check_labels) != self.list_labels.__len__():
                logging.error(
                    f"Missing labels in {self.cfg.data.labels_dir}/{self.cfg.label}"
                )
                raise ValueError(
                    f"Missing labels in {self.cfg.data.labels_dir}/{self.cfg.label}"
                )
        logging.info(f"Data img & labels check passed successfully")
        logging.info(
            f"Simple MLC dataloader initialized with: {self.cfg.data.img_dir}")
        logging.info(f"Total data in this dataloader: {self.data.__len__()}")
        logging.info(f"Used scene list: {self.cfg.get('scene_list', 'NA')}")
        logging.info(
            f"Labels dir: {os.path.join(self.cfg.data.labels_dir, self.cfg.label)}"
        )

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, idx):
        # ! iteration per each self.data given a idx
        label_fn = self.list_labels[idx]
        image_fn = self.list_imgs[idx]

        if os.path.exists(image_fn + '.jpg'):
            image_fn += '.jpg'
        elif os.path.exists(image_fn + '.png'):
            image_fn += '.png'

        img = np.array(Image.open(image_fn), np.float32)[..., :3] / 255.

        if os.path.exists(label_fn + '.npy'):
            label = np.load(label_fn + '.npy')
        elif os.path.exists(label_fn + '.npz'):
            label = np.load(label_fn + '.npz')["phi_coords"]
        else:
            raise ValueError(f"Label file not found: {label_fn}")
        assert label.shape[1] == img.shape[1], f"Shape mismatch: {label_fn}, {label.shape}"
        
        if label.shape[0] == 4:
            # ! Then labels were compute from mlc [4, 1024]
            std = label[2:]
            label = label[:2]
        elif label.shape[0] == 3:
            # Then labels were compute from mlc [3, 1024]
            logging.warning(f"label.shape[0]: {label.shape[0]}")
            label = label[:2]
            std = np.hstack((label[3], label[3]))
        else:   
            logging.warning(f"label.shape[0]: {label.shape[0]}")
            assert label.shape[0] == 2, f"Shape mismatch: {label_fn}, {label.shape}"
            std = np.ones([2, label.shape[1]])

        # Random flip
        if self.cfg.get('flip', False) and np.random.randint(2) == 0:
            img = np.flip(img, axis=1)
            label = np.flip(label, axis=len(label.shape) - 1)

        # Random horizontal rotate
        if self.cfg.get('rotate', False):
            dx = np.random.randint(img.shape[1])
            img = np.roll(img, dx, axis=1)
            label = np.roll(label, dx, axis=len(label.shape) - 1)

        # Random gamma augmentation
        if self.cfg.get('gamma', False):
            p = np.random.uniform(1, 2)
            if np.random.randint(2) == 0:
                p = 1 / p
            img = img**p

        x = torch.FloatTensor(img.transpose([2, 0, 1]).copy())
        label = torch.FloatTensor(label.copy())
        std = torch.FloatTensor(std.copy())
        return (x, label, std)


def get_mvl_simple_dataloader(cfg, device='cpu'):
    loader = DataLoader(
        MVLSimpleDataLoader(cfg),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=True if device != 'cpu' else False,
        worker_init_fn=lambda x: np.random.seed(),
    )
    return loader