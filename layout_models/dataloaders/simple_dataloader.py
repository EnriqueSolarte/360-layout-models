
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
from geometry_perception_utils.spherical_utils import phi_coords2xyz
from omegaconf import OmegaConf


class SimpleDataloader(data.Dataset):
    '''
    The general dataloader uses the following hyperparameters to load data:  
    - img_dir, - label_dir, - scene_list. The latter is defines in 
    a room: [list of frames] format. 
    '''

    def __init__(self, cfg):
        [setattr(self, key, val) for key, val in cfg.items()]
        logging.info(f"{self.__module__} initialized")
        [logging.info(f"{key}: {val}") for key, val in cfg.items()]

        # List of scenes defined in a list file
        if self.scene_list == '' or self.scene_list is None or self.scene_list == 'None':
            #  Reading from available labels data in labels_dir
            self.list_frames = os.listdir(self.labels_dir)
            self.list_rooms = None
        else:
            assert os.path.exists(
                self.scene_list), f"No found {self.scene_list}"
            raw_data = json.load(open(self.scene_list))
            # keys are rooms
            self.list_rooms = list(raw_data.keys())
            # values are list of frames per room
            self.list_frames = [raw_data[room] for room in self.list_rooms]
            # all frames
            self.list_frames = [
                item for sublist in self.list_frames for item in sublist
            ]
        np.random.seed(self.seed)
        if self.size < 0:
            # Assuming negative size means all data
            self.selected_fr = self.list_frames
        elif self.size < 1:
            # fraction of data
            np.random.shuffle(self.list_frames)
            self.selected_fr = self.list_frames[:int(self.size *
                                                     self.list_frames.__len__())]
        else:
            # exact number of data
            np.random.shuffle(self.list_frames)
            self.selected_fr = self.list_frames[:self.size]
        #  By default this dataloader iterates by frames
        #  Pre compute list of files to speed-up iterations
        self.pre_compute_list_files()

    def pre_compute_list_files(self):
        # * Data Directories
        self.list_imgs = []
        self.list_labels = []

        [(self.list_imgs.append(os.path.join(self.img_dir, f"{scene}")),
          self.list_labels.append(os.path.join(self.labels_dir, f"{scene}"))
          )
         for scene in self.selected_fr]
        logging.info(
            f"Total data in this dataloader: {self.selected_fr.__len__()}")

    def __len__(self):
        return self.selected_fr.__len__()

    def get_image(self, idx):
        # * Load image
        image_fn = self.list_imgs[idx]
        if os.path.exists(image_fn + '.jpg'):
            image_fn += '.jpg'
        elif os.path.exists(image_fn + '.png'):
            image_fn += '.png'
        else:
            raise ValueError(f"Image file not found: {image_fn}")

        return np.array(Image.open(image_fn), np.float32)[..., :3] / 255.

    def get_label(self, idx):
        label_fn = self.list_labels[idx]
        # * Load label
        if os.path.exists(label_fn + '.npy'):
            label = np.load(label_fn + '.npy')
        elif os.path.exists(label_fn + '.npz'):
            label = np.load(label_fn + '.npz')["phi_coords"]
        else:
            raise ValueError(f"Label file not found: {label_fn}")
        return label

    def __getitem__(self, idx):
        # ! iteration per each self.data given a idx

        img = self.get_image(idx)
        label = self.get_label(idx)

        # Random flip
        if self.flip and np.random.randint(2) == 0:
            img = np.flip(img, axis=1)
            label = np.flip(label, axis=len(label.shape) - 1)

        # Random horizontal rotate
        if self.rotate:
            dx = np.random.randint(img.shape[1])
            img = np.roll(img, dx, axis=1)
            label = np.roll(label, dx, axis=len(label.shape) - 1)

        # Random gamma augmentation
        if self.gamma:
            p = np.random.uniform(1, 2)
            if np.random.randint(2) == 0:
                p = 1 / p
            img = img**p

        x = torch.FloatTensor(img.transpose([2, 0, 1]).copy())
        label = torch.FloatTensor(label.copy())
        return dict(x=x, y=label)
