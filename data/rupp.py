# dataset for training ResUnet++ (RUPP)
# augmentation already done by scripts/preprocess.py

import os
import os.path as osp
from glob import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import pmap, read_rgb_255, read_mask


def _read_rgb_255(path):
    return read_rgb_255(path), path

def _read_mask(path):
    return read_mask(path), path

def _sort_by_name_inplace(ls):
    ls.sort(key=lambda x: x[1])


class RUPP(Dataset):
    def __init__(self, 
                 root_dir='/content/data/kvasir_seg_new/Kvasir-SEG/',
                 split='train'):

        self.root_dir = root_dir
        self.split = split
        xdirs = glob(osp.join(self.root_dir, split, 'images', '*.jpg'))
        ydirs = glob(osp.join(self.root_dir, split, 'masks', '*.jpg'))
        self.x = pmap(_read_rgb_255, xdirs, show_pbar=True)
        self.y = pmap(_read_mask,    ydirs, show_pbar=True)
        _sort_by_name_inplace(self.x)
        _sort_by_name_inplace(self.y)
        self.print_info()

    def print_info(self):
        print(f'Found {len(self)} images and segmentation maps')
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x, _ = self.x[idx]
        y, _ = self.y[idx]

        x = x.astype(np.float32) / 255
        x = torch.from_numpy(x).movedim(-1, 0)
        y = torch.from_numpy(y)
        return x, y
