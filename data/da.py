# dataset for ADVENT domain adaptation
# online augmentation

import os
import os.path as osp
from glob import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import pmap, read_rgb_255, read_mask, get_files_of_extension


def _read_rgb_255(path):
    return read_rgb_255(path), path

def _read_mask(path):
    return read_mask(path).astype(np.float32), path

def _sort_by_name_inplace(ls):
    ls.sort(key=lambda x: x[1])



class ADVENT(Dataset):
    def __init__(self, 
                 root_dir='/content/data/kvasir_seg_new/Kvasir-SEG/',
                 transform=None,
                 get_mask=True,
                 ):

        self.root_dir = root_dir
        imgs_dir = osp.join(self.root_dir, 'images')
        xdirs = get_files_of_extension(imgs_dir,  '.jpg', '.png')
        self.x = pmap(_read_rgb_255, xdirs, show_pbar=True)
        _sort_by_name_inplace(self.x)

        self.get_mask = get_mask
        if self.get_mask:
            masks_dir = osp.join(self.root_dir, 'masks')
            ydirs = get_files_of_extension(masks_dir, '.jpg', '.png')
            self.y = pmap(_read_mask, ydirs, show_pbar=True)
            _sort_by_name_inplace(self.y)

        self.img_transform, self.mask_transform, self.pair_transform = None, None, None
        if transform is not None:
            self.img_transform = transform.get('img')
            self.mask_transform = transform.get('mask')
            self.pair_transform = transform.get('pair')

        self.print_info()

    def print_info(self):
        print(f'Found {len(self)} samples.')
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x, _ = self.x[idx]
        if self.img_transform is not None:
            x = self.img_transform(x)

        if self.get_mask:
            y, _ = self.y[idx]
            if self.mask_transform is not None:
                y = self.mask_transform(y)
            if self.pair_transform is not None:
                x, y = self.pair_transform([x, y])
            return x, y

        return x
