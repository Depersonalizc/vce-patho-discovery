# dataset for training ResUnet++ (RUPP)
# augmentation already done by scripts/preprocess.py

import os
import os.path as osp
from glob import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import read_rgb_255, read_gray_255

sglob = lambda x: sorted(glob(x))


class RUPP(Dataset):
    def __init__(self, 
                 root_dir='/content/data/kvasir_seg_new/Kvasir-SEG/',
                 split='train'):

        self.root_dir = root_dir
        self.split = split
        self.x = sglob(osp.join(self.root_dir, split, 'images', '*.jpg'))
        self.y = sglob(osp.join(self.root_dir, split, 'masks', '*.jpg'))
        self.print_info()

    def print_info(self):
        print(f'Found {len(self)} images and segmentation maps')
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = read_rgb_255(self.x[idx])
        y = read_gray_255(self.y[idx])

        x = x.astype(np.float32) / 255
        y = (y > 128)

        x = torch.from_numpy(x).movedim(-1, 0)
        y = torch.from_numpy(y)
        return x, y
