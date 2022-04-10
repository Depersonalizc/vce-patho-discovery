import os
import os.path as osp
import cv2
import numpy as np
import pandas as pd
from concurrent import futures
from functools import partial
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
import torch

from utils.utils import pmap


def get_weighted_sampler(labels):
    cls_weights = 1. / np.bincount(labels)
    samp_weights = cls_weights[labels]
    sampler = WeightedRandomSampler(samp_weights, len(samp_weights))
    return sampler


class KvasirCapsuleBinary(Dataset):

    NORMAL_CLS = ['Normal clean mucosa', 'Pylorus', 
                  'Ampulla of Vatar', 'Ileocecal Valve']  # class 0

    def __init__(self, metadata_dir, images_dir, fold, transform=None, to_ram=True):

        self.fold = fold
        self.images_dir = images_dir
        self.metadata_dir = metadata_dir
        self.read_metadata()
        self.read_labels()
        self.to_ram = to_ram
        if self.to_ram:
            self.read_images()

        self.transform = transform

    def read_metadata(self):
        metadata = pd.read_csv(self.metadata_dir)
        video_ids = sorted(metadata['video_id'].unique())
        self.fold_video_ids = [video_ids[:22], video_ids[22:]][self.fold]
        self.metadata = metadata.loc[metadata['video_id'].isin(self.fold_video_ids)]

    def read_images(self):
        def get_tuple(idx):
            fname = self.metadata.iloc[idx, :].filename
            image = cv2.imread(osp.join(self.images_dir, fname))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image, idx

        tuples = pmap(get_tuple, range(len(self.metadata)), show_pbar=True)
        tuples = sorted(tuples, key=lambda tup: tup[1])
        self.images = [image for image, _ in tuples]  # (H, W, RGB)

    def read_labels(self):
        normal = self.metadata['finding_class'].isin(self.NORMAL_CLS)
        self.labels = (~normal).to_numpy(dtype=np.uint8)  # 0: normal, 1: abnormal

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        if self.to_ram:
            image = self.images[index]
        else:
            fname = self.metadata.iloc[index, :].filename
            image = cv2.imread(osp.join(self.images_dir, fname))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image, label
