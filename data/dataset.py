import os
import os.path as osp
import cv2
import numpy as np
import pandas as pd
from concurrent import futures
from functools import partial
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler

from utils.utils import pmap


def get_weighted_sampler(labels):
    cls_weights = 1. / np.bincount(labels)
    samp_weights = cls_weights[labels]
    sampler = WeightedRandomSampler(samp_weights, len(samp_weights))
    return sampler


class KvasirCapsuleBinary(Dataset):

    NORMAL_CLS = ['Normal clean mucosa', 'Pylorus', 
                  'Ampulla of Vatar', 'Ileocecal Valve']  # class 0

    def __init__(self, metadata_dir, images_dir, fold, transform=None):

        self.fold = fold
        self.images_dir = images_dir
        self.metadata_dir = metadata_dir
        self.read_metadata()

        self.get_labels()
        self.get_images()

        self.transform = transform

    def read_metadata(self):
        metadata = pd.read_csv(self.metadata_dir)
        video_ids = sorted(metadata['video_id'].unique())
        self.fold_video_ids = [video_ids[:22], video_ids[22:]][self.fold]
        self.metadata = metadata.loc[metadata['video_id'].isin(self.fold_video_ids)]

    def get_images(self):
        def get_tuple(idx):
            fname = self.metadata.iloc[idx, :].filename
            image = cv2.imread(osp.join(self.images_dir, fname))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image, idx

        tuples = pmap(get_tuple, range(len(self.metadata)), show_pbar=True)
        tuples = sorted(tuples, key=lambda tup: tup[1])
        self.images = [image for image, _ in tuples]  # (H, W, RGB)

    def get_labels(self):
        normal = self.metadata['finding_class'].isin(self.NORMAL_CLS)
        self.labels = (~normal).to_numpy(dtype=np.uint8)  # 0: normal, 1: abnormal

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label
    

class KvasirCapsuleBinaryTest(Dataset):

    NORMAL_CLS = ['Normal clean mucosa', 'Pylorus', 
                  'Ampulla of Vatar', 'Ileocecal Valve']  # class 0

    def __init__(self, metadata_dir, images_dir, fold, to_ram = True,transform=None):

        self.fold = fold
        self.images_dir = images_dir
        self.metadata_dir = metadata_dir
        self.read_metadata()
        self.to_ram = to_ram
        self.get_labels()
        if self.to_ram:
            
            self.get_images()

        self.transform = transform

    def read_metadata(self):
        metadata = pd.read_csv(self.metadata_dir)
        video_ids = sorted(metadata['video_id'].unique())
        self.fold_video_ids = [video_ids[:22], video_ids[22:]][self.fold]
        self.metadata = metadata.loc[metadata['video_id'].isin(self.fold_video_ids)]

    def get_images(self):
        def get_tuple(idx):
            fname = self.metadata.iloc[idx, :].filename
            image = cv2.imread(osp.join(self.images_dir, fname))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image, idx

        tuples = pmap(get_tuple, range(len(self.metadata)), show_pbar=True)
        tuples = sorted(tuples, key=lambda tup: tup[1])
        self.images = [image for image, _ in tuples]  # (H, W, RGB)

    def get_labels(self):
        normal = self.metadata['finding_class'].isin(self.NORMAL_CLS)
        self.labels = (~normal).to_numpy(dtype=np.uint8)  # 0: normal, 1: abnormal

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.to_ram:
            image, label = self.images[index], self.labels[index]
        else:
            fname = self.metadata.iloc[index, :].filename
            image = cv2.imread(osp.join(self.images_dir, fname))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            label = self.metadata.iloc[index,:]['finding_class'] in self.NORMAL_CLS
        if self.transform:
            image = self.transform(image)
        return image, label
