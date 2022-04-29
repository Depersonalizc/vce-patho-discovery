from functools import partial

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.transforms import functional as TF


class RandomFlip:
    """Random horizontal + vertical flip"""
    def __init__(self, ph=0.5, pv=0.5):
        self.ph = ph
        self.pv = pv

    def __call__(self, ab):
        a, b = ab
        if torch.rand(1) < self.ph:
            a, b = TF.hflip(a), TF.hflip(b)
        if torch.rand(1) < self.pv:
            a, b = TF.vflip(a), TF.vflip(b)
        return a, b


class RandomSE2:
    """Random rigid motions in SE(2)"""
    se2 = partial(TF.affine, scale=1.0, shear=[0.0, 0.0],
                  interpolation=TF.InterpolationMode.NEAREST)

    def __init__(self, degrees, trans_x, trans_y):
        self.degrees = degrees
        self.trans_x = trans_x
        self.trans_y = trans_y

    def __call__(self, ab):
        a, b = ab
        h, w = a.shape[-2:]
        angle = np.random.randint(-self.degrees, self.degrees+1)
        trans_x = int(np.random.uniform(-self.trans_x*w, self.trans_x*w))
        trans_y = int(np.random.uniform(-self.trans_y*h, self.trans_y*h))
        tsfm = partial(self.se2, angle=angle, translate=[trans_x, trans_y])
        a, b = tsfm(a), tsfm(b)
        return a, b
