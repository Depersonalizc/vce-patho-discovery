import os
import os.path as osp
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset



class KvasirCapsuleVideo(Dataset):

  def __init__(self, video_dir, video_id, metadata_dir, to_ram=False, transform=None):
    self.video_id = video_id
    self.frames_dir = osp.join(video_dir, video_id)
    self.metadata_dir = metadata_dir
    self.to_ram = to_ram
    self.transform = transform

    self.frame_names = os.listdir(self.frames_dir)
    self.frame_names.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    self.read_metadata()
    if self.to_ram:
      self.read_frames()

  def __len__(self):
    return len(self.frame_names)

  def __getitem__(self, idx):
    frame_number = idx + 1
    ret = {
        'frame_number': frame_number,
        'name': self.frame_names[idx]
    }
    
    # get frame
    if self.to_ram:
      frame = self.frames[idx]
    else:
      frame_dir = osp.join(self.frames_dir, self.frame_names[idx])
      frame = cv2.imread(frame_dir)
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if self.transform is not None:
      frame = self.transform(frame)
    ret['frame'] = frame
      

    # get label
    ret['bbox'] = []
    if frame_number in self._frames_with_cls:
      row = self.metadata.loc[self.metadata['frame_number']==frame_number]
      ret['class'] = row['finding_class'].item()
      if not np.isnan(row['x1']).item():
        ret['bbox'] = [row['x1'], row['y1'], row['x3'], row['y3']]
    else:
      ret['class'] = 'Normal'

    return ret

  def read_metadata(self):
    metadata = pd.read_csv(self.metadata_dir)
    self.metadata = metadata.loc[metadata['video_id']==self.video_id]
    self._frames_with_cls = set(self.metadata['frame_number'])

  def read_frames(self):
    pass

