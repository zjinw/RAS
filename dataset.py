import torch
import numpy as np
import cv2 as cv
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import nrrd
import os


class RAHeart(Dataset):
    """ RA Dataset """

    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []

        if split == 'train':
            with open(self._base_dir + 'train.list', 'r') as f:
                self.image_list = f.readlines()
        elif split == 'vaild':
            with open(self._base_dir + 'vaild.list', 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.strip() for item in self.image_list]

        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]

        image, nrrd_options = nrrd.read() #image_path
        label, nrrd_options = nrrd.read() #label_path

        if image.shape != [240, 160, 96]:
            w, h, d = image.shape
            label = np.pad(label, ((int((240 - w) / 2), int((240 - w) / 2)),
                                   (int((160 - h) / 2), int((160 - h) / 2)),
                                   (int((96 - d) / 2), int((96 - d) / 2))),
                           'constant',
                           constant_values=0)
            image = np.pad(image, ((int((240 - w) / 2), int((240 - w) / 2)),
                                   (int((160 - h) / 2), int((160 - h) / 2)),
                                   (int((96 - d) / 2), int((96 - d) / 2))),
                           'constant',
                           constant_values=0)

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample






