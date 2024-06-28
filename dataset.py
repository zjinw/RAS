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

        image, nrrd_options = nrrd.read()
        label, nrrd_options = nrrd.read()


        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample






