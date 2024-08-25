import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF


class NPY_datasets(Dataset):
    def __init__(self, path_Data, train=True):
        super(NPY_datasets, self)
        self.train=train
        if train:
            images_list = os.listdir(path_Data + 'train/images/')
            masks_list = os.listdir(path_Data + 'train/masks/')
            images_list = sorted(images_list)
            masks_list = sorted(masks_list)
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data + 'train/images/' + images_list[i]
                mask_path = path_Data + 'train/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = transforms.Compose([
                myNormalize('isic18', train=True),
                myToTensor(),
                myRandomHorizontalFlip(p=0.5),
                myRandomVerticalFlip(p=0.5),
                myRandomRotation(p=0.5, degree=[0, 360]),
                myResize(256, 256)
            ])
        else:
            images_list = os.listdir(path_Data + 'val/images/')
            masks_list = os.listdir(path_Data + 'val/masks/')
            images_list = sorted(images_list)
            masks_list = sorted(masks_list)
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data + 'val/images/' + images_list[i]
                mask_path = path_Data + 'val/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = transforms.Compose([
                myNormalize('isic18', train=False),
                myToTensor(),
                myResize(256, 256)
            ])

    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
        img, msk = self.transformer((img, msk))
        if self.train:
            return img, msk
        else:
            return img, msk, Path(img_path).stem

    def __len__(self):
        return len(self.data)


class myToTensor:
    def __init__(self):
        pass

    def __call__(self, data):
        image, mask = data
        return torch.tensor(image).permute(2, 0, 1), torch.tensor(mask).permute(2, 0, 1)


class myResize:
    def __init__(self, size_h=256, size_w=256):
        self.size_h = size_h
        self.size_w = size_w

    def __call__(self, data):
        image, mask = data
        return TF.resize(image, [self.size_h, self.size_w]), TF.resize(mask, [self.size_h, self.size_w])


class myRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            return TF.hflip(image), TF.hflip(mask)
        else:
            return image, mask


class myRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            return TF.vflip(image), TF.vflip(mask)
        else:
            return image, mask


class myRandomRotation:
    def __init__(self, p=0.5, degree=[0, 360]):
        self.angle = random.uniform(degree[0], degree[1])
        self.p = p

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            return TF.rotate(image, self.angle), TF.rotate(mask, self.angle)
        else:
            return image, mask


class myNormalize:
    def __init__(self, data_name, train=True):
        if data_name == 'isic18':
            if train:
                self.mean = 157.561
                self.std = 26.706
            else:
                self.mean = 149.034
                self.std = 32.022
        elif data_name == 'isic17':
            if train:
                self.mean = 159.922
                self.std = 28.871
            else:
                self.mean = 148.429
                self.std = 25.748
        else:
            if train:
                self.mean = 156.2899
                self.std = 26.5457
            else:
                self.mean = 149.8485
                self.std = 35.3346

    def __call__(self, data):
        img, msk = data
        img_normalized = (img - self.mean) / self.std
        img_normalized = ((img_normalized - np.min(img_normalized))
                          / (np.max(img_normalized) - np.min(img_normalized))) * 255.
        return img_normalized, msk
