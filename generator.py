import pandas as pd

from albumentations import Compose, Blur, OneOf, OpticalDistortion, GridDistortion, HorizontalFlip, ShiftScaleRotate, IAASharpen, IAAEmboss
from albumentations.pytorch import ToTensor
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import torch
import pydicom
import os
import cv2
import pandas as pd
import numpy as np
import pydicom
import cv2

class IntracranialDataset(Dataset):
    def __init__(self, df, base_path, training):
        self.df = df
        self.base_path = base_path
        self.training = training
        self.shape = (3, 512, 512)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        # Generate indexes of the batch
        name = self.df.index[idx]
        
        if (self.training):
            y = torch.tensor(self.df.values[idx])
            X = self._read(self.base_path + name + ".dcm")
            a = self.augment()
            X = a(image=X)["image"]
            X = X.reshape((3, 512, 512))

            return X, y
        else:
            X = self._read(self.base_path + name + ".dcm")
            a = self.augmentTest()
            X = a(image=X)["image"]
            X = X.reshape((3, 512, 512))

            return X

    def augment(self):
        return Compose([
            OpticalDistortion(distort_limit=0.02, shift_limit=0.02, border_mode=0, value=0, p=0.1),
            GridDistortion(num_steps=9, distort_limit=0.1, border_mode=0, value=0, p=0.1),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=8, p=.25),
            OneOf([
                IAASharpen(),
                IAAEmboss(),
            ], p=0.1),
            HorizontalFlip(p=0.6),
            ToTensor()
        ], p=0.7)

    def augmentTest(self):
        return Compose([
            OpticalDistortion(distort_limit=0.02, shift_limit=0.02, border_mode=0, value=0, p=0.1),
            GridDistortion(num_steps=9, distort_limit=0.1, border_mode=0, value=0, p=0.1),
            ShiftScaleRotate(shift_limit=0.03125, scale_limit=0.05, rotate_limit=4, p=.15),
            OneOf([
                IAASharpen(),
                IAAEmboss(),
            ], p=0.05),
            HorizontalFlip(p=0.6),
            ToTensor()
        ], p=0.9)

    def correct_dcm(self, dcm):
        x = dcm.pixel_array + 1000
        px_mode = 4096
        x[x>=px_mode] = x[x>=px_mode] - px_mode
        dcm.PixelData = x.tobytes()
        dcm.RescaleIntercept = -1000

    def window_image(self, dcm, window_center, window_width):    
        if (dcm.BitsStored == 12) and (dcm.PixelRepresentation == 0) and (int(dcm.RescaleIntercept) > -100):
            self.correct_dcm(dcm)
        img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
        
        img = cv2.resize(img, self.shape[1:], interpolation=cv2.INTER_LINEAR)
       
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        img = np.clip(img, img_min, img_max)
        return img

    def bsb_window(self, dcm):
        brain_img = self.window_image(dcm, 40, 80)
        subdural_img = self.window_image(dcm, 80, 200)
        soft_img = self.window_image(dcm, 40, 380)
        
        brain_img = (brain_img - 0) / 80
        subdural_img = (subdural_img - (-20)) / 200
        soft_img = (soft_img - (-150)) / 380
        bsb_img = np.array([brain_img, subdural_img, soft_img]).transpose(1,2,0)
        return bsb_img

    def _read(self, path):
        dcm = pydicom.dcmread(path)
        try:
            img = self.bsb_window(dcm)
        except ValueError:
            img = np.zeros((3, 512, 512))
        return img
