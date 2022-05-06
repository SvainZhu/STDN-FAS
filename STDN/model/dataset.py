import numpy as np
from PIL import Image
from torch.utils import data
import torch
import cv2
import math
import os
import random
from .warp import generate_offset_map

## ---------------------- Dataloaders ---------------------- ##
# class Dataset_Csv(data.Dataset):
#     "Characterizes a dataset for PyTorch"
# 
#     def __init__(self, config, li_data_samples, sp_data_samples, labels, mode='train', transform=None):
#         "Initialization"
#         self.config = config
#         self.li_data_samples = li_data_samples
#         self.sp_data_samples = sp_data_samples
#         self.labels = labels
#         self.mode = mode
#         self.transform = transform
# 
#     def __len__(self):
#         "Denotes the total number of samples"
#         return len(self.li_data_samples)
# 
#     def __getitem__(self, index):
#         "Generates one sample of data"
# 
#         imsize = self.config.IMAGE_SIZE
# 
#         if self.mode == 'test':
#             im_name = self.li_data_samples[index]
#             image = Image.open(im_name)
#             image = image.resize((imsize, imsize))
#             image = np.array(image, dtype=np.float32)
#             image = torch.from_numpy(image / 255)
#             im_label = int(self.labels[index])
# 
#             return image, im_name, im_label
# 
#         else:
#             lm_reverse_list = np.array([17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
#                                         27, 26, 25, 24, 23, 22, 21, 20, 19, 18,
#                                         28, 29, 30, 31, 36, 35, 34, 33, 32,
#                                         46, 45, 44, 43, 48, 47, 40, 39, 38, 37, 42, 41,
#                                         55, 54, 53, 52, 51, 50, 49, 60, 59, 58, 57, 56, 65, 64, 63, 62, 61, 68, 67, 66],
#                                        np.int32) - 1
# 
#             # live
#             im_name = self.li_data_samples[index]
#             lm_name = im_name[:-3] + 'npy'
#             image = Image.open(im_name)
#             width, height = image.size
#             image_li = image.resize((imsize, imsize))
#             image_li = np.array(image_li, np.float32)
#             lm_li = np.load(lm_name) / width
#             if np.random.rand() > 0.5:
#                 image_li = cv2.flip(image_li, 1)
#                 lm_li[:, 0] = 1 - lm_li[:, 0]
#                 lm_li = lm_li[lm_reverse_list, :]
# 
#             # spoof
#             im_name = self.sp_data_samples[index]
#             lm_name = im_name[:-3] + 'npy'
#             image = Image.open(im_name)
#             width, height = image.size
#             image_sp = image.resize((imsize, imsize))
#             image_sp = np.array(image_sp, np.float32)
#             lm_sp = np.load(lm_name) / width
#             if np.random.rand() > 0.5:
#                 image_sp = cv2.flip(image_sp, 1)
#                 lm_sp[:, 0] = 1 - lm_sp[:, 0]
#                 lm_sp = lm_sp[lm_reverse_list, :]
# 
#             # offset map
#             reg_map_sp = torch.from_numpy(generate_offset_map(lm_sp, lm_li).astype(np.float32))
# 
#             image_sp = torch.from_numpy(np.array(image_sp, np.float32) / 255)
#             image_li = torch.from_numpy(np.array(image_li, np.float32) / 255)
# 
#             image = torch.stack([image_li, image_sp], dim=0)
# 
#             return image, reg_map_sp


class Dataset_Csv_train(data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, config, li_data_samples, sp_data_samples,  transform=None):
        "Initialization"
        self.config = config
        self.li_data_samples = li_data_samples
        self.sp_data_samples = sp_data_samples
        self.transform = transform

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.li_data_samples)

    def __getitem__(self, index):
        "Generates one sample of data"

        imsize = self.config.IMAGE_SIZE
        lm_reverse_list = np.array([17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
                                    27, 26, 25, 24, 23, 22, 21, 20, 19, 18,
                                    28, 29, 30, 31, 36, 35, 34, 33, 32,
                                    46, 45, 44, 43, 48, 47, 40, 39, 38, 37, 42, 41,
                                    55, 54, 53, 52, 51, 50, 49, 60, 59, 58, 57, 56, 65, 64, 63, 62, 61, 68, 67, 66],
                                   np.int32) - 1

        # live
        im_name = self.li_data_samples[index]
        lm_name = im_name[:-3] + 'npy'
        image = Image.open(im_name)
        width, height = image.size
        image_li = image.resize((imsize, imsize))
        image_li = np.array(image_li, np.float32)
        lm_li = np.load(lm_name) / width
        if np.random.rand() > 0.5:
            image_li = cv2.flip(image_li, 1)
            lm_li[:, 0] = 1 - lm_li[:, 0]
            lm_li = lm_li[lm_reverse_list, :]

        # spoof
        im_name = self.sp_data_samples[index]
        lm_name = im_name[:-3] + 'npy'
        image = Image.open(im_name)
        width, height = image.size
        image_sp = image.resize((imsize, imsize))
        image_sp = np.array(image_sp, np.float32)
        lm_sp = np.load(lm_name) / width
        if np.random.rand() > 0.5:
            image_sp = cv2.flip(image_sp, 1)
            lm_sp[:, 0] = 1 - lm_sp[:, 0]
            lm_sp = lm_sp[lm_reverse_list, :]

        # offset map
        reg_map_sp = torch.from_numpy(generate_offset_map(lm_sp, lm_li).astype(np.float32))

        image_sp = torch.from_numpy(np.array(image_sp, np.float32) / 255)
        image_li = torch.from_numpy(np.array(image_li, np.float32) / 255)

        image = torch.stack([image_li, image_sp], dim=0)

        return image, reg_map_sp


class Dataset_Csv_test(data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, config, data_samples,  labels, transform=None):
        "Initialization"
        self.config = config
        self.data_samples = data_samples
        self.transform = transform
        self.labels = labels

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.data_samples)

    def __getitem__(self, index):
        "Generates one sample of data"

        imsize = self.config.IMAGE_SIZE

        im_name = self.data_samples[index]
        image = Image.open(im_name)
        image = image.resize((imsize, imsize))
        image = np.array(image, np.float32)
        image = torch.from_numpy(np.array(image, np.float32) / 255)
        im_label = int(self.labels[index])


        return image, im_name, im_label

## ---------------------- end of Dataloaders ---------------------- ##
