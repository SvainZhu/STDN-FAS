import torch.utils.data as data
import numpy as np
import cv2
import torch
from PIL import Image

class ImageLabelFilelist_train(data.Dataset):
    def __init__(self, images_r, images_s, transform=None):
        self.images_r = images_r
        self.images_s = images_s
        self.transform = transform

    def __getitem__(self, index):
        impath, _, __ = self.images_r[index]
        image_r = Image.open(impath).convert('RGB')
        if self.transform is not None:
            image_r = self.transform(image_r)

        impath, _, __ = self.images_s[index]
        image_s = Image.open(impath).convert('RGB')
        if self.transform is not None:
            image_s = self.transform(image_s)
        return image_r, image_s

    def __len__(self):
        return len(self.images_r)

class ImageLabelFilelist_test(data.Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __getitem__(self, index):
        impath, _, label = self.images[index]
        image = Image.open(impath).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, int(label)

    def __len__(self):
        return len(self.images)
