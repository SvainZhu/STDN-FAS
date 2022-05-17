import torch.utils.data as data
import numpy as np
import cv2
import torch
from PIL import Image

class ImageLabelFilelist(data.Dataset):
    def __init__(self, images, mode='train', transform=None):
        self.images = images
        self.mode = mode
        self.transform = transform

    def __getitem__(self, index):
        if self.mode == 'train':
            impath_r, _, label_r, impath_s, _, label_s = self.images[index]
            image_r = Image.open(impath_r).convert('RGB')
            if self.transform is not None:
                image_r = self.transform(image_r)
            image_s = Image.open(impath_s).convert('RGB')
            if self.transform is not None:
                image_s = self.transform(image_s)
            return image_r, image_s
        else:
            impath, _, label = self.images[index]
            image = Image.open(impath).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            return image, int(label)
    def __len__(self):
        return len(self.images)
