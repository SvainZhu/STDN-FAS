import os
import torch
import pandas as pd
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
import math
import os 
from glob import glob


def crop_face_from_scene(image, scale):
    y1,x1,w,h = 0, 0, image.shape[1], image.shape[0]
    y2=y1+w
    x2=x1+h
    y_mid=(y1+y2)/2.0
    x_mid=(x1+x2)/2.0
    w_scale=scale/1.5*w
    h_scale=scale/1.5*h
    h_img, w_img = image.shape[0], image.shape[1]
    y1=y_mid-w_scale/2.0
    x1=x_mid-h_scale/2.0
    y2=y_mid+w_scale/2.0
    x2=x_mid+h_scale/2.0
    y1=max(math.floor(y1),0)
    x1=max(math.floor(x1),0)
    y2=min(math.floor(y2),w_img)
    x2=min(math.floor(x2),h_img)
    region=image[x1:x2,y1:y2]
    return region


class ImageLabelFileList_train(Dataset):
    
    def __init__(self, root_csv, transform=None, scale_up=1.5, scale_down=1.0, img_size=256, map_size=32, UUID=-1):
        self.images = pd.read_csv(root_csv, delimiter=",", header=None)
        self.transform = transform
        self.scale_up = scale_up
        self.scale_down = scale_down
        self.img_size = img_size
        self.map_size = map_size
        self.UUID = UUID

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        impath, mappath, label = self.images.iloc[index]
        sample = self.read_image_x(impath, mappath, label)
        if self.transform:
            sample = self.transform(sample)
        sample["UUID"] = self.UUID
        return sample

    def read_image_x(self, impath, mappath, label):
        image, map = cv2.imread(impath, cv2.IMREAD_COLOR), cv2.imread(mappath, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        face_scale = np.random.randint(int(self.scale_down*10), int(self.scale_up*10))
        face_scale = face_scale/10.0
        if label == 1:
            map = cv2.resize(crop_face_from_scene(map, face_scale), (self.map_size, self.map_size))
        else:
            map = np.zeros((self.map_size, self.map_size))
        # RGB
        image = cv2.resize(crop_face_from_scene(image, face_scale), (self.img_size, self.img_size))

        sample = {'image_x': image, 'map_x': map, 'label': label}
        return sample