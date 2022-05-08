import torch.utils.data as data
import numpy as np
import cv2
import torch

class ImageLabelFilelist_train(data.Dataset):
    def __init__(self, images_r, images_s, transform=None):
        self.images_r = images_r
        self.images_s = images_s
        self.transform = transform

    def __getitem__(self, index):
        impath, mappath, label = self.images_r[index]
        image_r, map_r = cv2.imread(impath), cv2.imread(mappath, 0)
        sample_r = {'image': image_r, 'map': map_r, 'label': label}
        if self.transform is not None:
            sample_r = self.transform(sample_r)

        impath, mappath, label = self.images_s[index]
        image_s, map_s = cv2.imread(impath), cv2.imread(mappath, 0)
        sample_s = {'image': image_s, 'map': map_s, 'label': label}
        if self.transform is not None:
            sample_s = self.transform(sample_s)
        return sample_r['image'], sample_s['image']

    def __len__(self):
        return len(self.images_r)

class ImageLabelFilelist_test(data.Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __getitem__(self, index):
        image, map, label = self.images[index]
        image = cv2.imread(image)
        image = torch.from_numpy(np.array(image, np.float32) / 255.0)
        return image, int(label)

    def __len__(self):
        return len(self.images)
