import torch.utils.data as data
import numpy as np
import cv2
# from PIL import Image

class ImageLabelFilelist(data.Dataset):
    def __init__(self, images, mode='train', transform=None):
        self.images = images
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.mode == 'train':
            impath_r, mappath_r, label_r, impath_s, mappath_s, label_s = self.images[index]
            sample_r = self.read_images(impath_r, mappath_r, label_r)
            sample_s = self.read_images(impath_s, mappath_s, label_s)
            return sample_r['image'], sample_r['map'], sample_r['landmark'], sample_s['image'], sample_s['map'], sample_s['landmark']
            # return sample_r['image'], sample_r['map'], sample_r['landmark'], sample_s['image'], sample_s['map'], sample_s[
            #     'landmark']

        else:
            impath, mappath, label = self.images[index]
            sample = self.read_images(impath, mappath, label)
            return sample['image'], int(sample['label'])


    def read_images(self, im_path, map_path, label):
        image, map = cv2.imread(im_path, cv2.IMREAD_COLOR), cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        landmark = im_path.split('.jp')[0] + '.npy'
        landmark = np.load(landmark) / 256
        sample = {'image': image, 'map': map, 'landmark': landmark, 'label': label}

        if self.transform is not None:
            sample = self.transform(sample)
        return sample
