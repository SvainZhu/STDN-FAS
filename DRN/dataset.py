import torch.utils.data as data
import os.path
import csv
from PIL import Image

def default_loader(path):
    return Image.open(path).convert('RGB')


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)

    return imlist


def default_csv_reader(csvfile):
    imlist, label_list = [], []
    with open(csvfile, 'r') as rf:
        csv_reader = csv.reader(rf)
        for line in csv_reader:
            imlist.append(line[0])
            label_list.append(line[1])
    return imlist, label_list


class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        impath = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.imlist)


class ImageLabelFilelist(data.Dataset):
    def __init__(self, images_r, images_s, transform=None, loader=default_loader):
        self.images_r = images_r
        self.images_s = images_s
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        impath, mappath, label = self.images_r[index]
        image_r, map_r = self.loader(impath), Image.open(mappath).convert('L').resize((32, 32))
        if self.transform is not None:
            image_r, map_r = self.transform(image_r), self.transform(map_r)

        impath, mappath, label = self.images_s[index]
        image_s, map_s = self.loader(impath), Image.open(mappath).convert('L').resize((32, 32))
        if self.transform is not None:
            image_s, map_s = self.transform(image_s), self.transform(map_s)
        return image_r, map_r, image_s, map_s

    def __len__(self):
        return len(self.images_r)


class ImageFileCsv(data.Dataset):
    def __init__(self, csv_file, transform=None,
                 csv_reader=default_csv_reader, loader=default_loader):
        self.imlist, _ = csv_reader(csv_file)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        impath = self.imlist[index]
        img = self.loader(impath)
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.imlist)


class ImageLabelFileCsv(data.Dataset):
    def __init__(self, csv_file, transform=None,
                 csv_reader=default_csv_reader, loader=default_loader):
        self.imlist, self.label_list = csv_reader(csv_file)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        impath = self.imlist[index]
        label = self.label_list[index]
        img = self.loader(impath)
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imlist)


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = sorted(make_dataset(root))
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in: " + root + "\n"
                                                               "Supported image extensions are: " +
                                ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)