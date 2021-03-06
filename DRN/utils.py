from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import transforms
# from dataset import ImageFilelist, ImageFolder, ImageFileCsv, ImageLabelFileCsv, ImageLabelFilelist
from dataset import ImageLabelFilelist_train, ImageLabelFilelist_test
import torch
import os
import math
import torchvision.utils as vutils
import yaml
import numpy as np
import torch.nn.init as init
import time
import csv
import random
import cv2
# Methods
# get_all_data_loaders      : primary data loader interface (load trainA, testA, trainB, testB)
# get_data_loader_list      : list-based data loader
# get_data_loader_folder    : folder-based data loader
# get_data_loader_csv       : csv-based data loader
# get_config                : load yaml file
# eformat                   :
# write_2images             : save output image
# prepare_sub_folder        : create checkpoints and images folders for saving outputs
# write_loss
# slerp
# get_slerp_interp
# get_model_list
# get_scheduler
# weights_init

def get_all_data_loaders(conf):
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']
    if 'new_size' in conf:
        new_size_a = new_size_b = conf['new_size']
    else:
        new_size_a = conf['new_size_a']
        new_size_b = conf['new_size_b']
    height = conf['crop_image_height']
    width = conf['crop_image_width']

    # if 'data_root' in conf:
    #     train_loader_a = get_data_loader_folder(os.path.join(conf['data_root'], 'trainA'), batch_size, True,
    #                                           new_size_a, height, width, num_workers, True)
    #     test_loader_a = get_data_loader_folder(os.path.join(conf['data_root'], 'testA'), batch_size, False,
    #                                          new_size_a, new_size_a, new_size_a, num_workers, True)
    #     train_loader_b = get_data_loader_folder(os.path.join(conf['data_root'], 'trainB'), batch_size, True,
    #                                           new_size_b, height, width, num_workers, True)
    #     test_loader_b = get_data_loader_folder(os.path.join(conf['data_root'], 'testB'), batch_size, False,
    #                                          new_size_b, new_size_b, new_size_b, num_workers, True)
    # elif 'data_list_train_a' in conf:
    #     train_loader_a = get_data_loader_list(conf['data_folder_train_a'], conf['data_list_train_a'], batch_size, True,
    #                                             new_size_a, height, width, num_workers, True)
    #     test_loader_a = get_data_loader_list(conf['data_folder_test_a'], conf['data_list_test_a'], batch_size, False,
    #                                             new_size_a, new_size_a, new_size_a, num_workers, True)
    #     train_loader_b = get_data_loader_list(conf['data_folder_train_b'], conf['data_list_train_b'], batch_size, True,
    #                                             new_size_b, height, width, num_workers, True)
    #     test_loader_b = get_data_loader_list(conf['data_folder_test_b'], conf['data_list_test_b'], batch_size, False,
    #                                             new_size_b, new_size_b, new_size_b, num_workers, True)
    # else:
    train_loader = get_data_loader_csv(conf['data_csv_train'], batch_size, True, num_workers, 'train')
    test_loader = get_data_loader_csv(conf['data_csv_test'], batch_size, False, num_workers, 'test')
    return train_loader, test_loader


# def get_data_loader_list(root, file_list, batch_size, train, num_workers=4, crop=True):
#     transform_list = []
#     transform_list = transform_list + [RandomErasing(),] if crop else transform_list
#     transform_list = transform_list + [RandomHorizontalFlip(),] if train else transform_list
#     transform_list = transform_list + [ToTensor(),]
#     transform_list = transform_list + [Cutout(),] if crop else transform_list
#     transform_list = transform_list + [Normaliztion(),]
#     transform = transforms.Compose(transform_list)
#
#
#     dataset = ImageFilelist(root, file_list, transform=transform)
#     loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
#     return loader

# def get_data_loader_csv(csv_file, batch_size, train, new_size=None,
#                            height=256, width=256, num_workers=4, crop=True):
#     transform_list = [transforms.ToTensor(),
#                       transforms.Normalize((0.5, 0.5, 0.5),
#                                            (0.5, 0.5, 0.5))]
#     transform_list = [transforms.RandomCrop((height, width))] + transform_list if crop else transform_list
#     transform_list = [transforms.Resize(new_size)] + transform_list if new_size is not None else transform_list
#     transform_list = [transforms.RandomHorizontalFlip()] + transform_list if train else transform_list
#     transform = transforms.Compose(transform_list)
#     dataset = ImageFileCsv(csv_file, transform=transform)
#     loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
#     return loader

def get_data_loader_csv(csv_file, batch_size, train, num_workers=4, mode='train'):
    if mode == 'train':
        transform_list = []
        # transform_list = transform_list + [RandomErasing(), ] if crop else transform_list
        # transform_list = transform_list + [RandomHorizontalFlip(), ] if train else transform_list
        transform_list = transform_list + [ToTensor(), ]
        # transform_list = transform_list + [Cutout(), ] if crop else transform_list
        transform_list = transform_list + [Normaliztion()]
        transform = transforms.Compose(transform_list)

        list_r, list_s = [], []
        frame_reader = open(csv_file, 'r')
        csv_reader = csv.reader(frame_reader)

        for f in csv_reader:
            if int(f[2]) == 1:
                list_r.append(f)
            else:
                list_s.append(f)
        len_r = len(list_r)
        len_s = len(list_s)
        if len_r < len_s:
            while len(list_r) < len_s:
                list_r += random.sample(list_r, len(list_r))
            list_r = list_r[:len_s]
        elif len_r > len_s:
            while len(list_s) < len_r:
                list_s += random.sample(list_s, len(list_s))
            list_s = list_s[:len_r]

        dataset = ImageLabelFilelist_train(list_r, list_s, transform=transform)
    else:
        list_data = []
        frame_reader = open(csv_file, 'r')
        csv_reader = csv.reader(frame_reader)
        for f in csv_reader:
            list_data.append(f)
        dataset = ImageLabelFilelist_test(list_data, transform=None)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
    return loader

def get_data_loader_folder(input_folder, batch_size, train, new_size=None,
                           height=256, width=256, num_workers=4, crop=True):
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]
    transform_list = [transforms.RandomCrop((height, width))] + transform_list if crop else transform_list
    transform_list = [transforms.Resize(new_size)] + transform_list if new_size is not None else transform_list
    transform_list = [transforms.RandomHorizontalFlip()] + transform_list if train else transform_list
    transform = transforms.Compose(transform_list)
    dataset = ImageFolder(input_folder, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
    return loader


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)


def eformat(f, prec):
    s = "%.*e"%(prec, f)
    mantissa, exp = s.split('e')
    # add 1 to digits as 1 is taken by sign +/-
    return "%se%d"%(mantissa, int(exp))


def __write_images(image_outputs, display_image_num, file_name):
    image_outputs = [images.expand(-1, 3, -1, -1) for images in image_outputs] # expand gray-scale images to 3 channels
    image_tensor = torch.cat([images[:display_image_num] for images in image_outputs], 0)
    image_grid = vutils.make_grid(image_tensor.data, nrow=display_image_num, padding=0, normalize=True)
    vutils.save_image(image_grid, file_name, nrow=1)


def write_2images(image_outputs, display_image_num, image_directory, postfix):
    n = len(image_outputs)
    __write_images(image_outputs[0:n//2], display_image_num, '%s/gen_r2s_%s.jpg' % (image_directory, postfix))
    __write_images(image_outputs[n//2:n], display_image_num, '%s/gen_s2r_%s.jpg' % (image_directory, postfix))


def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory


def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer) \
               if not callable(getattr(trainer, attr)) and not attr.startswith("__") and ('loss' in attr or 'grad' in attr or 'nwd' in attr)]
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)


def slerp(val, low, high):
    """
    original: Animating Rotation with Quaternion Curves, Ken Shoemake
    https://arxiv.org/abs/1609.04468
    Code: https://github.com/soumith/dcgan.torch/issues/14, Tom White
    """
    omega = np.arccos(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


def get_slerp_interp(nb_latents, nb_interp, z_dim):
    """
    modified from: PyTorch inference for "Progressive Growing of GANs" with CelebA snapshot
    https://github.com/ptrblck/prog_gans_pytorch_inference
    """

    latent_interps = np.empty(shape=(0, z_dim), dtype=np.float32)
    for _ in range(nb_latents):
        low = np.random.randn(z_dim)
        high = np.random.randn(z_dim)  # low + np.random.randn(512) * 0.7
        interp_vals = np.linspace(0, 1, num=nb_interp)
        latent_interp = np.array([slerp(v, low, high) for v in interp_vals],
                                 dtype=np.float32)
        latent_interps = np.vstack((latent_interps, latent_interp))

    return latent_interps[:, :, np.newaxis, np.newaxis]


# Get model list for resume
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def get_scheduler(optimizer, hyperparameters, iterations=-1):
    if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
        scheduler = None # constant scheduler
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'], last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))



class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''

    def __init__(self, probability=0.5, sl=0.01, sh=0.05, r1=0.5, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, sample):
        img, map, label = sample['image'], sample['map'], sample['label']

        if random.uniform(0, 1) < self.probability:
            attempts = np.random.randint(1, 3)
            for attempt in range(attempts):
                area = img.shape[0] * img.shape[1]

                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, 1 / self.r1)

                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))

                if w < img.shape[1] and h < img.shape[0]:
                    x1 = random.randint(0, img.shape[0] - h)
                    y1 = random.randint(0, img.shape[1] - w)

                    img[x1:x1 + h, y1:y1 + w, 0] = self.mean[0]
                    img[x1:x1 + h, y1:y1 + w, 1] = self.mean[1]
                    img[x1:x1 + h, y1:y1 + w, 2] = self.mean[2]

        return {'image': img, 'map': map, 'label': label}


# Tensor
class Cutout(object):
    def __init__(self, length=50):
        self.length = length

    def __call__(self, sample):
        img, map, label = sample['image'], sample['map'], sample['label']
        h, w = img.shape[1], img.shape[2]  # Tensor [1][2],  nparray [0][1]
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        length_new = np.random.randint(1, self.length)

        y1 = np.clip(y - length_new // 2, 0, h)
        y2 = np.clip(y + length_new // 2, 0, h)
        x1 = np.clip(x - length_new // 2, 0, w)
        x2 = np.clip(x + length_new // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return {'image': img, 'map': map, 'label': label}


class Normaliztion(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """

    def __call__(self, sample):
        image, map, label = sample['image'], sample['map'], sample['label']
        new_image = image / 255.0  # [-1,1]
        new_map = map / 255.0  # [0,1]
        return {'image': new_image, 'map': new_map, 'label': label}


class RandomHorizontalFlip(object):
    """Horizontally flip the given Image randomly with a probability of 0.5."""

    def __call__(self, sample):
        image, map, label = sample['image'], sample['map'], sample['label']

        new_image = np.zeros((224, 224, 3))
        new_map = np.zeros((32, 32))

        p = random.random()
        if p < 0.5:
            # print('Flip')

            new_image = cv2.flip(image, 1)
            new_map = cv2.flip(map, 1)

            return {'image': new_image, 'map': new_map, 'label': label}
        else:
            # print('no Flip')
            return {'image': image, 'map': map, 'label': label}


class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        image, map, label = sample['image'], sample['map'], sample['label']

        # swap color axis because
        # numpy image: (batch_size) x H x W x C
        # torch image: (batch_size) x C X H X W
        image = image[:, :, ::-1].transpose((2, 0, 1))
        image = np.array(image)

        map = np.array(map)

        label_np = np.array([0], dtype=np.long)
        label_np[0] = label

        return {'image': torch.from_numpy(image.astype(np.float)).float(),
                'map': torch.from_numpy(map.astype(np.float)).float(),
                'label': torch.from_numpy(label_np.astype(np.long)).long()}

