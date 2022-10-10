import os
import torch
import cv2
from .load_train import ImageLabelFileList_train as train_dataload
from .load_valtest import ImageLabelFileList_valtest as valtest_dataload


class data_merge(object):

    def __init__(self, image_dir):
        self.dic = {}
        self.image_dir = image_dir
        CASIA_MFSD_root = os.path.join(self.image_dir, "CASIA_FASD")
        self.dic["CASIA_MFSD"] = CASIA_MFSD_root
        # Replay_attack
        RE_root = os.path.join(self.image_dir, "REPLAY_ATTACK")
        self.dic["REPLAY_ATTACK"] = CASIA_MFSD_root
        # MSU_MFSD
        CASIA_MFSD_root = os.path.join(self.image_dir, "MSU_MFSD")
        self.dic["MSU_MFSD"] = CASIA_MFSD_root
        # OULU
        CASIA_MFSD_root = os.path.join(self.image_dir, "OULU")
        self.dic["OULU"] = CASIA_MFSD_root

    def get_single_dataset(self, data_name="", train=True, img_size=256, map_size=32, transform=None, debug_subset_size=None, UUID=-1):
        if train:
            data_dir = self.dic[data_name]
            if data_name in ["OULU", "CASIA_MFSD", "Replay_attack", "MSU_MFSD"]:
                data_set = train_dataload(os.path.join(data_dir, "CSV/2.0/train_1_6.csv"), transform=transform, img_size=img_size, map_size=map_size)
            else:
                raise("Load data Error!!!")
            if debug_subset_size is not None:
                data_set = torch.utils.data.Subset(data_set, range(0, debug_subset_size))
        else:
            data_dir = self.dic[data_name]
            if data_name in ["OULU", "CASIA_MFSD", "Replay_attack", "MSU_MFSD"]:
                data_set = valtest_dataload(os.path.join(data_dir, "CSV/2.0/test_1_6.csv"), transform=transform,
                                          img_size=img_size, map_size=map_size)
            else:
                raise ("Load data Error!!!")
            if debug_subset_size is not None:
                data_set = torch.utils.data.Subset(data_set, range(0, debug_subset_size))
        print("Loading {}, number: {}".format(data_name, len(data_set)))
        return data_set

    def get_datasets(self, train=True, protocol="1", img_size=256, map_size=32, transform=None, debug_subset_size=None):
        if protocol == "O_C_I_to_M":
            data_name_list_train = ["OULU", "CASIA_MFSD", "Replay_attack"]
            data_name_list_test = ["MSU_MFSD"]
        elif protocol == "O_M_I_to_C":
            data_name_list_train = ["OULU", "MSU_MFSD", "Replay_attack"]
            data_name_list_test = ["CASIA_MFSD"]
        elif protocol == "O_C_M_to_I":
            data_name_list_train = ["OULU", "CASIA_MFSD", "MSU_MFSD"]
            data_name_list_test = ["Replay_attack"]
        elif protocol == "I_C_M_to_O":
            data_name_list_train = ["MSU_MFSD", "CASIA_MFSD", "Replay_attack"]
            data_name_list_test = ["OULU"] 
        elif protocol == "M_I_to_C":
            data_name_list_train = ["MSU_MFSD", "Replay_attack"]
            data_name_list_test = ["CASIA_MFSD"]
        elif protocol == "M_I_to_O":
            data_name_list_train = ["MSU_MFSD", "Replay_attack"]
            data_name_list_test = ["OULU"]
        sum_n = 0
        data_set_sum = {}
        if train:
            for i in range(len(data_name_list_train)):
                data_tmp = self.get_single_dataset(data_name=data_name_list_train[i], train=True, img_size=img_size, map_size=map_size, transform=transform, debug_subset_size=debug_subset_size, UUID=i)
                data_set_sum[data_name_list_train[i]] = data_tmp
                sum_n += len(data_tmp)
        else:
            for i in range(len(data_name_list_test)):
                data_tmp = self.get_single_dataset(data_name=data_name_list_test[i], train=False, img_size=img_size, map_size=map_size, transform=transform, debug_subset_size=debug_subset_size, UUID=i)
                data_set_sum[data_name_list_test[i]] = data_tmp
                sum_n += len(data_tmp)
        print("Total number: {}".format(sum_n))
        return data_set_sum
