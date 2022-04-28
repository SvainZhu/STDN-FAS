import os
import csv
import json
import random
import pandas as pd

def listdir(path, sample_num, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, sample_num, list_name)
        else:
            for file in random.sample(os.listdir(path), sample_num):
                file_path = os.path.join(path, file)
                list_name.append(file_path)

def base_process(image_dir, map_dir, image_csv, map_csv, sample_num, label_identifier):
    map_csv = open(map_csv, 'a', encoding='utf-8', newline='')
    map_csv_writer = csv.writer(map_csv)
    with open(image_csv, 'a', encoding='utf-8', newline='') as f:
        image_csv_writer = csv.writer(f)
        images_list= []
        listdir(image_dir, sample_num, images_list)
        for image_path in images_list:
            if label_identifier in range(-1, 2):
                label = label_identifier
            else:
                label = 1 if label_identifier in image_path else 0
            map_path = image_path.replace(image_dir, map_dir)
            map_path = map_path.replace('-', '_')
            map_path = map_path.split('.')[0] + '_depth1D.jpg'
            image_csv_writer.writerow([image_path, label])
            map_csv_writer.writerow([map_path, label])

    map_csv.close()


def Oulu_process(image_root, protocol, sub_protocol, crop_size, sample_num):
    image_root = os.path.join(image_root, crop_size)
    map_root = image_root.replace('Image', 'Face_Depth_Map')

    image_dir, map_dir, protocol_dir, image_csv_dir, map_csv_dir = {}, {}, {}, {}, {}
    for dataset_name in ['Train_files', 'Dev_files', 'Test_files']:
        image_dir[dataset_name] = os.path.join(image_root, dataset_name)
        map_dir[dataset_name] = os.path.join(map_root, dataset_name)

    for dataset_name in ['Train', 'Dev', 'Test']:
        protocol_dir = 'E:/zsw/Data/OULU/Protocols/Protocol_%s/%s%s.txt' % (protocol, dataset_name, sub_protocol)
        image_csv_dir = r'E:/zsw/Data/OULU/CSV_RS/%s/%s_%s%s_%s.csv' % (crop_size, dataset_name.lower(), protocol, sub_protocol, sample_num)
        map_csv_dir = r'E:/zsw/Data/OULU/CSV_RS/%s/%s_map_%s%s_%s.csv' % (
        crop_size, dataset_name.lower(), protocol, sub_protocol, sample_num)

        set = pd.read_csv(protocol_dir, delimiter=',', header=None)
        for i in range(len(set)):
            video_name = str(set.iloc[i, 1])
            labels = int(set.iloc[i, 0])
            dataset_name_s = dataset_name+'_files'
            images_path = os.path.join(image_dir[dataset_name_s], video_name)
            map_path = os.path.join(map_dir[dataset_name_s], video_name)
            base_process(images_path, map_path, image_csv_dir, map_csv_dir, sample_num, labels)


def Siw_process(image_root, protocol, sub_protocol, crop_size, sample_num):
    image_root = os.path.join(image_root, crop_size)
    map_root = image_root.replace('Image', 'Face_Depth_Map')

    image_dir, map_dir, protocol_dir, image_csv_dir, map_csv_dir = {}, {}, {}, {}, {}
    for dataset_name in ['Train', 'Test']:
        image_dir[dataset_name] = os.path.join(image_root, dataset_name)
        map_dir[dataset_name] = os.path.join(map_root, dataset_name)

    for dataset_name in ['Train', 'Test']:
        protocol_dir = 'E:/zsw/Data/OULU/Protocols/Protocol_%s/%s%s.txt' % (protocol, dataset_name, sub_protocol)
        image_csv_dir = r'E:/zsw/Data/OULU/CSV_RS/%s/%s_%s%s_%s.csv' % (
        crop_size, dataset_name.lower(), protocol, sub_protocol, sample_num)
        map_csv_dir = r'E:/zsw/Data/OULU/CSV_RS/%s/%s_map_%s%s_%s.csv' % (
            crop_size, dataset_name.lower(), protocol, sub_protocol, sample_num)

        set = pd.read_csv(protocol_dir, delimiter=',', header=None)
        for i in range(len(set)):
            video_name = str(set.iloc[i, 1])
            labels = int(set.iloc[i, 0])
            dataset_name_s = dataset_name + '_files'
            images_path = os.path.join(image_dir[dataset_name_s], video_name)
            map_path = os.path.join(map_dir[dataset_name_s], video_name)
            base_process(images_path, map_path, image_csv_dir, map_csv_dir, sample_num, labels)