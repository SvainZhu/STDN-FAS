import os
import csv
import json
import random
import pandas as pd

interval = 6


def base_process(image_dir, map_dir, image_csv, map_csv):
    map_csv_a = open(map_csv, 'a', encoding='utf-8', newline='')
    map_csv_writer = csv.writer(map_csv_a)
    with open(image_csv, 'a', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        for i in os.listdir(map_dir):
            file_path1 = os.path.join(map_dir, i)
            if i == 'attack_face':
                label = 0
            else:
                label = 1
            for j in os.listdir(file_path1):
                file_path2 = os.path.join(file_path1, j)
                count = -1
                for k in os.listdir(file_path2):
                    count += 1
                    if count % interval != 0:
                        continue
                    face_name = '_'.join(k.split('_')[:-1]) + '.jpg'
                    map_path = os.path.join(file_path2, k)
                    image_path = os.path.join(os.path.join(image_dir, i), j)
                    face_path = os.path.join(image_path, face_name)
                    csv_writer.writerow([face_path, label])
                    map_csv_writer.writerow([map_path, label])

                else:
                    continue
    map_csv_a.close()


def Oulu_process(crop_size):
    Protocol = '1'
    sub_Protocol = ''

    train_image_dir = 'E:/zsw/Data/OULU/CropFace256/%s/Train_files/' % crop_size
    val_image_dir = 'E:/zsw/Data/OULU/CropFace256/%s/Dev_files/' % crop_size
    test_image_dir = 'E:/zsw/Data/OULU/CropFace256/%s/Test_files/' % crop_size

    train_map_dir = 'E:/zsw/Data/OULU/Face_Depth_Map/%s/Train_files/' % crop_size
    val_map_dir = 'E:/zsw/Data/OULU/Face_Depth_Map/%s/Dev_files/' % crop_size
    test_map_dir = 'E:/zsw/Data/OULU/Face_Depth_Map/%s/Test_files/' % crop_size

    train_list = 'E:/zsw/Data/OULU/Protocols/Protocol_%s/Train%s.txt' % (Protocol, sub_Protocol)
    val_list = 'E:/zsw/Data/OULU/Protocols/Protocol_%s/Dev%s.txt' % (Protocol, sub_Protocol)
    test_list = 'E:/zsw/Data/OULU/Protocols/Protocol_%s/Test%s.txt' % (Protocol, sub_Protocol)

    train_csv = r'E:/zsw/Data/OULU/CSV_rsf/%s/train_%s%s_%s.csv' % (
    crop_size, Protocol, sub_Protocol, interval)  # the train split file
    val_csv = r'E:/zsw/Data/OULU/CSV_rsf/%s/val_%s%s_%s.csv' % (
    crop_size, Protocol, sub_Protocol, interval)  # the validation split file
    test_csv = r'E:/zsw/Data/OULU/CSV_rsf/%s/test_%s%s_%s.csv' % (crop_size, Protocol, sub_Protocol, interval)

    train_map_csv = r'E:/zsw/Data/OULU/CSV_rsf/%s/train_map_%s%s_%s.csv' % (
        crop_size, Protocol, sub_Protocol, interval)  # the train split file
    val_map_csv = r'E:/zsw/Data/OULU/CSV_rsf/%s/val_map_%s%s_%s.csv' % (
        crop_size, Protocol, sub_Protocol, interval)  # the validation split file
    test_map_csv = r'E:/zsw/Data/OULU/CSV_rsf/%s/test_map_%s%s_%s.csv' % (crop_size, Protocol, sub_Protocol, interval)

    def oulu_base_process(image_dir, map_dir, list, image_csv, map_csv):
        set = pd.read_csv(list, delimiter=',', header=None)
        map_csv_a = open(map_csv, 'a', encoding='utf-8', newline='')
        map_csv_writer = csv.writer(map_csv_a)
        with open(image_csv, 'a', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            for i in range(len(set)):
                video_name = str(set.iloc[i, 1])
                labels = int(set.iloc[i, 0])
                if labels == 1:
                    labels = 1
                else:
                    labels = 0

                faces_name = os.listdir(os.path.join(image_dir, video_name))
                for face_name in random.sample(faces_name, 1):
                    face_name = face_name.split('.')[0] + '.jpg'
                    map_name = face_name.split('.')[0].replace('-', '_') + '_depth1D.jpg'
                    map_path = os.path.join(os.path.join(map_dir, video_name), map_name)
                    face_path = os.path.join(os.path.join(image_dir, video_name), face_name)
                    csv_writer.writerow([face_path, labels])
                    map_csv_writer.writerow([map_path, labels])
        map_csv_a.close()
        return 0

    oulu_base_process(image_dir=train_image_dir, map_dir=train_map_dir, list=train_list,
                      image_csv=train_csv, map_csv=train_map_csv)
    oulu_base_process(image_dir=val_image_dir, map_dir=val_map_dir, list=val_list,
                      image_csv=test_csv, map_csv=test_map_csv)
    # oulu_base_process(image_dir=test_image_dir, map_dir=test_map_dir, list=test_list,
    #                   image_csv=test_csv, map_csv=test_map_csv)


def SiW_process(crop_size):
    Protocol = '1'
    sub_Protocol = ''

    train_image_dir = 'E:/zsw/Data/SiW/CropFace256/%s/Train/' % crop_size
    test_image_dir = 'E:/zsw/Data/SiW/CropFace256/%s/Test/' % crop_size

    train_map_dir = 'E:/zsw/Data/SiW/Face_Depth_Map/%s/Train/' % crop_size
    test_map_dir = 'E:/zsw/Data/SiW/Face_Depth_Map/%s/Test/' % crop_size

    train_csv = r'E:/zsw/Data/SiW/CSV_rsf/%s/train_%s%s_%s.csv' % (
    crop_size, Protocol, sub_Protocol, interval)  # the train split file
    test_csv = r'E:/zsw/Data/SiW/CSV_rsf/%s/test_%s%s_%s.csv' % (crop_size, Protocol, sub_Protocol, interval)

    train_map_csv = r'E:/zsw/Data/SiW/CSV_rsf/%s/train_map_%s%s_%s.csv' % (
        crop_size, Protocol, sub_Protocol, interval)  # the train split file
    test_map_csv = r'E:/zsw/Data/SiW/CSV_rsf/%s/test_map_%s%s_%s.csv' % (crop_size, Protocol, sub_Protocol, interval)

    def siw_base_process(image_dir, map_dir, image_csv, map_csv, frames_num, type_ids, medium_ids):
        map_csv_a = open(map_csv, 'a', encoding='utf-8', newline='')
        map_csv_writer = csv.writer(map_csv_a)
        with open(image_csv, 'a', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            for i in os.listdir(map_dir):
                file_path1 = os.path.join(map_dir, i)
                if i == 'attack_face':
                    label = 0
                else:
                    label = 1
                for j in os.listdir(file_path1):
                    file_path2 = os.path.join(file_path1, j)
                    for k in os.listdir(file_path2):
                        property_list = k.split('-')
                        if int(property_list[2]) in type_ids and int(property_list[3]) in medium_ids:
                            file_path3 = os.path.join(file_path2, k)
                            count = -1
                            for map in os.listdir(file_path3):
                                count += 1
                                if count >= frames_num:
                                    break
                                if count % interval != 0:
                                    continue
                                face_name = '_'.join(map.split('_')[:-1]) + '.jpg'
                                map_path = os.path.join(file_path3, map)
                                face_path = os.path.join(os.path.join(os.path.join(os.path.join(image_dir, i), j), k),
                                                         face_name)
                                csv_writer.writerow([face_path, label])
                                map_csv_writer.writerow([map_path, label])

        map_csv_a.close()

    if Protocol == '2':
        train_medium_ids, test_medium_ids = [1, 2, 3, 4], []
        train_medium_ids.remove(int(sub_Protocol[1:]))
        test_medium_ids.append(int(sub_Protocol[1:]))
        siw_base_process(image_dir=train_image_dir, map_dir=train_map_dir, image_csv=train_csv, map_csv=train_map_csv,
                         frames_num=999999, type_ids=[3], medium_ids=train_medium_ids)
        siw_base_process(image_dir=test_image_dir, map_dir=test_map_dir, image_csv=test_csv, map_csv=test_map_csv,
                         frames_num=999999, type_ids=[3], medium_ids=test_medium_ids)
    elif Protocol == '3':
        if sub_Protocol == '_1':
            siw_base_process(image_dir=train_image_dir, map_dir=train_map_dir, image_csv=train_csv,
                             map_csv=train_map_csv,
                             frames_num=999999, type_ids=[2], medium_ids=[1, 2])
            siw_base_process(image_dir=test_image_dir, map_dir=test_map_dir, image_csv=test_csv, map_csv=test_map_csv,
                             frames_num=999999, type_ids=[3], medium_ids=[1, 2, 3, 4])
        else:
            siw_base_process(image_dir=train_image_dir, map_dir=train_map_dir, image_csv=train_csv,
                             map_csv=train_map_csv, frames_num=999999, type_ids=[3], medium_ids=[1, 2, 3, 4])
            siw_base_process(image_dir=test_image_dir, map_dir=test_map_dir, image_csv=test_csv, map_csv=test_map_csv,
                             frames_num=999999, type_ids=[2], medium_ids=[1, 2])
    else:
        siw_base_process(image_dir=train_image_dir, map_dir=train_map_dir, image_csv=train_csv,
                         map_csv=train_map_csv,
                         frames_num=60, type_ids=[1, 2], medium_ids=[1, 2])
        siw_base_process(image_dir=train_image_dir, map_dir=train_map_dir, image_csv=train_csv, map_csv=train_map_csv,
                         frames_num=60, type_ids=[3], medium_ids=[1, 2, 3, 4])
        siw_base_process(image_dir=test_image_dir, map_dir=test_map_dir, image_csv=test_csv,
                         map_csv=test_map_csv,
                         frames_num=999999, type_ids=[1, 2], medium_ids=[1, 2])
        siw_base_process(image_dir=test_image_dir, map_dir=test_map_dir, image_csv=test_csv, map_csv=test_map_csv,
                         frames_num=999999, type_ids=[3], medium_ids=[1, 2, 3, 4])


def CASIA_FASD_process(crop_size):
    Protocol = '1'  # 1: wrapped photo attack; 2: cut photo attack; 0: video attack
    train_map_dir = "E:/zsw/Data/CASIA_FASD/CASIA_FASD_DepthMap/%s/train_release/" % crop_size
    test_map_dir = "E:/zsw/Data/CASIA_FASD/CASIA_FASD_DepthMap/%s/test_release/" % crop_size

    train_image_dir = "E:/zsw/Data/CASIA_FASD/CASIA_FASD_CropFace256/%s/train_release/" % crop_size
    test_image_dir = "E:/zsw/Data/CASIA_FASD/CASIA_FASD_CropFace256/%s/test_release/" % crop_size

    train_csv = r'E:/zsw/Data/CASIA_FASD/CSV_rsf/%s/train_%s_%s.csv' % (crop_size, Protocol, interval)
    test_csv = r'E:/zsw/Data/CASIA_FASD/CSV_rsf/%s/test_%s_%s.csv' % (crop_size, Protocol, interval)

    train_map_csv = r'E:/zsw/Data/CASIA_FASD/CSV_rsf/%s/train_map_%s_%s.csv' % (crop_size, Protocol, interval)
    test_map_csv = r'E:/zsw/Data/CASIA_FASD/CSV_rsf/%s/test_map_%s_%s.csv' % (crop_size, Protocol, interval)

    def CASIA_FASD_base_process(image_dir, map_dir, image_csv, map_csv, type_id):
        map_csv_a = open(map_csv, 'a', encoding='utf-8', newline='')
        map_csv_writer = csv.writer(map_csv_a)
        with open(image_csv, 'a', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            for i in os.listdir(map_dir):
                file_path1 = os.path.join(map_dir, i)
                if i == 'attack_face':
                    label = 0
                else:
                    label = 1
                for j in os.listdir(file_path1):
                    file_path2 = os.path.join(file_path1, j)
                    if int(j) % 3 != int(type_id) and i == 'attack_face':
                        continue
                    faces_name = os.listdir(file_path2)
                    for k in random.sample(faces_name, 1):
                        face_name = '_'.join(k.split('_')[:-1]) + '.jpg'
                        map_path = os.path.join(file_path2, k)
                        image_path = os.path.join(os.path.join(image_dir, i), j)
                        face_path = os.path.join(image_path, face_name)
                        csv_writer.writerow([face_path, label])
                        map_csv_writer.writerow([map_path, label])

                    else:
                        continue
        map_csv_a.close()

    CASIA_FASD_base_process(image_dir=train_image_dir, map_dir=train_map_dir,
                            image_csv=train_csv, map_csv=train_map_csv, type_id=Protocol)
    CASIA_FASD_base_process(image_dir=test_image_dir, map_dir=test_map_dir,
                            image_csv=test_csv, map_csv=test_map_csv, type_id=Protocol)


def MSU_MFSD_process(crop_size):
    Protocol = 'ipad'  # ipad: HR video attack; iphone: Mobile video attack; printed: Printed attack
    train_map_dir = "E:/zsw/Data/MSU_MFSD/MSU_MFSD_DepthMap/%s/train/" % crop_size
    test_map_dir = "E:/zsw/Data/MSU_MFSD/MSU_MFSD_DepthMap/%s/test/" % crop_size

    train_image_dir = "E:/zsw/Data/MSU_MFSD/MSU_MFSD_CropFace256/%s/train/" % crop_size
    test_image_dir = "E:/zsw/Data/MSU_MFSD/MSU_MFSD_CropFace256/%s/test/" % crop_size

    train_csv = r'E:/zsw/Data/MSU_MFSD/CSV_rsf/%s/train_%s_%s.csv' % (crop_size, Protocol, interval)
    test_csv = r'E:/zsw/Data/MSU_MFSD/CSV/%s/test_%s%s_%s.csv' % (crop_size, Protocol, interval)

    train_map_csv = r'E:/zsw/Data/MSU_MFSD/CSV_rsf/%s/train_map_%s_%s.csv' % (crop_size, Protocol, interval)
    test_map_csv = r'E:/zsw/Data/MSU_MFSD/CSV_rsf/%s/test_map_%s_%s.csv' % (crop_size, Protocol, interval)

    # if not os.path.exists(train_csv):
    #     os.makedirs(train_csv)

    def MSU_MFSD_base_process(image_dir, map_dir, image_csv, map_csv, type_id):
        map_csv_a = open(map_csv, 'a', encoding='utf-8', newline='')
        map_csv_writer = csv.writer(map_csv_a)
        with open(image_csv, 'a', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            for i in os.listdir(map_dir):
                file_path1 = os.path.join(map_dir, i)
                if i == 'attack_face':
                    label = 0
                else:
                    label = 1
                for j in os.listdir(file_path1):
                    file_path2 = os.path.join(file_path1, j)
                    if j.split('_')[4] != type_id and i == 'attack_face':
                        continue
                    faces_name = os.listdir(file_path2)
                    for k in random.sample(faces_name, 1):
                        face_name = '_'.join(k.split('_')[:-1]) + '.jpg'
                        map_path = os.path.join(file_path2, k)
                        image_path = os.path.join(os.path.join(image_dir, i), j)
                        face_path = os.path.join(image_path, face_name)
                        csv_writer.writerow([face_path, label])
                        map_csv_writer.writerow([map_path, label])

                    else:
                        continue
        map_csv_a.close()

    MSU_MFSD_base_process(image_dir=train_image_dir, map_dir=train_map_dir,
                          image_csv=train_csv, map_csv=train_map_csv, type_id=Protocol)
    MSU_MFSD_base_process(image_dir=test_image_dir, map_dir=test_map_dir,
                          image_csv=test_csv, map_csv=test_map_csv, type_id=Protocol)


def RE_process(crop_size):
    Protocol = 'print'  # print: Printed Photo attack; mobile: Video attack; highdef: Digital Photo attack
    train_map_dir = "E:/zsw/Data/RE/RE_DepthMap/%s/train/" % crop_size
    devel_map_dir = "E:/zsw/Data/RE/RE_DepthMap/%s/devel/" % crop_size
    test_map_dir = "E:/zsw/Data/RE/RE_DepthMap/%s/test/" % crop_size

    train_image_dir = "E:/zsw/Data/RE/RE_CropFace256/%s/train/" % crop_size
    devel_image_dir = "E:/zsw/Data/RE/RE_CropFace256/%s/devel/" % crop_size
    test_image_dir = "E:/zsw/Data/RE/RE_CropFace256/%s/test/" % crop_size

    train_csv = r'E:/zsw/Data/RE/CSV_rsf/%s/train_%s_%s.csv' % (crop_size, Protocol, interval)
    test_csv = r'E:/zsw/Data/RE/CSV_rsf/%s/test_%s%s_%s.csv' % (crop_size, Protocol, interval)

    train_map_csv = r'E:/zsw/Data/RE/CSV_rsf/%s/train_map_%s_%s.csv' % (crop_size, Protocol, interval)
    test_map_csv = r'E:/zsw/Data/RE/CSV_rsf/%s/test_map_%s_%s.csv' % (crop_size, Protocol, interval)

    # if not os.path.exists(train_csv):
    #     os.makedirs(train_csv)

    def RE_base_process(image_dir, map_dir, image_csv, map_csv, type_id):
        map_csv_a = open(map_csv, 'a', encoding='utf-8', newline='')
        map_csv_writer = csv.writer(map_csv_a)
        with open(image_csv, 'a', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            for i in os.listdir(map_dir):
                file_path1 = os.path.join(map_dir, i)
                if i == 'attack_face':
                    label = 0
                else:
                    label = 1
                for j in os.listdir(file_path1):
                    file_path2 = os.path.join(file_path1, j)
                    if j.split('_')[1] != type_id and i == 'attack_face':
                        continue
                    faces_name = os.listdir(file_path2)
                    for k in random.sample(faces_name, 1):
                        face_name = '_'.join(k.split('_')[:-1]) + '.jpg'
                        map_path = os.path.join(file_path2, k)
                        image_path = os.path.join(os.path.join(image_dir, i), j)
                        face_path = os.path.join(image_path, face_name)
                        csv_writer.writerow([face_path, label])
                        map_csv_writer.writerow([map_path, label])

                    else:
                        continue
        map_csv_a.close()

    RE_base_process(image_dir=train_image_dir, map_dir=train_map_dir,
                    image_csv=train_csv, map_csv=train_map_csv, type_id=Protocol)
    RE_base_process(image_dir=devel_image_dir, map_dir=devel_map_dir,
                    image_csv=train_csv, map_csv=train_map_csv, type_id=Protocol)
    RE_base_process(image_dir=test_image_dir, map_dir=test_map_dir,
                    image_csv=test_csv, map_csv=test_map_csv, type_id=Protocol)


if __name__ == '__main__':
    # Modify the following directories to yourselves

    crop_size = '2.0'
    # Oulu_process(crop_size)
    # SiW_process(crop_size)
    CASIA_FASD_process(crop_size)
    # RE_process(crop_size)

    # MSU_MFSD_process(crop_size)
