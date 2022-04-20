from __future__ import print_function, division
import torch.optim as optim
import sys
import time
import numpy as np
import os
import torch
from torch.optim import lr_scheduler
import torch.nn as nn
from torch.utils import data
from random import shuffle
from model.model import Generator, Discrinator, Discrinator_s
from model.dataset import Dataset_Csv_train, Dataset_Csv_test
from model.config  import Config
from model.utils   import Error, plotResults
from model.loss    import l1_loss, l2_loss
from model.warp    import warping
from model.statistic import calculate_statistic
import torch.nn.functional as F
from albumentations import *
from sklearn.metrics import roc_auc_score, accuracy_score
import csv
import random


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


def train_model(config, dataloader, model_dir, num_epochs=20, current_epoch=0):
    best_ACER = 1.0
    train_num = 300
    bsize = config.BATCH_SIZE
    imsize = config.IMAGE_SIZE
    im2size = 160
    im3size = 40
    pre_epoch = current_epoch - 1

    gen = Generator(in_c=3, out_c=3)
    gen.train()
    gen.load_state_dict(torch.load('./model_out/STDN_OULU_STDN_1_6/%s/gen.ckpt' %pre_epoch))
    gen = torch.nn.DataParallel(gen.cuda())
    optimizer_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_AVERAGE_DECAY)
    exp_lr_scheduler_gen = lr_scheduler.StepLR(optimizer_gen, step_size=config.NUM_EPOCHS_PER_DECAY, gamma=config.GAMMA)

    dis_1 = Discrinator(in_c=3, out_c=3)
    dis_1.train()
    dis_1.load_state_dict(torch.load('./model_out/STDN_OULU_STDN_1_6/%s/dis_1.ckpt' %pre_epoch))
    dis_1 = torch.nn.DataParallel(dis_1.cuda())
    optimizer_dis_1 = optim.Adam(dis_1.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_AVERAGE_DECAY)
    exp_lr_scheduler_dis_1 = lr_scheduler.StepLR(optimizer_dis_1, step_size=config.NUM_EPOCHS_PER_DECAY,
                                                 gamma=config.GAMMA)

    dis_2 = Discrinator(in_c=3, out_c=3)
    dis_2.train()
    dis_2.load_state_dict(torch.load('./model_out/STDN_OULU_STDN_1_6/%s/dis_2.ckpt' %pre_epoch))
    dis_2 = torch.nn.DataParallel(dis_2.cuda())
    optimizer_dis_2 = optim.Adam(dis_2.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_AVERAGE_DECAY)
    exp_lr_scheduler_dis_2 = lr_scheduler.StepLR(optimizer_dis_2, step_size=config.NUM_EPOCHS_PER_DECAY,
                                                 gamma=config.GAMMA)

    dis_3 = Discrinator(in_c=3, out_c=3)
    dis_3.train()
    dis_3.load_state_dict(torch.load('./model_out/STDN_OULU_STDN_1_6/%s/dis_3.ckpt' %pre_epoch))
    dis_3 = torch.nn.DataParallel(dis_3.cuda())
    optimizer_dis_3 = optim.Adam(dis_3.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_AVERAGE_DECAY)
    exp_lr_scheduler_dis_3 = lr_scheduler.StepLR(optimizer_dis_3, step_size=config.NUM_EPOCHS_PER_DECAY,
                                                 gamma=config.GAMMA)
    criterion = nn.BCEWithLogitsLoss().cuda()


    for epoch in range(current_epoch, num_epochs):

        epoch_start = time.time()
        if not os.path.exists(model_dir + str(epoch)):
            os.makedirs(model_dir + str(epoch))
        gen_out_path = os.path.join(model_dir + str(epoch), 'gen.ckpt')
        dis_1_out_path = os.path.join(model_dir + str(epoch), 'dis_1.ckpt')
        dis_2_out_path = os.path.join(model_dir + str(epoch), 'dis_2.ckpt')
        dis_3_out_path = os.path.join(model_dir + str(epoch), 'dis_3.ckpt')

        log.write('------------------------------------------------------------------------\n')
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                exp_lr_scheduler_gen.step()
                exp_lr_scheduler_dis_1.step()
                exp_lr_scheduler_dis_2.step()
                exp_lr_scheduler_dis_3.step()
                gen.train()
                dis_1.train()
                dis_2.train()
                dis_3.train()

                for i, (image, reg_map_sp) in enumerate(dataloader[phase]):
                    optimizer_gen.zero_grad()
                    optimizer_dis_1.zero_grad()
                    optimizer_dis_2.zero_grad()
                    optimizer_dis_3.zero_grad()
                    image, reg_map_sp = image.cuda(), reg_map_sp.cuda()
                    image = image.permute(1, 0, 4, 2, 3)
                    image = image.reshape(bsize * 2, 3, imsize, imsize)
                    image2 = F.interpolate(image, [im2size, im2size])
                    image3 = F.interpolate(image, [im3size, im3size])  #

                    # disentangle the spoof traces / step 1
                    reg_map_sp = reg_map_sp.permute(0, 3, 1, 2)
                    M, s, b, C, T = gen(image)
                    recon1 = (1 - s) * image - b - F.interpolate(C, [imsize, imsize]) - T  # reconstruct the live face
                    trace = image - recon1  # disentangle the spoof trace
                    trace_warp = warping(trace[bsize:, ...], reg_map_sp, imsize)  # get warpped trace

                    # synthesis the new spoof image
                    synth1 = image[:bsize, ...] + trace_warp
                    image_d1 = torch.cat([image, recon1[bsize:, ...], synth1], 0)

                    # input the feature into dis / step 2
                    dis_1l, dis_1s = dis_1(image_d1)

                    recon2 = F.interpolate(recon1, [im2size, im2size])
                    synth2 = F.interpolate(synth1, [im2size, im2size])
                    image_d2 = torch.cat([image2, recon2[bsize:, ...], synth2], 0)
                    dis_2l, dis_2s = dis_2(image_d2)

                    recon3 = F.interpolate(recon1, [im3size, im3size])
                    synth3 = F.interpolate(synth1, [im3size, im3size])
                    image_d3 = torch.cat([image3, recon3[bsize:, ...], synth3], 0)
                    dis_3l, dis_3s = dis_2(image_d3)

                    # hard mode / step 3
                    s_hard = s * torch.Tensor(bsize * 2, 1, 1, 1).uniform_(0.1, 0.8).cuda()
                    b_hard = b * torch.Tensor(bsize * 2, 1, 1, 1).uniform_(0.1, 0.8).cuda()
                    C_hard = C * torch.Tensor(bsize * 2, 1, 1, 1).uniform_(0.1, 0.8).cuda()
                    T_hard = T * torch.Tensor(bsize * 2, 1, 1, 1).uniform_(0.1, 0.8).cuda()
                    recon_hard1 = (1 - s_hard) * image - b - F.interpolate(C, [imsize, imsize]) - T
                    recon_hard2 = (1 - s) * image - b_hard - F.interpolate(C, [imsize, imsize]) - T
                    recon_hard3 = (1 - s) * image - b - F.interpolate(C_hard, [imsize, imsize]) - T
                    recon_hard4 = (1 - s) * image - b - F.interpolate(C, [imsize, imsize]) - T_hard

                    recon_hard_s1 = recon_hard1 if torch.gt(torch.Tensor(1).uniform_(0, 1)[0], 0.5) else recon_hard2
                    recon_hard_s2 = recon_hard3 if torch.gt(torch.Tensor(1).uniform_(0, 1)[0], 0.5) else recon_hard4
                    recon_hard = recon_hard_s1 if torch.gt(torch.Tensor(1).uniform_(0, 1)[0], 0.5) else recon_hard_s2

                    image_a1 = torch.cat([image[:bsize, ...], recon_hard[bsize:, ...]], dim=0).detach()
                    image_a2 = torch.cat([image[:bsize, ...], synth1], dim=0).detach()
                    dec = torch.gt(torch.Tensor(1).uniform_(0, 1)[0], 0.5)
                    image_a = image_a1 if dec else image_a2
                    M_a, s_a, b_a, C_a, T_a = gen(image_a)
                    traces_a = s_a * image + b_a + F.interpolate(C_a, [imsize, imsize]) + T_a

                    # loss computation
                    d1_rl, _, d1_sl, _ = torch.split(dis_1l, dis_1l.shape[0]//4)
                    d2_rl, _, d2_sl, _ = torch.split(dis_2l, dis_2l.shape[0]//4)
                    d3_rl, _, d3_sl, _ = torch.split(dis_3l, dis_3l.shape[0]//4)
                    _, d1_rs, _, d1_ss = torch.split(dis_1s, dis_1s.shape[0]//4)
                    _, d2_rs, _, d2_ss = torch.split(dis_2s, dis_2s.shape[0]//4)
                    _, d3_rs, _, d3_ss = torch.split(dis_3s, dis_3s.shape[0]//4)

                    # loss for step 1
                    M_li, M_sp = torch.split(M, M.shape[0]//2, dim=0)
                    esr_loss = l1_loss(M_li, -1) + l1_loss(M_sp, 1)
                    gan_loss = l2_loss(d1_sl, 1) + l2_loss(d2_sl, 1) + l2_loss(d3_sl, 1) + \
                               l2_loss(d1_ss, 1) + l2_loss(d2_ss, 1) + l2_loss(d3_ss, 1)
                    reg_loss_li = l2_loss(s[:bsize, ...], 0) + l2_loss(b[:bsize, ...], 0) + l2_loss(C[:bsize, ...], 0) \
                                  + l2_loss(T[:bsize, ...], 0)
                    reg_loss_sp = l2_loss(s[bsize:, ...], 0) + l2_loss(b[bsize:, ...], 0) + l2_loss(C[bsize:, ...], 0) \
                                  + l2_loss(T[bsize:, ...], 0)
                    reg_loss = reg_loss_li * 10 + reg_loss_sp * 1e-4
                    g_loss = esr_loss * 50 + gan_loss + reg_loss

                    # loss for step2
                    d_loss = (l2_loss(d1_rl, 1) + l2_loss(d2_rl, 1) + l2_loss(d3_rl, 1) + \
                              l2_loss(d1_rs, 1) + l2_loss(d2_rs, 1) + l2_loss(d3_rs, 1) + \
                              l2_loss(d1_sl, 0) + l2_loss(d2_sl, 0) + l2_loss(d3_sl, 0) + \
                              l2_loss(d1_ss, 0) + l2_loss(d2_ss, 0) + l2_loss(d3_ss, 0)) / 4

                    # loss for step3.
                    esr_loss_a = l1_loss(M_a[:bsize, ...], -1) + l1_loss(M_a[bsize:, ...], 1)
                    pixel_loss = l1_loss(traces_a[:bsize, ...], trace_warp.detach())
                    a_loss_1 = esr_loss_a * 5 + pixel_loss * 0.0  # #
                    a_loss_2 = esr_loss_a * 5 + pixel_loss * 0.1  # #
                    a_loss = a_loss_1 if dec else a_loss_2

                    gen_loss = g_loss + a_loss
                    dis_loss = d_loss
                    gen_loss.backward(retain_graph=True)
                    dis_loss.backward(retain_graph=True)
                    optimizer_gen.step()
                    optimizer_dis_1.step()
                    optimizer_dis_2.step()
                    optimizer_dis_3.step()
                    if (i + 1) % train_num == 0:
                        log.write(
                            '**Epoch {}/{} Train {}/{}: g_loss: {:.4f} d_loss: {:.4f} a_loss: {:.4f}\n'.format(epoch, num_epochs - 1,
                            i, len(dataloader[phase]), g_loss, d_loss, a_loss))

            else:
                gen.eval()
                dis_1.eval()
                dis_2.eval()
                dis_3.eval()

                y_scores, y_labels = [], []
                for i, (image, im_name, label) in enumerate(dataloader[phase]):
                    image, label = image.cuda(), torch.LongTensor(label).cuda()
                    image = image.permute(0, 3, 1, 2)
                    M, s, b, C, T = gen(image)
                    score = torch.mean(M, dim=(1, 2, 3))
                    label = label.data.cpu().numpy()
                    score = score.data.cpu().numpy()
                    score = np.where(score > 0.75, 0, 1)
                    y_scores.extend(score)
                    y_labels.extend(label)
                    if (i + 1) % train_num == 0:
                        accuracy = accuracy_score(y_labels, y_scores)
                        log.write(
                            '**Epoch {}/{} Test {}/{}: Accuracy: {:.4f}\n'.format(epoch, num_epochs - 1,
                                                                                                i,
                                                                                                len(dataloader[phase]),
                                                                                                accuracy))



                APCER, NPCER, ACER, HTER, AUC = val_model(y_scores, y_labels, current_epoch=epoch, num_epochs=num_epochs)
                log.write('Epoch {}/{} Time {}s\n'.format(epoch, num_epochs - 1, time.time() - epoch_start))
                log.write('***************************************************')
                if ACER < best_ACER:
                    torch.save(gen.module.state_dict(), gen_out_path)
                    torch.save(dis_1.module.state_dict(), dis_1_out_path)
                    torch.save(dis_2.module.state_dict(), dis_2_out_path)
                    torch.save(dis_3.module.state_dict(), dis_3_out_path)


def val_model(scores, labels, current_epoch, num_epochs):
    labels, scores = np.array(labels), np.array(scores)
    APCER, NPCER, ACER, ACC, HTER = calculate_statistic(scores, labels)
    AUC = roc_auc_score(labels, scores)
    log.write(
        '\n *********Epoch {}/{}: APCER: {:.4f} NPCER: {:.4f} ACER: {:.4f} HTER: {:.4f} AUC: {:.4f} \n'.format(
            current_epoch,
            num_epochs - 1,
            APCER, NPCER, ACER, HTER, AUC))
    log.write('***************************************************\n')
    return APCER, NPCER, ACER, HTER, AUC


def get_train_data(csv_file):
    list_li, list_sp = [], []
    frame_reader = open(csv_file, 'r')
    csv_reader = csv.reader(frame_reader)

    for f in csv_reader:
        img_path = f[0]
        label = int(f[1])
        if label == 0:
            list_sp.append(img_path)
        else:
            list_li.append(img_path)
    len_li = len(list_li)
    len_sp = len(list_sp)
    if len_li < len_sp:
        while len(list_li) < len_sp:
            list_li += random.sample(list_li, len(list_li))
        list_li = list_li[:len_sp]
    elif len_li > len_sp:
        while len(list_sp) < len_li:
            list_sp += random.sample(list_sp, len(list_sp))
        list_sp = list_sp[:len_li]

    log.write(str(len(list_sp) + len(list_li)) + '\n')
    return list_li, list_sp

def get_test_data(csv_file):
    image_list = []
    label_list = []
    frame_reader = open(csv_file, 'r')
    csv_reader = csv.reader(frame_reader)

    for f in csv_reader:
        image_list.append(f[0])
        label_list.append(f[1])

    log.write(str(len(image_list)) + '\n')
    return image_list, label_list




if __name__ == '__main__':

    config = Config(gpu='1',
                    database='OULU',
                    protocol='_1')

    # Modify the following directories to yourselves
    os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_USAGE
    database = config.DATABASE      #OULU, CASIA_FASD, MSU_MFSD, RE
    start = time.time()
    current_epoch = 8
    batch_size = config.BATCH_SIZE
    Protocol = config.PROTOCOL
    crop_size = config.CROP_SIZE
    interval = config.INTERVAL

    train_csv = r'E:/zsw/Data/%s/CSV/%s/train%s_%s.csv' % (database, crop_size, Protocol, interval)  # The train split file
    test_csv = r'E:/zsw/Data/%s/CSV/%s/test%s_%s.csv' % (database, crop_size, Protocol, interval)  # The validation split file

    train_map_csv = r'E:/zsw/Data/%s/CSV/%s/train_map%s_%s.csv' % (database, crop_size, Protocol, interval)  # The train split file
    test_map_csv = r'E:/zsw/Data/%s/CSV/%s/test_map%s_%s.csv' % (
        database, crop_size, Protocol, interval)  # The validation split file

    #  Output path
    model_dir = 'model_out/STDN_%s_%s%s_%s/' % (database, crop_size, Protocol, interval)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    log_name = model_dir.split('/')[-2] + '.log'
    log_dir = os.path.join(model_dir, log_name)
    if os.path.exists(log_dir):
        os.remove(log_dir)
        print('The log file is exit!')

    log = Logger(log_dir, sys.stdout)
    log.write('model : ViT   batch_size : 16 frames : 6 \n')
    log.write('pretrain : False   input_size : 224*224\n')

    use_cuda = torch.cuda.is_available()  # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU

    # Data loading parameters
    params = {'shuffle': False, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    log.write('loading train data' + '\n')
    train_list_li, train_list_sp = get_train_data(train_csv)
    ziplist = list(zip(train_list_li, train_list_sp))
    shuffle(ziplist)
    train_list_li, train_list_sp = zip(*ziplist)

    log.write('loading val data' + '\n')
    test_list, test_labels = get_test_data(test_csv)

    train_set = Dataset_Csv_train(config, train_list_li, train_list_sp, transform=None)
    test_set = Dataset_Csv_test(config, test_list, test_labels, transform=None)


    image_datasets = {}
    # over sampling
    image_datasets['train'] = data.DataLoader(train_set, batch_size=batch_size, **params, drop_last=True)
    image_datasets['test'] = data.DataLoader(test_set, batch_size=batch_size, **params, drop_last=True)


    dataloaders = {x: image_datasets[x] for x in ['train', 'test']}
    datasets_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

    train_model(config=config, dataloader=dataloaders, model_dir=model_dir,
                num_epochs=config.MAX_EPOCH,
                current_epoch=current_epoch)

    elapsed = (time.time() - start)
    log.write('Total time is {}.\n'.format(elapsed))
