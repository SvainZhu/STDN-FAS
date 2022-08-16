from __future__ import print_function, division
import sys
import time
import numpy as np
import os
import torch
import torch.nn.functional as F
import argparse
import shutil
import tensorboardX
from model_test import Generator, MultiScaleDis, FeatureEstimator
from loss import l1_loss, l2_loss
from statistic import calculate_statistic, calculate_accuracy_score, calculate_roc_auc_score

from utils import get_all_data_loaders, get_scheduler, weights_init, get_model_list, prepare_sub_folder, write_loss, get_config, write_2images

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

def train_model(config, dataloader, checkpoint_dir, image_dir, max_epochs=20, current_epoch=0):
    def resume(checkpoint_dir, config):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        gen.load_state_dict(state_dict)
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        dis.load_state_dict(state_dict)
        # Load estimator
        last_model_name = get_model_list(checkpoint_dir, "est")
        state_dict = torch.load(last_model_name)
        est.load_state_dict(state_dict)
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        dis_opt.load_state_dict(state_dict['dis'])
        gen_opt.load_state_dict(state_dict['gen'])
        est_opt.load_state_dict(state_dict['est'])
        # Reinitilize schedulers
        dis_scheduler = get_scheduler(dis_opt, config, iterations)
        gen_scheduler = get_scheduler(gen_opt, config, iterations)
        est_scheduler = get_scheduler(est_opt, config, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(snapshot_dir, currect_epoch):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (currect_epoch + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (currect_epoch + 1))
        est_name = os.path.join(snapshot_dir, 'est_%08d.pt' % (currect_epoch + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save(gen.state_dict(), gen_name)
        torch.save(dis.state_dict(), dis_name)
        torch.save(est.state_dict(), est_name)
        torch.save({'gen': gen_opt.state_dict(), 'dis': dis_opt.state_dict(), 'est': est_opt.state_dict()}, opt_name)


    best_ACER = 1.0
    best_epoch = 0
    display_size = config['display_size']

    gen_config = config['gen']
    dis_config = config['dis']
    est_config = config['est']

    gen = Generator(config['input_channel'], config['input_channel'], gen_config['feature_c'],
                    gen_config['n_downsample'], gen_config['n_block'], gen_config['norm'], gen_config['act'], gen_config['pad_type'])
    dis = MultiScaleDis(dis_config['input_c'], dis_config['output_c'], dis_config['num_scales'],
                    dis_config['n_layer'], dis_config['norm'], dis_config['act'], dis_config['pad_type'])
    est = FeatureEstimator(est_config['input_c'], est_config['output_c'],
                           est_config['n_layer'], est_config['norm'], est_config['act'], est_config['pad_type'])
    gen = gen.cuda()
    dis = dis.cuda()
    est = est.cuda()

    # Setup the optimizers
    beta1, beta2, lr, weight_decay = config['beta1'], config['beta2'], config['lr'], config['weight_decay']
    dis_opt = torch.optim.Adam(dis.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    est_opt = torch.optim.Adam(est.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)

    dis_scheduler = get_scheduler(dis_opt, config)
    gen_scheduler = get_scheduler(gen_opt, config)
    est_scheduler = get_scheduler(est_opt, config)

    # Initialize the network weight
    init = config['init']
    gen.apply(weights_init(init))
    dis.apply(weights_init(init))
    est.apply(weights_init(init))
    iterations = resume(checkpoint_directory, config=config) if opts.resume else 0
    for epoch in range(current_epoch, max_epochs):
        epoch_start = time.time()
        log.write('------------------------------------------------------------------------\n')
        # Each epoch has a training and validation phase
        # for phase in ['test']:
        for phase in ['train', 'test']:
            if phase == 'train':
                gen_opt.step()
                dis_opt.step()
                est_opt.step()
                gen_scheduler.step()
                dis_scheduler.step()
                est_scheduler.step()
                gen.train()
                dis.train()
                est.train()
                for i, (images_r, maps_r, images_s, maps_s) in enumerate(dataloader['train_loader']):
                    gen_opt.zero_grad()
                    dis_opt.zero_grad()
                    est_opt.zero_grad()

                    images_r, maps_r, images_s, maps_s = images_r.cuda(), maps_r.cuda(), images_s.cuda(), maps_s.cuda()

                    # disentangle the content-style feature of live and spoof faces
                    content_r, style_r = gen.encode(images_r)
                    content_s, style_s = gen.encode(images_s)

                    # reconstruct the liveness faces and synthesize the spoof faces
                    recon_r, inpaint_re_r, inpaint_trace_r, spoof_trace_r = gen.decode(content_s, style_r, 'reconstruct')
                    synth_s, inpaint_re_s, inpaint_trace_s, spoof_trace_s = gen.decode(content_r, style_s, 'synthesize')

                    content_recon_r, style_recon_r = gen.encode(recon_r)
                    content_synth_s, style_synth_s = gen.encode(synth_s)
                    recon_r_exchange, inpaint_re_r_exchange, inpaint_trace_r_exchange, spoof_trace_r_exchange = gen.decode(content_synth_s, style_recon_r, 'synthesize')
                    recon_s_exchange, inpaint_re_s_exchange, inpaint_trace_s_exchange, spoof_trace_s_exchange = gen.decode(content_recon_r, style_synth_s, 'synthesize')

                    # discriminate the image
                    dis_recon_r = dis(recon_r)
                    dis_synth_s = dis(synth_s)
                    dis_recon_r_exchange = dis(recon_r_exchange)
                    dis_recon_s_exchange = dis(recon_s_exchange)

                    # estimate the feature style
                    est_r = est(style_r)
                    est_s = est(style_s)
                    est_recon_r = est(style_recon_r)
                    est_synth_s = est(style_synth_s)

                    # compute the loss
                    # loss for step 1
                    gan_loss = 0
                    for i in range(dis_config['num_scales']):
                        gan_loss += (l2_loss(dis_recon_r[i], 0) + l2_loss(dis_synth_s[i], 1))
                    est_loss = l1_loss(est_r, maps_r) + l1_loss(est_s, maps_s)
                    reg_loss_r = l2_loss(spoof_trace_r, 0)
                    reg_loss_s = l2_loss(spoof_trace_s, 0)
                    reg_loss = config['reg_loss_s_w'] * reg_loss_s + config['reg_loss_r_w'] * reg_loss_r
                    pixel_recon_loss = l2_loss(images_r, recon_r_exchange) + l2_loss(images_s, recon_s_exchange)
                    gen_loss = config['gan_w'] * gan_loss + config['reg_w'] * reg_loss + pixel_recon_loss + \
                               config['pixel_recon_w'] * pixel_recon_loss

                    # loss for step2
                    dis_loss = 0
                    for i in range(dis_config['num_scales']):
                        loss = (l2_loss(dis_recon_r[i], 0) + l2_loss(dis_synth_s[i], 1) + l2_loss(dis_recon_r_exchange[i], 0) + l2_loss(dis_recon_s_exchange[i], 1)) / 4.0
                        dis_loss += loss

                    # loss for step3
                    est_recon_loss = l1_loss(est_recon_r, maps_r) + l1_loss(est_synth_s, maps_s)
                    est_total_loss = config['est_w'] * est_loss + config['est_recon_w'] * est_recon_loss


                    gen_loss.backward(retain_graph=True)
                    dis_loss.backward(retain_graph=True)
                    est_total_loss.backward(retain_graph=True)
                    gen_opt.step()
                    dis_opt.step()
                    est_opt.step()

                    if (iterations + 1) % config['log_iter'] == 0:
                        log.write(
                            '**Epoch {}/{} Train {}/{}: gen_loss: {:.4f} dis_loss: {:.4f} est_loss: {:.4f}\n'.format(epoch,
                            max_epochs - 1, iterations, len(loader['train_loader']), gen_loss, dis_loss, est_total_loss))

                    # Write images

                    if (iterations + 1) % config['image_save_iter'] == 0:
                        # train_display_images_r = torch.stack(
                        #     [loader['train_loader'].dataset[i][0] for i in range(display_size)]).cuda()
                        # train_display_images_s = torch.stack(
                        #     [loader['train_loader'].dataset[i][1] for i in range(display_size)]).cuda()
                        with torch.no_grad():
                            # # disentangle the content-style feature of live and spoof faces
                            # content_r, style_r = gen.encode(train_display_images_r)
                            # content_s, style_s = gen.encode(train_display_images_s)
                            #
                            # # reconstruct the liveness faces and synthesize the spoof faces
                            # recon_r = gen.decode(content_s, style_r)
                            # synth_s = gen.decode(content_r, style_s)
                            #
                            # content_recon_r, style_recon_r = gen.encode(recon_r)
                            # content_synth_s, style_synth_s = gen.encode(synth_s)
                            # recon_r_exchange = gen.decode(content_synth_s, style_recon_r)
                            # recon_s_exchange = gen.decode(content_recon_r, style_synth_s)
                            #
                            # # estimate the feature style
                            # est_r, est_s = est(style_r), est(style_s)
                            # est_recon_r, est_synth_s = est(style_recon_r), est(style_synth_s)
                            est_r, est_s = F.interpolate(est_r, (256, 256)), F.interpolate(est_s, (256, 256))
                            est_recon_r, est_synth_s = F.interpolate(est_recon_r, (256, 256)), F.interpolate(est_synth_s, (256, 256))
                            train_image_outputs = [images_r, inpaint_re_r, inpaint_trace_r, spoof_trace_r, synth_s, recon_r_exchange, est_r, est_synth_s,
                                                   images_s, inpaint_re_s, inpaint_trace_s, spoof_trace_s, recon_r, recon_s_exchange, est_s, est_recon_r]
                            write_2images(train_image_outputs, display_size, image_dir,
                                          'train_%08d' % (iterations + 1))

                    iterations += 1
                    if iterations >= config['max_iters']:
                        sys.exit('Finish training')



            else:
                gen.eval()
                dis.eval()
                est.eval()

                y_scores, y_labels = [], []
                iterations = 0
                for i, (image, label) in enumerate(loader['test_loader']):
                    image, label = image.cuda(), torch.LongTensor(label).cuda()
                    content, style = gen.encode(image)
                    recon, inpaint_re, inpaint_trace, spoof_trace = gen.decode(content, style,
                                                                                                'reconstruct')

                    style_est = est(style)
                    score = l1_loss(inpaint_trace, 0) + l1_loss(spoof_trace, 0) - l1_loss(style_est, 0)
                    score = score.data.cpu().numpy().reshape((-1))
                    score = np.where(score > 0, 0, 1)
                    label = label.data.cpu().numpy()

                    y_scores.extend(score)
                    y_labels.extend(label)
                    if (iterations + 1) % config['log_iter'] == 0:
                        accuracy = calculate_accuracy_score(y_labels, y_scores)
                        log.write(
                            '**Epoch {}/{} Test {}/{}: Accuracy: {:.4f}\n'.format(epoch, max_epochs - 1,
                                                                                  iterations, len(loader['test_loader']), accuracy))
                    iterations += 1

                APCER, NPCER, ACER, HTER, AUC = val_model(y_scores, y_labels, current_epoch=epoch,
                                                          num_epochs=max_epochs)
                log.write('Epoch {}/{} Time {}s\n'.format(epoch, max_epochs - 1, time.time() - epoch_start))
                log.write('***************************************************')
                if ACER < best_ACER:
                    best_ACER = ACER
                    best_epoch = epoch
                save(checkpoint_dir, epoch)
                log.write('Best epoch is {}. The ACER of best epoch is {}, current epoch is {}\n'.format(best_epoch,
                                                                                                         best_ACER,
                                                                                                         epoch))


def val_model(scores, labels, current_epoch, num_epochs):
    labels, scores = np.array(labels), np.array(scores)
    APCER, NPCER, ACER, ACC, HTER = calculate_statistic(scores, labels)
    AUC = calculate_roc_auc_score(labels, scores)
    log.write(
        '\n *********Epoch {}/{}: APCER: {:.4f} NPCER: {:.4f} ACER: {:.4f} HTER: {:.4f} AUC: {:.4f} \n'.format(
            current_epoch,
            num_epochs - 1,
            APCER, NPCER, ACER, HTER, AUC))
    log.write('***************************************************\n')
    return APCER, NPCER, ACER, HTER, AUC

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='OULU.yaml', help='Path to the config file.')
    parser.add_argument('--output_path', type=str, default='./results_test0/', help="outputs path")
    parser.add_argument("--resume", type=bool, default=False, help="pretrain resume")
    opts = parser.parse_args()

    # Load experiment setting
    config = get_config(opts.config)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['GPU_USAGE'])
    use_cuda = torch.cuda.is_available()  # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU
    start = time.time()
    current_epoch = 0

    # Setup logger and output folders
    model_name = os.path.splitext(os.path.basename(opts.config))[0]
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
    output_directory = os.path.join(opts.output_path + "/outputs", model_name)
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)

    log_name = model_name + '.log'
    log_dir = os.path.join(checkpoint_directory, log_name)
    if os.path.exists(log_dir):
        os.remove(log_dir)
        print('The log file is exit!')

    log = Logger(log_dir, sys.stdout)
    log.write('model: MMDR   batch_size: 1 frames_sample: 3 \n')
    log.write('pretrain: False   input_size: 256*56*3\n')

    # Data loading parameters
    log.write('loading train data' + '\n')
    loader = get_all_data_loaders(config)


    train_model(config=config, dataloader=loader, checkpoint_dir=checkpoint_directory, image_dir=image_directory,
                max_epochs=config['max_epochs'],
                current_epoch=current_epoch)

    elapsed = (time.time() - start)
    log.write('Total time is {}.\n'.format(elapsed))
