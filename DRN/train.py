import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from utils import get_all_data_loaders, prepare_sub_folder, write_loss, get_config, write_2images, Timer
import argparse
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import sys
import numpy as np
import tensorboardX
import shutil
from trainer import MMDR_Trainer
from statistic import calculate_statistic, calculate_accuracy_score, calculate_roc_auc_score

try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='OULU.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='./results/conv/', help="outputs path")
parser.add_argument("--resume", action="store_true")
parser.add_argument('--trainer', type=str, default='MMDR', help="MMDR/?")
opts = parser.parse_args()


# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']

# Setup model and data loader
if opts.trainer == 'MMDR':
    trainer = MMDR_Trainer(config)
# elif opts.trainer == 'UNIT':
#     trainer = UNIT_Trainer(config)
else:
    sys.exit("Only support MMDR")

trainer.cuda()
train_loader, test_loader = get_all_data_loaders(config)
train_display_images_r = torch.stack([train_loader.dataset[i][0] for i in range(display_size)]).cuda()
train_display_images_s = torch.stack([train_loader.dataset[i][2] for i in range(display_size)]).cuda()

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

# Start training
iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
while True:
    for it, (images_r, maps_r, images_s, maps_s) in enumerate(train_loader):
        trainer.update_learning_rate()
        images_r, maps_r, images_s, maps_s = images_r.cuda().detach(), maps_r.cuda().detach(), images_s.cuda().detach(), maps_s.cuda().detach()

        with Timer("Elapsed time in update: %f"):
            # Main training code
            trainer.dis_update(images_r, images_s, config)
            trainer.gen_update(images_r, images_s, config)
            trainer.est_update(images_r, images_s, maps_r, maps_s, config)
            torch.cuda.synchronize()

        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            write_loss(iterations, trainer, train_writer)

        # Write images
        if (iterations + 1) % config['image_save_iter'] == 0:
            with torch.no_grad():
                train_image_outputs = trainer.sample(train_display_images_r, train_display_images_s)
                write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))

        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations)

        iterations += 1
        if iterations >= max_iter:
            sys.exit('Finish training')


# def val_model(scores, labels, current_iter, num_iters):
#     labels, scores = np.array(labels), np.array(scores)
#     APCER, NPCER, ACER, ACC, HTER = calculate_statistic(scores, labels)
#     AUC = calculate_roc_auc_score(labels, scores)
#     log.write(
#         '\n *********Iters {}/{}: APCER: {:.4f} NPCER: {:.4f} ACER: {:.4f} HTER: {:.4f} AUC: {:.4f} \n'.format(
#             current_iter,
#             num_iters - 1,
#             APCER, NPCER, ACER, HTER, AUC))
#     log.write('***************************************************\n')
#     return APCER, NPCER, ACER, HTER, AUC