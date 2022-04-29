"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from model import AdaINGen, MultiScaleDis, VAEGen
from utils import weights_init, get_model_list, vgg_preprocess, load_vgg16, get_scheduler
from torch.autograd import Variable
import torch
import torch.nn as nn
import os

class MMDR_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(MMDR_Trainer, self).__init__()
        lr = hyperparameters['lr']
        gen_config = hyperparameters['gen']
        dis_config = hyperparameters['dis']
        # Initiate the networks
        self.gen_r = AdaINGen(hyperparameters['input_dim_r'], gen_config['dim'], gen_config['style_dim'],
                              gen_config['n_downsample'], gen_config['n_resblock'], gen_config['mlp_dim'],
                              gen_config['act'], gen_config['pad_type'])  # auto-encoder for type real
        self.gen_s = AdaINGen(hyperparameters['input_dim_s'], gen_config['dim'], gen_config['style_dim'],
                              gen_config['n_downsample'], gen_config['n_resblock'], gen_config['mlp_dim'],
                              gen_config['act'], gen_config['pad_type'])  # auto-encoder for type spoof
        self.dis_r = MultiScaleDis(hyperparameters['input_dim_s'], dis_config['dim'], dis_config['num_scales'],
                              dis_config['n_layer'], dis_config['gan_type'], dis_config['norm'],
                              dis_config['act'], dis_config['pad_type'])  # discriminator for type real
        self.dis_s = MultiScaleDis(hyperparameters['input_dim_r'], dis_config['dim'], dis_config['num_scales'],
                              dis_config['n_layer'], dis_config['gan_type'], dis_config['norm'],
                              dis_config['act'], dis_config['pad_type'])  # discriminator for type spoof
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_c = hyperparameters['gen']['style_dim']

        # fix the noise used in sampling
        display_size = int(hyperparameters['display_size'])
        self.style_r_fake = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        self.style_s_fake = torch.randn(display_size, self.style_dim, 1, 1).cuda()

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_r.parameters()) + list(self.dis_s.parameters())
        gen_params = list(self.gen_r.parameters()) + list(self.gen_s.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_r.apply(weights_init('gaussian'))
        self.dis_s.apply(weights_init('gaussian'))


    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_r, x_s):
        self.eval()
        style_s_fake = Variable(self.style_s_fake)
        style_r_fake = Variable(self.style_r_fake)
        content_s, style_s = self.gen_s.encode(x_s)
        content_r, style_r = self.gen_r.encode(x_r)
        x_rs = self.gen_r.decode(content_s, style_r)
        x_sr = self.gen_s.decode(content_r, style_s)
        self.train()
        return x_rs, x_sr

    def gen_update(self, x_r, x_s, hyperparameters):
        self.gen_opt.zero_grad()
        style_r_fake = Variable(torch.randn(x_r.size(0), self.style_dim, 1, 1).cuda())
        style_s_fake = Variable(torch.randn(x_s.size(0), self.style_dim, 1, 1).cuda())

        # encode
        content_r, style_r = self.gen_r.encode(x_r)
        content_s, style_s = self.gen_s.encode(x_s)
        # decode (within domain)
        x_r_recon = self.gen_r.decode(content_r, style_r)
        x_s_recon = self.gen_s.decode(content_s, style_s)
        # decode (cross type)
        x_sr = self.gen_r.decode(content_s, style_r)
        x_rs = self.gen_s.decode(content_r, style_s)

        # encode again
        content_s_recon, style_r_recon = self.gen_r.encode(x_sr)
        content_r_recon, style_s_recon = self.gen_b.encode(x_rs)
        # decode again (if needed)
        x_rsr = self.gen_r.decode(content_r_recon, style_r) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_srs = self.gen_s.decode(content_s_recon, style_s) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # reconstruction loss
        self.loss_gen_recon_x_r = self.recon_criterion(x_r_recon, x_r)
        self.loss_gen_recon_x_s = self.recon_criterion(x_s_recon, x_s)
        self.loss_gen_recon_style_r = self.recon_criterion(style_r_recon, style_r)
        self.loss_gen_recon_style_s = self.recon_criterion(style_s_recon, style_s)
        self.loss_gen_recon_content_r = self.recon_criterion(content_r_recon, content_r)
        self.loss_gen_recon_content_s = self.recon_criterion(content_s_recon, content_s)
        self.loss_gen_cycrecon_x_r = self.recon_criterion(x_rsr, x_r) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        self.loss_gen_cycrecon_x_s = self.recon_criterion(x_srs, x_s) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0
        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_b
        self.loss_gen_total.backward()
        self.gen_opt.step()

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def sample(self, x_a, x_b):
        self.eval()
        s_a1 = Variable(self.s_a)
        s_b1 = Variable(self.s_b)
        s_a2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2 = [], [], [], [], [], []
        for i in range(x_a.size(0)):
            c_a, s_a_fake = self.gen_a.encode(x_a[i].unsqueeze(0))
            c_b, s_b_fake = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_a_recon.append(self.gen_a.decode(c_a, s_a_fake))
            x_b_recon.append(self.gen_b.decode(c_b, s_b_fake))
            x_ba1.append(self.gen_a.decode(c_b, s_a1[i].unsqueeze(0)))
            x_ba2.append(self.gen_a.decode(c_b, s_a2[i].unsqueeze(0)))
            x_ab1.append(self.gen_b.decode(c_a, s_b1[i].unsqueeze(0)))
            x_ab2.append(self.gen_b.decode(c_a, s_b2[i].unsqueeze(0)))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
        x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
        self.train()
        return x_a, x_a_recon, x_ab1, x_ab2, x_b, x_b_recon, x_ba1, x_ba2

    def dis_update(self, x_a, x_b, hyperparameters):
        self.dis_opt.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # encode
        c_a, _ = self.gen_a.encode(x_a)
        c_b, _ = self.gen_b.encode(x_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)
        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)
