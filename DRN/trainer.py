from model import AdaINGen, MultiScaleDis, VAEGen, DepthEstimator
from utils import weights_init, get_model_list, get_scheduler
from torch.autograd import Variable
import torch
import torch.nn as nn
import os
import torch.nn.functional as F

class MMDR_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(MMDR_Trainer, self).__init__()
        lr = hyperparameters['lr']
        self.style_dim = hyperparameters['gen']['style_dim']
        gen_config = hyperparameters['gen']
        dis_config = hyperparameters['dis']
        est_config = hyperparameters['est']
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

        self.est = DepthEstimator(hyperparameters['input_dim_r'], est_config['dim'],
                              est_config['n_layer'], est_config['layer_type'], est_config['norm'],
                              est_config['act'], gen_config['pad_type']).cuda()  # depth map estimator for style feature

        # fix the noise used in sampling
        display_size = int(hyperparameters['display_size'])
        self.style_r_fake = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        self.style_s_fake = torch.randn(display_size, self.style_dim, 1, 1).cuda()

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_r.parameters()) + list(self.dis_s.parameters())
        gen_params = list(self.gen_r.parameters()) + list(self.gen_s.parameters())
        est_params = list(self.est.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.est_opt = torch.optim.Adam([p for p in est_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])

        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)
        self.est_scheduler = get_scheduler(self.est_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_r.apply(weights_init('gaussian'))
        self.dis_s.apply(weights_init('gaussian'))


    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_r, x_s):
        # self.eval()
        self.train()
        content_s, style_s = self.gen_s.encode(x_s)
        content_r, style_r = self.gen_r.encode(x_r)
        x_rs = self.gen_r.decode(content_s, style_r)
        x_sr = self.gen_s.decode(content_r, style_s)

        map_rs = self.est(x_rs)
        map_sr = self.est(x_sr)
        return x_rs, x_sr, map_rs, map_sr

    def gen_update(self, x_r, x_s, hyperparameters):
        self.gen_opt.zero_grad()

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
        content_r_recon, style_s_recon = self.gen_s.encode(x_rs)
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
        self.loss_gen_adv_r = self.dis_r.calc_gen_loss(x_sr)
        self.loss_gen_adv_s = self.dis_s.calc_gen_loss(x_rs)
        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_r + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_s + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_r + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_style_r + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_content_r + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_s + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_style_s + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_content_s + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_r + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_s
        self.loss_gen_total.backward()
        self.gen_opt.step()

    def sample(self, x_r, x_s):
        self.eval()
        xs_r_recon, xs_s_recon, xs_sr, xs_rs = [], [], [], []
        maps_r, maps_s, maps_sr, maps_rs = [], [], [], []
        for i in range(x_r.size(0)):
            content_r, style_r = self.gen_r.encode(x_r[i].unsqueeze(0))
            content_s, style_s = self.gen_s.encode(x_s[i].unsqueeze(0))
            x_r_recon = self.gen_r.decode(content_r, style_r)
            x_s_recon = self.gen_s.decode(content_s, style_s)
            x_sr = self.gen_r.decode(content_r, style_s)
            x_rs = self.gen_s.decode(content_s, style_r)
            xs_r_recon.append(x_r_recon)
            xs_s_recon.append(x_s_recon)
            xs_sr.append(x_sr)
            xs_rs.append(x_rs)
            maps_r.append(self.est(x_r_recon))
            maps_s.append(self.est(x_s_recon))
            maps_sr.append(self.est(x_sr))
            maps_rs.append(self.est(x_rs))

        x_r_recon, x_s_recon, x_sr, x_rs = torch.cat(xs_r_recon), torch.cat(xs_s_recon), torch.cat(xs_sr), torch.cat(xs_rs)
        maps_r, maps_s, maps_sr, maps_rs = torch.cat(maps_r), torch.cat(maps_s), torch.cat(maps_sr), torch.cat(maps_rs)
        maps_r, maps_s, maps_sr, maps_rs = F.interpolate(maps_r, (256, 256)), F.interpolate(maps_s, (256, 256)), \
                                           F.interpolate(maps_sr, (256, 256)), F.interpolate(maps_rs, (256, 256))

        self.train()
        return x_r, x_r_recon, x_sr, maps_r, maps_sr, x_s, x_s_recon, x_rs, maps_s, maps_rs

    def dis_update(self, x_r, x_s, hyperparameters):
        self.dis_opt.zero_grad()
        # style_r_fake = Variable(torch.randn(x_r.size(0), self.style_dim, 1, 1).cuda())
        # style_s_fake = Variable(torch.randn(x_s.size(0), self.style_dim, 1, 1).cuda())
        # encode
        content_r, style_r = self.gen_r.encode(x_r)
        content_s, style_s = self.gen_s.encode(x_s)
        # decode (cross domain)
        x_sr = self.gen_r.decode(content_s, style_r)
        x_rs = self.gen_s.decode(content_r, style_s)
        # D loss
        self.loss_dis_r = self.dis_r.calc_dis_loss(x_sr.detach(), x_r)
        self.loss_dis_s = self.dis_s.calc_dis_loss(x_rs.detach(), x_s)
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_r + hyperparameters['gan_w'] * self.loss_dis_s
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def est_update(self, x_r, x_s, gt_r, gt_s, hyperparameters):
        gt_r = F.interpolate(gt_r, (32, 32))
        gt_s = F.interpolate(gt_s, (32, 32))
        self.est_opt.zero_grad()
        # encode
        content_r, style_r = self.gen_r.encode(x_r)
        content_s, style_s = self.gen_s.encode(x_s)
        # decode (cross domain)
        x_sr = self.gen_r.decode(content_s, style_r)
        x_rs = self.gen_s.decode(content_r, style_s)
        map_r = self.est(x_r)
        map_s = self.est(x_s)
        map_rs = self.est(x_rs)
        map_sr = self.est(x_sr)
        # E loss
        self.loss_est_r = self.est.calc_map_loss(map_r, gt_r)
        self.loss_est_s = self.est.calc_map_loss(map_s, gt_s)
        self.loss_est_rs = self.est.calc_map_loss(map_rs, gt_s)
        self.loss_est_sr = self.est.calc_map_loss(map_sr, gt_r)
        self.loss_est_total = hyperparameters['est_o_w'] * self.loss_est_r + hyperparameters['est_o_w'] * self.loss_est_s \
                              + hyperparameters['est_s_w'] * self.loss_est_rs + hyperparameters['est_s_w'] * self.loss_est_sr
        self.loss_est_total.backward()
        self.est_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()
        if self.est_scheduler is not None:
            self.est_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_r.load_state_dict(state_dict['r'])
        self.gen_s.load_state_dict(state_dict['s'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_r.load_state_dict(state_dict['r'])
        self.dis_s.load_state_dict(state_dict['s'])
        # Load estimator
        last_model_name = get_model_list(checkpoint_dir, "est")
        state_dict = torch.load(last_model_name)
        self.est.load_state_dict(state_dict)
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        self.est_opt.load_state_dict(state_dict['est'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        self.est_scheduler = get_scheduler(self.est_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        est_name = os.path.join(snapshot_dir, 'est_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'r': self.gen_r.state_dict(), 's': self.gen_s.state_dict()}, gen_name)
        torch.save({'r': self.dis_r.state_dict(), 's': self.dis_s.state_dict()}, dis_name)
        torch.save(self.est.state_dict(), opt_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict(), 'est': self.est_opt.state_dict()}, opt_name)
