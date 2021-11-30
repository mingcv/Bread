import argparse
import datetime
import os
import traceback

import kornia
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

import models
from datasets import LowLightDataset, LowLightFDataset
from models import PSNR, SSIM, CosineLR
from tools import SingleSummaryWriter
from tools import saver, mutils


def get_args():
    parser = argparse.ArgumentParser('Breaking Downing the Darkness')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus being used')
    parser.add_argument('--num_workers', type=int, default=12, help='num_workers of dataloader')
    parser.add_argument('--batch_size', type=int, default=1, help='The number of images per batch among all devices')
    parser.add_argument('-m1', '--model1', type=str, default='INet',
                        help='Model1 Name')
    parser.add_argument('-m2', '--model2', type=str, default='NSNet',
                        help='Model2 Name')
    parser.add_argument('-m3', '--model3', type=str, default='NSNet',
                        help='Model3 Name')

    parser.add_argument('-m1w', '--model1_weight', type=str, default=None,
                        help='Model Name')
    parser.add_argument('-m2w', '--model2_weight', type=str, default=None,
                        help='Model Name')

    parser.add_argument('--comment', type=str, default='default',
                        help='Project comment')
    parser.add_argument('--graph', action='store_true')
    parser.add_argument('--no_sche', action='store_true')
    parser.add_argument('--sampling', action='store_true')

    parser.add_argument('--slope', type=float, default=2.)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optim', type=str, default='adam', help='select optimizer for training, '
                                                                  'suggest using \'admaw\' until the'
                                                                  ' very final stage then switch to \'sgd\'')
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
    parser.add_argument('--save_interval', type=int, default=500, help='Number of steps between saving')
    parser.add_argument('--data_path', type=str, default='./data/LOL',
                        help='the root folder of dataset')
    parser.add_argument('--log_path', type=str, default='logs/')
    parser.add_argument('--saved_path', type=str, default='logs/')
    args = parser.parse_args()
    return args


class ModelNSNet(nn.Module):
    def __init__(self, model1, model2, model3):
        super().__init__()
        self.texture_loss = models.SSIML1Loss(channels=1)
        self.model_ianet = model1(in_channels=1, out_channels=1)
        self.model_nsnet = model2(in_channels=2, out_channels=1)
        self.model_fusenet = model3(in_channels=3, out_channels=1)

        assert opt.model1_weight is not None
        self.load_weight(self.model_ianet, opt.model1_weight)
        self.load_weight(self.model_nsnet, opt.model2_weight)
        self.model_ianet.eval()
        self.model_nsnet.eval()
        self.eps = 1e-2

    def load_weight(self, model, weight_pth):
        state_dict = torch.load(weight_pth)
        ret = model.load_state_dict(state_dict, strict=True)
        print(ret)

    def noise_syn(self, illumi, strength):
        return torch.exp(-illumi) * strength

    def forward(self, image, image_gt, training=True):
        texture_nss = []
        with torch.no_grad():
            if training:
                image = image.squeeze(0)
                image_gt = image_gt.repeat(8, 1, 1, 1)

            texture_in, _, _ = torch.split(kornia.color.rgb_to_ycbcr(image), 1, dim=1)
            texture_gt, _, _ = torch.split(kornia.color.rgb_to_ycbcr(image_gt), 1, dim=1)

            texture_in_down = F.interpolate(texture_in, scale_factor=0.5, mode='bicubic', align_corners=True)
            illumi = self.model_ianet(texture_in_down)
            illumi = F.interpolate(illumi, scale_factor=2, mode='bicubic', align_corners=True)
            noisy_gt = texture_in / torch.clamp_min(illumi, self.eps)

            for strength in [0, 0.05, 0.1]:
                illumi = torch.clamp(illumi, 0., 1.)
                attention = self.noise_syn(illumi, strength=strength)
                texture_res = self.model_nsnet(torch.cat([noisy_gt, attention], dim=1))
                texture_ns = noisy_gt + texture_res
                texture_nss.append(texture_ns)

            texture_nss = torch.cat(texture_nss, dim=1).detach()

        texture_fuse = self.model_fusenet(texture_nss)
        restor_loss = self.texture_loss(texture_fuse, texture_gt)
        psnr = PSNR(texture_fuse, texture_gt)
        ssim = SSIM(texture_fuse, texture_gt).item()
        return noisy_gt, texture_nss, texture_fuse, texture_res, illumi, restor_loss, psnr, ssim


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    timestamp = mutils.get_formatted_time()
    opt.saved_path = opt.saved_path + f'/{opt.comment}/{timestamp}'
    opt.log_path = opt.log_path + f'/{opt.comment}/{timestamp}/tensorboard/'
    os.makedirs(opt.log_path, exist_ok=True)
    os.makedirs(opt.saved_path, exist_ok=True)

    training_params = {'batch_size': opt.batch_size,
                       'shuffle': True,
                       'drop_last': True,
                       'num_workers': opt.num_workers}

    val_params = {'batch_size': 1,
                  'shuffle': False,
                  'drop_last': True,
                  'num_workers': opt.num_workers}

    training_set = LowLightFDataset(os.path.join(opt.data_path, 'train'), image_split='images_aug')
    training_generator = DataLoader(training_set, **training_params)

    val_set = LowLightDataset(os.path.join(opt.data_path, 'eval'))
    val_generator = DataLoader(val_set, **val_params)

    model1 = getattr(models, opt.model1)
    model2 = getattr(models, opt.model2)
    model3 = getattr(models, opt.model3)
    writer = SingleSummaryWriter(opt.log_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

    model = ModelNSNet(model1, model2, model3)
    print(model)

    if opt.num_gpus > 0:
        model = model.cuda()
        if opt.num_gpus > 1:
            model = nn.DataParallel(model)

    if opt.optim == 'adam':
        optimizer = torch.optim.Adam(model.model_fusenet.parameters(), opt.lr)
    else:
        optimizer = torch.optim.SGD(model.model_fusenet.parameters(), opt.lr, momentum=0.9, nesterov=True)

    scheduler = CosineLR(optimizer, opt.lr, opt.num_epochs)
    epoch = 0
    step = 0
    model.model_fusenet.train()

    num_iter_per_epoch = len(training_generator)

    try:
        for epoch in range(opt.num_epochs):
            last_epoch = step // num_iter_per_epoch
            if epoch < last_epoch:
                continue

            epoch_loss = []
            progress_bar = tqdm(training_generator)

            saver.base_url = os.path.join(opt.saved_path, 'results', '%03d' % epoch)
            if not opt.sampling:
                for iter, (data, target, name) in enumerate(progress_bar):
                    if iter < step - last_epoch * num_iter_per_epoch:
                        progress_bar.update()
                        continue
                    try:
                        if opt.num_gpus == 1:
                            data = data.cuda()
                            target = target.cuda()

                        optimizer.zero_grad()

                        noisy_gt, texture_nss, texture_fuse, texture_res, \
                        illumi, restor_loss, psnr, ssim = model(data, target, training=True)

                        loss = restor_loss
                        loss.backward()
                        optimizer.step()

                        epoch_loss.append(float(loss))

                        progress_bar.set_description(
                            'Step: {}. Epoch: {}/{}. Iteration: {}/{}. restor_loss: {:.5f},  psnr: {:.5f}, ssim: {:.5f}'.format(
                                step, epoch, opt.num_epochs, iter + 1, num_iter_per_epoch, restor_loss.item(), psnr,
                                ssim))
                        writer.add_scalar('Loss/train', loss, step)
                        writer.add_scalar('PSNR/train', psnr, step)
                        writer.add_scalar('SSIM/train', ssim, step)

                        # log learning_rate
                        current_lr = optimizer.param_groups[0]['lr']
                        writer.add_scalar('learning_rate', current_lr, step)

                        step += 1

                    except Exception as e:
                        print('[Error]', traceback.format_exc())
                        print(e)
                        continue

            if not opt.no_sche:
                scheduler.step()

            if epoch % opt.val_interval == 0:
                model.model_fusenet.eval()
                loss_ls = []
                psnrs = []
                ssims = []

                for iter, (data, target, name) in enumerate(val_generator):
                    with torch.no_grad():
                        if opt.num_gpus == 1:
                            data = data.cuda()
                            target = target.cuda()

                        noisy_gt, texture_nss, texture_fuse, texture_res, \
                        illumi, restor_loss, psnr, ssim = model(data, target, training=False)
                        texture_gt, _, _ = torch.split(kornia.color.rgb_to_ycbcr(target), 1, dim=1)

                        saver.save_image(noisy_gt, name=os.path.splitext(name[0])[0] + '_in')
                        saver.save_image(texture_nss.transpose(0, 1), name=os.path.splitext(name[0])[0] + '_ns')
                        saver.save_image(texture_fuse, name=os.path.splitext(name[0])[0] + '_fuse')
                        saver.save_image(texture_res, name=os.path.splitext(name[0])[0] + '_res')
                        saver.save_image(illumi, name=os.path.splitext(name[0])[0] + '_ill')
                        saver.save_image(target, name=os.path.splitext(name[0])[0] + '_gt')

                        loss = restor_loss
                        loss_ls.append(loss.item())
                        psnrs.append(psnr)
                        ssims.append(ssim)

                loss = np.mean(np.array(loss_ls))
                psnr = np.mean(np.array(psnrs))
                ssim = np.mean(np.array(ssims))

                print(
                    'Val. Epoch: {}/{}. Loss: {:1.5f}, psnr: {:.5f}, ssim: {:.5f}'.format(
                        epoch, opt.num_epochs, loss, psnr, ssim))
                writer.add_scalar('Loss/val', loss, step)
                writer.add_scalar('PSNR/val', psnr, step)
                writer.add_scalar('SSIM/val', ssim, step)

                save_checkpoint(model, f'{opt.model3}_{"%03d" % epoch}_{psnr}_{ssim}_{step}.pth')

                model.model_fusenet.train()
            if opt.sampling:
                exit(0)
    except KeyboardInterrupt:
        save_checkpoint(model, f'{opt.model3}_{epoch}_{step}_keyboardInterrupt.pth')
        writer.close()
    writer.close()


def save_checkpoint(model, name):
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.model_fusenet.state_dict(), os.path.join(opt.saved_path, name))
    else:
        torch.save(model.model_fdnet.state_dict(), os.path.join(opt.saved_path, name))


if __name__ == '__main__':
    opt = get_args()
    train(opt)
