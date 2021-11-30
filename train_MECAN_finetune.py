# original author: signatrix
# adapted from https://github.com/signatrix/efficientdet/blob/master/train.py
# modified by Zylo117

import argparse
import datetime
import os
import traceback

import kornia
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

import models
from datasets import MEFDataset, LowLightDataset, LowLightDatasetReverse
from models import PSNR, SSIM, CosineLR
from tools import SingleSummaryWriter
from tools import saver, mutils


def get_args():
    parser = argparse.ArgumentParser('Breaking Downing the Darkness')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus being used')
    parser.add_argument('--num_workers', type=int, default=12, help='num_workers of dataloader')
    parser.add_argument('--batch_size', type=int, default=1, help='The number of images per batch among all devices')
    parser.add_argument('-m', '--model', type=str, default='INet',
                        help='Model Name')
    parser.add_argument('-mw', '--model_weight', type=str, default=None,
                        help='Model weight')
    parser.add_argument('-ts', '--targets_split', type=str, default='targets',
                        help='dir of targets')
    parser.add_argument('--comment', type=str, default='default',
                        help='Project comment')
    parser.add_argument('--graph', action='store_true')
    parser.add_argument('--scratch', action='store_true')
    parser.add_argument('--sampling', action='store_true')
    parser.add_argument('--test_on_start', action='store_true')

    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--no_sche', action='store_true')

    parser.add_argument('--optim', type=str, default='adam', help='select optimizer for training, '
                                                                  'suggest using \'admaw\' until the'
                                                                  ' very final stage then switch to \'sgd\'')
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
    parser.add_argument('--save_interval', type=int, default=500, help='Number of steps between saving')
    parser.add_argument('--data_path1', type=str, default='./data/SICE',
                        help='the root folder of dataset')
    parser.add_argument('--data_path2', type=str, default='./data/LOL',
                        help='the root folder of dataset')
    parser.add_argument('--log_path', type=str, default='logs/')

    parser.add_argument('--saved_path', type=str, default='logs/')
    args = parser.parse_args()
    return args


def compute_gradient(img):
    gradx = img[..., 1:, :] - img[..., :-1, :]
    grady = img[..., 1:] - img[..., :-1]
    return gradx, grady


class ModelINet(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.color_loss = models.SSIML1Loss(channels=1)
        self.restor_loss = models.SSIML1Loss(channels=3)
        self.model_canet = model(in_channels=4, out_channels=2)
        self.eps = 1e-2
        self.load_weight(self.model_canet, opt.model_weight)

    def load_weight(self, model, weight_pth):
        state_dict = torch.load(weight_pth)
        ret = model.load_state_dict(state_dict, strict=True)
        print(ret)

    def forward(self, image, image_gt, training=True):
        texture_in, cb_in, cr_in = torch.split(kornia.color.rgb_to_ycbcr(image), 1, dim=1)
        texture_gt, cb_gt, cr_gt = torch.split(kornia.color.rgb_to_ycbcr(image_gt), 1, dim=1)

        colors = self.model_canet(torch.cat([texture_in, cb_in, cr_in, texture_gt], dim=1))
        cb, cr = torch.split(colors, 1, dim=1)

        color_loss1 = self.color_loss(cb, cb_gt)
        color_loss2 = self.color_loss(cr, cr_gt)

        image_out = kornia.color.ycbcr_to_rgb(torch.cat([texture_gt, cb, cr], dim=1))
        restor_loss = self.restor_loss(image_out, image_gt)

        psnr = PSNR(image_out, image_gt)
        ssim = SSIM(image_out, image_gt).item()
        return image_out, color_loss1, color_loss2, restor_loss, psnr, ssim


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    # params.project_name = params.project_name + str(time.time()).replace('.', '')
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

    training_set1 = MEFDataset(os.path.join(opt.data_path1, 'train'))
    training_set2 = LowLightDataset(os.path.join(opt.data_path2, 'train'), color_tuning=True,
                                    targets_split=opt.targets_split)
    training_set3 = LowLightDatasetReverse(os.path.join(opt.data_path2, 'train'), color_tuning=True,
                                           targets_split=opt.targets_split)
    training_set = torch.utils.data.ConcatDataset([training_set1, training_set2, training_set3])
    training_generator = DataLoader(training_set, **training_params)

    # val_set = MEFDataset(os.path.join(opt.data_path1, 'eval'))
    val_set = LowLightDataset(os.path.join(opt.data_path2, 'eval'), color_tuning=True)
    # val_set = torch.utils.data.ConcatDataset([val_set1, val_set2])
    val_generator = DataLoader(val_set, **val_params)

    model = getattr(models, opt.model)

    model = ModelINet(model)
    print(model)

    writer = SingleSummaryWriter(opt.log_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')


    if opt.num_gpus > 0:
        model = model.cuda()
        if opt.num_gpus > 1:
            model = nn.DataParallel(model)

    if opt.optim == 'adam':
        optimizer = torch.optim.Adam(model.model_canet.parameters(), opt.lr)
    else:
        optimizer = torch.optim.SGD(model.model_canet.parameters(), opt.lr, momentum=0.9, nesterov=True)

    scheduler = CosineLR(optimizer, opt.lr, opt.num_epochs)
    epoch = 0
    step = 0
    model.model_canet.train()

    num_iter_per_epoch = len(training_generator)

    try:
        for epoch in range(opt.num_epochs):
            last_epoch = step // num_iter_per_epoch
            if epoch < last_epoch:
                continue

            epoch_loss = []
            progress_bar = tqdm(training_generator)
            if not opt.sampling and not opt.test_on_start:
                for iter, (data, target, name1, name2) in enumerate(progress_bar):
                    if iter < step - last_epoch * num_iter_per_epoch:
                        progress_bar.update()
                        continue
                    try:
                        if opt.num_gpus == 1:
                            data, target = data.cuda(), target.cuda()
                        optimizer.zero_grad()

                        image_out, color_loss1, color_loss2, \
                        restor_loss, psnr, ssim = model(data, target, training=True)
                        loss = color_loss1 + color_loss2 + restor_loss
                        loss.backward()
                        optimizer.step()

                        epoch_loss.append(float(loss))

                        progress_bar.set_description(
                            'Step: {}. Epoch: {}/{}. Iteration: {}/{}. color_loss1: {:1.5f}, color_loss2: {:1.5f}, restor_loss: {:1.5f}, psnr: {:.5f}, ssim: {:.5f}'.format(
                                step, epoch, opt.num_epochs, iter + 1, num_iter_per_epoch,
                                color_loss1.item(), color_loss2.item(),
                                restor_loss.item(), psnr, ssim))
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
                    # scheduler.step(np.mean(epoch_loss))

            if opt.no_sche:
                scheduler.step()

            saver.base_url = os.path.join(opt.saved_path, 'results', '%03d' % epoch)

            if epoch % opt.val_interval == 0:
                model.model_canet.eval()
                loss_ls = []
                psnrs = []
                ssims = []

                for iter, (data, target, name1, name2) in enumerate(val_generator):
                    with torch.no_grad():
                        if opt.num_gpus == 1:
                            data, target = data.cuda(), target.cuda()

                        image_out, color_loss1, color_loss2, restor_loss, \
                        psnr, ssim = model(data, target, training=False)
                        saver.save_image(data, name=os.path.splitext(name1[0])[0] + '_im1')
                        saver.save_image(target, name=os.path.splitext(name2[0])[0] + '_im2')
                        saver.save_image(image_out, name=os.path.splitext(name2[0])[0] + '_im2_pred')

                        loss = restor_loss + color_loss1 + color_loss2
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

                save_checkpoint(model, f'{opt.model}_{"%03d" % epoch}_{psnr}_{ssim}_{step}.pth')

                model.model_canet.train()

            opt.test_on_start = False
            if opt.sampling:
                exit(0)
    except KeyboardInterrupt:
        save_checkpoint(model, f'{opt.model}_{epoch}_{step}_keyboardInterrupt.pth')
        writer.close()
    writer.close()


def save_checkpoint(model, name):
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.model_canet.state_dict(), os.path.join(opt.saved_path, name))
    else:
        torch.save(model.model_canet.state_dict(), os.path.join(opt.saved_path, name))


if __name__ == '__main__':
    opt = get_args()
    train(opt)
