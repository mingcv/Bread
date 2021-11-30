import argparse
import os

import kornia
import torch
import torch.nn.functional as F
import tqdm
from torch import nn
from torch.utils.data import DataLoader

import models
from datasets import LowLightDataset
from tools import saver, mutils
from models import PSNR, SSIM
import numpy as np


def get_args():
    parser = argparse.ArgumentParser('Breaking Downing the Darkness')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus being used')
    parser.add_argument('--num_workers', type=int, default=12, help='num_workers of dataloader')
    parser.add_argument('--batch_size', type=int, default=4, help='The number of images per batch among all devices')
    parser.add_argument('-m1', '--model1', type=str, default='IANet', help='Model1 Name')
    parser.add_argument('-m2', '--model2', type=str, default='NSNet', help='Model2 Name')
    parser.add_argument('-m3', '--model3', type=str, default='FuseNet', help='Model3 Name')
    parser.add_argument('-m4', '--model4', type=str, default=None, help='Model4 Name')

    parser.add_argument('-m1w', '--model1_weight', type=str, default=None, help='Model weight of IAN')
    parser.add_argument('-m2w', '--model2_weight', type=str, default=None, help='Model weight of ANSN')
    parser.add_argument('-m3w', '--model3_weight', type=str, default=None, help='Model weight of CAN')
    parser.add_argument('-m4w', '--model4_weight', type=str, default=None, help='Model weight of NFM')

    parser.add_argument('--mef', action='store_true', help='using color adation based MEF data or not')
    parser.add_argument('--gc', action='store_true', help='using gamma correction or not')
    parser.add_argument('--save_extra', action='store_true', help='save intermediate outputs or not')

    parser.add_argument('--comment', type=str, default='default',
                        help='Project comment')

    parser.add_argument('--alpha', '-a', type=float, default=0.10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--optim', type=str, default='adamw', help='select optimizer for training, '
                                                                   'suggest using \'admaw\' until the'
                                                                   ' very final stage then switch to \'sgd\'')
    parser.add_argument('--data_path', type=str, default='./data/LOL/eval',
                        help='the root folder of dataset')
    parser.add_argument('--log_path', type=str, default='logs/')
    parser.add_argument('--saved_path', type=str, default='logs/')
    args = parser.parse_args()
    return args


class ModelBreadNet(nn.Module):
    def __init__(self, model1, model2, model3, model4):
        super().__init__()
        self.eps = 1e-6
        self.model_ianet = model1(in_channels=1, out_channels=1)
        self.model_nsnet = model2(in_channels=2, out_channels=1)
        self.model_canet = model3(in_channels=4, out_channels=2) if opt.mef else model3(in_channels=6, out_channels=2)
        self.model_fdnet = model4(in_channels=3, out_channels=1) if opt.model4 else None
        self.load_weight(self.model_ianet, opt.model1_weight)
        self.load_weight(self.model_nsnet, opt.model2_weight)
        self.load_weight(self.model_canet, opt.model3_weight)
        self.load_weight(self.model_fdnet, opt.model4_weight)

    def load_weight(self, model, weight_pth):
        if model is not None:
            state_dict = torch.load(weight_pth)
            ret = model.load_state_dict(state_dict, strict=True)
            print(ret)

    def noise_syn_exp(self, illumi, strength):
        return torch.exp(-illumi) * strength

    def forward(self, image, image_gt):
        # Color space mapping
        texture_in, cb_in, cr_in = torch.split(kornia.color.rgb_to_ycbcr(image), 1, dim=1)
        texture_gt, _, _ = torch.split(kornia.color.rgb_to_ycbcr(image_gt), 1, dim=1)

        # Illumination prediction
        texture_in_down = F.interpolate(texture_in, scale_factor=0.5, mode='bicubic', align_corners=True)
        texture_illumi = self.model_ianet(texture_in_down)
        texture_illumi = F.interpolate(texture_illumi, scale_factor=2, mode='bicubic', align_corners=True)

        # Illumination adjustment
        texture_illumi = torch.clamp(texture_illumi, 0., 1.)
        texture_ia = texture_in / torch.clamp_min(texture_illumi, self.eps)
        texture_ia = torch.clamp(texture_ia, 0., 1.)

        # Noise suppression and fusion
        texture_nss = []
        for strength in [0., 0.05, 0.1]:
            attention = self.noise_syn_exp(texture_illumi, strength=strength)
            texture_res = self.model_nsnet(torch.cat([texture_ia, attention], dim=1))
            texture_ns = texture_ia + texture_res
            texture_nss.append(texture_ns)
        texture_nss = torch.cat(texture_nss, dim=1).detach()
        texture_fd = self.model_fdnet(texture_nss)

        # Gamma correction to align the brightness with ground truth;
        # other methods involved in our main paper are also conducted the same correction for evaluation.
        if opt.gc:
            max_psnr = 0
            best = None
            for ga in np.arange(0.1, 2.0, 0.01):
                tx_en = texture_fd ** ga
                psnr = PSNR(tx_en, texture_gt)
                if psnr > max_psnr:
                    max_psnr = psnr
                    best = tx_en

            texture_fd = torch.clamp(best, 0, 1)

        # Color adaption
        if not opt.mef:
            image_ia_ycbcr = kornia.color.rgb_to_ycbcr(torch.clamp(image / (texture_illumi + self.eps), 0, 1))
            _, cb_ia, cr_ia = torch.split(image_ia_ycbcr, 1, dim=1)
            colors = self.model_canet(torch.cat([texture_in, cb_in, cr_in, texture_fd, cb_ia, cr_ia], dim=1))
        else:
            colors = self.model_canet(
                torch.cat([texture_in, cb_in, cr_in, texture_fd], dim=1))

        cb_out, cr_out = torch.split(colors, 1, dim=1)
        cb_out = torch.clamp(cb_out, 0, 1)
        cr_out = torch.clamp(cr_out, 0, 1)

        # Color space mapping
        image_out = kornia.color.ycbcr_to_rgb(
            torch.cat([texture_fd, cb_out, cr_out], dim=1))
        image_out = torch.clamp(image_out, 0, 1)

        # Calculating image quality metrics
        psnr = PSNR(image_out, image_gt)
        ssim = SSIM(image_out, image_gt).item()

        return texture_ia, texture_nss, texture_fd, image_out, texture_illumi, texture_res, psnr, ssim


def evaluation(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    timestamp = mutils.get_formatted_time()
    opt.saved_path = opt.saved_path + f'/{opt.comment}/{timestamp}'
    os.makedirs(opt.saved_path, exist_ok=True)

    val_params = {'batch_size': 1,
                  'shuffle': False,
                  'drop_last': False,
                  'num_workers': opt.num_workers}

    val_set = LowLightDataset(opt.data_path)

    val_generator = DataLoader(val_set, **val_params)
    val_generator = tqdm.tqdm(val_generator)

    model1 = getattr(models, opt.model1)
    model2 = getattr(models, opt.model2)
    model3 = getattr(models, opt.model3)
    model4 = getattr(models, opt.model4) if opt.model4 else None

    model = ModelBreadNet(model1, model2, model3, model4)
    print(model)

    if opt.num_gpus > 0:
        model = model.cuda()
        if opt.num_gpus > 1:
            model = nn.DataParallel(model)

    model.eval()
    psnrs, ssims, fns = [], [], []
    for iter, (data, target, name) in enumerate(val_generator):
        saver.base_url = os.path.join(opt.saved_path, 'results')
        with torch.no_grad():
            if opt.num_gpus == 1:
                data = data.cuda()
                target = target.cuda()
            texture_in, _, _ = torch.split(kornia.color.rgb_to_ycbcr(data), 1, dim=1)
            texture_gt, _, _ = torch.split(kornia.color.rgb_to_ycbcr(target), 1, dim=1)
            texture_ia, texture_nss, texture_fd, image_out, \
            texture_illumi, texture_res, psnr, ssim = model(data, target)
            if opt.save_extra:
                saver.save_image(data, name=os.path.splitext(name[0])[0] + '_im_in')
                saver.save_image(target, name=os.path.splitext(name[0])[0] + '_im_gt')

                saver.save_image(texture_in, name=os.path.splitext(name[0])[0] + '_y_in')
                saver.save_image(texture_gt, name=os.path.splitext(name[0])[0] + '_y_gt')

                saver.save_image(texture_ia, name=os.path.splitext(name[0])[0] + '_ia')
                for i in range(texture_nss.shape[1]):
                    saver.save_image(texture_nss[:, i, ...], name=os.path.splitext(name[0])[0] + f'_ns_{i}')
                saver.save_image(texture_fd, name=os.path.splitext(name[0])[0] + '_fd')

                saver.save_image(texture_illumi, name=os.path.splitext(name[0])[0] + '_illumi')
                saver.save_image(texture_res, name=os.path.splitext(name[0])[0] + '_res')

                saver.save_image(image_out, name=os.path.splitext(name[0])[0] + '_out')
            else:
                saver.save_image(image_out, name=os.path.splitext(name[0])[0] + '_Bread')

            psnrs.append(psnr)
            ssims.append(ssim)
            fns.append(name[0])

    results = list(zip(psnrs, ssims, fns))
    results.sort(key=lambda item: item[0])
    for r in results:
        print(*r)
    psnr = np.mean(np.array(psnrs))
    ssim = np.mean(np.array(ssims))
    print('psnr: ', psnr, ', ssim: ', ssim)


if __name__ == '__main__':
    opt = get_args()
    evaluation(opt)
