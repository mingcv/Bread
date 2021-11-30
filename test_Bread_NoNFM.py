import argparse
import os

import kornia
import torch
import torch.nn.functional as F
import tqdm
from torch import nn
from torch.utils.data import DataLoader

import models
from datasets import LowLightDatasetTest
from tools import saver, mutils


def get_args():
    parser = argparse.ArgumentParser('Breaking Downing the Darkness')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus being used')
    parser.add_argument('--num_workers', type=int, default=12, help='num_workers of dataloader')
    parser.add_argument('--batch_size', type=int, default=4, help='The number of images per batch among all devices')
    parser.add_argument('-m1', '--model1', type=str, default='IAN', help='Model1 Name')
    parser.add_argument('-m2', '--model2', type=str, default='ANSN', help='Model2 Name')
    parser.add_argument('-m3', '--model3', type=str, default='FuseNet', help='Model3 Name')

    parser.add_argument('-m1w', '--model1_weight', type=str, default=None, help='Model weight of IAN')
    parser.add_argument('-m2w', '--model2_weight', type=str, default=None, help='Model weight of ANSN')
    parser.add_argument('-m3w', '--model3_weight', type=str, default=None, help='Model weight of CAN')

    parser.add_argument('--mef', action='store_true')
    parser.add_argument('--save_extra', action='store_true', help='save intermediate outputs or not')

    parser.add_argument('--comment', type=str, default='default',
                        help='Project comment')

    parser.add_argument('--alpha', '-a', type=float, default=0.10)

    parser.add_argument('--data_path', type=str, default='./data/test',
                        help='the root folder of dataset')
    parser.add_argument('--log_path', type=str, default='logs/')
    parser.add_argument('--saved_path', type=str, default='logs/')
    args = parser.parse_args()
    return args


class ModelBreadNet(nn.Module):
    def __init__(self, model1, model2, model3):
        super().__init__()
        self.eps = 1e-6
        self.model_ianet = model1(in_channels=1, out_channels=1)
        self.model_nsnet = model2(in_channels=2, out_channels=1)
        self.model_canet = model3(in_channels=4, out_channels=2) if opt.mef else model3(in_channels=6, out_channels=2)

        self.load_weight(self.model_ianet, opt.model1_weight)
        self.load_weight(self.model_nsnet, opt.model2_weight)
        self.load_weight(self.model_canet, opt.model3_weight)

    def load_weight(self, model, weight_pth):
        if model is not None:
            state_dict = torch.load(weight_pth)
            ret = model.load_state_dict(state_dict, strict=True)
            print(ret)

    def noise_syn_exp(self, illumi, strength):
        return torch.exp(-illumi) * strength

    def forward(self, image):
        # Color space mapping
        texture_in, cb_in, cr_in = torch.split(kornia.color.rgb_to_ycbcr(image), 1, dim=1)

        # Illumination prediction
        texture_in_down = F.interpolate(texture_in, scale_factor=0.5, mode='bicubic', align_corners=True)
        texture_illumi = self.model_ianet(texture_in_down)
        texture_illumi = F.interpolate(texture_illumi, scale_factor=2, mode='bicubic', align_corners=True)

        # Illumination adjustment
        texture_illumi = torch.clamp(texture_illumi, 0., 1.)
        texture_ia = texture_in / torch.clamp_min(texture_illumi, self.eps)
        texture_ia = torch.clamp(texture_ia, 0., 1.)

        # Noise suppression and fusion
        attention = self.noise_syn_exp(texture_illumi, strength=opt.alpha)
        texture_res = self.model_nsnet(torch.cat([texture_ia, attention], dim=1))
        texture_ns = texture_ia + texture_res

        # Further preserve the texture under brighter illumination
        texture_ns = texture_illumi * texture_in + (1 - texture_illumi) * texture_ns
        texture_ns = torch.clamp(texture_ns, 0, 1)

        # Color adaption
        colors = self.model_canet(
            torch.cat([texture_in, cb_in, cr_in, texture_ns], dim=1))
        cb_out, cr_out = torch.split(colors, 1, dim=1)
        cb_out = torch.clamp(cb_out, 0, 1)
        cr_out = torch.clamp(cr_out, 0, 1)

        # Color space mapping
        image_out = kornia.color.ycbcr_to_rgb(
            torch.cat([texture_ns, cb_out, cr_out], dim=1))

        # Further preserve the color under brighter illumination
        img_fusion = texture_illumi * image + (1 - texture_illumi) * image_out
        _, cb_fuse, cr_fuse = torch.split(kornia.color.rgb_to_ycbcr(img_fusion), 1, dim=1)
        image_out = kornia.color.ycbcr_to_rgb(
            torch.cat([texture_ns, cb_fuse, cr_fuse], dim=1))
        image_out = torch.clamp(image_out, 0, 1)

        return texture_ia, texture_ns, image_out, texture_illumi, texture_res


def test(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    timestamp = mutils.get_formatted_time()
    opt.saved_path = opt.saved_path + f'/{opt.comment}/{timestamp}'
    os.makedirs(opt.saved_path, exist_ok=True)

    test_params = {'batch_size': 1,
                   'shuffle': False,
                   'drop_last': False,
                   'num_workers': opt.num_workers}

    test_set = LowLightDatasetTest(opt.data_path)

    test_generator = DataLoader(test_set, **test_params)
    test_generator = tqdm.tqdm(test_generator)

    model1 = getattr(models, opt.model1)
    model2 = getattr(models, opt.model2)
    model3 = getattr(models, opt.model3)

    model = ModelBreadNet(model1, model2, model3)
    print(model)

    if opt.num_gpus > 0:
        model = model.cuda()
        if opt.num_gpus > 1:
            model = nn.DataParallel(model)

    model.eval()

    for iter, (data, subset, name) in enumerate(test_generator):
        saver.base_url = os.path.join(opt.saved_path, 'results', subset[0])
        with torch.no_grad():
            if opt.num_gpus == 1:
                data = data.cuda()
            texture_in, _, _ = torch.split(kornia.color.rgb_to_ycbcr(data), 1, dim=1)

            texture_ia, texture_ns, image_out, texture_illumi, texture_res = model(data)

            if opt.save_extra:
                saver.save_image(data, name=os.path.splitext(name[0])[0] + '_im_in')
                saver.save_image(texture_in, name=os.path.splitext(name[0])[0] + '_y_in')
                saver.save_image(texture_ia, name=os.path.splitext(name[0])[0] + '_ia')
                saver.save_image(texture_ns, name=os.path.splitext(name[0])[0] + '_ns')

                saver.save_image(texture_illumi, name=os.path.splitext(name[0])[0] + '_illumi')
                saver.save_image(texture_res, name=os.path.splitext(name[0])[0] + '_res')
                saver.save_image(image_out, name=os.path.splitext(name[0])[0] + '_out')
            else:
                saver.save_image(image_out, name=os.path.splitext(name[0])[0] + '_Bread')

if __name__ == '__main__':
    opt = get_args()
    test(opt)
