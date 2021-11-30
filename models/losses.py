import torch
import torch.nn as nn
from pytorch_msssim import SSIM, MS_SSIM
from torch.nn import L1Loss, MSELoss
from torchvision.models import vgg16
import torch.nn.functional as F


def compute_gradient(img):
    gradx = img[..., 1:, :] - img[..., :-1, :]
    grady = img[..., 1:] - img[..., :-1]
    return gradx, grady


class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, predict, target):
        predict_gradx, predict_grady = compute_gradient(predict)
        target_gradx, target_grady = compute_gradient(target)

        return self.loss(predict_gradx, target_gradx) + self.loss(predict_grady, target_grady)


class SSIMLoss(nn.Module):
    def __init__(self, channels):
        super(SSIMLoss, self).__init__()
        self.ssim = SSIM(data_range=1., size_average=True, channel=channels)

    def forward(self, output, target):
        ssim_loss = 1 - self.ssim(output, target)
        return ssim_loss


class SSIML1Loss(nn.Module):
    def __init__(self, channels):
        super(SSIML1Loss, self).__init__()
        self.l1_loss_func = nn.L1Loss()
        self.ssim = SSIM(data_range=1., size_average=True, channel=channels)
        self.alpha = 1.4

    def forward(self, output, target):
        l1_loss = self.l1_loss_func(output, target)
        ssim_loss = 1 - self.ssim(output, target)
        total_loss = l1_loss + self.alpha * ssim_loss
        return total_loss


class GradSSIML1Loss(nn.Module):
    def __init__(self, channels):
        super(GradSSIML1Loss, self).__init__()
        self.l1_loss_func = nn.L1Loss()
        self.ssim = SSIM(data_range=1., size_average=True, channel=channels)
        self.grad_loss_func = GradientLoss()
        self.alpha = 1.4

    def forward(self, output, target):
        l1_loss = self.l1_loss_func(output, target)
        ssim_loss = 1 - self.ssim(output, target)
        grad_loss = self.grad_loss_func(output, target)
        total_loss = l1_loss + self.alpha * ssim_loss + 0.2 * grad_loss
        return total_loss


class SSIML2Loss(nn.Module):
    def __init__(self, channels):
        super(SSIML2Loss, self).__init__()
        self.l2_loss_func = nn.MSELoss()
        self.ssim = SSIM(data_range=1., size_average=True, channel=channels)
        self.alpha = 1.

    def forward(self, output, target):
        l2_loss = self.l2_loss_func(output, target)
        ssim_loss = 1 - self.ssim(output, target)
        total_loss = l2_loss + self.alpha * ssim_loss
        return total_loss


class MSSSIML1Loss(nn.Module):
    def __init__(self, channels):
        super(MSSSIML1Loss, self).__init__()
        self.l1_loss_func = nn.L1Loss()
        self.ms_ssim = MS_SSIM(data_range=1., size_average=True, channel=channels)
        self.alpha = 1.0

    def forward(self, output, target):
        ms_ssim_loss = 1 - self.ms_ssim(output, target)
        l1_loss = self.l1_loss_func(output, target)
        total_loss = l1_loss + self.alpha * ms_ssim_loss
        return total_loss


class MSSSIML2Loss(nn.Module):
    def __init__(self, channels):
        super(MSSSIML2Loss, self).__init__()
        self.l2_loss_func = nn.MSELoss()
        self.ms_ssim = MS_SSIM(data_range=1., size_average=True, channel=channels)
        # self.alpha = 0.84
        self.alpha = 1.2

    def forward(self, output, target):
        l2_loss = self.l2_loss_func(output, target)
        ms_ssim_loss = 1 - self.ms_ssim(output, target)
        total_loss = l2_loss + self.alpha * ms_ssim_loss
        return total_loss


class PerLoss(torch.nn.Module):
    def __init__(self):
        super(PerLoss, self).__init__()
        vgg_model = vgg16(pretrained=True).features[:16]
        vgg_model = vgg_model.to('cuda')
        for param in vgg_model.parameters():
            param.requires_grad = False

        self.vgg_layers = vgg_model

        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, data, gt):
        loss = []
        if data.shape[1] == 1:
            data = data.repeat(1, 3, 1, 1)
            gt = gt.repeat(1, 3, 1, 1)

        dehaze_features = self.output_features(data)
        gt_features = self.output_features(gt)
        for dehaze_feature, gt_feature in zip(dehaze_features, gt_features):
            loss.append(F.mse_loss(dehaze_feature, gt_feature))

        return sum(loss) / len(loss)


class PerL1Loss(torch.nn.Module):
    def __init__(self):
        super(PerL1Loss, self).__init__()
        self.l1_loss_func = nn.L1Loss()
        self.per_loss_func = PerLoss().to('cuda')

    def forward(self, output, target):
        l1_loss = self.l1_loss_func(output, target)
        per_loss = self.per_loss_func(output, target)
        # total_loss = l1_loss + 0.04 * per_loss
        total_loss = l1_loss + 0.2 * per_loss
        return total_loss


class MSPerL1Loss(torch.nn.Module):
    def __init__(self, channels):
        super(MSPerL1Loss, self).__init__()
        self.l1_loss_func = nn.L1Loss()
        self.ms_ssim = MS_SSIM(data_range=1., size_average=True, channel=channels)
        self.per_loss_func = PerLoss().to('cuda')

    def forward(self, output, target):
        ms_ssim_loss = 1 - self.ms_ssim(output, target)
        l1_loss = self.l1_loss_func(output, target)
        per_loss = self.per_loss_func(output, target)
        total_loss = l1_loss + 1.2 * ms_ssim_loss + 0.04 * per_loss
        return total_loss


class MSPerL2Loss(torch.nn.Module):
    def __init__(self):
        super(MSPerL2Loss, self).__init__()
        self.l2_loss_func = nn.MSELoss()
        self.ms_ssim = MS_SSIM(data_range=1., size_average=True, channel=3)
        self.per_loss_func = PerLoss().to('cuda')

    def forward(self, output, target):
        ms_ssim_loss = 1 - self.ms_ssim(output, target)
        l2_loss = self.l2_loss_func(output, target)
        per_loss = self.per_loss_func(output, target)
        total_loss = l2_loss + 0.16 * ms_ssim_loss + 0.2 * per_loss
        return total_loss


class TVLoss(torch.nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, data):
        w_variance = torch.sum(torch.pow(data[:, :, :, :-1] - data[:, :, :, 1:], 2))
        h_variance = torch.sum(torch.pow(data[:, :, :-1, :] - data[:, :, 1:, :], 2))

        count_h = self._tensor_size(data[:, :, 1:, :])
        count_w = self._tensor_size(data[:, :, :, 1:])

        tv_loss = h_variance / count_h + w_variance / count_w
        return tv_loss

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def safe_div(a, b, eps=1e-2):
    return a / torch.clamp_min(b, eps)


class WTVLoss(torch.nn.Module):
    def __init__(self):
        super(WTVLoss, self).__init__()
        self.eps = 1e-2

    def forward(self, data, aux):
        data_dw = data[:, :, :, :-1] - data[:, :, :, 1:]
        data_dh = data[:, :, :-1, :] - data[:, :, 1:, :]
        aux_dw = torch.abs(aux[:, :, :, :-1] - aux[:, :, :, 1:])
        aux_dh = torch.abs(aux[:, :, :-1, :] - aux[:, :, 1:, :])

        w_variance = torch.sum(torch.pow(safe_div(data_dw, aux_dw, self.eps), 2))
        h_variance = torch.sum(torch.pow(safe_div(data_dh, aux_dh, self.eps), 2))

        count_h = self._tensor_size(data[:, :, 1:, :])
        count_w = self._tensor_size(data[:, :, :, 1:])

        tv_loss = h_variance / count_h + w_variance / count_w
        return tv_loss

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class WTVLoss2(torch.nn.Module):
    def __init__(self):
        super(WTVLoss2, self).__init__()
        self.eps = 1e-2
        self.criterion = nn.MSELoss()

    def forward(self, data, aux):
        N, C, H, W = data.shape

        data_dw = F.pad(torch.abs(data[:, :, :, :-1] - data[:, :, :, 1:]), (1, 0, 0, 0))
        data_dh = F.pad(torch.abs(data[:, :, :-1, :] - data[:, :, 1:, :]), (0, 0, 1, 0))
        aux_dw = F.pad(torch.abs(aux[:, :, :, :-1] - aux[:, :, :, 1:]), (1, 0, 0, 0))
        aux_dh = F.pad(torch.abs(aux[:, :, :-1, :] - aux[:, :, 1:, :]), (0, 0, 1, 0))

        data_d = data_dw + data_dh
        aux_d = aux_dw + aux_dh

        loss1 = self.criterion(data_d, aux_d)
        # loss2 = torch.norm(data_d / (aux_d + self.eps), p=1) / (C * H * W)
        loss2 = torch.norm(data_d / (aux_d + self.eps)) / (C * H * W)
        return loss1 * 0.5 + loss2 * 4.0


class MSTVPerL1Loss(torch.nn.Module):
    def __init__(self):
        super(MSTVPerL1Loss, self).__init__()
        self.l1_loss_func = nn.L1Loss()
        self.ms_ssim = MS_SSIM(data_range=1., size_average=True, channel=3)
        self.per_loss_func = PerLoss().to('cuda')
        self.tv_loss_func = TVLoss()

    def forward(self, output, target):
        ms_ssim_loss = 1 - self.ms_ssim(output, target)
        l1_loss = self.l1_loss_func(output, target)
        per_loss = self.per_loss_func(output, target)
        tv_loss = self.tv_loss_func(output)
        total_loss = l1_loss + 1.2 * ms_ssim_loss + 0.04 * per_loss + 1e-7 * tv_loss
        return total_loss


if __name__ == "__main__":
    MSTVPerL1Loss()
