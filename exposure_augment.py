import math
import os

import PIL.Image as Image
import numpy as np
import torch
import torchvision.transforms as vtrans
import tqdm


def main(fip, fod):
    max_overex_rate = 0.25
    steps = 20
    num_gen = 4

    im = Image.open(fip)
    im = vtrans.ToTensor()(im)
    im_max = torch.flatten(torch.max(im, dim=0, keepdim=True).values)
    mag = 1. / torch.topk(im_max, math.floor(len(im_max) * max_overex_rate + 1)).values
    mag = mag[range(0, len(mag), int(len(mag) * (1. / steps)))]
    mag_diff = torch.diff(mag, 1)
    mag = mag[:-1]

    top_mag_diff = torch.topk(mag_diff, num_gen).values
    min_gain = top_mag_diff[top_mag_diff > 0][-1]
    min_mag = mag[0]
    max_mag = mag[mag_diff > min_gain][-1]
    fn, ext = os.path.basename(fip).split('.')
    bar.set_description(f'{fn}: {min_gain}')
    ma = np.arange(1, min_mag - min_gain, min_gain * 2)
    if len(ma) > num_gen:
        mags = np.append(np.linspace(1, min_mag - min_gain, num_gen),
                         np.linspace(min_mag, max_mag, num_gen))
    elif len(ma) == num_gen:
        mags = np.append(ma, np.linspace(min_mag, max_mag, num_gen))
    else:
        mags = np.linspace(1, max_mag, num_gen * 2)

    im = Image.open(fip)
    im_raw = vtrans.ToTensor()(im)

    for i, mag in enumerate(mags):
        im = im_raw * mag
        im.clamp_max_(1.)
        fop = os.path.join(fod, f'{fn}_{i}.{ext}')

        if not os.path.exists(fop):
            vtrans.ToPILImage()(im).save(fop)


if __name__ == '__main__':
    # one needs to download it online
    fid = './data/LOL/train/images'
    fod = './data/LOL/train/images_aug'
    os.makedirs(fod, exist_ok=True)

    bar = tqdm.tqdm(os.listdir(fid))
    for fn in bar:
        fip = os.path.join(fid, fn)
        main(fip, fod)
