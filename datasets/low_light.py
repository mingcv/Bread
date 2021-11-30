import os
import random

import torch
import torch.utils.data as data
import torchvision.transforms as T
from PIL import Image


class LowLightFDataset(data.Dataset):
    def __init__(self, root, image_split='images_aug', targets_split='targets', training=True):
        self.root = root
        self.num_instances = 8
        self.img_root = os.path.join(root, image_split)
        self.target_root = os.path.join(root, targets_split)
        self.training = training
        print('----', image_split, targets_split, '----')
        self.imgs = list(sorted(os.listdir(self.img_root)))
        self.gts = list(sorted(os.listdir(self.target_root)))

        names = [img_name.split('_')[0] + '.' + img_name.split('.')[-1] for img_name in self.imgs]
        self.imgs = list(
            filter(lambda img_name: img_name.split('_')[0] + '.' + img_name.split('.')[-1] in self.gts, self.imgs))

        self.gts = list(filter(lambda gt: gt in names, self.gts))

        print(len(self.imgs), len(self.gts))
        self.preproc = T.Compose(
            [T.ToTensor()]
        )
        self.preproc_gt = T.Compose(
            [T.ToTensor()]
        )

    def __getitem__(self, idx):
        fn, ext = self.gts[idx].split('.')
        imgs = []
        for i in range(self.num_instances):
            img_path = os.path.join(self.img_root, f"{fn}_{i}.{ext}")
            imgs += [self.preproc(Image.open(img_path).convert("RGB"))]

        if self.training:
            random.shuffle(imgs)
        gt_path = os.path.join(self.target_root, self.gts[idx])
        gt = Image.open(gt_path).convert("RGB")
        gt = self.preproc_gt(gt)

        # print(img_path, gt_path)
        return torch.stack(imgs, dim=0), gt, fn

    def __len__(self):
        return len(self.gts)


class LowLightFDatasetEval(data.Dataset):
    def __init__(self, root, targets_split='targets', training=True):
        self.root = root
        self.num_instances = 1
        self.img_root = os.path.join(root, 'images')
        self.target_root = os.path.join(root, targets_split)
        self.training = training

        self.imgs = list(sorted(os.listdir(self.img_root)))
        self.gts = list(sorted(os.listdir(self.target_root)))

        self.imgs = list(filter(lambda img_name: img_name in self.gts, self.imgs))
        self.gts = list(filter(lambda gt: gt in self.imgs, self.gts))

        print(len(self.imgs), len(self.gts))
        self.preproc = T.Compose(
            [T.ToTensor()]
        )
        self.preproc_gt = T.Compose(
            [T.ToTensor()]
        )

    def __getitem__(self, idx):
        fn, ext = self.gts[idx].split('.')
        imgs = []
        for i in range(self.num_instances):
            img_path = os.path.join(self.img_root, f"{fn}.{ext}")
            imgs += [self.preproc(Image.open(img_path).convert("RGB"))]

        gt_path = os.path.join(self.target_root, self.gts[idx])
        gt = Image.open(gt_path).convert("RGB")
        gt = self.preproc_gt(gt)

        # print(img_path, gt_path)
        return torch.stack(imgs, dim=0), gt, fn

    def __len__(self):
        return len(self.gts)


class LowLightDataset(data.Dataset):
    def __init__(self, root, targets_split='targets', color_tuning=False):
        self.root = root
        self.img_root = os.path.join(root, 'images')
        self.target_root = os.path.join(root, targets_split)
        self.color_tuning = color_tuning
        self.imgs = list(sorted(os.listdir(self.img_root)))
        self.gts = list(sorted(os.listdir(self.target_root)))

        self.imgs = list(filter(lambda img_name: img_name in self.gts, self.imgs))
        self.gts = list(filter(lambda gt: gt in self.imgs, self.gts))

        print(len(self.imgs), len(self.gts))
        self.preproc = T.Compose(
            [T.ToTensor()]
        )
        self.preproc_gt = T.Compose(
            [T.ToTensor()]
        )

    def __getitem__(self, idx):
        fn, ext = self.gts[idx].split('.')

        img_path = os.path.join(self.img_root, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        img = self.preproc(img)

        gt_path = os.path.join(self.target_root, self.gts[idx])
        gt = Image.open(gt_path).convert("RGB")
        gt = self.preproc_gt(gt)

        if self.color_tuning:
            return img, gt, 'a' + self.imgs[idx], 'a' + self.imgs[idx]
        else:
            return img, gt, fn

    def __len__(self):
        return len(self.imgs)


class LowLightDatasetReverse(data.Dataset):
    def __init__(self, root, targets_split='targets', color_tuning=False):
        self.root = root
        self.img_root = os.path.join(root, 'images')
        self.target_root = os.path.join(root, targets_split)
        self.color_tuning = color_tuning
        self.imgs = list(sorted(os.listdir(self.img_root)))
        self.gts = list(sorted(os.listdir(self.target_root)))

        self.imgs = list(filter(lambda img_name: img_name in self.gts, self.imgs))
        self.gts = list(filter(lambda gt: gt in self.imgs, self.gts))

        print(len(self.imgs), len(self.gts))
        self.preproc = T.Compose(
            [T.ToTensor()]
        )
        self.preproc_gt = T.Compose(
            [T.ToTensor()]
        )

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_root, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        img = self.preproc(img)

        gt_path = os.path.join(self.target_root, self.gts[idx])
        gt = Image.open(gt_path).convert("RGB")
        gt = self.preproc_gt(gt)

        if self.color_tuning:
            return gt, img, 'a' + self.imgs[idx], 'a' + self.imgs[idx]
        else:
            fn, ext = os.path.splitext(self.imgs[idx])
            return gt, img, '%03d' % int(fn) + ext

    def __len__(self):
        return len(self.imgs)
