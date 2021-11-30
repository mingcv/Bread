import os
import random

import torch.utils.data as data
import torchvision.transforms as T
from PIL import Image


class MEFDataset(data.Dataset):
    def __init__(self, root):
        self.img_root = root

        self.numbers = list(sorted(os.listdir(self.img_root)))
        print(len(self.numbers))

        self.preproc = T.Compose(
            [T.ToTensor()]
        )

    def __getitem__(self, idx):
        number = self.numbers[idx]
        im_dir = os.path.join(self.img_root, number)
        fn1, fn2 = tuple(random.sample(os.listdir(im_dir), k=2))
        fp1 = os.path.join(im_dir, fn1)
        fp2 = os.path.join(im_dir, fn2)
        img1 = Image.open(fp1).convert("RGB")
        img2 = Image.open(fp2).convert("RGB")
        img1 = self.preproc(img1)
        img2 = self.preproc(img2)

        fn1 = f'{number}_{fn1}'
        fn2 = f'{number}_{fn2}'
        return img1, img2, fn1, fn2

    def __len__(self):
        return len(self.numbers)
