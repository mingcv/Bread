import os

import torch.utils.data as data
import torchvision.transforms as T
from PIL import Image


class LowLightDatasetTest(data.Dataset):
    def __init__(self, root, reside=False):
        self.root = root
        self.items = []

        subsets = os.listdir(root)
        for subset in subsets:
            img_root = os.path.join(root, subset)
            img_names = list(sorted(os.listdir(img_root)))

            for img_name in img_names:
                self.items.append((
                    os.path.join(img_root, img_name),
                    subset,
                    img_name
                ))

        self.preproc = T.Compose(
            [T.ToTensor()]
        )
        self.preproc_raw = T.Compose(
            [T.ToTensor()]
        )

    def __getitem__(self, idx):
        img_path, subset, img_name = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((img.width // 8 * 8, img.height // 8 * 8), Image.ANTIALIAS)
        img_raw = self.preproc_raw(img)

        return img_raw, subset, img_name

    def __len__(self):
        return len(self.items)
