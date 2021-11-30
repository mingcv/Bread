import torch
import torch.nn as nn
import os
import time
from tools import mutils

saved_grad = None
saved_name = None

base_url = './results'
os.makedirs(base_url, exist_ok=True)


def normalize_tensor_mm(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())


def normalize_tensor_sigmoid(tensor):
    return nn.functional.sigmoid(tensor)


def save_image(tensor, name=None, save_path=None, exit_flag=False, timestamp=False, norm=False):
    import torchvision.utils as vutils
    os.makedirs(base_url, exist_ok=True)
    if norm:
        tensor = normalize_tensor_mm(tensor)
    grid = vutils.make_grid(tensor.detach().cpu(), nrow=4)

    if save_path:
        vutils.save_image(grid, save_path)
    else:
        if timestamp:
            vutils.save_image(grid, f'{base_url}/{name}_{mutils.get_timestamp()}.png')
        else:
            vutils.save_image(grid, f'{base_url}/{name}.png')
    if exit_flag:
        exit(0)


def save_feature(tensor, name, exit_flag=False, timestamp=False):
    import torchvision.utils as vutils
    # tensors = [tensor, normalize_tensor_mm(tensor), normalize_tensor_sigmoid(tensor)]
    tensors = [tensor]
    titles = ['original', 'min-max', 'sigmoid']
    os.makedirs(base_url, exist_ok=True)
    if timestamp:
        name += '_' + str(time.time()).replace('.', '')

    for index, tensor in enumerate(tensors):
        _data = tensor.detach().cpu().squeeze(0).unsqueeze(1)
        num_per_row = 8
        grid = vutils.make_grid(_data, nrow=num_per_row)
        vutils.save_image(grid, f'{base_url}/{name}_{titles[index]}.png')
    if exit_flag:
        exit(0)
