import os
import sys
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim
import torch
import numpy as np
from errno import EEXIST
from os import makedirs, path
import random
from math import exp
from torch.autograd import Variable


class LogWriter:
    def __init__(self, file_path, train=True):
        os.makedirs(file_path, exist_ok=True)
        self.file_path = os.path.join(file_path, "train.txt" if train else "test.txt")

    def write(self, text):
        # 打印到控制台
        print(text)
        # 追加到文件
        with open(self.file_path, 'a') as file:
            file.write(text + '\n')


def loss_fn(pred, target, loss_type='L2', lambda_value=0.7):
    target = target.detach()
    pred = pred.float()
    target  = target.float()
    if loss_type == 'L2':
        loss = F.mse_loss(pred, target)
    elif loss_type == 'L1':
        loss = F.l1_loss(pred, target)
    elif loss_type == 'SSIM':
        loss = 1 - ssim(pred, target, data_range=1, size_average=True)
    elif loss_type == 'Fusion1':
        loss = lambda_value * F.mse_loss(pred, target) + (1-lambda_value) * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion2':
        loss = lambda_value * F.l1_loss(pred, target) + (1-lambda_value) * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion3':
        loss = lambda_value * F.mse_loss(pred, target) + (1-lambda_value) * F.l1_loss(pred, target)
    elif loss_type == 'Fusion4':
        loss = lambda_value * F.l1_loss(pred, target) + (1-lambda_value) * (1 - ms_ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion_hinerv':
        loss = lambda_value * F.l1_loss(pred, target) + (1-lambda_value)  * (1 - ms_ssim(pred, target, data_range=1, size_average=True, win_size=5))
    return loss

def strip_lowerdiag(L):
    if L.shape[1] == 3:
        uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")
        uncertainty[:, 0] = L[:, 0, 0]
        uncertainty[:, 1] = L[:, 0, 1]
        uncertainty[:, 2] = L[:, 0, 2]
        uncertainty[:, 3] = L[:, 1, 1]
        uncertainty[:, 4] = L[:, 1, 2]
        uncertainty[:, 5] = L[:, 2, 2]

    elif L.shape[1] == 2:
        uncertainty = torch.zeros((L.shape[0], 3), dtype=torch.float, device="cuda")
        uncertainty[:, 0] = L[:, 0, 0]
        uncertainty[:, 1] = L[:, 0, 1]
        uncertainty[:, 2] = L[:, 1, 1]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def build_rotation_2d(r):
    '''
    Build rotation matrix in 2D.
    '''
    R = torch.zeros((r.size(0), 2, 2), device='cuda')
    R[:, 0, 0] = torch.cos(r)[:, 0]
    R[:, 0, 1] = -torch.sin(r)[:, 0]
    R[:, 1, 0] = torch.sin(r)[:, 0]
    R[:, 1, 1] = torch.cos(r)[:, 0]
    return R

def build_scaling_rotation_2d(s, r, device):
    L = torch.zeros((s.shape[0], 2, 2), dtype=torch.float, device='cuda')
    R = build_rotation_2d(r, device)
    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L = R @ L
    return L
    
def build_covariance_from_scaling_rotation_2d(scaling, scaling_modifier, rotation, device):
    '''
    Build covariance metrix from rotation and scale matricies.
    '''
    L = build_scaling_rotation_2d(scaling_modifier * scaling, rotation, device)
    actual_covariance = L @ L.transpose(1, 2)
    return actual_covariance

def build_triangular(r):
    R = torch.zeros((r.size(0), 2, 2), device=r.device)
    R[:, 0, 0] = r[:, 0]
    R[:, 1, 0] = r[:, 1]
    R[:, 1, 1] = r[:, 2]
    return R


def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)




def get_expon_lr_func(
        lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper





def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def my_ssim(img1, img2, window_size=11, size_average=True, weights=None):
    channel = img1.size(-3)
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    if weights is None:
        weights = torch.ones_like(img1)
    
    img1 = img1 * weights
    img2 = img2 * weights

    return _ssim(img1, img2, window, window_size, channel, size_average)



def get_linear_noise_func(
        lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = lr_init * (1 - t) + lr_final * t
        return delay_rate * log_lerp

    return helper

