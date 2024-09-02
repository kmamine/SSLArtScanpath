from ast import Not
from pathlib import Path
import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
import copy
import torch

from torch import nn
import torch.nn.functional as F
from functools import wraps
from PIL import Image, ImageOps, ImageFilter
from torch import nn, optim
import torch
import torchvision

def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * args.learning_rate_weights
    optimizer.param_groups[1]['lr'] = lr * args.learning_rate_biases



def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


class EMA():
    def __init__(self, beta=0.99):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        current_params.data = ema_updater.update_average(old_weight, up_weight)


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class BarlowTwins(nn.Module):
    def __init__(self, model,batch):
        super().__init__()
        #self.args = args
        self.backbone = model
        #self.backbone.fc = nn.Identity()
        self.batch = batch
        # projector
        sizes = [1280,2048,2048,1280] 
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))

        self.projector = nn.Sequential(*layers)
        self.target_encoder = None
        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)
        self.target_ema_updater = EMA()
        #self.forward(torch.randn(2, 3, 224, 224, device='cuda'),record=True)

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.backbone)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.backbone)

    def forward(self, y1=None, y2=None, record =False):

        if record :
            target_encoder = self._get_target_encoder()
            z1 = self.backbone(y1)
            return (z1).squeeze()

        z1 = self.backbone(y1)
        z2 = self.backbone(y2)
        target_encoder = self._get_target_encoder()
        
        z1 = nn.MaxPool2d(kernel_size = z1.shape[-1],stride=z1.shape[-1])(z1) 
        z2 = nn.MaxPool2d(kernel_size = z2.shape[-1],stride=z2.shape[-1])(z2) 
       
    
        hidden1 = self.projector(z1.squeeze()) 
        #hidden2 = z

        # empirical cross-correlation matrix
        c = self.bn(z1.squeeze()).T @ self.bn(z2.squeeze())

        # sum the cross-correlation matrix between all gpus
        c.div_(self.batch)
        #torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + 0.5 * off_diag

        return loss,hidden1.detach()#,hidden2



if __name__=='__main__':
    import torchvision

    import timeit

    model = torchvision.models.mobilenet_v2().features
    
    twins = BarlowTwins(model,2).cuda()
    start_init = timeit.default_timer()
    inp1 = torch.rand(2,3,224,224).cuda()
    inp2 = torch.rand(2,3,224,224).cuda()
    outs =twins(inp1, inp2)
    twins.update_moving_average()
    print("\n taken time: ", (timeit.default_timer()- start_init))
