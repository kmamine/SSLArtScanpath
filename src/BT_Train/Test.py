from BT_Train.Run import *
from dataset import * 
import torch
from torch import nn
from tqdm.auto import tqdm
import os
import numpy as np
from torch.utils import data
from torchvision import utils
from torch.nn.modules.activation import Sigmoid, ReLU
import torch.nn.functional as F
import timeit
import csv
import pandas as pd
#np.random.seed(42)
#torch.manual_seed(42)
import copy

from PIL import Image, ImageOps, ImageFilter

transform11 = transforms.Compose([
    #transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),

    transforms.RandomCrop(224, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'),

    #transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
])



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = torchvision.models.mobilenet_v2(pretrained =True).features
model.to(device)

twins = BarlowTwins(model,128)
twins.cuda()

weight = torch.load('/media/amine/DataDisk8T/maroine/BarlowTwins/model-1/model-o-420.pt')

twins.eval()
img = Image.open('/media/amine/DataDisk8T/maroine/BarlowTwins/Dataset/Style4/1.jpg')
img=  img.convert("RGB") 
img = transform11(img).cuda()
out1 = twins(torch.rand([10,3,224,224]).cuda(),record=True)


twins.load_state_dict(weight,strict=True)
out1 = twins.backbone.forward(img.unsqueeze(0))
img = Image.open('/media/amine/DataDisk8T/maroine/BarlowTwins/Dataset/Original/1.jpg')
img=  img.convert("RGB") 
img = transform11(img).cuda()
out2 = twins.backbone.forward(img.unsqueeze(0))

print(nn.MaxPool2d(kernel_size = out1.shape[-1],stride=out1.shape[-1])(out1).squeeze().sum())
print(nn.MaxPool2d(kernel_size = out2.shape[-1],stride=out2.shape[-1])(out2).squeeze().sum())