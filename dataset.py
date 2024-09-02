import numpy as np
import open3d as o3d
import pandas as pd
import torch
import os
import random
import math
from torch.utils import data
from torchvision import transforms
import cv2




from PIL import Image, ImageOps, ImageFilter
import torchvision.transforms as transforms
import random
'''
#####
Adapted from https://github.com/facebookresearch/barlowtwins
#####
'''


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class Transform:
    def __init__(self):
        self.transform = transforms.Compose([
            #transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),

            transforms.RandomCrop(224, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'),

            transforms.RandomHorizontalFlip(p=0.5),
            #transforms.RandomApply(
            #    [transforms.ColorJitter(brightness=0.4, contrast=0.4,
            #                            saturation=0.2, hue=0.1)],
            #    p=0.8
            #),
            #transforms.RandomGrayscale(p=0.2),
            #GaussianBlur(p=1.0),
            #Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform1 = transforms.Compose([
            #transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),

            transforms.RandomCrop(224, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'),

            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                       saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            #GaussianBlur(p=1.0),
            #Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x,style=True):
        if style:
            y1 = self.transform(x)
        else: 
            y1 = self.transform1(x)
        #print(y1.shape)
        #print(y2.shape)

        
        return y1

class Custom(data.Dataset):
    def __init__(self, img_dir='/media/amine/DataDisk8T/maroine/BarlowTwins/Dataset/Style1/', numebr_of_samples = -1, start = 0, transform=Transform()):

        self.img_dir = img_dir
        self.list_of_img = os.listdir(os.path.join(self.img_dir))
        self.list_of_img.sort()
        #print(self.list_of_img)
        self.list_of_img = self.list_of_img[:numebr_of_samples]
        self.tranform = transform
   
    def __len__(self):

        return len(self.list_of_img)


    def __getitem__(self, idx):
         
        try:
            styles = np.random.choice(5, size=3)
            #img = cv2.imread(img_path)
            #print(styles)
            img_path = os.path.join(self.img_dir.replace('Style1','Original'),self.list_of_img[idx])
            img_path_style0 = os.path.join(self.img_dir.replace('Style1','Style'+str(styles[0]+1)),self.list_of_img[idx])
            img_path_style1 = os.path.join(self.img_dir.replace('Style1','Style'+str(styles[1]+1)),self.list_of_img[idx])
            img_path_style2 = os.path.join(self.img_dir.replace('Style1','Style'+str(styles[2]+1)),self.list_of_img[idx])


            
            img__style0 = Image.open(img_path_style0)
            img__style0=  img__style0.convert("RGB")   
            
            img__style1 = Image.open(img_path_style1)
            img__style1=  img__style1.convert("RGB")    

            img__style2 = Image.open(img_path_style2)
            img__style2=  img__style2.convert("RGB")   

            img = Image.open(img_path)
            img=  img.convert("RGB") 
        except:
            print(img_path)
            img = img__style0

        if self.tranform is not None: 
            #
            img = self.tranform(img,style=False)
            img__style0 = self.tranform(img__style0)
            img__style1 = self.tranform(img__style1)
            img__style2 = self.tranform(img__style2)
        #  
        

        return img__style0,img__style1,img__style2,img

train_ds_s = Custom(transform=Transform())
#dataset = dsets.ImageFolder('/media/amine/DataDisk8T/amine/wikiart_dataset_reduced/', Transform())
"""
from tqdm.auto import tqdm

for i in range(1000) :
    indices = np.random.choice(len(train_ds_s), size=10000)
    loader = torch.utils.data.DataLoader(train_ds_s,
                                            batch_size=128,
 
                                            sampler=data.SubsetRandomSampler(indices))

    pbar = tqdm(loader)
    print(i)
    for datat in pbar: 
        img__style0,img__style1,img__style2,img= datat
"""