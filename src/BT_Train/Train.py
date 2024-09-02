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


def train_(model, train_d, epochs=1000,batchsize=128):
    twins = BarlowTwins(model,batchsize)
    twins.cuda()
    optimizer = torch.optim.Adam(twins.parameters(), lr=0.0005)
    

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,  factor=0.5, patience=5, threshold=0.0001)
    for epoch in range(epochs): 
        indices = np.random.choice(len(train_d)-2, size=8000)
        train_loader = torch.utils.data.DataLoader(train_d,
                                                batch_size=batchsize,
                                                sampler=data.SubsetRandomSampler(indices))
        start_init = timeit.default_timer()
        twins.train()
        running_loss = 0.0

        running_loss1 = 0.0
        pbar = tqdm(train_loader)
        
              
        for list_batch in pbar:
            i = 1
            lenght = len(list_batch[:])
            memory_bank = []
            outs_l = 0.0
            for stylebatch in list_batch[:-1]:

                if i<(lenght-1):
                    image1 = stylebatch
                    image2 = list_batch[np.random.randint(lenght-i)+i]         
                else : 
                    image1 = stylebatch
                    image2 = list_batch[-1]  
                
                outs,h1 = twins(image1.to(device), image2.to(device))  
                #find another name       
                memory_bank.append(h1) 

                optimizer.zero_grad()
                outs.backward(retain_graph=True)
                optimizer.step()
                twins.update_moving_average()
                outs_l = outs_l +outs.item()
                i =i+1
            m_loss = 0.0
            h2 = twins(image2.cuda(),record = True) 
            for h in memory_bank:
                m_loss = loss_fn(h2,h)/lenght +m_loss

            optimizer.zero_grad()
            m_loss.mean().backward()
            optimizer.step()
            
            pbar.set_description("batch  {}".format(m_loss.mean().item()))
            pbar.refresh()
            running_loss = m_loss.mean().item() + running_loss
            running_loss1 = outs_l + running_loss1

        scheduler.step(running_loss/(8000/batchsize))
        print("Taken time for taraining: ", format((timeit.default_timer()- start_init)/60,".3f"))
        print('*****************************')
        print('[Epoch: %d,], Loss Train                 :::    %.10f' %
            (epoch + 1, running_loss/(8000/batchsize)))
        print('[Epoch: %d,], Loss* Train                 :::    %.10f' %
            (epoch + 1, running_loss1/(8000/batchsize)))

        save_path = f'./model-1/model-o-{epoch}.pt'
        os.makedirs('./model-1',exist_ok= True)
        torch.save(twins.state_dict(), save_path) 




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = torchvision.models.mobilenet_v2().features
model.to(device)

train_d = Custom()
batchsize = 128
print('Train dataset size: ', len(train_d))

#weight = torch.load('/media/amine/DataDisk8T/maroine/BarlowTwins/model-1/model-o-258.pt')
#model.load_state_dict(weight,strict=False)
train_(model, train_d, epochs=1000,batchsize=128)
