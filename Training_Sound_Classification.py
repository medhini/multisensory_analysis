#!/usr/bin/env python
# coding: utf-8

# In[2]:

import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import h5py  
import numpy as np
import os, sys
from scipy.misc import imresize, imsave
import random
import cv2

sys.path.append('data/process/')
import category_getter

class AudioDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, train, frames_len=40, transform=None, h5_file='/media/jeff/Backup/CS598PS/data_nice_2597.h5', transform_label=None):
        """
        Args:
            train (bool): Whether or not to use training data
            frames (int): Number of video frames per video sample
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.train = train
        self.transform = transform
        self.frames_len = frames_len
        
        labels_lst = ['Animal','Channel, environment and background','Human sounds','Music',
                         'Natural sounds','Sounds of things','Source-ambiguous sounds']
        
        dataset = h5py.File(h5_file,'r')
        if self.train:
            self.sounds_train = np.array(dataset['sounds_train'])
            self.filenames = np.load('/data/jz/CS598PS/data/data_nice_filenames.npy')
            
            train_cg = category_getter.CategoryGetter("data/process/unbalanced_train_segments.csv")
            labels_train = []
            for idx in range(len(self.filenames)):
                filename = str(self.filenames[idx])
                youtube_id = '_'.join(filename.split("_")[:-2])
                label = np.array([0,0,0,0,0,0,0])
                for id in train_cg.get_general_categories_for_video(youtube_id):
                    label[labels_lst.index(train_cg.ontology.get_record_for_id(id)["name"])] = 1
                labels_train.append(label)
            labels_train = np.array(labels_train)
            self.labels = labels_train
        else:
            self.sounds_test = np.array(dataset['sounds_test'])
            self.filenames = np.load('/data/jz/CS598PS/data/filenames_nice_test.npy')
            
            test_cg = category_getter.CategoryGetter("data/process/eval_segments.csv")
            labels_test = []
            for idx in range(len(self.filenames)):
                filename = str(self.filenames[idx])
                youtube_id = '_'.join(filename.split("_")[:-2])
                label = np.array([0,0,0,0,0,0,0])
                for id in test_cg.get_general_categories_for_video(youtube_id):
                    label[labels_lst.index(test_cg.ontology.get_record_for_id(id)["name"])] = 1
                labels_test.append(label)
            labels_test = np.array(labels_test)
            self.labels = labels_test
        dataset.close()
        
    def __len__(self):
        if self.train:
            return len(self.sounds_train)
        return len(self.sounds_test)

    def __getitem__(self, idx):
        if self.train:
            audio = self.sounds_train[idx]
            label = self.labels[idx]
        else:
            audio = self.sounds_test[idx]
            label = self.labels[idx]
            
        audio = audio[int(30*220500/100.0):int(30*220500/100.0)+88200]
    
        return (audio, label)


# In[ ]:


import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Block2(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, kernel_size, stride, downsample=None):
        super(Block2, self).__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, groups=1, bias=True)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Block3(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, kernel_size=(1,1,1), stride=1, downsample=None, padding=0):
        super(Block3, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=1, groups=1, bias=True)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(1,1,1), stride=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

def Linear(in_features, out_features, dropout=0.):
    m = nn.Linear(in_features, out_features)
    m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)

class alignment(nn.Module):
    def __init__(self):
        super(alignment, self).__init__()
        """Sound Features"""
        self.conv1_1 = nn.Conv1d(2, 64, 65, stride=4, padding=0, dilation=1, groups=1, bias=True)
        self.pool1_1 = nn.MaxPool1d(4, stride=4)

        self.s_net_1 = self._make_layer(Block2, 64, 128, 15, 4, 1)
        self.s_net_2 = self._make_layer(Block2, 128, 128, 15, 4, 1)
        self.s_net_3 = self._make_layer(Block2, 128, 256, 15, 4, 1)
        
        self.pool1_2 = nn.MaxPool1d(3, stride=3)
        self.conv1_2 = nn.Conv1d(256, 128, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
        
        """Fuse Features"""
        self.fractional_maxpool = nn.FractionalMaxPool2d((3,1), output_size=(10, 1))
        self.conv3_2 = nn.Conv3d(192, 512, (1, 1, 1))
        self.conv3_3 = nn.Conv3d(512, 128, (1, 1, 1))
        self.joint_net_1 = self._make_layer(Block3, 128, 128, (3,3,3), (2,2,2), 2)
        self.joint_net_2 = self._make_layer(Block3, 128, 256, (3,3,3), (1,2,2), 2)
        self.joint_net_3 = self._make_layer(Block3, 256, 512, (3,3,3), (1,2,2), 2)

        #TODO: Global avg pooling, fc and sigmoid
        self.fc = Linear(3200,7)

    def _make_layer(self, block, in_channels, out_channels, kernel_size, stride, blocks):
        downsample = None
        if stride != 1 or in_channels != out_channels * block.expansion:
            if isinstance(kernel_size, int):
                downsample = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels * block.expansion, kernel_size, stride),
                    nn.BatchNorm1d(out_channels * block.expansion),
                )
                layers = []
                layers.append(block(in_channels, out_channels, kernel_size, stride, downsample))
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(in_channels, out_channels * block.expansion, kernel_size, stride, padding=1),
                    nn.BatchNorm3d(out_channels * block.expansion),
                )
                layers = []
                layers.append(block(in_channels, out_channels, kernel_size, stride, downsample, padding=1))

        
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, sounds):
        batchsize = sounds.shape[0]
        sounds = sounds.view(batchsize, 2, -1)
        
        out_s = self.conv1_1(sounds)
        out_s = self.pool1_1(out_s)

        out_s = self.s_net_1(out_s)
        out_s = self.s_net_2(out_s)
        out_s = self.s_net_3(out_s)

        out_s = self.pool1_2(out_s)
        out_s = self.conv1_2(out_s)
        
        out_joint = out_s.view(out_s.shape[0], -1)
        out_joint = self.fc(out_joint)
        out_joint = torch.sigmoid(out_joint)
        return out_joint

# In[2]:


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

transform = transforms.Compose([
transforms.ToPILImage(),
# transforms.RandomHorizontalFlip(),
transforms.RandomCrop(224),
transforms.ToTensor()])

train_dataset = AudioDataset(train=True,transform=transform,h5_file='/data/jz/CS598PS/data/data_nice_2597.h5')
test_dataset = AudioDataset(train=False,transform=transform,h5_file='/data/jz/CS598PS/data/data_nice_2597.h5')

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=4)


# In[14]:


model_align = alignment().cuda()
# checkpoint = torch.load("nice_350.pth")
# model_align.load_state_dict(checkpoint.state_dict())


# # Training

# In[15]:


loss_fn = nn.BCELoss()
optimizer_align = optim.Adam(model_align.parameters(), lr = 5e-5)
# optimizer_align = optim.SGD(model_align.parameters(), lr = 1e-4, momentum=0.9)

for epoch in range(200):
    accs = []
    losses = []
    model_align.train()
    for batch_idx, (sounds, labels) in enumerate(train_loader):
        sounds_v = Variable(sounds.type(torch.FloatTensor)).cuda()
        labels_v = Variable(labels.type(torch.FloatTensor)).cuda()
        
        optimizer_align.zero_grad()
        aligned_res = model_align(sounds_v)
        loss = loss_fn(aligned_res, labels_v)
        loss.backward()
        optimizer_align.step()
        losses.append(loss.item())
        accs.append(np.mean((torch.argmax(aligned_res,1) == torch.argmax(labels_v,1)).detach().cpu().numpy()))
    print("Epoch :", epoch, np.mean(losses), np.mean(accs))
    if (epoch + 1)%25 == 0:
        accs = []
        losses = []
        model_align.eval()
        for batch_idx, (sounds, labels) in enumerate(test_loader):
            with torch.no_grad():
                sounds_v = Variable(sounds.type(torch.FloatTensor)).cuda()
                labels_v = Variable(labels.type(torch.FloatTensor)).cuda()
                aligned_res = model_align(sounds_v)
                loss = loss_fn(aligned_res, labels_v)
                losses.append(loss.item())
                accs.append(np.mean((torch.argmax(aligned_res,1) == torch.argmax(labels_v,1)).detach().cpu().numpy()))
        print("Validation :", epoch, np.mean(losses), np.mean(accs))
torch.save(model_align, 'nice_sound_classification_200.pth')
