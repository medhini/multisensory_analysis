#!/usr/bin/env python
# coding: utf-8

# In[1]:
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import h5py  
import numpy as np
import os 
from scipy.misc import imresize
import skvideo.io
import cv2
import random
import soundfile as sf

class AudioDataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, train, frames_len=40, transform=None, h5_file='/media/jeff/Backup/CS598PS/data_2682.h5', transform_label=None):
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
        
        dataset = h5py.File(h5_file, 'r')
        if self.train:
            self.videos_train = np.array(dataset['videos_train'])
            self.sounds_train = np.array(dataset['sounds_train'])
        else:
            self.videos_test = np.array(dataset['videos_test'])
            self.sounds_test = np.array(dataset['sounds_test'])
        
        
    def __len__(self):
        if self.train:
            return len(self.videos_train)
        return len(self.videos_test)

    def __getitem__(self, idx):
        if self.train:
            image = self.videos_train[idx]
            audio = self.sounds_train[idx]
        else:
            image = self.videos_test[idx]
            audio = self.sounds_test[idx]

        # Randomly sample 4 seconds from 10 second clip
        if random.random() < 0.5:
            start = random.randint(0,10) # Start frame
        else:
            start = random.randint(50,60)
        new_image = np.zeros((self.frames_len,256,256,1), dtype=np.uint8)
        for i in range(self.frames_len):
            new_image[i] = np.expand_dims(image[start+i],2)
        
        # Randomly align or misalign audio sample
        if random.random() < 0.5: # align
            audio = audio[int(start*220500/100.0):int(start*220500/100.0)+88200]
            label = 0
        else: # misalign
            if start < 30: # Add shift
                shift = random.randint(20, 60-start) # frame shift amount
                start = start+shift
            else: # Subtract shift
                shift = random.randint(20, start) # frame shift amount
                start = start-shift
            audio = audio[int(start*220500/100.0):int(start*220500/100.0)+88200]
            label = 1
            
        transform_image = np.zeros((self.frames_len,1,224,224))
        if self.transform:
            for i in range(self.frames_len):
                transform_image[i] = self.transform(new_image[i]) # Transform image frames
        
        return (transform_image, audio, label)





