#!/usr/bin/env python
# coding: utf-8

# In[1]:
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torchvisions
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


# In[2]:

def process_data():
	videos = []
	sounds = []
	for filename in os.listdir('videos'): 
	    if '.mp4' in filename: # video files
	        cap = cv2.VideoCapture(os.path.join('videos',filename))
	        
	        frames = []
	        ret = True # flag for remaining frames in cap
	        while(ret):
	            ret, frame = cap.read()
	            if ret:
	                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # convert rgb to grayscale
	                frames.append(imresize(frame,(256,256))) # resize to 256x256
	        frames = np.array([frames[int(i*len(frames)/100)] for i in range(100)]) # subsample 100 frames
	        videos.append(frames)
	    elif '.flac' in filename: # audio files
	        data, samplerate = sf.read(os.path.join('videos',filename))
	        data = data[::2] # Subsample to 22050 hertz
	        sounds.append(data)
	        
	videos = np.array(videos)
	sounds = np.array(sounds)
	print(videos.shape, sounds.shape)


	# In[3]:
	hf = h5py.File('data.h5', 'w')
	hf.create_dataset('videos_train', data=videos)
	hf.create_dataset('sounds_train', data=sounds)
	hf.create_dataset('videos_test', data=videos)
	hf.create_dataset('sounds_test', data=sounds)
	hf.close()


# In[40]:
process_data()

dataset = h5py.File('data.h5')
videos_train = np.array(dataset['videos_train'])
sounds_train = np.array(dataset['sounds_train'])
videos_test = np.array(dataset['videos_test'])
sounds_test = np.array(dataset['sounds_test'])

class AudioDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, train, frames_len=40, transform=None, transform_label = None):
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
        
    def __len__(self):
        if self.train:
            return len(videos_train)
        return len(videos_train)

    def __getitem__(self, idx):
        if self.train:
            image = videos_train[idx]
            audio = sounds_train[idx]
        else:
            image = videos_test[idx]
            audio = sounds_test[idx]

        # Randomly sample 4 seconds from 10 second clip
        start = random.randint(0, 100-self.frames_len) # Start frame
        new_image = np.zeros((self.frames_len,256,256,1), dtype=np.uint8)
        for i in range(self.frames_len):
            new_image[i] = np.expand_dims(image[start+i],2)
        
        # Randomly align or misalign audio sample
        if random.random() < 0.5: # align
            audio = audio[int(start*220500/100.0):int(start*220500/100.0)+88200]
            label = 0
        else: # misalign
            shift = random.randint(20, 60) # frame shift amount
            if random.random() < 0.5: # Add shift
                start = np.clip(start-shift, 0, 100-self.frames_len)
            else: # Subtract shift
                start = np.clip(start+shift, 0, 100-self.frames_len)
            audio = audio[int(start*220500/100.0):int(start*220500/100.0)+88200]
            label = 1
            
        transform_image = np.zeros((self.frames_len,1,224,224), dtype=np.uint8)
        if self.transform:
            for i in range(self.frames_len):
                transform_image[i] = self.transform(new_image[i]) # Transform image frames
            
        return (new_image, audio, label)


# In[41]:


# Image preprocessing modules
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224),
    transforms.ToTensor()])

# CIFAR-10 dataset
train_dataset = AudioDataset(train=True,transform=transform)
test_dataset = AudioDataset(train=False,transform=transforms.ToTensor())

# # Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=5, 
                                           shuffle=True, num_workers=4)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=5, 
                                           shuffle=False, num_workers=4)


# In[49]:


for images, sounds, labels in train_loader:
    print(images.shape)
    print(sounds.shape)
    print(labels.shape)



# In[ ]:




