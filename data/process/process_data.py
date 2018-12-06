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

import category_getter

def process_data(data_folder = '/media/jeff/Backup/CS598PS/train_nice_6340/'):
    '''
    Returns 
        - numpy array of videos of 10 seconds at 10 frames/s (Nx100x224x224), 
        - numpy array of sounds at 22050 hertz for 10 seconds
        - numpy array of labels (index of labels from labels_lst)
        - numpy array of filenames
    
    @params data_folder: string of data_folder of videos
    '''
    filenames = []
    videos = []
    sounds = []
    labels = []
    labels_lst = ['Animal','Channel, environment and background','Human sounds','Music',
                 'Natural sounds','Sounds of things','Source-ambiguous sounds']
    cg = category_getter.CategoryGetter("unbalanced_train_segments.csv")
    for folder in os.listdir(data_folder): 
        try:
            for filename in os.listdir(os.path.join(data_folder, folder)):
                if '.mp4' in filename: # video files
                    clip_name = filename.split('.mp4')[0]
                    if os.path.exists(os.path.join(data_folder,folder, clip_name + '.flac')):
                        data, samplerate = sf.read(os.path.join(data_folder,folder, clip_name + '.flac'))
                        if samplerate == 44100 and len(data)==441000: # make sure audio sample rate is consistent
                            data = data[::2] # Subsample to 22050 hertz
                            cap = cv2.VideoCapture(os.path.join(data_folder,folder,filename))
                            frames = []
                            ret = True # flag for remaining frames in cap
                            while(ret):
                                ret, frame = cap.read()
                                if ret:
                                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # convert rgb to grayscale
                                    frames.append(imresize(frame,(256,256))) # resize to 256x256
                            if len(frames)>100: # guarantee video has at least 100 frames for 10 seconds
                                frames = np.array([frames[int(i*len(frames)/100.)] for i in range(100)]) # subsample 100 frames
                                youtube_id = '_'.join(filename.split("_")[:-2])
                                labels.append([labels_lst.index(cg.ontology.get_record_for_id(id)["name"]) for id in cg.get_general_categories_for_video(youtube_id)])
                                videos.append(frames)
                                sounds.append(data)
                                filenames.append(filename)
        except Exception as e: # If any videos have some errors of DS_store
            print(e)
            print(filename)
            
    videos = np.array(videos)
    sounds = np.array(sounds)
    labels = np.array(labels)
    filenames = np.array(filenames)
    return videos, sounds, labels, filenames

if __name__ == '__main__':
    videos_train, sounds_train = process_data('/media/jeff/Backup/CS598PS/train_nice_6340/')
    videos_test, sounds_test = process_data('/media/jeff/Backup/CS598PS/test/')
    
    print("Training set shapes: ", videos_train.shape, sounds_train.shape)
    print("Testing set shapes: ", videos_test.shape, sounds_test.shape)
    
    hf = h5py.File('/media/jeff/Backup/CS598PS/data.h5', 'w') # h5 file to save data to
    hf.create_dataset('videos_train', data=videos_train)
    hf.create_dataset('sounds_train', data=sounds_train)
    hf.create_dataset('videos_test', data=videos_test)
    hf.create_dataset('sounds_test', data=sounds_test)
    hf.close()