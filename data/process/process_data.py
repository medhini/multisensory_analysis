#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import os 
from scipy.misc import imresize
import scipy.signal
import cv2
import soundfile as sf
import multiprocessing
import sys

N_THREADS = 8
MIN_SAMPLE_RATE = 22050 # Hz
VIDEO_LENGTH = 10 # Seconds
FRAME_RATE = 20 # FPS

def process_one_file(args):
    try:
        video_path, audio_path = args
        unused, clip_name = os.path.split(video_path)

        # Do audio
        data, samplerate = sf.read(audio_path)
        if samplerate >= MIN_SAMPLE_RATE and len(data) >= MIN_SAMPLE_RATE * VIDEO_LENGTH: # make sure audio sample rate is consistent
            if samplerate >= MIN_SAMPLE_RATE: # Subsample to MIN_SAMPLE_RATE if we're above.
                data = scipy.signal.resample(data[::2], MIN_SAMPLE_RATE * VIDEO_LENGTH)
        else:
            return None

        # Do video
        cap = cv2.VideoCapture(video_path)
        frames = []
        ret = True # flag for remaining frames in cap
        while(ret):
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # convert rgb to grayscale
                frames.append(imresize(frame,(256,256))) # resize to 256x256
        desired_n_frames = FRAME_RATE * VIDEO_LENGTH
        if len(frames) > desired_n_frames: # guarantee video has at least 100 frames for 10 seconds
            frames = np.array([frames[int(i*len(frames)/float(desired_n_frames))] for i in range(desired_n_frames)]) # subsample 100 frames
            youtube_id = '_'.join(clip_name.split("_")[:-2])
            return (frames, data, clip_name)
        else:
            return None
    except Exception as e: # If any videos have some errors of DS_store
        print(e)
        print(args)
        return None

def process_data(data_folder):
    '''
    Returns 
        - numpy array of videos of VIDEO_LENGTH seconds at FRAME_RATE frames/s (Nx(VIDEO_LENGTH*FRAME_RATE)x224x224), 
        - numpy array of sounds at MIN_SAMPLE_RATE hz for VIDEO_LENGTH seconds
        - numpy array of labels (index of labels from labels_lst)
        - numpy array of filenames
    
    @params data_folder: string of data_folder of videos
    '''
    to_process = []
    for folder in os.listdir(data_folder):
        for filename in os.listdir(os.path.join(data_folder, folder)):
            if '.mp4' in filename: # video files
                clip_name = filename.split('.mp4')[0]
                video_path = os.path.join(data_folder, folder, filename)
                audio_path = os.path.join(data_folder, folder, clip_name + '.flac')
                if os.path.exists(audio_path):
                    to_process.append((video_path, audio_path))

    thread_pool = multiprocessing.Pool(N_THREADS)
    results = thread_pool.map(process_one_file, to_process)
    results = [data for data in results if data is not None]
    videos, sounds, clip_names = zip(*results)

    videos = np.array(videos)
    sounds = np.array(sounds)
    clip_names = np.array(clip_names)
    return videos, sounds, clip_names

if __name__ == '__main__':
    if len(sys.argv) <= 3:
        print("Usage:", sys.argv[0], "[train data folder] [test data folder] [out file name]")
        sys.exit(0)
    
    videos_train, sounds_train, clip_names_train = process_data(sys.argv[1])
    videos_test, sounds_test, clip_names_test = process_data(sys.argv[2])
    print("Training set shapes:", videos_train.shape, sounds_train.shape)
    print("Testing set shapes:", videos_test.shape, sounds_test.shape)

    np.savez_compressed(
        sys.argv[3],
        videos_train=videos_train, 
        sounds_train=sounds_train,
        clip_names_train=clip_names_train,
        videos_test=videos_test,
        sounds_test=sounds_test,
        clip_names_test=clip_names_test
    )
