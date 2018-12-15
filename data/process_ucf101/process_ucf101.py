#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import os
import scipy.signal
import cv2
import soundfile as sf
import multiprocessing
import skimage
import skimage.transform
import sys
import warnings

N_THREADS = 12
MIN_SAMPLE_RATE = 22050 # Hz
MIN_VIDEO_LENGTH = 6 # Seconds
FRAME_RATE = 10 # FPS
RESIZE_SIZE = 256
TEST_SET_FILE = os.path.dirname(os.path.realpath(__file__)) + "/testlist01.txt"

def process_one_file(args):
    try:
        video_path, audio_path = args

        # Do audio
        data, samplerate = sf.read(audio_path)
        if samplerate >= MIN_SAMPLE_RATE and len(data) >= MIN_SAMPLE_RATE * MIN_VIDEO_LENGTH: # make sure audio sample rate is consistent
            if samplerate >= MIN_SAMPLE_RATE: # Subsample to MIN_SAMPLE_RATE if we're above.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    data = scipy.signal.resample(data[::2], MIN_SAMPLE_RATE * MIN_VIDEO_LENGTH)
        else:
            return None
        if len(data.shape) < 2: # Is mono?  If so, duplicate the single channel is it's "stereo"
            data = np.vstack((data, data))

        # Do video
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_s = total_frames / fps
        if total_s < MIN_VIDEO_LENGTH:
            return None

        frames = []
        ret = True # flag for remaining frames in cap
        while(ret):
            ret, frame = cap.read() # w x h x n_colors shape matrix
            if ret:
                #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # convert rgb to grayscale
                # resize to 256x256
                resized = skimage.transform.resize(frame, (RESIZE_SIZE, RESIZE_SIZE), mode="constant", clip=True, anti_aliasing=True)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    frames.append(skimage.img_as_ubyte(resized))
        

        desired_n_frames = int(total_s * FRAME_RATE)
        frames = np.array([frames[int(i*len(frames)/float(desired_n_frames))] for i in range(desired_n_frames)]) # subsample desired # of frames
        return (frames, data)
    except Exception as e: # If any videos have some errors of DS_store
        print(e)
        print(args)
        return None

def process_data(videos_dir, audio_dir):
    '''
    Returns
        - numpy array of videos of VIDEO_LENGTH seconds at FRAME_RATE frames/s (Nx(VIDEO_LENGTH*FRAME_RATE)x224x224), 
        - numpy array of sounds at MIN_SAMPLE_RATE hz for VIDEO_LENGTH seconds
        - numpy array of labels (index of labels from labels_lst)
        - numpy array of filenames
    
    @params videos_dir, audio_dir: dir where video and audio is stored respectively
    '''
    to_process = []
    categories = []
    clip_names = []
    for audio_file in os.listdir(audio_dir):
        if audio_file.endswith(".wav"):
            category = audio_file.split("_")[1]
            base_name = audio_file.rstrip(".wav")
            video_file = base_name + ".avi"
            video_path = os.path.join(videos_dir, category, video_file)
            audio_path = os.path.join(audio_dir, audio_file)
            to_process.append((video_path, audio_path))
            categories.append(category)
            clip_names.append(video_file)

    thread_pool = multiprocessing.Pool(N_THREADS)
    results = thread_pool.map(process_one_file, to_process)
    good_indices = [i for i in range(len(results)) if results[i] is not None]
    print( "%d good videos out of %d" % (len(good_indices), len(to_process)) )

    results = [results[i] for i in good_indices]
    videos, sounds = zip(*results)
    categories = np.array([categories[i] for i in good_indices])
    clip_names = np.array([clip_names[i] for i in good_indices])

    return np.array(videos), np.array(sounds), categories, clip_names

def get_test_clips(test_set_file):
    with open(test_set_file, "r") as f:
        return set([os.path.split(path.strip())[1] for path in f])

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage:", sys.argv[0], "[videos folder] [audio folder] [out file name]")
        sys.exit(0)
    
    videos, sounds, categories, clip_names = process_data(sys.argv[1], sys.argv[2])
    test_clips = get_test_clips(TEST_SET_FILE)
    train_indices = []
    test_indices = []
    for i, clip_name in enumerate(clip_names):
        if clip_name in test_clips:
            test_indices.append(i)
        else:
            train_indices.append(i)

    np.savez_compressed(
        sys.argv[3],
        videos_train=videos[train_indices], 
        sounds_train=sounds[train_indices],
        categories_train=categories[train_indices],
        clip_names_train=clip_names[train_indices],

        videos_test=videos[test_indices], 
        sounds_test=sounds[test_indices],
        categories_test=categories[test_indices],
        clip_names_test=clip_names[test_indices]
    )
