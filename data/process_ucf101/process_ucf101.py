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
import random

random.seed(0)
N_THREADS = 32
MIN_SAMPLE_RATE = 22050 # Hz
MIN_VIDEO_LENGTH = 6 # Seconds
FRAME_RATE = 10 # FPS
RESIZE_SIZE = 256

TRAIN_SET_FILE = os.path.dirname(os.path.realpath(__file__)) + "/trainlist01.txt"
TEST_SET_FILE = os.path.dirname(os.path.realpath(__file__)) + "/testlist01.txt"
CATEGORY_DEF_FILE = os.path.dirname(os.path.realpath(__file__)) + "/classInd.txt"

N_CLASSES = 101
N_TRAIN_PER_CLASS = 20
N_TEST_PER_CLASS = 5

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
        if fps <= 0:
            print("Warning: couldn't get fps from", video_path, "; skipping...")
            return None
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

def process_data(video_paths, categories, audio_dir):
    '''
    Returns
        - numpy array of videos of VIDEO_LENGTH seconds at FRAME_RATE frames/s (Nx(VIDEO_LENGTH*FRAME_RATE)x224x224), 
        - numpy array of sounds at MIN_SAMPLE_RATE hz for VIDEO_LENGTH seconds
        - numpy array of labels (index of labels from labels_lst)
        - numpy array of filenames
    
    @params videos_dir, audio_dir: dir where video and audio is stored respectively
    '''
    to_process = []
    result_categories = []
    clip_names = []
    for i, video_path in enumerate(video_paths):
        clip_name = os.path.split(video_path)[1]
        audio_name = clip_name.rstrip(".avi") + ".wav"
        audio_path = os.path.join(audio_dir, audio_name)
        if os.path.exists(audio_path):
            to_process.append((video_path, audio_path))
            result_categories.append(categories[i])
            clip_names.append(clip_name)

    thread_pool = multiprocessing.Pool(N_THREADS)
    results = thread_pool.map(process_one_file, to_process)
    good_indices = [i for i in range(len(results)) if results[i] is not None]
    print( "%d good videos out of %d" % (len(good_indices), len(to_process)) )

    results = [results[i] for i in good_indices]
    videos, sounds = zip(*results)
    categories = np.array([categories[i] for i in good_indices])
    clip_names = np.array([clip_names[i] for i in good_indices])

    return np.array(videos), np.array(sounds), categories, clip_names

def get_category_to_int_map():
    int_for_category = {}
    with open(CATEGORY_DEF_FILE, "r") as f:
        for line in f:
            if len(line) > 1:
                index, category = line.strip().split(" ")
                int_for_category[category] = int(index)
    return int_for_category

def get_videos_train(train_set_file, videos_dir):
    """Gets train videos, grouped by category
    
    Arguments:
        train_set_file {string} -- path to the training set file
        videos_dir {string} -- path to the video directory
    
    Returns:
        string[][] - lists of video paths, grouped by category
    """
    videos_for_category = {i: [] for i in range(1, N_CLASSES + 1)}
    with open(train_set_file, "r") as f:
        for line in f:
            if len(line) > 1:
                path, category = line.split(" ")
                sys_path = os.path.join(videos_dir, path)
                videos_for_category[int(category)].append(sys_path)
    return videos_for_category

def get_videos_test(test_set_file, videos_dir):
    videos_for_category = {i: [] for i in range(1, N_CLASSES + 1)}
    int_for_category = get_category_to_int_map()
    with open(test_set_file, "r") as f:
        for line in f:
            if len(line) > 1:
                category, video_path = line.strip().split("/")
                index = int_for_category[category]
                sys_path = os.path.join(videos_dir, category, video_path)
                videos_for_category[index].append(sys_path)
    return videos_for_category

def sample_videos(videos_by_category, n_per_category):
    videos = []
    categories = []
    for category, videos_in_category in videos_by_category.items():
        videos.extend(random.sample(videos_in_category, n_per_category))
        categories.extend([category] * n_per_category)
    return videos, categories

def main(argv):
    if len(argv) < 3:
        print("Usage:", sys.argv[0], "[videos folder] [audio folder] [out file name]")
        return 1

    video_dir = sys.argv[1]
    audio_dir = sys.argv[2]
    out_path = sys.argv[3]

    # We do test first because it's smaller, so it will fail faster if there is a problem
    print("Processing TEST videos...")
    videos_by_category_test = get_videos_test(TEST_SET_FILE, video_dir)
    paths_test, categories_test = sample_videos(videos_by_category_test, N_TEST_PER_CLASS)
    data_test = process_data(paths_test, categories_test, audio_dir)
    print("done")

    print("Processing TRAIN videos...")
    videos_by_category_train = get_videos_train(TRAIN_SET_FILE, video_dir)
    paths_train, categories_train = sample_videos(videos_by_category_train, N_TRAIN_PER_CLASS)
    data_train = process_data(paths_train, categories_train, audio_dir)
    print("done")

    print("Saving arrays...")
    np.savez_compressed(
        out_path,
        videos_train=data_train[0], 
        sounds_train=data_train[1],
        categories_train=data_train[2],
        clip_names_train=data_train[3],

        videos_test=data_test[0], 
        sounds_test=data_test[1],
        categories_test=data_test[2],
        clip_names_test=data_test[3],
    )
    return 0

if __name__ == '__main__':
    main(sys.argv)
