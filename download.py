import subprocess
import sys
import csv
import random

import skvideo.io
import cv2
import pafy

# Output settings
AUDIO_CODEC = "flac"
AUDIO_CONTAINER = "flac"
VIDEO_CODEC = "h264"
VIDEO_CONTAINER = "mp4"

# Random video 
RANDOM_SEED = 0
NUM_VIDEOS = 10

def main(argv):
    if len(argv) < 3:
        print("Usage: %s [csv file of youtube videos] [output dir]" % argv[0])
        print("Adjust other parameters by modifying the consts at the top of the script")
        return 1
    input_path = argv[1]
    output_dir = argv[2]

    user_permission = query_yes_no(
        "This will download %d video files and %d audio files to %s.  Continue?" % (NUM_VIDEOS, NUM_VIDEOS, output_dir)
    )
    if not user_permission:
        return 0

    youtube_clips = []
    with open(input_path) as f:
        for line in f:
            if not line.startswith("#"):
                youtube_id, trim_start, trim_end = line.strip().split(",")[:3] # The first three columns
                youtube_clips.append([ youtube_id, float(trim_start), float(trim_end) ]) 
    
    random.seed(RANDOM_SEED)
    to_download = random.sample(youtube_clips, NUM_VIDEOS)
    for clip in to_download:
        download_one_clip(clip[0], clip[1], clip[2], output_dir)

    return 0

def query_yes_no(question, default="no"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        choice = input(question + prompt)
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")
    
def download_one_clip(youtube_id, trim_start, trim_end, output_dir):
    """
    youtube_id: string; youtube id to download
    trim_start: float; start position in seconds
    trim_end: float; end position in seconds
    output_dir: string; output directory
    """
    if not output_dir.endswith("/"):
        output_dir += "/"
    youtube_url = "https://www.youtube.com/watch?v=%s" % youtube_id
    duration = trim_end - trim_start
    video = pafy.new(youtube_url)
    video_filepath = "%s%s_%d_%d.%s" % (output_dir, youtube_id, int(trim_start)*1000, int(trim_end) * 1000, VIDEO_CONTAINER)
    audio_filepath = "%s%s_%d_%d.%s" % (output_dir, youtube_id, int(trim_start)*1000, int(trim_end) * 1000, AUDIO_CONTAINER)

    video_download_args = ["ffmpeg", "-n",
        "-ss", str(trim_start), # The beginning of the trim window
        "-i", video.getbestvideo().url,   # Specify the input video URL
        "-t", str(duration),    # Specify the duration of the output
        "-f", VIDEO_CONTAINER,  # Specify the format (container) of the video
        "-framerate", "30",     # Specify the framerate
        "-vcodec", VIDEO_CODEC, # Specify the output encoding
        video_filepath]

    process = subprocess.Popen(video_download_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(stderr)
    else:
        print(video_filepath)

    audio_download_args = ["ffmpeg", "-n",
        "-ss", str(trim_start),  # The beginning of the trim window
        "-i", video.getbestaudio().url,    # Specify the input video URL
        "-t", str(duration),     # Specify the duration of the output
        "-vn",                   # Suppress the video stream
        "-ac", "2",              # Set the number of channels
        "-sample_fmt", "s16",    # Specify the bit depth
        "-acodec", AUDIO_CODEC,  # Specify the output encoding
        "-ar", "44100",          # Specify the audio sample rate
        audio_filepath]

    process = subprocess.Popen(audio_download_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(stderr)
    else:
        print(audio_filepath)

if __name__ == "__main__":
    main(sys.argv)
