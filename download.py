import subprocess
import sys
import csv
import random
import os

import cv2
import pafy

from youtube_clip import YouTubeClip
import process_ontology

# Output settings
AUDIO_CODEC = "flac"
AUDIO_CONTAINER = "flac"
VIDEO_CODEC = "h264"
VIDEO_CONTAINER = "mp4"

# Random video settings
CATEGORY_WHITELIST = ["Gunshot, gunfire", "Musical instrument", "Explosion", "Computer keyboard", "Sneeze"] # AudioSet terms to use.  If empty, uses all of them.
RANDOM_SEED = 0
NUM_VIDEOS = 10000
START_VIDEO = 6341

def main(argv):
    # Parse args
    if len(argv) < 3:
        print("Usage: %s [csv file of youtube videos] [output dir]" % argv[0])
        print("Adjust other parameters by modifying the consts at the top of the script")
        return 1

    input_path = argv[1]
    output_dir = argv[2]

    # Check and confirm download settings
    ontology_records = process_ontology.get_records(CATEGORY_WHITELIST)
    if len(ontology_records) > 0:
        print("Video content whitelist:", [record["name"] for record in ontology_records])
    wanted_ids = set([record["id"] for record in ontology_records])

    user_permission = query_yes_no(
        "This will download %d video files and %d audio files to %s.  Continue?" % (NUM_VIDEOS, NUM_VIDEOS, output_dir)
    )
    if not user_permission:
        return 0

    # Parse available clips
    print("Reading available clips...")
    youtube_clips = read_clips(input_path)
    print("Sampling clips...")
    to_download = sample_clips(youtube_clips, NUM_VIDEOS, wanted_ids)
    print("Starting downloads.")
    for (i, clip) in enumerate(to_download):
        if i >= START_VIDEO:
            download_one_clip(clip, output_dir, i)

    return 0

def query_yes_no(question, default="no"):
    """Asks a yes/no question via stdin and return the user's answer.
    
    Arguments:
        question {string} -- string presented to user via stdout
    
    Keyword Arguments:
        default {string} -- the presumed answer if the user just hits Enter.
            Valid options include "yes"; "no"; and None, meaning an answer is required.
            (default: {"no"})
    
    Returns:
        {boolean} -- whether the user answered in the affimative
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

def read_clips(input_path):
    youtube_clips = []
    with open(input_path) as f:
        lines = [line for line in f if not line.startswith("#")]
        csv_reader = csv.reader(lines, skipinitialspace=True)
        for row in csv_reader:
            youtube_clips.append(YouTubeClip(*row))
    return youtube_clips

def sample_clips(youtube_clips, num_clips, wanted_ids=[]):
    if len(wanted_ids) > 0:
        whitelist_clips = []
        for clip in youtube_clips:
            for label in clip.labels:
                if label in wanted_ids:
                    whitelist_clips.append(clip)
                    break
    else:
        whitelist_clips = youtube_clips

    random.seed(RANDOM_SEED)
    return random.sample(whitelist_clips, num_clips)

def download_one_clip(clip, output_dir, index=None):
    """Downloads one YouTube clip.
    
    Arguments:
        clip {YouTubeClip} -- clip to download
        output_dir {string} -- output directory
    
    Keyword Arguments:
        index {int} -- Number to prepend to print statements (default: {None})
    
    Returns:
        {boolean} -- whether download succeeded
    """
    if not output_dir.endswith("/"):
        output_dir += "/"
    output_dir += clip.labels[0].replace("/", "_") + "/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_filepath = "%s%s.%s" % (output_dir, clip.to_string(), VIDEO_CONTAINER)
    audio_filepath = "%s%s.%s" % (output_dir, clip.to_string(), AUDIO_CONTAINER)

    
    youtube_url = "https://www.youtube.com/watch?v=%s" % clip.id
    video = video_url = audio_url = None
    try:
        video = pafy.new(youtube_url)
        video_url = video.getbestvideo().url
        audio_url = video.getbestaudio().url
    except KeyboardInterrupt as interrupt:
        print("Download interrupted.  You should start from video #", index, "with random seed",
            RANDOM_SEED, "next time.")
        sys.exit(0)
    except:
        print("Error:", youtube_url, "is invalid or unavailable", file=sys.stderr)
        return False

    try:
        video_download_args = ["ffmpeg", "-n",
            "-ss", str(clip.trim_start), # The beginning of the trim window
            "-i", video_url,             # Specify the input video URL
            "-t", str(clip.get_duration()),    # Specify the duration of the output
            "-f", VIDEO_CONTAINER,  # Specify the format (container) of the video
            "-framerate", "30",     # Specify the framerate
            "-vcodec", VIDEO_CODEC, # Specify the output encoding
            video_filepath]

        success = True
        process = subprocess.Popen(video_download_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(stderr.decode('utf-8'), file=sys.stderr)
            success = False
        else:
            print(index, video_filepath)

        audio_download_args = ["ffmpeg", "-n",
            "-ss", str(clip.trim_start),  # The beginning of the trim window
            "-i", audio_url,              # Specify the input video URL
            "-t", str(clip.get_duration()),     # Specify the duration of the output
            "-vn",                   # Suppress the video stream
            "-ac", "2",              # Set the number of channels
            "-sample_fmt", "s16",    # Specify the bit depth
            "-acodec", AUDIO_CODEC,  # Specify the output encoding
            "-ar", "44100",          # Specify the audio sample rate
            audio_filepath]

        process = subprocess.Popen(audio_download_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(stderr.decode('utf-8'), file=sys.stderr)
            success = False
        else:
            print(index, audio_filepath)
        
        return success
    except KeyboardInterrupt as interrupt:
        try:
            os.remove(video_filepath)
            os.remove(audio_filepath)
        except FileNotFoundError:
            pass
        finally:
            print("Download interrupted.  You should start from video #", index, "with random seed",
                RANDOM_SEED, "next time.")
            sys.exit(0)

if __name__ == "__main__":
    main(sys.argv)
