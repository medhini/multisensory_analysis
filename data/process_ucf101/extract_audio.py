import sys
import os
import subprocess
import multiprocessing

VIDEO_FORMAT = ".avi"
AUDIO_FORMAT = ".wav"

N_THREADS = 12

def extract_one_file(args):
    in_path, out_path = args
    return_value = subprocess.call(
        ["ffmpeg", "-i", in_path, "-vn", out_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    if return_value is 0:
        print(in_path, "-->", out_path)
    else:
        print(in_path, ": failed")

if len(sys.argv) < 3:
    print("Usage: %s [input dir] [output dir]" % sys.argv[0])

input_dir = sys.argv[1]
output_dir = sys.argv[2]
thread_pool = multiprocessing.Pool(N_THREADS)
args = []
for root, dirs, files in os.walk(input_dir):
    for file_name in files:
        if file_name.endswith(VIDEO_FORMAT):
            in_path = os.path.join(root, file_name)
            base_name = file_name.rstrip(VIDEO_FORMAT)
            out_path = os.path.join(output_dir, base_name + AUDIO_FORMAT)
            args.append((in_path, out_path))
thread_pool.map(extract_one_file, args)
