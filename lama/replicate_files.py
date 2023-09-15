"""Replicate files in a directory to a specified amount, optionally randomize
order.
Note: script expect original files named from 0 to (n-1), where n is number of
original files."""
import os
import sys
from PIL import Image
import random


default_dir = '/home/duk3/Workspace/WPI/Summer2023/ws/lama/predict_data/static_data/tmp'
default_replicate_times = 0


if __name__ == '__main__':
    # Parameters
    dir = sys.argv[1] if len(sys.argv) > 1 else default_dir
    random_order = True
    replicate_times = int(sys.argv[2]) if len(sys.argv) > 2 \
        else default_replicate_times

    #
    files = os.listdir(dir)
    for i in range(1, replicate_times+1):
        random.shuffle(files)
        for f in files:
            name, ext = os.path.splitext(f)
            new_file = str(int(name) + i*len(files)) + ext
            img = Image.open(os.path.join(dir, f))
            img.save(os.path.join(dir, new_file))
            print(f"Copied {f} to {new_file}")
