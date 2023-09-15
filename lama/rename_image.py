#!/usr/bin/env python3

import os

# Set the directory where the images are stored
image_dir = '/home/jc-merlab/lama/results/09_04_2023/rcnn'  # Current directory

# List all files in the directory
all_files = sorted(os.listdir(image_dir))

# Filter out files that are JPEG images
image_files = [f for f in all_files if f.endswith('.rgb.jpg')]

# Sort files to rename them in ascending order
#image_files.sort(key=lambda f: int(f.split('.')[0]))

# Rename each file
for old_name in image_files:
    # Extract the numerical part from the filename
    #num = int(old_name.split('.')[0])

    # Create the new name by replacing ".rgb.jpg" with ".jpg"
    new_name = old_name.replace(".rgb.jpg", ".jpg")

    # Full path to old and new files
    old_path = os.path.join(image_dir, old_name)
    new_path = os.path.join(image_dir, new_name)

    # Rename file
    os.rename(old_path, new_path)

    print(f"Renamed {old_name} to {new_name}")
