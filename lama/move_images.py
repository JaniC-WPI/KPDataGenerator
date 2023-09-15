import os
import shutil

# Path where the mixed image files are stored
source_dir = '/home/jc-merlab/lama/results/raw_and_mask'  # Update this path as needed

# Path where you want to store the separated JPG files
destination_dir = '/home/jc-merlab/lama/results/09_04_2023/rcnn'  # Update this path as needed

# Create destination directory if it doesn't exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# List all files in the source directory
all_files = sorted(os.listdir(source_dir))

# Filter out JPG files
jpg_files = [f for f in all_files if f.lower().endswith('.jpg')]

# Move JPG files to the destination directory
for jpg in jpg_files:
    source_path = os.path.join(source_dir, jpg)
    destination_path = os.path.join(destination_dir, jpg)
    
    # Move the file
    shutil.move(source_path, destination_path)
    
    print(f"Moved {jpg} to {destination_dir}")

print("All JPG files have been moved.")

