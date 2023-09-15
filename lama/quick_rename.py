import os

# Set the directories where the images and JSON files are stored
image_dir = '/home/jc-merlab/lama/results/raw_and_mask'  # Update if your images are in a different folder
json_dir = '/home/jc-merlab/lama/results/json'  # Update if your JSON files are in a different folder

# List all files in the directories
image_files = os.listdir(image_dir)
json_files = os.listdir(json_dir)

# Filter out the relevant image and JSON files
jpg_files = sorted([f for f in image_files if f.endswith('.jpg')])
png_files = sorted([f for f in image_files if f.endswith('.png')])
json_files = sorted([f for f in json_files if f.endswith('.json')])

# Check if the lengths are matching
if len(jpg_files) != len(png_files) or len(jpg_files) != len(json_files):
    print("The number of jpg, png and json files do not match. Please check the directories.")
    exit()

# Rename each set of files
for i, (jpg, png, jsn) in enumerate(zip(jpg_files, png_files, json_files)):
    new_name = f"{i:06}"

    # Rename jpg file
    os.rename(os.path.join(image_dir, jpg), os.path.join(image_dir, f"{new_name}.rgb.jpg"))
    # Rename png file
    os.rename(os.path.join(image_dir, png), os.path.join(image_dir, f"{new_name}.png"))
    # Rename json file
    os.rename(os.path.join(json_dir, jsn), os.path.join(json_dir, f"{new_name}.json"))

    print(f"Renamed {jpg}, {png}, and {jsn} to {new_name}.jpg, {new_name}.png, and {new_name}.json")

print("All files have been renamed.")
