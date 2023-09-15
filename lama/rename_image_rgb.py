import os
import json

folder_path = 'path_to_your_folder'  # Replace with the path to your folder containing the images and json files

# Get the list of all image and json files in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

# Sort the files for processing
image_files.sort()
json_files.sort()

# Ensure equal number of image and json files
if len(image_files) != len(json_files):
    raise ValueError("Number of image files doesn't match the number of json files.")

# Rename each image and update the corresponding json file
for idx, (img_file, json_file) in enumerate(zip(image_files, json_files)):
    new_idx = idx + 1

    # Rename image file
    new_img_name = f"{new_idx}.jpg"
    os.rename(os.path.join(folder_path, img_file), os.path.join(folder_path, new_img_name))

    # Update json file
    with open(os.path.join(folder_path, json_file), 'r') as jf:
        data = json.load(jf)
        data['id'] = new_idx
        data['image_rgb'] = new_img_name

    new_json_name = f"{new_idx}.json"
    with open(os.path.join(folder_path, new_json_name), 'w') as jf:
        json.dump(data, jf, indent=4)

    # Rename json file if its name was changed
    if json_file != new_json_name:
        os.rename(os.path.join(folder_path, json_file), os.path.join(folder_path, new_json_name))

print("Files renamed and json updated successfully!")

