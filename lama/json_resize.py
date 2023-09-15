import os
import json

folder_path = '/home/jc-merlab/lama/predict_data/2023-09-13/prediction/kprcnn_ur_dataset_json/'  # Modify this to the path where your JSON files are stored

# Iterate through each JSON file in the folder
for json_file in os.listdir(folder_path):
    if json_file.endswith(".json"):
        json_path = os.path.join(folder_path, json_file)
        
        # Load JSON content
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Modify bounding box values
        for i, (bbox, keypoint) in enumerate(zip(data['bboxes'], data['keypoints'])):
            kpx, kpy, _ = keypoint[0]  # Assuming there's one keypoint per bounding box
            data['bboxes'][i] = [
                kpx - 10,
                kpy - 10,
                kpx + 10,
                kpy + 10
            ]
        
        # Save the modified content back to the file
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)