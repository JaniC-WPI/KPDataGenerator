#!/usr/bin/env python3

import cv2
import json
import os

data_dir = '/home/jc-merlab/lama/predict_data/2023-09-13/prediction/kprcnn_ur_dataset_json'
dest_dir = '/home/jc-merlab/lama/predict_data/2023-09-13/prediction/kprcnn_ur_dataset_json-vis'

save_flag = True
visualize_flag = False
show_kp_idx = True

# Create destination directory if it does not exist
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

def visualize(image, keypoints, bboxes, image_name):
    if image is None:
        print(f"Image not found for {image_name}")
        return

    for i in range(len(keypoints)):
        # Draw keypoints
        u = round(keypoints[i][0][0])
        v = round(keypoints[i][0][1])
        image = cv2.circle(image, (u, v), radius=5, color=(0, 0, 255), thickness=-1)
        
        # Draw bounding boxes
        u_bb1 = round(bboxes[i][0])
        v_bb1 = round(bboxes[i][1])
        u_bb2 = round(bboxes[i][2])
        v_bb2 = round(bboxes[i][3])
        image = cv2.rectangle(image, (u_bb1, v_bb1), (u_bb2, v_bb2), color=(0, 255, 0), thickness=3)
        
        # Draw index of kp
        global show_kp_idx
        if show_kp_idx:
            # font
            font = cv2.FONT_HERSHEY_SIMPLEX
            # org
            org = (u, v)
            # fontScale
            fontScale = 1
            # Blue color in BGR
            color = (0, 255, 0)
            # Line thickness of 2 px
            thickness = 2 
            image = cv2.putText(image, str(i), org, font, 
                       fontScale, color, thickness, cv2.LINE_AA)
    
    # Visualize
    global visualize_flag
    if visualize_flag:
        cv2.imshow('img', image)
        cv2.waitKey(0)
        cv2.destroyWindow('img')
    
    # Save the image
    global save_flag
    if save_flag:
        save_path = os.path.join(dest_dir, image_name)
        cv2.imwrite(save_path, image)
        print(f'Saved {save_path}')

if __name__ == '__main__':
    data_files = os.listdir(data_dir)  # All files in the data folder
    
    # Filter for JSON files
    json_files = sorted([f for f in data_files if f.endswith('.json')])
    
    # Loop through each JSON file
    for f in json_files:
        json_path = os.path.join(data_dir, f)
        
        with open(json_path, 'r') as f_json:
            data = json.load(f_json)
            image_path = os.path.join(data_dir, data['image_rgb'])
            image = cv2.imread(image_path)
            
            # Extract just the file name without directory to use as new image name
            image_name = os.path.basename(data['image_rgb'])
            
            visualize(image, data['keypoints'], data['bboxes'], image_name)
            
    cv2.destroyAllWindows()
