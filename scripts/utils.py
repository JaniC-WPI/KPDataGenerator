import errno
import os
import time
from ruamel.yaml import YAML
from pathlib import Path
from datetime import datetime
import albumentations as A
from os.path import expanduser
import shutil

import torch
import torch.distributed as dist


class DataPrePro:
    def __init__(self):

        # to generalize home directory. User can change their parent path without entering their home directory
        self.home = expanduser("~")

        # specifying variables to save dataset in folders according to date and time to keep track
        today = datetime.now()
        self.year = str(today.strftime('%Y'))
        self.month = str(today.strftime('%m'))
        self.day = str(today.strftime('%d'))
        self.h = str(today.hour)
        self.m = str(today.minute)       
        self.parent_path = self.home + "/Pictures/" + "Data/"
        self.root_dir = self.parent_path + self.year + "-" + self.month + "-" + self.day + "/"

    def train_transform(self):
        return A.Compose([
            A.Sequential([
                A.RandomRotate90(p=1), # Random rotation of an image by 90 degrees zero or more times
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, brightness_by_max=True, always_apply=False, p=1), # Random change of brightness & contrast
            ], p=1)
        ],
        keypoint_params=A.KeypointParams(format='xy'), # More about keypoint formats used in albumentations library read at https://albumentations.ai/docs/getting_started/keypoints_augmentation/
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bboxes_labels']) # Bboxes should have labels, read more at https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
    )


    def read_yaml(self):
        path = Path('config/hyperparams.yaml')
        yaml = YAML(typ='safe')
        data = yaml.load(path)
        hp = data['hyperparameters']

        return hp

    def curr_split(self, mode):
        latest_split_folder = self.parent_path + "split_folder_latest"
        current_split_folder = self.parent_path + "split_folder_current"

        if mode == 'train':    
            if os.path.exists(current_split_folder):
                shutil.rmtree(current_split_folder)
                print("current folder removed")

            if os.path.exists(latest_split_folder):
                shutil.copytree(latest_split_folder, current_split_folder)

            new_split_folder_name = self.parent_path + "split_folder_output" + "-" + self.year + "-" + self.month + "-" + self.day
            os.rename(latest_split_folder,new_split_folder_name)
            
            return current_split_folder

        else:
            return current_split_folder






