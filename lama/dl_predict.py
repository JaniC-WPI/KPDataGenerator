#!/usr/bin/env python3

"""
Predicts using R-CNN model.
"""

import os
import numpy as np
import cv2
import torch
import torchvision
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import json


int_stream = '0000000'


def preprocess_img(img):
    """
    Converts image to tensor to put into Model.
    """
    img_tmp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return F.to_tensor(img_tmp)


def visualize(image, bboxes, keypoints, image_original=None, bboxes_original=None, keypoints_original=None):
    fontsize = 18

    for bbox in bboxes:
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])
        image = cv2.rectangle(image.copy(), start_point, end_point, (0,255,0), 2)

    for idx, kps in enumerate(keypoints):
        for kp in kps:
            image = cv2.circle(image.copy(), tuple(kp), 2, (255,0,0), 10)

    if image_original is None and keypoints_original is None:
        # plt.figure(figsize=(40,40))
        # plt.imshow(image)

        return image


class KpDLVisualizer():
    """
    Visualize prediction from KP detection network
    """

    def __init__(self, model_path, img_dir, save_img=False, save_dir=None,
                 save_json=False, json_dir=None, vis=False, threshold=0.7,
                 num_kps=0, calc_stats=False, save_stats=False):
        """
        model_path: str: path to keypoint detection model
        im_plt: unused
        save_img: save the prediction to file
        save_dir: if [save_img] is True, this will be the path to save images
        """
        self.model_path = model_path
        self.img_dir = img_dir

        self.save_img = save_img
        self.save_dir = save_dir

        self.save_json = save_json
        self.json_dir = json_dir

        self.vis = vis

        # Create folder if not exists
        if self.save_img and not os.path.exists(self.save_dir):
            print('Save folder not found. Creating one...')
            os.mkdir(save_dir)

        self.threshold = threshold
        self.num_kps = num_kps
        self.calc_stats = calc_stats
        self.save_stats = save_stats
        self.stats = {'success': 0, 'fail': 0, 'num_kps': []}

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = torch.load(self.model_path).to(self.device)
        self.model.eval()

    def visualize(self, img):
        img_tensor = preprocess_img(img).to(self.device)
        with torch.no_grad():
            outputs = self.model([img_tensor])
            img_tensor_tup = (
                img_tensor.permute(1, 2, 0).detach().cpu().numpy() * 255
            ).astype(np.uint8)
            scores = outputs[0]['scores'].detach().cpu().numpy()
            # Indexes of boxes with scores > 0.7
            high_scores_idxs = np.where(scores > self.threshold)[0].tolist()
            # Indexes of boxes left after applying NMS (iou_threshold=0.3)
            post_nms_idxs = torchvision.ops.nms(
                outputs[0]['boxes'][high_scores_idxs],
                outputs[0]['scores'][high_scores_idxs], 0.3).cpu().numpy()

            keypoints = []
            for kps in outputs[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
                keypoints.append([list(map(int, kp[:2])) for kp in kps])

            bboxes = []
            for bbox in outputs[0]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
                bboxes.append(list(map(int, bbox.tolist())))

            json_data = None
            if self.save_json:
                json_data = self.gen_json(keypoints, bboxes)

            if self.calc_stats:
                if len(keypoints) == self.num_kps:
                    self.stats['success'] += 1
                else:
                    self.stats['fail'] += 1
                self.stats['num_kps'].append(len(keypoints))
            return visualize(img_tensor_tup, bboxes, keypoints), json_data

    def gen_json(self, keypoints, bboxes):
        print(keypoints)
        print(np.array(keypoints).shape)
        data = {
            "id": 0,  # placeholder
            "image_rgb": "",  # placeholder
            "bboxes": bboxes,
            "keypoints": [
                [
                    [
                        keypoints[i][i][0],
                        keypoints[i][i][1],
                        1
                    ]
                ] for i in range(len(keypoints[0]))
            ],
        }
        raise TypeError("lolol")
        return data

    def run(self):
        """Use cv2 to plot each image with keypoints and bounding boxes."""
        # Get all image files
        img_files = os.listdir(self.img_dir)
        img_files = [i for i in img_files if i.endswith('jpg')
                     or i.endswith('png')]

        for img_file in img_files:
            # process file names
            img_path = os.path.join(self.img_dir, img_file)

            raw_img = cv2.imread(img_path)
            vis_img, data = self.visualize(raw_img)

            name, ext = os.path.splitext(img_file)
            
            if self.save_json:
                data["id"] = int(name)
                data["image_rgb"] = img_file
                json_obj = json.dumps(data, indent=4)
                filename = os.path.join(self.json_dir, name + ".json")
                with open(filename, "w") as outfile:
                    outfile.write(json_obj)

            if self.save_img:
                save_img_path = os.path.join(self.save_dir, img_file)
                cv2.imwrite(save_img_path, vis_img)
                print(f'kp_gen(): Saved image {save_img_path}')

            if self.vis:
                cv2.imshow('Camera', vis_img)
                cv2.waitKey(3000)  # 3 sec
                cv2.destroyWindow('Camera')

        if self.vis:
            cv2.destroyAllWindows()

        if self.calc_stats:
            print('Stats:')
            print(f"Success: {self.stats['success']}/{len(self.stats['num_kps'])} ({100*self.stats['success']/len(self.stats['num_kps']):.2f}%)")
            if self.save_stats:
                with open(os.path.join(self.save_dir, '0_stats.txt'), 'w') as f:
                    f.write('Stats:\n')
                    f.write(f"Success: {self.stats['success']}/{len(self.stats['num_kps'])} ({100*self.stats['success']/len(self.stats['num_kps']):.2f}%)\n")
                print('Saved stats.txt')
            plt.hist(self.stats['num_kps'])
            plt.show()
            plt.title('Number of keypoints detected')


if __name__ == '__main__':
    date = '08_22_2023'
    model = 'keypointsrcnn_weights_ld_b1_e25_vorigami'
    # static_data dir
    # img_dir = '/home/jc-merlab/lama/rcnn_test_sets/origami/static_data/data/raw'
    # rcnn test set dir
    img_dir = f'/home/jc-merlab/lama/results/{date}/rcnn'
    # non_masked_254
    # img_dir = '/home/jc-merlab/lama/rcnn_test_sets/origami/non_masked_254/data'
    
    model_path = f'/home/jc-merlab/lama/results/{date}/trained_models/{model}.pth'
    
    # static data dir
    # save_dir = f'/home/jc-merlab/lama/rcnn_test_sets/origami/static_data/results/{date}/{model}'
    # rcnn test set dir
    save_dir = f'/home/jc-merlab/lama/rcnn_test_sets/origami/rcnn/results/{date}/{model}'
    # non_masked_254
    # save_dir = f'/home/jc-merlab/lama/rcnn_test_sets/origami/non_masked_254/results/{date}/{model}'
    
    json_dir = save_dir

    # Check GPU memory
    t = torch.cuda.get_device_properties(0).total_memory
    print(t)
    torch.cuda.empty_cache()

    r = torch.cuda.memory_reserved(0)
    print(r)
    a = torch.cuda.memory_allocated(0)
    print(a)

    KpDLVisualizer(
        model_path=model_path,
        img_dir=img_dir,
        save_img=True,
        save_dir=save_dir,
        save_json=False,
        json_dir=json_dir,
        vis=False,
        threshold=0.1,
        num_kps=4,
        calc_stats=True,
        save_stats=True).run()
