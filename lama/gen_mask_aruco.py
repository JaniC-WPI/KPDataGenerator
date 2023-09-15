# Original sources:
#   https://python-academia.com/en/opencv-aruco/
#   https://pyimagesearch.com/2020/12/21/detecting-aruco-markers-with-opencv-and-python/
# https://stackoverflow.com/questions/72772170/convert-an-image-to-black-and-white-mask-using-opencv-python
import os
import cv2
from cv2 import aruco
import numpy as np
from shapely import affinity
from shapely.geometry import Polygon
import json


class ArucoMaskGen:

    def __init__(self, raw_folder, dest_folder, full_folder,
                 num_markers, scale_factor=(1.45, 1.45),
                 gen_full_data=True, save_kp=False, json_folder=None,
                 save=True, visualize=False):
        self.raw_folder = raw_folder
        self.raw_files = sorted(os.listdir(raw_folder))  # all files in the data folder
        self.dest_folder = dest_folder
        self.full_folder = full_folder

        self.scale_factor = scale_factor
        self.num_markers = num_markers

        self.visualize = visualize

        self.gen_full_data = gen_full_data
        self.save_kp = save_kp  # Save keypoints and bboxes to json
        self.json_folder = json_folder if json_folder is not None \
            else self.dest_folder

        self.save = save

        self.dict_aruco = aruco.Dictionary_get(aruco.DICT_4X4_50)
        self.parameters = aruco.DetectorParameters_create()

    def gen_mask(self, img):
        """
        Generate mask of markers and return keypoints
        """
        kps = []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected = aruco.detectMarkers(
            gray, self.dict_aruco, parameters=self.parameters)

        # Sort ids in increasing order
        corners = list(zip(*sorted(list(zip(corners, ids)), key=lambda e: e[1])))[0]

        # Convert all points to int32 for cv2 to work
        corners = [c.astype(np.int32) for c in corners]

        mask = np.zeros(img.shape[:2], np.uint8)  # black by default
        # Enlarge the mask to account for the white border
        # Method 1: dilation
        # cv2.fillPoly(mask, corners, 255)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        # mask = cv2.dilate(mask, kernel, iterations=1)
        # Method 2: scale the bounding polygon
        for i in range(len(corners)):
            polygon = Polygon(corners[i][0])
            scaled_polygon = affinity.scale(polygon,
                                            xfact=self.scale_factor[0],
                                            yfact=self.scale_factor[1])
            corners[i][0] = \
                np.array(scaled_polygon.exterior.coords[:-1], np.int32)
            # keypoint [[x,y,visibility]]
            kps.append([list(scaled_polygon.centroid.coords[0]) + [1]])
            cv2.fillPoly(mask, corners, 255)

        if self.visualize:
            masked_img = np.copy(img)
            mask_3ch = cv2.merge((mask, mask, mask))  # 3 channel
            # cv2.fillPoly(masked_img, corners, color=(255, 255, 255))
            masked_img = cv2.bitwise_and(masked_img,
                                         cv2.bitwise_not(mask_3ch))
            # Visualizes original, mask, masked image
            vis_img = np.concatenate((img, mask_3ch, masked_img), axis=1)
            # Resize to fit screen
            vis_img = self.resize_with_aspect_ratio(vis_img, height=480)

            cv2.imshow('visualize', vis_img)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()

        return mask, kps

    def resize_with_aspect_ratio(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        """
        Reference: https://stackoverflow.com/questions/35180764/opencv-python-image-too-big-to-display
        """
        dim = None
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        return cv2.resize(image, dim, interpolation=inter)

    def run(self):
        skipped_files = []
        for f in self.raw_files:
            raw_file_name, extension = os.path.splitext(f)
            mask_file_name = raw_file_name + '_mask' + extension
            img = cv2.imread(os.path.join(self.raw_folder, f))
            mask, kps = self.gen_mask(img)
            if len(kps) < self.num_markers:
                # Skip file if not detected all markers
                print(f'Skipped file {raw_file_name}')
                skipped_files.append(raw_file_name)
                continue
            if self.save:
                cv2.imwrite(
                    os.path.join(self.dest_folder, mask_file_name), mask)
                print(f'Saved mask: {mask_file_name}')
                if self.gen_full_data:
                    # Also copy raw and mask to a final folder
                    cv2.imwrite(
                        os.path.join(self.full_folder, mask_file_name), mask)
                    cv2.imwrite(
                        os.path.join(self.full_folder, f), img)
                    print(f'Copied to full folder: {raw_file_name}')
            if self.save_kp:
                # keypoint json
                data = {
                    "id": int(raw_file_name),
                    "image_rgb": f,
                    "bboxes": [
                        [
                            kp[0][0]-20,
                            kp[0][1]-20,
                            kp[0][0]+20,
                            kp[0][1]+20
                        ] for kp in kps
                    ],
                    "keypoints": kps,
                }

                # if len(data["bboxes"]) < self.robot.num_joints:
                #     rospy.logerr('Invalid length of [data]')

                # save [data] to json file
                json_obj = json.dumps(data, indent=4)
                json_filepath = os.path.join(
                    self.json_folder, f'{raw_file_name}.json')
                with open(json_filepath, "w") as outfile:
                    outfile.write(json_obj)
                print(f'Saved json: {json_filepath}')
        print(f'Mask generation completed. Number of skipped files: \
                {len(skipped_files)}/{len(self.raw_files)}')
        print(f'Success rate: {100-100*len(skipped_files)/len(self.raw_files)}%')
        # print('Details:')
        # print(skipped_files)


if __name__ == "__main__":
    # raw_folder = '/home/duk3/Workspace/WPI/Summer2023/ws/lama/aruco_data_panda/'
    # dest_folder = '/home/duk3/Workspace/WPI/Summer2023/ws/lama/aruco_masked/'
    folder = '2023-09-13'
    raw_folder = f'/home/jc-merlab/lama/predict_data/{folder}/raw/'
    dest_folder = f'/home/jc-merlab/lama/predict_data/{folder}/mask/'
    json_folder = f'/home/jc-merlab/lama/predict_data/{folder}/json/'
    full_folder = f'/home/jc-merlab/lama/predict_data/{folder}/raw_and_mask/'

    ArucoMaskGen(raw_folder,
                 dest_folder,
                 full_folder,
                 3,
                 scale_factor=(1.6, 1.6),
                 gen_full_data=True,
                 save_kp=True,
                 json_folder=json_folder,
                 save=True,
                 visualize=False).run()
