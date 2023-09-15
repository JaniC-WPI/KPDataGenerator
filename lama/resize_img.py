import cv2
import os
import sys


def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
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


if __name__ == '__main__':
    try:
        data_dir = sys.argv[1]
        dest_dir = sys.argv[2]
        width = int(sys.argv[3])
        if len(sys.argv) > 4:
            ext = sys.argv[4]
        else:
            ext = '.jpg'
    except IndexError:
        print('Incorrect syntax. Usage:')
        print('python3 resize_img.py data_dir dest_dir width [ext]')

    if not os.path.exists(dest_dir):
        print(f'Created dir {dest_dir}')
        os.mkdir(dest_dir)

    img_files = os.listdir(data_dir)
    img_files = [i for i in img_files if i.endswith(ext)]
    for img_file in img_files:
        img = cv2.imread(os.path.join(data_dir, img_file))
        resized_img = resize_with_aspect_ratio(img, width=width)
        cv2.imwrite(os.path.join(dest_dir, img_file), resized_img)
        print(f'Resized img {os.path.join(dest_dir, img_file)}')
