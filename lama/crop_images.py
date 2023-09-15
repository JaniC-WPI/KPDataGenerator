"""Crops all images in a dir at specified region."""
import os
import sys
from PIL import Image


default_src_dir = '/home/jc-merlab/lama/train_data/tmp'
default_dest_dir = '/home/jc-merlab/lama/train_data/tmp'


if __name__ == '__main__':
    # Parameters
    # src_dir = sys.argv[1] if len(sys.argv) > 1 else default_src_dir
    # dest_dir = sys.argv[2] if len(sys.argv) > 2 else default_dest_dir
    src_dir = default_src_dir
    dest_dir = default_dest_dir
    region = (0, 0, 480, 480)

    #
    img_files = [f for f in os.listdir(src_dir) if f.endswith('.png') or
                 f.endswith('.jpg')]
    for img_file in img_files:
        img = Image.open(os.path.join(src_dir, img_file))
        img = img.crop(region)
        # Visualize
        # img.show()
        img.save(os.path.join(dest_dir, img_file))
        print(f"Cropped {img_file}")
