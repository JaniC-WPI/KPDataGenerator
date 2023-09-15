import os
import random
from PIL import Image


src_dir = '/home/jc-merlab/lama/train_data/ur10_dataset/raw'
output_root_dir = '/home/jc-merlab/lama/train_data/ur10_dataset'


if __name__ == '__main__':
    # Params:
    # Number of files in each folder
    eval_len = 2500
    val_len = 5000
    visual_test_len = 120
    
    src_files = os.listdir(src_dir)
    
    if len(src_files) < (eval_len + val_len + visual_test_len):
        print("Not sufficient data.")
       
    tmp = set(src_files)
    # Sample visual_test_src
    visual_test_src_files = random.sample(tmp, visual_test_len)
    tmp = tmp.difference(visual_test_src_files)
    # Sample eval_src
    eval_src_files = random.sample(tmp, eval_len)
    tmp = tmp.difference(eval_src_files)
    # Sample val_src
    val_src_files = random.sample(tmp, val_len)
    tmp = tmp.difference(val_src_files)
    
    # Save data
    for f in visual_test_src_files:
        image = Image.open(os.path.join(src_dir, f))
        image.save(os.path.join(os.path.join(output_root_dir, 'visual_test_source'), f))
    print(f'Done saving visual_test_source. Total files: {len(visual_test_src_files)}')
    for f in eval_src_files:
        image = Image.open(os.path.join(src_dir, f))
        image.save(os.path.join(os.path.join(output_root_dir, 'eval_source'), f))
    print(f'Done saving eval_source. Total files: {len(eval_src_files)}')
    for f in val_src_files:
        image = Image.open(os.path.join(src_dir, f))
        image.save(os.path.join(os.path.join(output_root_dir, 'val_source'), f))
    print(f'Done saving val_source. Total files: {len(val_src_files)}')
    for f in tmp:
        image = Image.open(os.path.join(src_dir, f))
        image.save(os.path.join(os.path.join(output_root_dir, 'train'), f))
    print(f'Done saving train. Total files: {len(tmp)}')
