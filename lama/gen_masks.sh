#!/bin/bash

dataset="ur10_dataset"

echo "Processing visual_test"

python3 bin/gen_mask_dataset.py $(pwd)/configs/data_gen/random_thin_512.yaml train_data/$dataset/visual_test_source/ train_data/$dataset/visual_test/random_thin_512/ --ext jpg

python3 bin/gen_mask_dataset.py $(pwd)/configs/data_gen/random_thick_512.yaml train_data/$dataset/visual_test_source/ train_data/$dataset/visual_test/random_thick_512/ --ext jpg

python3 bin/gen_mask_dataset.py $(pwd)/configs/data_gen/random_medium_512.yaml train_data/$dataset/visual_test_source/ train_data/$dataset/visual_test/random_medium_512/ --ext jpg

echo "Processing val"

python3 bin/gen_mask_dataset.py $(pwd)/configs/data_gen/random_thin_512.yaml train_data/$dataset/val_source/ train_data/$dataset/val/random_thin_512/ --ext jpg

python3 bin/gen_mask_dataset.py $(pwd)/configs/data_gen/random_thick_512.yaml train_data/$dataset/val_source/ train_data/$dataset/val/random_thick_512/ --ext jpg

python3 bin/gen_mask_dataset.py $(pwd)/configs/data_gen/random_medium_512.yaml train_data/$dataset/val_source/ train_data/$dataset/val/random_medium_512/ --ext jpg

echo "Processing eval"

python3 bin/gen_mask_dataset.py $(pwd)/configs/data_gen/random_thin_512.yaml train_data/$dataset/eval_source/ train_data/$dataset/eval/random_thin_512/ --ext jpg

python3 bin/gen_mask_dataset.py $(pwd)/configs/data_gen/random_thick_512.yaml train_data/$dataset/eval_source/ train_data/$dataset/eval/random_thick_512/ --ext jpg

python3 bin/gen_mask_dataset.py $(pwd)/configs/data_gen/random_medium_512.yaml train_data/$dataset/eval_source/ train_data/$dataset/eval/random_medium_512/ --ext jpg
