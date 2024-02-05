import os
import shutil
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

data_root = '/mnt/sphere/projects/contour_integration/pathfinder_full/curv_contour_length_combined/'

def move_train_val_test(train: List, val: List, test: List) -> None:
    """Move train, val, test files to corresponding destination"""
    l_files = [train, val, test]
    l_paths = [train_path, val_path, test_path]
    k = 0
    for l_f, dest in zip(l_files, l_paths):
        if l_f:
            for f in tqdm(l_f):
                new_file_path = f.split('/imgs/')
                
                new_loc = os.path.join(dest,
                                    f.split(data_root)[-1].strip('/').split('/')[0],
                                    f.split("imgs/")[-1].replace("/", "_"))
                shutil.copy(f, new_loc)

def move_train_val_test2(train, val, test):
    filepaths_splits = [train, val, test]
    folder_split_names = ["train", "val", "test"]
    for filepaths, folder_split_name in zip(filepaths_splits, folder_split_names):
        for filepath in tqdm(filepaths):
            # filepath example: /mnt/sphere/projects/contour_integration/pathfinder_full/curv_contour_length_combined/curv_contour_length_combined_neg/imgs/2/sample_4283.png
            # new_filepath example: /mnt/sphere/projects/contour_integration/pathfinder_full/curv_contour_length_combined/train/curv_contour_length_combined_neg/2_sample_4283.png
            # The below code converts filepath to new_filepath
            split_path = filepath.split('pathfinder_full/curv_contour_length_combined/')
            pos_or_neg, filename = split_path[1].split('/imgs/')
            filename = filename.replace("/", "_")
            new_file_path = os.path.join(data_root, folder_split_name, pos_or_neg, filename)
            shutil.copy(filepath, new_file_path)

if __name__ == "__main__":
    train_txt = "/mnt/sphere/projects/contour_integration/pathfinder_full/curv_contour_length_combined/train.txt"
    val_txt = "/mnt/sphere/projects/contour_integration/pathfinder_full/curv_contour_length_combined/val.txt"
    test_txt = "/mnt/sphere/projects/contour_integration/pathfinder_full/curv_contour_length_combined/test.txt"

    train_path = os.path.join(data_root, "train")
    val_path = os.path.join(data_root, "val")
    test_path = os.path.join(data_root, "test")

    data_paths = [train_path, val_path, test_path]

    destpaths = {}
    for path in data_paths:
        path_pos = Path(os.path.join(path, "curv_contour_length_combined"))
        path_neg = Path(os.path.join(path, "curv_contour_length_combined_neg"))
        print(path_pos, path_neg)
        path_pos.mkdir(exist_ok=True, parents=True)
        path_neg.mkdir(exist_ok=True, parents=True)
        destpaths
    with open(train_txt, 'r') as f:
        train_f = f.readlines()
        train_f = [i.strip() for i in train_f]

    with open(val_txt, "r") as f:
        val_f = f.readlines()
        val_f = [i.strip() for i in val_f]

    with open(test_txt, "r") as f:
        test_f = f.readlines()
        test_f = [i.strip() for i in test_f]

    move_train_val_test2(train_f, val_f, test_f)