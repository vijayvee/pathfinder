import os
import shutil
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm


def move_train_val_test(train: List, val: List, test: List) -> None:
    """Move train, val, test files to corresponding destination"""
    l_files = [train, val, test]
    l_paths = [train_path, val_path, test_path]

    for l_f, dest in zip(l_files, l_paths):
        if l_f:
            for f in tqdm(l_f):
                new_loc = os.path.join(dest,
                                    f.split(data_root)[-1].strip('/').split('/')[0],
                                    f.split("imgs/")[-1].replace("/", "_"))
                shutil.copy(f, new_loc)


if __name__ == "__main__":
    train_txt = "<PATH_TO_TRAIN>"
    val_txt = "<PATH_TO_VAL>"
    test_txt = "<PATH_TO_TEST>"

    data_root = '<PATH_TO_DATA_ROOT>'

    train_path = os.path.join(data_root, "train")
    val_path = os.path.join(data_root, "val")
    test_path = os.path.join(data_root, "test")

    data_paths = [train_path, val_path, test_path]

    for path in data_paths:
        path_pos = Path(os.path.join(path, "curv_contour_length_9"))
        path_neg = Path(os.path.join(path, "curv_contour_length_9_neg"))
        print(path_pos, path_neg)
        path_pos.mkdir(exist_ok=True, parents=True)
        path_neg.mkdir(exist_ok=True, parents=True)

    with open(train_txt, 'r') as f:
        train_f = f.readlines()
        train_f = [i.strip() for i in train_f]

    with open(val_txt, "r") as f:
        val_f = f.readlines()
        val_f = [i.strip() for i in val_f]

    with open(test_txt, "r") as f:
        test_f = f.readlines()
        test_f = [i.strip() for i in test_f]

    move_train_val_test(train_f, val_f, test_f)