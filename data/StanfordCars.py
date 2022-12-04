# coding:utf-8
# Author : klein
# CONTACT: zhangchy2@shanghaitech.edu.cn
# SOFTWARE: PyCharm
# FILE : StanfordCars.py
# DATE : 2022/8/28 13:01

import torch
from torch.utils.data import Dataset
import torchvision
from config.config import scars_root
from scipy import io
from utils.utils import *
import numpy as np
import os
from PIL import Image
from collections import Counter
import pickle as pkl


# 196: 98 98
class StanfordDataset(Dataset):
    def __init__(self, root_path, image_and_anno, transform=None):
        super(StanfordDataset, self).__init__()
        self.images_path = list(image_and_anno.keys())
        self.targets = np.array(list(image_and_anno.values()))
        self.transform = transform
        self.image_and_anno = image_and_anno
        self.root_path = root_path

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        _path = os.path.join(self.root_path, self.images_path[item])
        image = Image.open(_path).convert('RGB')
        label = self.targets[item]
        # label = label.astype(np.float16)
        if self.transform is not None:
            image = self.transform(image)

        return image, label, item


def get_stanford_datasets(transform_train, transform_val, num_labeled_classes,
                          num_unlabeled_classes, ratio, regenerate=False):
    train_data_dir = os.path.join(scars_root, "cars_train")
    val_data_dir = os.path.join(scars_root, "cars_test")
    dump_path = os.path.join("data/splits", f'stanford-labeled-{num_labeled_classes}'
                                            f'-unlabeled-{num_unlabeled_classes}-ratio-{int(ratio)}.pkl')
    if regenerate:
        generate(ratio, num_labeled_classes, dump_path)

    if not os.path.exists(dump_path):
        assert FileNotFoundError("Please generate label and unlabel index first!!!")

    with open(dump_path, "rb") as f:
        indices = pkl.load(f)
        train_indices_lab, train_indices_unlab, test_indice = indices["lab"], indices["unlab"], indices["test"]

    train_label_dataset = StanfordDataset(train_data_dir, train_indices_lab, transform=transform_train)
    train_unlabel_dataset = StanfordDataset(train_data_dir, train_indices_unlab, transform=transform_train)
    val_unlab_dataset = StanfordDataset(train_data_dir, train_indices_unlab, transform=transform_val)
    test_dataset = StanfordDataset(val_data_dir, test_indice, transform=transform_val)

    labeled_classes = np.arange(50)

    # seen class
    val_indices_seen = np.where(np.isin(np.array(val_unlab_dataset.targets), labeled_classes))[0]
    val_seen_dataset = torch.utils.data.Subset(val_unlab_dataset, val_indices_seen)

    # testset
    test_indices_seen = np.where(np.isin(np.array(test_dataset.targets), labeled_classes))[0]
    test_seen_dataset = torch.utils.data.Subset(test_dataset, test_indices_seen)
    all_datasets = {"train_label_dataset": train_label_dataset, "train_unlabel_dataset": train_unlabel_dataset,
                    "val_dataset": val_unlab_dataset, "val_seen_dataset": val_seen_dataset,
                    "test_dataset": test_dataset, "test_seen_dataset": test_seen_dataset}
    return all_datasets


def generate(ratio, num_labeled_classes, dump_path):
    devkit = io.loadmat(os.path.join(scars_root, "devkit/cars_train_annos.mat"))
    test_devkit = io.loadmat(os.path.join(scars_root, "devkit/cars_test_annos_withlabels.mat"))

    labeled_classes = range(num_labeled_classes)
    train_lab = {}
    train_unlab = {}
    test = {}
    lens = devkit["annotations"].shape[1]
    test_lens = test_devkit["annotations"].shape[1]
    ClassCount = {}
    # count
    for i in range(lens):
        class_idx = devkit["annotations"][0, i][-2][0][0] - 1
        ClassCount[class_idx] = ClassCount.get(class_idx, 0) + 1

    for key, value in ClassCount.items():
        ClassCount[key] = int(value * ratio / 100)

    for i in range(lens):
        image_name = devkit["annotations"][0, i][-1][0]
        class_idx = devkit["annotations"][0, i][-2][0][0] - 1
        if class_idx in labeled_classes and ClassCount[class_idx] > 0:
            train_lab[image_name] = class_idx
            ClassCount[class_idx] -= 1
        else:
            train_unlab[image_name] = class_idx

    for i in range(test_lens):
        image_name = test_devkit["annotations"][0, i][-1][0]
        test[image_name] = test_devkit["annotations"][0, i][-2][0][0] - 1

    with open(dump_path, "wb") as f:
        pkl.dump({"lab": train_lab, "unlab": train_unlab, "test": test}, f)

    print(f"Total dataset lens: {lens}, Train lab lens: {len(train_lab.keys())}")


if __name__ == "__main__":
    test_devkit = io.loadmat(os.path.join(scars_root, "devkit/cars_test_annos_withlabels.mat"))
    test = {}
    test_lens = test_devkit["annotations"].shape[1]
    for i in range(test_lens):
        image_name = test_devkit["annotations"][0, i][-1][0]
        test[image_name] = test_devkit["annotations"][0, i][-2][0][0] - 1
    dump_path = os.path.join("data/splits", f'stanford-labeled-98-unlabeled-98-ratio-50.pkl')

    with open(dump_path, "rb") as f:
        indices = pkl.load(f)

    indices["test"] = test
    with open(dump_path, "wb") as f:
        pkl.dump(indices, f)
