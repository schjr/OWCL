# coding:utf-8
# Author : shijr
# CONTACT: shijr@shanghaitech.edu.cn
# SOFTWARE: VSCode
# FILE : cifar_con.py
# DATE : 2022/11/9 14:14


import torch
from torchvision.datasets import CIFAR10, CIFAR100
import numpy as np
import os
import pickle as pkl
from config.config import cifar_10_root, cifar_100_root

import ipdb

class CustomCIFAR10(CIFAR10):

    def __init__(self, *args, **kwargs):
        super(CustomCIFAR10, self).__init__(*args, **kwargs)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label, uq_idx

    def __len__(self):
        return len(self.targets)


# Given CIFAR100 Dataset, we take the first 
# num_labelled_classes classes as labelled classes.
# We then take a ratio of val_ratio of data in each 
# classes in training set as validation data. The rest 
# are training data. The seperation is fair in terms of 
# each class.    

class CustomCIFAR100(CIFAR100):

    def __init__(self, *args, **kwargs):
        super(CustomCIFAR100, self).__init__(*args, **kwargs)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label, uq_idx

    def __len__(self):
        return len(self.targets)


def get_cifar_continual_datasets(transform_train, transform_val, num_labeled_classes,
                       num_unlabeled_classes, dataset="CIFAR100",
                       regenerate=False, val_ratio=10, choice_ratio=100, label_val_ratio=10):
    if dataset == "CIFAR10":
        train_dataset = CustomCIFAR10(cifar_10_root, train=True, transform=transform_train)
        val_dataset = CustomCIFAR10(cifar_10_root, train=True, transform=transform_val)
        test_dataset = CustomCIFAR10(cifar_10_root, train=False, transform=transform_val)

    elif dataset == "CIFAR100":
        train_dataset = CustomCIFAR100(cifar_100_root, train=True, transform=transform_train)
        val_dataset = CustomCIFAR100(cifar_100_root, train=True, transform=transform_val)
        test_dataset = CustomCIFAR100(cifar_100_root, train=False, transform=transform_val)

    else:
        raise NotImplementedError("cifar10 or cifar100")

    # split dataset
    dump_path = os.path.join("data/splits", f'{dataset}-labeled-{num_labeled_classes}-label_val_ratio-{label_val_ratio}'
                                            f'-unlabeled-{num_unlabeled_classes}-choice_ratio-{choice_ratio}-replay.pkl')
    labeled_classes = range(num_labeled_classes)

    if regenerate:
        train_indices_lab = []
        val_indices_lab = []
        train_indices_unlab = []
        val_indices_unlab = []
        for lc in labeled_classes:
            idx = np.nonzero(np.array(train_dataset.targets) == lc)[0]  # labelled里面每一类的位置
            all_val_indices = np.random.choice(idx, int(label_val_ratio * len(idx) / 100), False)
            val_indices = np.random.choice(all_val_indices, int(choice_ratio * len(all_val_indices) / 100), False)
            val_indices_lab.extend(val_indices)

            all_train_indices = np.array(list(set(idx) - set(val_indices)))
            train_indices = np.random.choice(all_train_indices, int(choice_ratio * len(all_train_indices) / 100), False)
            train_indices_lab.extend(train_indices)
        
        total_num_classes = num_labeled_classes + num_unlabeled_classes
        for ulc in range(num_labeled_classes, total_num_classes):
            idx = np.nonzero(np.array(train_dataset.targets) == ulc)[0]  # unlabelled里面每一类的位置
            val_indices = np.random.choice(idx, int(val_ratio * len(idx) / 100), False)
            val_indices_unlab.extend(val_indices)
            train_indices = np.array(list(set(idx) - set(val_indices)))
            train_indices_unlab.extend(train_indices)

        train_indices_lab = np.array(train_indices_lab)
        train_indices_unlab = np.array(train_indices_unlab)
        val_indices_lab = np.array(val_indices_lab)
        val_indices_unlab = np.array(val_indices_unlab)

        with open(dump_path, "wb") as f:
            pkl.dump({"train_lab": train_indices_lab, "train_unlab": train_indices_unlab, "val_lab" : val_indices_lab, "val_unlab": val_indices_unlab}, f)
    else:
        if not os.path.exists(dump_path):
            raise FileNotFoundError(f"Dump_path does not exists: {dump_path}")

        with open(dump_path, "rb") as f:
            indices = pkl.load(f)
        train_indices_lab, train_indices_unlab, val_indices_lab, val_indices_unlab = indices["train_lab"], indices["train_unlab"], indices["val_lab"], indices["val_unlab"]

    train_label_dataset = torch.utils.data.Subset(train_dataset, train_indices_lab)
    train_unlabel_dataset = torch.utils.data.Subset(train_dataset, train_indices_unlab)
    val_unlab_dataset = torch.utils.data.Subset(val_dataset, val_indices_unlab)
    val_seen_dataset = torch.utils.data.Subset(val_dataset, val_indices_lab)

    all_datasets = {"train_label_dataset": train_label_dataset, "train_unlabel_dataset": train_unlabel_dataset,
                    "val_dataset": val_unlab_dataset, "val_seen_dataset": val_seen_dataset,
                    "test_dataset": test_dataset}
    return all_datasets


if __name__ == "__main__":
    pass


# def get_cifar_continual_datasets(transform_train, transform_val, num_labeled_classes,
#                                  num_unlabeled_classes, dataset="CIFAR100",
#                                  regenerate=False, val_ratio=10):
#     if dataset == "CIFAR10":
#         train_dataset = CustomCIFAR10(cifar_10_root, train=True, transform=transform_train)
#         val_dataset = CustomCIFAR10(cifar_10_root, train=True, transform=transform_val)
#         test_dataset = CustomCIFAR10(cifar_10_root, train=False, transform=transform_val)
#
#     elif dataset == "CIFAR100":
#         train_dataset = CustomCIFAR100(cifar_100_root, train=True, transform=transform_train)
#         val_dataset = CustomCIFAR100(cifar_100_root, train=True, transform=transform_val)
#         test_dataset = CustomCIFAR100(cifar_100_root, train=False, transform=transform_val)
#
#     else:
#         raise NotImplementedError("cifar10 or cifar100")
#
#     # split dataset
#     dump_path = os.path.join("data/splits", f'{dataset}-labeled-{num_labeled_classes}'
#                                             f'-unlabeled-{num_unlabeled_classes}-continual.pkl')
#     labeled_classes = range(num_labeled_classes)
#
#     if regenerate:
#         train_indices_lab = []
#         val_indices_lab = []
#         train_indices_unlab = []
#         val_indices_unlab = []
#         for lc in labeled_classes:
#             idx = np.nonzero(np.array(train_dataset.targets) == lc)[0]  # labelled里面每一类的位置
#             val_indices = np.random.choice(idx, int(val_ratio * len(idx) / 100), False)
#             val_indices_lab.extend(val_indices)
#             train_indices = np.array(list(set(idx) - set(val_indices)))
#             train_indices_lab.extend(train_indices)
#
#         total_num_classes = num_labeled_classes + num_unlabeled_classes
#         for ulc in range(num_labeled_classes, total_num_classes):
#             idx = np.nonzero(np.array(train_dataset.targets) == ulc)[0]  # unlabelled里面每一类的位置
#             val_indices = np.random.choice(idx, int(val_ratio * len(idx) / 100), False)
#             val_indices_unlab.extend(val_indices)
#             train_indices = np.array(list(set(idx) - set(val_indices)))
#             train_indices_unlab.extend(train_indices)
#
#         train_indices_lab = np.array(train_indices_lab)
#         train_indices_unlab = np.array(train_indices_unlab)
#         val_indices_lab = np.array(val_indices_lab)
#         val_indices_unlab = np.array(val_indices_unlab)
#
#         with open(dump_path, "wb") as f:
#             pkl.dump({"train_lab": train_indices_lab, "train_unlab": train_indices_unlab, "val_lab": val_indices_lab,
#                       "val_unlab": val_indices_unlab}, f)
#     else:
#         if not os.path.exists(dump_path):
#             raise FileNotFoundError(f"Dump_path does not exists: {dump_path}")
#
#         with open(dump_path, "rb") as f:
#             indices = pkl.load(f)
#         train_indices_lab, train_indices_unlab, val_indices_lab, val_indices_unlab = indices["train_lab"], indices[
#             "train_unlab"], indices["val_lab"], indices["val_unlab"]
#
#     train_label_dataset = torch.utils.data.Subset(train_dataset, train_indices_lab)
#     train_unlabel_dataset = torch.utils.data.Subset(train_dataset, train_indices_unlab)
#     val_unlab_dataset = torch.utils.data.Subset(val_dataset, val_indices_unlab)
#     val_seen_dataset = torch.utils.data.Subset(val_dataset, val_indices_lab)
#
#     all_datasets = {"train_label_dataset": train_label_dataset, "train_unlabel_dataset": train_unlabel_dataset,
#                     "val_dataset": val_unlab_dataset, "val_seen_dataset": val_seen_dataset,
#                     "test_dataset": test_dataset}
#     return all_datasets
